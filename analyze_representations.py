"""
Phase 4: Representation Mechanism Analysis
============================================
Loads per-round checkpoints from Phase 1 / Phase 3, extracts penultimate-layer
features on a shared validation subset, then computes:
  - Linear CKA  (scratch_t  vs  pretrained)
  - Wasserstein / Sinkhorn distance  (scratch_t  vs  pretrained)

Usage:
    python analyze_representations.py \
        --config configs/cifar10_coldstart.yaml \
        --data_folder data_lib \
        --dataset cifar10 \
        --query_size 20 \
        --scratch_agent ShannonEntropy_coldstart \
        --pretrained_agent ShannonEntropy_coldstart_pretrained \
        --pretrained_config configs/cifar10_coldstart_pretrained.yaml \
        --run_id 1 \
        --val_samples 1000 \
        --rounds 0 1 2 3 4 5 6 7 8 9 10 \
        --output_dir results/coldstart
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import experiment_util  # noqa: F401 — sets up sys.path and device


# ── Feature extraction ─────────────────────────────────────────────────────

def build_model_from_config(config_path, dataset_obj, model_rng):
    """Construct a classifier matching the config (without loading weights)."""
    with open(config_path) as f:
        config = yaml.load(f, yaml.Loader)
    from classifiers.classifier import construct_model
    model, _ = construct_model(model_rng, dataset_obj, config["classifier"])
    return model


@torch.no_grad()
def extract_features(model, x, batch_size=256):
    """Run x through model._encode to get penultimate-layer features."""
    model.eval()
    device = next(model.parameters()).device
    feats = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(device)
        f = model._encode(batch)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


def load_checkpoint_model(ckpt_path, config_path, dataset_obj, model_seed=1):
    """Load a per-round checkpoint and return the ready model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_rng = torch.Generator()
    model_rng.manual_seed(model_seed)
    model = build_model_from_config(config_path, dataset_obj, model_rng)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ── Metrics ────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """
    Linear CKA between feature matrices X, Y  (n × d_x, n × d_y).
    Kornblith et al., 2019.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def sinkhorn_distance(X, Y, reg=0.05, max_iter=100):
    """
    Entropy-regularised Wasserstein distance via Sinkhorn iteration.
    Falls back to a simple MMD proxy if input is too large.
    """
    try:
        import ot
        n = len(X)
        a = np.ones(n) / n
        b = np.ones(n) / n
        M = ot.dist(X, Y, metric="sqeuclidean")
        return float(ot.sinkhorn2(a, b, M, reg=reg, numItermax=max_iter))
    except ImportError:
        pass

    # Fallback: simple maximum-mean-discrepancy with RBF kernel
    from scipy.spatial.distance import cdist
    n = min(len(X), 2000)
    idx = np.random.choice(len(X), n, replace=False)
    X, Y = X[idx], Y[idx]
    sigma = np.median(cdist(X, X, "sqeuclidean"))
    if sigma == 0:
        sigma = 1.0
    Kxx = np.exp(-cdist(X, X, "sqeuclidean") / (2 * sigma))
    Kyy = np.exp(-cdist(Y, Y, "sqeuclidean") / (2 * sigma))
    Kxy = np.exp(-cdist(X, Y, "sqeuclidean") / (2 * sigma))
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: representation analysis")
    parser.add_argument("--config", default="configs/cifar10_coldstart.yaml")
    parser.add_argument("--pretrained_config", default="configs/cifar10_coldstart_pretrained.yaml")
    parser.add_argument("--data_folder", default="data_lib")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--query_size", type=int, default=20)
    parser.add_argument("--scratch_agent", default="ShannonEntropy_coldstart")
    parser.add_argument("--pretrained_agent", default="ShannonEntropy_coldstart_pretrained")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="Number of test images for feature extraction")
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=list(range(11)))
    parser.add_argument("--output_dir", default="results/coldstart")
    parser.add_argument("--pool_seed", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset (for x_test and shape info) ──
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)
    from core.helper_functions import get_dataset_by_name
    pool_rng = np.random.default_rng(args.pool_seed + args.run_id)
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset_obj = DatasetClass(args.data_folder, config, pool_rng, encoded=False)

    x_eval = dataset_obj.x_test[:args.val_samples]
    print(f"Evaluation subset: {x_eval.shape}")

    init_dir = f"init{dataset_obj.initial_points_per_class}"

    # ── Reference features from pretrained model (fixed across rounds) ──
    pretrained_ckpt_dir = os.path.join(
        "runs", dataset_obj.name, init_dir, str(args.query_size),
        args.pretrained_agent, f"run_{args.run_id}", "checkpoints")

    if os.path.isdir(pretrained_ckpt_dir):
        best_round = max(args.rounds)
        pretrained_ckpt = os.path.join(pretrained_ckpt_dir, f"round_{best_round:03d}.pt")
        print(f"Loading pretrained reference: {pretrained_ckpt}")
        pretrained_model = load_checkpoint_model(
            pretrained_ckpt, args.pretrained_config, dataset_obj,
            model_seed=args.model_seed + args.run_id)
        pretrained_model.to(experiment_util.device)
        feat_ref = extract_features(pretrained_model, x_eval)
        del pretrained_model
    else:
        print(f"[WARN] Pretrained checkpoints not found at {pretrained_ckpt_dir}")
        print("       Will compute CKA / OT of scratch model against itself at round 0.")
        feat_ref = None

    # ── Extract features per scratch round ──
    scratch_ckpt_dir = os.path.join(
        "runs", dataset_obj.name, init_dir, str(args.query_size),
        args.scratch_agent, f"run_{args.run_id}", "checkpoints")

    cka_values, ot_values, valid_rounds = [], [], []
    for r in args.rounds:
        ckpt_path = os.path.join(scratch_ckpt_dir, f"round_{r:03d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  round {r}: checkpoint not found, skipping")
            continue
        model = load_checkpoint_model(
            ckpt_path, args.config, dataset_obj,
            model_seed=args.model_seed + args.run_id)
        model.to(experiment_util.device)
        feat_scratch = extract_features(model, x_eval)
        del model

        if feat_ref is None:
            if r == args.rounds[0]:
                feat_ref = feat_scratch.copy()
            else:
                feat_ref = feat_ref  # use round-0 scratch as baseline

        cka = linear_cka(feat_scratch, feat_ref)
        ot_dist = sinkhorn_distance(feat_scratch, feat_ref)
        cka_values.append(cka)
        ot_values.append(ot_dist)
        valid_rounds.append(r)
        print(f"  round {r:>3d}  |  CKA = {cka:.4f}  |  OT = {ot_dist:.6f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(valid_rounds, cka_values, "o-", color="tab:blue", linewidth=2)
    axes[0].set_xlabel("AL Round")
    axes[0].set_ylabel("Linear CKA")
    axes[0].set_title("CKA(scratch_t, pretrained)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(valid_rounds, ot_values, "s-", color="tab:red", linewidth=2)
    axes[1].set_xlabel("AL Round")
    axes[1].set_ylabel("Sinkhorn Distance")
    axes[1].set_title("OT(scratch_t, pretrained)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "representation_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close(fig)

    # ── Save raw numbers ──
    import csv
    csv_path = os.path.join(args.output_dir, "representation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "cka", "sinkhorn_distance"])
        for r, c, o in zip(valid_rounds, cka_values, ot_values):
            writer.writerow([r, f"{c:.6f}", f"{o:.8f}"])
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
