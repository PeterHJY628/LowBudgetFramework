"""
Generate all visualizations for Experiment 1.

1. Budget (fraction) vs downstream validation loss curve
2. AL performance curves under different sampling methods
3. t-SNE of representations at the turning-point budget

Usage:
    python visualize.py --seed 1
    python visualize.py --seed 1 --seeds 1,2,3   (aggregate multiple seeds)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.manifold import TSNE

from core.helper_functions import get_dataset_by_name
from core.data import normalize

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="Primary seed to visualize")
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds for aggregation (e.g. 1,2,3)")
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--data_folder", type=str, default=None,
                    help="Data folder (required for t-SNE)")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--tsne_samples", type=int, default=5000,
                    help="Number of samples for t-SNE visualization")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()

seed_list = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
primary_seed = seed_list[0]
seed_dir = os.path.join(args.results_dir, f"seed_{primary_seed}")
plot_dir = os.path.join(seed_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

AGENT_COLORS = {
    "RandomAgent": "#7f8c8d",
    "MarginScore": "#2980b9",
    "ShannonEntropy": "#27ae60",
    "LeastConfident": "#e74c3c",
}
AGENT_LABELS = {
    "RandomAgent": "Random",
    "MarginScore": "Margin",
    "ShannonEntropy": "Entropy",
    "LeastConfident": "LeastConfident",
}


# ========================================================================
# Plot 1: Budget (fraction) vs downstream validation loss
# ========================================================================
def plot_budget_vs_loss():
    print("Generating Plot 1: Budget vs Validation Loss...")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    all_fracs, all_losses, all_accs = [], [], []
    for seed in seed_list:
        eval_path = os.path.join(args.results_dir, f"seed_{seed}",
                                 "downstream_eval", "eval_results.json")
        if not os.path.exists(eval_path):
            print(f"  Warning: {eval_path} not found, skipping seed {seed}")
            continue
        with open(eval_path) as f:
            data = json.load(f)

        fracs = [r["fraction"] for r in data["eval_results"]]
        losses = [r["val_loss"] for r in data["eval_results"]]
        accs = [r["val_acc"] for r in data["eval_results"]]
        all_fracs.append(fracs)
        all_losses.append(losses)
        all_accs.append(accs)

    if not all_fracs:
        print("  No data found. Skipping plot 1.")
        return

    fracs = np.array(all_fracs[0])
    mean_loss = np.mean(all_losses, axis=0)
    mean_acc = np.mean(all_accs, axis=0)

    # Load turning point info from primary seed
    with open(os.path.join(seed_dir, "downstream_eval", "eval_results.json")) as f:
        primary_data = json.load(f)
    tp_idx = primary_data["turning_point_index"]
    tp_frac = primary_data["turning_point_fraction"]

    fracs_pct = fracs * 100

    color_loss = "#e74c3c"
    color_acc = "#2980b9"

    ax1.plot(fracs_pct, mean_loss, 'o-', color=color_loss, linewidth=2, markersize=5, label="Val Loss")
    if len(all_losses) > 1:
        std_loss = np.std(all_losses, axis=0)
        ax1.fill_between(fracs_pct, mean_loss - std_loss, mean_loss + std_loss,
                         color=color_loss, alpha=0.15)
    ax1.axvline(x=tp_frac * 100, color="#e67e22", linestyle="--", linewidth=2,
                label=f"Turning Point ({tp_frac:.0%})")
    ax1.set_xlabel("SimCLR Pre-training Data Budget (%)", fontsize=12)
    ax1.set_ylabel("Downstream Validation Loss", fontsize=12, color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.plot(fracs_pct, mean_acc, 's--', color=color_acc, linewidth=2, markersize=5, label="Val Acc")
    if len(all_accs) > 1:
        std_acc = np.std(all_accs, axis=0)
        ax2.fill_between(fracs_pct, mean_acc - std_acc, mean_acc + std_acc,
                         color=color_acc, alpha=0.15)
    ax2.set_ylabel("Downstream Validation Accuracy", fontsize=12, color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title("SimCLR Pre-training Budget vs Downstream Performance", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "budget_vs_loss.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved to {out_path}")


# ========================================================================
# Plot 2: AL performance curves
# ========================================================================
def plot_al_performance():
    print("Generating Plot 2: AL Performance Curves...")

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    agents_found = []
    for agent_name in AGENT_COLORS:
        all_n_labeled = []
        all_accs = []
        all_losses = []

        for seed in seed_list:
            result_path = os.path.join(args.results_dir, f"seed_{seed}",
                                       "al_results", agent_name, "al_results.json")
            if not os.path.exists(result_path):
                continue
            with open(result_path) as f:
                data = json.load(f)
            all_n_labeled.append(data["n_labeled"])
            all_accs.append(data["accuracies"])
            all_losses.append(data["losses"])

        if not all_accs:
            continue

        agents_found.append(agent_name)
        color = AGENT_COLORS[agent_name]
        label = AGENT_LABELS.get(agent_name, agent_name)

        # Use the shortest run length for alignment
        min_len = min(len(a) for a in all_accs)
        n_labeled = np.array(all_n_labeled[0][:min_len])
        accs = np.array([a[:min_len] for a in all_accs])
        losses_arr = np.array([l[:min_len] for l in all_losses])

        mean_acc = np.mean(accs, axis=0)
        mean_loss = np.mean(losses_arr, axis=0)

        ax_acc.plot(n_labeled, mean_acc, '-', color=color, linewidth=2, label=label)
        ax_loss.plot(n_labeled, mean_loss, '-', color=color, linewidth=2, label=label)

        if len(accs) > 1:
            std_acc = np.std(accs, axis=0)
            std_loss = np.std(losses_arr, axis=0)
            ax_acc.fill_between(n_labeled, mean_acc - std_acc, mean_acc + std_acc,
                                color=color, alpha=0.12)
            ax_loss.fill_between(n_labeled, mean_loss - std_loss, mean_loss + std_loss,
                                 color=color, alpha=0.12)

    if not agents_found:
        print("  No AL results found. Skipping plot 2.")
        return

    for ax, ylabel, title in [
        (ax_acc, "Test Accuracy", "Active Learning: Accuracy vs Labeled Samples"),
        (ax_loss, "Test Loss", "Active Learning: Loss vs Labeled Samples"),
    ]:
        ax.set_xlabel("Number of Labeled Samples", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    fig.tight_layout()
    out_path = os.path.join(plot_dir, "al_performance.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved to {out_path}")


# ========================================================================
# Plot 3: t-SNE of representations at turning point
# ========================================================================
def plot_tsne():
    print("Generating Plot 3: t-SNE of Turning-Point Representations...")

    if args.data_folder is None:
        print("  --data_folder required for t-SNE. Skipping.")
        return

    eval_path = os.path.join(seed_dir, "downstream_eval", "eval_results.json")
    if not os.path.exists(eval_path):
        print(f"  {eval_path} not found. Skipping t-SNE.")
        return

    with open(eval_path) as f:
        eval_data = json.load(f)
    tp_frac = eval_data["turning_point_fraction"]
    tp_pct = int(round(tp_frac * 100))
    tp_ckpt = os.path.join(seed_dir, "simclr_checkpoints",
                           f"frac_{tp_pct:03d}", "model.pth.tar")

    if not os.path.exists(tp_ckpt):
        print(f"  Checkpoint {tp_ckpt} not found. Skipping t-SNE.")
        return

    config_path = args.config or os.path.join(os.path.dirname(__file__), '..', 'configs', 'cifar10.yaml')
    with open(config_path) as f:
        config = yaml.load(f, yaml.Loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DatasetClass = get_dataset_by_name("cifar10")
    dataset = DatasetClass(args.data_folder, config,
                           np.random.default_rng(primary_seed), encoded=False)
    config["n_classes"] = dataset.n_classes

    encoder = dataset.get_pretext_encoder(config, seed=primary_seed)
    encoder.load_state_dict(torch.load(tp_ckpt, map_location="cpu"))
    encoder = encoder.to(device).eval()

    # Sample a subset for t-SNE
    rng = np.random.default_rng(primary_seed + 3000)
    n = min(args.tsne_samples, len(dataset.x_test))
    ids = rng.choice(len(dataset.x_test), n, replace=False)
    x_subset = dataset.x_test[ids].to(device)
    y_subset = dataset.y_test[ids]
    labels = torch.argmax(y_subset, dim=1).numpy() if y_subset.dim() > 1 else y_subset.numpy()

    # Encode through backbone
    feats = []
    with torch.no_grad():
        for i in range(0, len(x_subset), 256):
            batch = x_subset[i:i+256]
            f = encoder.backbone(batch)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0)

    # Run t-SNE
    print(f"  Running t-SNE on {len(feats)} samples...")
    tsne = TSNE(n_components=2, random_state=primary_seed, perplexity=30, n_iter=1000)
    embeddings = tsne.fit_transform(feats)

    # Plot
    class_names = [f"Class {i}" for i in range(dataset.n_classes)]
    cifar10_names = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    if dataset.n_classes == 10:
        class_names = cifar10_names

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10", dataset.n_classes)

    for c in range(dataset.n_classes):
        mask = labels == c
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[cmap(c)], s=8, alpha=0.6, label=class_names[c])

    ax.set_title(f"t-SNE of SimCLR Representations (Budget: {tp_frac:.0%} data)", fontsize=13)
    ax.legend(fontsize=9, markerscale=3, loc="best", ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    out_path = os.path.join(plot_dir, "tsne_turning_point.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved to {out_path}")


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    plot_budget_vs_loss()
    plot_al_performance()
    plot_tsne()
    print(f"\nAll plots saved to {plot_dir}/")
