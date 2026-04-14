"""
Evaluate downstream classification for all trained SimCLR fraction models.
Finds the budget turning point where loss stabilizes.

Usage:
    python evaluate_downstream.py --data_folder ../data_lib --seed 1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from core.helper_functions import get_dataset_by_name, EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--eval_epochs", type=int, default=100,
                    help="Max epochs for downstream linear classifier training")
parser.add_argument("--val_size", type=int, default=2000,
                    help="Size of the fixed validation set for downstream eval")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = args.config or os.path.join(os.path.dirname(__file__), '..', 'configs', 'cifar10.yaml')
with open(config_path) as f:
    config = yaml.load(f, yaml.Loader)

seed_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
ckpt_base = os.path.join(seed_dir, "simclr_checkpoints")
eval_dir = os.path.join(seed_dir, "downstream_eval")
os.makedirs(eval_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

DatasetClass = get_dataset_by_name("cifar10")
dataset = DatasetClass(args.data_folder, config, np.random.default_rng(args.seed), encoded=False)
config["n_classes"] = dataset.n_classes

# --- Build fixed train/val/test splits for downstream evaluation ---
# Use dataset.x_train / y_train (already normalized tensors)
rng = np.random.default_rng(args.seed + 1000)  # offset to avoid collision with SimCLR subsampling
all_ids = np.arange(len(dataset.x_train))
rng.shuffle(all_ids)
val_ids = all_ids[:args.val_size]
train_ids = all_ids[args.val_size:]

x_down_train = dataset.x_train[train_ids].to(device)
y_down_train = dataset.y_train[train_ids].to(device)
x_down_val = dataset.x_train[val_ids].to(device)
y_down_val = dataset.y_train[val_ids].to(device)
x_test = dataset.x_test.to(device)
y_test = dataset.y_test.to(device)

print(f"Downstream splits: train={len(x_down_train)}, val={len(x_down_val)}, test={len(x_test)}")


def encode_data(encoder, x, batch_size=256):
    """Encode data through the SimCLR backbone, returning feature vectors."""
    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            f = encoder.backbone(batch)
            feats.append(f)
    return torch.cat(feats, dim=0)


def train_and_evaluate_linear(x_train, y_train, x_val, y_val, x_test, y_test,
                               feature_dim, n_classes, max_epochs=100, patience=10):
    """
    Train ONE linear classifier on encoded features using val loss for early stopping.
    Return val_loss, val_acc, test_acc from the same classifier.
    """
    head = nn.Linear(feature_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=patience, lower_is_better=True)

    y_train_cls = torch.argmax(y_train, dim=1) if y_train.dim() > 1 else y_train
    y_val_cls = torch.argmax(y_val, dim=1) if y_val.dim() > 1 else y_val
    y_test_cls = torch.argmax(y_test, dim=1) if y_test.dim() > 1 else y_test

    train_loader = DataLoader(TensorDataset(x_train, y_train_cls),
                              batch_size=256, shuffle=True)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(max_epochs):
        head.train()
        for bx, by in train_loader:
            logits = head(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(x_val)
            val_loss = criterion(val_logits, y_val_cls).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if early_stop.check_stop(val_loss):
            break

    # Restore best model and compute final metrics
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_pred = torch.argmax(head(x_val), dim=1)
        val_acc = (val_pred == y_val_cls).float().mean().item()

        test_logits = head(x_test)
        test_loss = criterion(test_logits, y_test_cls).item()
        test_pred = torch.argmax(test_logits, dim=1)
        test_acc = (test_pred == y_test_cls).float().mean().item()

    return best_val_loss, val_acc, test_loss, test_acc


def find_turning_point(fractions, losses):
    """
    Find the elbow/turning point using maximum curvature on the loss curve.
    Returns the index of the turning point.
    """
    if len(fractions) < 3:
        return 0

    x = np.array(fractions, dtype=float)
    y = np.array(losses, dtype=float)

    # Normalize to [0, 1] for fair curvature calculation
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

    # Compute discrete second derivative (curvature proxy)
    curvatures = []
    for i in range(1, len(x_norm) - 1):
        dx1 = x_norm[i] - x_norm[i-1]
        dx2 = x_norm[i+1] - x_norm[i]
        dy1 = (y_norm[i] - y_norm[i-1]) / dx1
        dy2 = (y_norm[i+1] - y_norm[i]) / dx2
        curvature = abs(dy2 - dy1) / ((dx1 + dx2) / 2)
        curvatures.append(curvature)

    # The turning point is where curvature is maximum
    tp_idx = np.argmax(curvatures) + 1  # +1 because curvatures starts at index 1
    return tp_idx


# --- Evaluate each fraction checkpoint ---
frac_dirs = sorted([d for d in os.listdir(ckpt_base)
                    if d.startswith("frac_") and os.path.isdir(os.path.join(ckpt_base, d))])

if not frac_dirs:
    print(f"No checkpoints found in {ckpt_base}")
    sys.exit(1)

results = []
for frac_dir in frac_dirs:
    frac_pct = int(frac_dir.split("_")[1])
    fraction = frac_pct / 100.0
    ckpt_path = os.path.join(ckpt_base, frac_dir, "model.pth.tar")

    if not os.path.exists(ckpt_path):
        print(f"Skipping {frac_dir}: no checkpoint found")
        continue

    print(f"\n--- Evaluating fraction={fraction:.2f} ({frac_dir}) ---")

    # Build and load SimCLR model
    encoder = dataset.get_pretext_encoder(config, seed=args.seed)
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder = encoder.to(device)

    # Encode datasets through the backbone (not the projection head)
    feat_train = encode_data(encoder, x_down_train)
    feat_val = encode_data(encoder, x_down_val)
    feat_test = encode_data(encoder, x_test)
    feature_dim = feat_train.shape[1]

    # Train ONE linear classifier, evaluate on both val and test
    val_loss, val_acc, test_loss, test_acc = train_and_evaluate_linear(
        feat_train, y_down_train, feat_val, y_down_val, feat_test, y_test,
        feature_dim, dataset.n_classes, max_epochs=args.eval_epochs
    )

    results.append({
        "fraction": fraction,
        "fraction_pct": frac_pct,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "feature_dim": feature_dim,
    })
    print(f"  val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
          f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

# --- Find turning point ---
fractions = [r["fraction"] for r in results]
val_losses = [r["val_loss"] for r in results]
tp_idx = find_turning_point(fractions, val_losses)
turning_point = results[tp_idx]

print(f"\n{'='*60}")
print(f"Turning point: fraction={turning_point['fraction']:.0%} "
      f"(val_loss={turning_point['val_loss']:.4f}, val_acc={turning_point['val_acc']:.4f})")
print(f"{'='*60}")

# --- Save results ---
output = {
    "seed": args.seed,
    "eval_results": results,
    "turning_point_index": tp_idx,
    "turning_point_fraction": turning_point["fraction"],
    "turning_point_fraction_pct": turning_point["fraction_pct"],
}
with open(os.path.join(eval_dir, "eval_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {eval_dir}/eval_results.json")
