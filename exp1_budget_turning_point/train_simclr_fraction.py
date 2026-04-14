"""
Train SimCLR on a specified fraction of CIFAR-10 data.

Usage:
    python train_simclr_fraction.py --data_folder ../data_lib --seed 1 --fraction 0.05
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import yaml
import torch
import numpy as np
from torch.utils.data import Subset

from core.helper_functions import get_dataset_by_name
from sim_clr.data import AugmentedDataset, get_train_dataloader_for_dataset, get_validation_dataloader_for_dataset
from sim_clr.loss import get_loss_for_dataset
from sim_clr.optim import get_optimizer_for_dataset
from sim_clr.training import adjust_learning_rate, simclr_train
from sim_clr.evaluate import linear_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--fraction", type=float, required=True,
                    help="Fraction of CIFAR-10 to use (0.05, 0.10, ..., 1.0)")
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--max_epochs", type=int, default=None,
                    help="Override max training epochs (default: from config)")
parser.add_argument("--patience", type=int, default=30,
                    help="Early stopping patience (epochs without loss improvement)")
parser.add_argument("--config", type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = args.config or os.path.join(os.path.dirname(__file__), '..', 'configs', 'cifar10.yaml')
with open(config_path) as f:
    config = yaml.load(f, yaml.Loader)

if args.max_epochs is not None:
    config["pretext_training"]["epochs"] = args.max_epochs

seed_dir = os.path.join(args.results_dir, f"seed_{args.seed}")
frac_pct = int(round(args.fraction * 100))
frac_str = f"frac_{frac_pct:03d}"
ckpt_dir = os.path.join(seed_dir, "simclr_checkpoints", frac_str)
os.makedirs(ckpt_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

DatasetClass = get_dataset_by_name("cifar10")
dataset = DatasetClass(args.data_folder, config, np.random.default_rng(args.seed), encoded=False)
config["n_classes"] = dataset.n_classes

model = dataset.get_pretext_encoder(config, seed=args.seed)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
model = model.to(device)

train_dataset, val_dataset = dataset.load_pretext_data()

# --- Subsample training data to the specified fraction ---
rng = np.random.default_rng(args.seed)
n_total = len(train_dataset)
n_subset = max(1, int(n_total * args.fraction))
indices = rng.choice(n_total, n_subset, replace=False).tolist()

subset_data = train_dataset.data[indices]
subset_targets = train_dataset.targets[indices] if isinstance(train_dataset.targets, torch.Tensor) \
    else torch.tensor(train_dataset.targets)[indices]

train_dataset.data = subset_data
train_dataset.targets = subset_targets
print(f"SimCLR training: fraction={args.fraction}, samples={len(train_dataset)}/{n_total}")

# --- Set transforms and wrap ---
train_dataset.transform = dataset.get_pretext_transforms(config)
val_dataset.transform = dataset.get_pretext_validation_transforms(config)
train_dataset = AugmentedDataset(train_dataset)
val_dataset = AugmentedDataset(val_dataset)

train_loader = get_train_dataloader_for_dataset(config, train_dataset)
val_loader = get_validation_dataloader_for_dataset(config, val_dataset)

# --- For linear evaluation during training ---
base_dataset, _ = dataset.load_pretext_data()
base_dataset.transform = dataset.get_pretext_validation_transforms(config)
base_dataset = AugmentedDataset(base_dataset)
base_loader = get_validation_dataloader_for_dataset(config, base_dataset)

criterion = get_loss_for_dataset(config, device).to(device)
optimizer = get_optimizer_for_dataset(config, model)

max_epochs = config["pretext_training"]["epochs"]
loss_history = []
acc_history = []

# --- Convergence detection on training loss ---
best_loss = float('inf')
epochs_no_improve = 0
converged_epoch = max_epochs
best_state = None

for epoch in range(max_epochs):
    lr = adjust_learning_rate(config, optimizer, epoch)
    loss = simclr_train(train_loader, model, criterion, optimizer, epoch, device)
    loss_history.append(loss)

    # Track convergence: if loss hasn't improved for `patience` epochs, stop
    if loss < best_loss - 1e-4:
        best_loss = loss
        epochs_no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 50 == 0 or epoch == max_epochs - 1:
        acc = linear_evaluate(base_loader, val_loader, model,
                              config["pretext_encoder"]["feature_dim"],
                              dataset.n_classes, device)
        acc_history.append({"epoch": epoch + 1, "acc": acc})
        model.train()
        print(f"[Epoch {epoch+1}/{max_epochs}] loss={loss:.4f}, lr={lr:.5f}, "
              f"linear_acc={acc:.2f}%, no_improve={epochs_no_improve}/{args.patience}")
    else:
        print(f"[Epoch {epoch+1}/{max_epochs}] loss={loss:.4f}, lr={lr:.5f}, "
              f"no_improve={epochs_no_improve}/{args.patience}")

    if epochs_no_improve >= args.patience:
        converged_epoch = epoch + 1
        print(f"\nConverged at epoch {converged_epoch} (no improvement for {args.patience} epochs)")
        break

# Restore best model if early stopped
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# Final linear evaluation
final_acc = linear_evaluate(base_loader, val_loader, model,
                            config["pretext_encoder"]["feature_dim"],
                            dataset.n_classes, device)
print(f"Final linear eval accuracy: {final_acc:.2f}%")

# --- Save checkpoint and training log ---
ckpt_path = os.path.join(ckpt_dir, "model.pth.tar")
torch.save(model.state_dict(), ckpt_path)

log = {
    "seed": args.seed,
    "fraction": args.fraction,
    "n_samples": n_subset,
    "max_epochs": max_epochs,
    "converged_epoch": converged_epoch,
    "patience": args.patience,
    "final_loss": loss_history[-1],
    "best_loss": best_loss,
    "final_linear_acc": final_acc,
    "loss_history": loss_history,
    "acc_history": acc_history,
}
with open(os.path.join(ckpt_dir, "train_log.json"), "w") as f:
    json.dump(log, f, indent=2)

print(f"Checkpoint saved to {ckpt_path}")
print(f"Training: {converged_epoch}/{max_epochs} epochs, best_loss={best_loss:.4f}")
