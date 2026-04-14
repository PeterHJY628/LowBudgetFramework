"""
Run active learning from the turning-point SimCLR model.

Encodes CIFAR-10 with the turning-point encoder, then runs AL
with a specified agent (random / margin / entropy / leastconfident)
using query_size = 5% of total data, until all data is labeled.

Usage:
    python run_al.py --data_folder ../data_lib --seed 1 --agent random
    python run_al.py --data_folder ../data_lib --seed 1 --agent margin
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import math
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from core.helper_functions import get_dataset_by_name, get_agent_by_name, EarlyStopping
from core.data import BaseDataset, normalize, to_torch
from classifiers.classifier import construct_model

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--agent", type=str, required=True,
                    choices=["random", "margin", "entropy", "leastconfident"])
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--query_frac", type=float, default=0.05,
                    help="Query size as fraction of total data (default: 0.05 = 5%%)")
parser.add_argument("--initial_per_class", type=int, default=1,
                    help="Initial labeled samples per class")
parser.add_argument("--classifier_epochs", type=int, default=50)
parser.add_argument("--turning_point_frac", type=float, default=None,
                    help="Override turning point fraction instead of reading from eval_results.json")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = args.config or os.path.join(os.path.dirname(__file__), '..', 'configs', 'cifar10.yaml')
with open(config_path) as f:
    config = yaml.load(f, yaml.Loader)

seed_dir = os.path.join(args.results_dir, f"seed_{args.seed}")

# --- Determine turning point ---
if args.turning_point_frac is not None:
    tp_frac = args.turning_point_frac
else:
    eval_path = os.path.join(seed_dir, "downstream_eval", "eval_results.json")
    with open(eval_path) as f:
        eval_data = json.load(f)
    tp_frac = eval_data["turning_point_fraction"]

tp_pct = int(round(tp_frac * 100))
tp_ckpt = os.path.join(seed_dir, "simclr_checkpoints", f"frac_{tp_pct:03d}", "model.pth.tar")
print(f"Using turning-point model: fraction={tp_frac:.0%}, checkpoint={tp_ckpt}")

# --- Load dataset and encode ---
torch.manual_seed(args.seed)
np.random.seed(args.seed)

DatasetClass = get_dataset_by_name("cifar10")
pool_rng = np.random.default_rng(args.seed)
dataset = DatasetClass(args.data_folder, config, pool_rng, encoded=False)
config["n_classes"] = dataset.n_classes

encoder = dataset.get_pretext_encoder(config, seed=args.seed)
encoder.load_state_dict(torch.load(tp_ckpt, map_location="cpu"))
encoder = encoder.to(device).eval()

print("Encoding dataset with turning-point SimCLR model...")


def encode_all(encoder, x, batch_size=256):
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size].to(device)
            f = encoder.backbone(batch)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


x_train_enc = encode_all(encoder, dataset.x_train)
x_test_enc = encode_all(encoder, dataset.x_test)

# Normalize encoded features
x_train_np = x_train_enc.numpy()
x_test_np = x_test_enc.numpy()
x_train_np, x_test_np = normalize(x_train_np, x_test_np, mode="mean_std")
x_train_enc = torch.tensor(x_train_np, dtype=torch.float32)
x_test_enc = torch.tensor(x_test_np, dtype=torch.float32)

feature_dim = x_train_enc.shape[1]
n_classes = dataset.n_classes
n_total = len(x_train_enc)
query_size = max(1, int(n_total * args.query_frac))

print(f"Encoded feature dim: {feature_dim}, n_total: {n_total}, query_size: {query_size}")

# --- Set up labeled/unlabeled pools ---
y_train = dataset.y_train
y_test = dataset.y_test

# Create validation split (fixed)
val_rng = np.random.default_rng(args.seed + 2000)
all_ids = np.arange(n_total)
val_rng.shuffle(all_ids)
val_cut = int(n_total * 0.04)
val_ids = all_ids[:val_cut]
pool_ids = all_ids[val_cut:]

x_val = x_train_enc[val_ids].to(device)
y_val = y_train[val_ids].to(device)
x_pool = x_train_enc[pool_ids]
y_pool = y_train[pool_ids]
x_test_d = x_test_enc.to(device)
y_test_d = y_test.to(device)

# Create seed set: initial_per_class samples per class
seed_rng = np.random.default_rng(args.seed)
shuffled_pool = np.arange(len(x_pool))
seed_rng.shuffle(shuffled_pool)

per_class_count = [0] * n_classes
labeled_mask = np.zeros(len(x_pool), dtype=bool)
for i in shuffled_pool:
    cls = int(torch.argmax(y_pool[i]).item())
    if per_class_count[cls] < args.initial_per_class:
        labeled_mask[i] = True
        per_class_count[cls] += 1
    if sum(per_class_count) >= args.initial_per_class * n_classes:
        break

labeled_ids = np.where(labeled_mask)[0].tolist()
unlabeled_ids = np.where(~labeled_mask)[0].tolist()

print(f"Initial labeled: {len(labeled_ids)}, unlabeled: {len(unlabeled_ids)}")


# --- Classifier training helpers ---
def build_classifier(feature_dim, n_classes, model_rng):
    model = nn.Linear(feature_dim, n_classes).to(device)
    return model


def train_classifier(model, x_labeled, y_labeled, x_val, y_val,
                     max_epochs=50, patience=5, lr=1e-3):
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=patience, lower_is_better=True)

    y_cls = torch.argmax(y_labeled, dim=1) if y_labeled.dim() > 1 else y_labeled
    y_val_cls = torch.argmax(y_val, dim=1) if y_val.dim() > 1 else y_val

    batch_size = min(64, len(x_labeled))
    loader = DataLoader(TensorDataset(x_labeled, y_cls), batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for bx, by in loader:
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val_cls).item()
        if early_stop.check_stop(val_loss):
            break

    return model


def evaluate_classifier(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1)
        y_cls = torch.argmax(y_test, dim=1) if y_test.dim() > 1 else y_test
        acc = (preds == y_cls).float().mean().item()
        loss = nn.CrossEntropyLoss()(logits, y_cls).item()
    return acc, loss


# --- Set up agent ---
AgentClass = get_agent_by_name(args.agent)

# Inject config for encoded mode
al_config = dict(config)
al_config["classifier_embedded"] = {"type": "Linear"}
al_config["optimizer_embedded"] = {"type": "NAdam", "lr": 1e-3, "weight_decay": 1e-5}
AgentClass.inject_config(al_config)

agent = AgentClass(args.seed, al_config, query_size)
print(f"Agent: {agent.name}, query_size: {query_size}")

# --- AL Loop ---
budget = len(unlabeled_ids)
n_rounds = math.ceil(budget / query_size)

accuracies = []
losses = []
n_labeled_history = []

# Initial evaluation
model_rng = torch.Generator()
model_rng.manual_seed(args.seed)

classifier = build_classifier(feature_dim, n_classes, model_rng)
x_lab = x_pool[labeled_ids].to(device)
y_lab = y_pool[labeled_ids].to(device)
classifier = train_classifier(classifier, x_lab, y_lab, x_val, y_val,
                               max_epochs=args.classifier_epochs)
acc, loss = evaluate_classifier(classifier, x_test_d, y_test_d)
accuracies.append(acc)
losses.append(loss)
n_labeled_history.append(len(labeled_ids))
print(f"[Init] labeled={len(labeled_ids)}, acc={acc:.4f}, loss={loss:.4f}")

per_class_instances = {c: per_class_count[c] for c in range(n_classes)}
initial_acc = acc

iterator = tqdm(range(n_rounds), desc=f"AL [{agent.name}]")
for round_i in iterator:
    if not unlabeled_ids:
        break

    x_unlabeled = x_pool[unlabeled_ids].to(device)
    x_labeled_t = x_pool[labeled_ids].to(device)
    y_labeled_t = y_pool[labeled_ids].to(device)

    actual_query = min(query_size, len(unlabeled_ids))
    agent.query_size = actual_query

    optimizer = torch.optim.NAdam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    action = agent.predict(
        x_unlabeled, x_labeled_t, y_labeled_t,
        per_class_instances, budget, len(labeled_ids),
        initial_acc, acc,
        classifier, optimizer
    )

    if isinstance(action, (int, np.integer)):
        action = [action]
    action = list(action)

    # Move selected samples from unlabeled to labeled
    selected_unlabeled_ids = [unlabeled_ids[a] for a in action]
    for uid in selected_unlabeled_ids:
        labeled_ids.append(uid)
        cls = int(torch.argmax(y_pool[uid]).item())
        per_class_instances[cls] = per_class_instances.get(cls, 0) + 1
    unlabeled_ids = [u for u in unlabeled_ids if u not in set(selected_unlabeled_ids)]

    # Retrain classifier from scratch
    classifier = build_classifier(feature_dim, n_classes, model_rng)
    x_lab = x_pool[labeled_ids].to(device)
    y_lab = y_pool[labeled_ids].to(device)
    classifier = train_classifier(classifier, x_lab, y_lab, x_val, y_val,
                                   max_epochs=args.classifier_epochs)

    acc, loss = evaluate_classifier(classifier, x_test_d, y_test_d)
    accuracies.append(acc)
    losses.append(loss)
    n_labeled_history.append(len(labeled_ids))

    iterator.set_postfix({"labeled": len(labeled_ids), "acc": f"{acc:.4f}"})

# --- Save results ---
al_dir = os.path.join(seed_dir, "al_results", agent.name)
os.makedirs(al_dir, exist_ok=True)

output = {
    "seed": args.seed,
    "agent": agent.name,
    "turning_point_fraction": tp_frac,
    "query_size": query_size,
    "query_frac": args.query_frac,
    "initial_per_class": args.initial_per_class,
    "n_labeled": n_labeled_history,
    "accuracies": accuracies,
    "losses": losses,
}
with open(os.path.join(al_dir, "al_results.json"), "w") as f:
    json.dump(output, f, indent=2)

# Also save as CSV for compatibility
import pandas as pd
df = pd.DataFrame({
    "n_labeled": n_labeled_history,
    "accuracy": accuracies,
    "loss": losses,
})
df.to_csv(os.path.join(al_dir, "al_results.csv"), index=False)

print(f"\nResults saved to {al_dir}/")
print(f"Final accuracy: {accuracies[-1]:.4f} with {n_labeled_history[-1]} labeled samples")
