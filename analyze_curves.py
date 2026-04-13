"""
Phase 2: Cold-Start Transition Analysis
========================================
Reads accuracy/loss CSVs from Phase 1 (and optionally Phase 3) runs,
plots smoothed curves with error bands, and locates the transition point
where Entropy stably beats Random.

Usage:
    python analyze_curves.py \
        --runs_dir runs \
        --dataset Cifar10 \
        --query_size 20 \
        --scratch_postfix coldstart \
        --pretrained_postfix coldstart_pretrained \
        --output_dir results/coldstart
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ── helpers ────────────────────────────────────────────────────────────────

def load_agent_curves(runs_dir, dataset, query_size, agent_name, postfix=None,
                      initial_per_class=None):
    """Return (index_array, accuracy_matrix[rounds, seeds]) from per-run CSVs."""
    folder_name = f"{agent_name}_{postfix}" if postfix else agent_name
    if initial_per_class is not None:
        base = os.path.join(runs_dir, dataset, f"init{initial_per_class}", str(query_size), folder_name)
    else:
        base = os.path.join(runs_dir, dataset, str(query_size), folder_name)
    acc_file = os.path.join(base, "accuracies.csv")
    if not os.path.exists(acc_file):
        raise FileNotFoundError(f"Missing {acc_file}")
    df = pd.read_csv(acc_file, header=0, index_col=0).dropna(axis=0)
    return np.array(df.index, dtype=float), df.values


def smooth(y, weight=0.6):
    """Exponential moving average."""
    s = np.empty_like(y)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = weight * s[i - 1] + (1 - weight) * y[i]
    return s


def plot_agent(ax, x, matrix, label, color, smooth_weight=0.0):
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1)
    if smooth_weight > 0:
        mean = smooth(mean, smooth_weight)
        std = smooth(std, smooth_weight)
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def find_crossover(x, mean_al, mean_rs, window=3):
    """
    Find the first round where AL mean is consistently above RS mean
    for `window` consecutive rounds.  Returns (index, x_value) or None.
    """
    diff = mean_al - mean_rs
    streak = 0
    for i, d in enumerate(diff):
        if d > 0:
            streak += 1
            if streak >= window:
                start = i - window + 1
                return start, x[start]
        else:
            streak = 0
    return None


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: cold-start curve analysis")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--query_size", type=int, default=20)
    parser.add_argument("--scratch_postfix", default="coldstart")
    parser.add_argument("--pretrained_postfix", default="coldstart_pretrained",
                        help="Set to empty string to skip pretrained curves")
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--output_dir", default="results/coldstart")
    parser.add_argument("--initial_per_class", type=int, default=2,
                        help="Points per class in seed set (for x-axis offset)")
    parser.add_argument("--n_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_pool = args.initial_per_class * args.n_classes

    ipc = args.initial_per_class

    # ── Load scratch curves ──
    x_rs, mat_rs = load_agent_curves(
        args.runs_dir, args.dataset, args.query_size,
        "RandomAgent", args.scratch_postfix, ipc)
    x_al, mat_al = load_agent_curves(
        args.runs_dir, args.dataset, args.query_size,
        "ShannonEntropy", args.scratch_postfix, ipc)

    x_rs = x_rs + seed_pool
    x_al = x_al + seed_pool

    # ── Plot accuracy ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_agent(axes[0], x_rs, mat_rs, "Random (scratch)", "grey", args.smooth)
    plot_agent(axes[0], x_al, mat_al, "Entropy (scratch)", "tab:blue", args.smooth)

    # ── Optionally overlay pretrained curves ──
    if args.pretrained_postfix:
        try:
            x_rs_p, mat_rs_p = load_agent_curves(
                args.runs_dir, args.dataset, args.query_size,
                "RandomAgent", args.pretrained_postfix, ipc)
            x_al_p, mat_al_p = load_agent_curves(
                args.runs_dir, args.dataset, args.query_size,
                "ShannonEntropy", args.pretrained_postfix, ipc)
            x_rs_p = x_rs_p + seed_pool
            x_al_p = x_al_p + seed_pool
            plot_agent(axes[0], x_rs_p, mat_rs_p, "Random (pretrained)", "silver",
                       args.smooth)
            plot_agent(axes[0], x_al_p, mat_al_p, "Entropy (pretrained)", "tab:orange",
                       args.smooth)
        except FileNotFoundError:
            print("Pretrained curves not found — skipping overlay.")

    axes[0].set_xlabel("# Labeled Samples")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title(f"{args.dataset} — Accuracy (query={args.query_size})")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    # ── Plot loss ──
    try:
        loss_rs = load_loss(args.runs_dir, args.dataset, args.query_size,
                            "RandomAgent", args.scratch_postfix, ipc)
        loss_al = load_loss(args.runs_dir, args.dataset, args.query_size,
                            "ShannonEntropy", args.scratch_postfix, ipc)
        xl = np.arange(loss_rs.shape[0]) * args.query_size + seed_pool
        plot_agent(axes[1], xl, loss_rs, "Random (scratch)", "grey", args.smooth)
        plot_agent(axes[1], xl, loss_al, "Entropy (scratch)", "tab:blue", args.smooth)
        axes[1].set_xlabel("# Labeled Samples")
        axes[1].set_ylabel("Test Loss")
        axes[1].set_title(f"{args.dataset} — Loss")
        axes[1].legend(fontsize="small")
        axes[1].grid(True, alpha=0.3)
    except FileNotFoundError:
        axes[1].text(0.5, 0.5, "Loss files not found",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "coldstart_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)

    # ── Find transition point ──
    mean_rs = np.mean(mat_rs, axis=1)
    mean_al = np.mean(mat_al, axis=1)
    result = find_crossover(x_al, mean_al, mean_rs, window=2)
    if result is not None:
        idx, x_val = result
        round_num = idx
        print(f"\n{'='*50}")
        print(f"Transition detected at round {round_num} "
              f"(~{int(x_val)} labeled samples)")
        print(f"  Suggested target rounds:")
        t_early = max(0, round_num - 3)
        t_trans = round_num
        t_late = min(len(mean_al) - 1, round_num + 3)
        print(f"    t_early = round {t_early}  ({int(x_al[t_early])} samples)")
        print(f"    t_trans = round {t_trans}  ({int(x_al[t_trans])} samples)")
        print(f"    t_late  = round {t_late}  ({int(x_al[t_late])} samples)")
        print(f"{'='*50}")
    else:
        print("\nNo stable crossover found. Entropy may not have overtaken Random yet.")
        print("Consider running more rounds or lowering the crossover window.")


def load_loss(runs_dir, dataset, query_size, agent_name, postfix=None,
              initial_per_class=None):
    folder_name = f"{agent_name}_{postfix}" if postfix else agent_name
    if initial_per_class is not None:
        base = os.path.join(runs_dir, dataset, f"init{initial_per_class}", str(query_size), folder_name)
    else:
        base = os.path.join(runs_dir, dataset, str(query_size), folder_name)
    loss_file = os.path.join(base, "losses.csv")
    if not os.path.exists(loss_file):
        raise FileNotFoundError(f"Missing {loss_file}")
    df = pd.read_csv(loss_file, header=0, index_col=0).dropna(axis=0)
    return df.values


if __name__ == "__main__":
    main()
