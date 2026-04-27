#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_SWITCH_RE = re.compile(r"^mode_switch_")


def load_run_curve(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected CSV format: {csv_path}")
    budget = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    accuracy = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    out = pd.DataFrame({"budget": budget, "accuracy": accuracy}).dropna(subset=["budget"])
    out["budget"] = out["budget"].astype(int)
    return out


def aggregate_method(mode_dir: Path) -> Tuple[pd.DataFrame, int]:
    run_paths = sorted(mode_dir.glob("run_*/accuracies.csv"))
    if not run_paths:
        raise FileNotFoundError(f"No run accuracies found in {mode_dir}")

    per_run = []
    for run_csv in run_paths:
        run_df = load_run_curve(run_csv).rename(columns={"accuracy": run_csv.parent.name})
        per_run.append(run_df.set_index("budget"))

    merged = pd.concat(per_run, axis=1)
    mean = merged.mean(axis=1, skipna=True)
    var = merged.var(axis=1, skipna=True, ddof=0)
    count = merged.count(axis=1)

    out = pd.DataFrame(
        {
            "budget": merged.index.values,
            "mean_accuracy": mean.values,
            "var_accuracy": var.values,
            "std_accuracy": np.sqrt(var.values),
            "num_runs_with_value": count.values,
        }
    ).sort_values("budget")
    return out, len(run_paths)


def compute_auc(summary_df: pd.DataFrame) -> float:
    valid = summary_df.dropna(subset=["mean_accuracy"]).sort_values("budget")
    if len(valid) < 2:
        return float("nan")
    return float(np.trapezoid(valid["mean_accuracy"].to_numpy(), valid["budget"].to_numpy()))


def infer_budget_step(budgets: np.ndarray) -> int | None:
    b = np.unique(np.asarray(budgets, dtype=int))
    b = b[b > 0]
    if b.size < 2:
        return None
    diffs = np.diff(np.sort(b))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(counts)])


def set_budget_ticks(ax, budgets: np.ndarray, step: int | None) -> None:
    b = np.unique(np.asarray(budgets, dtype=int))
    b = np.sort(b[b > 0])
    if b.size == 0:
        return
    start = int(b[0])
    if step is None or step <= 0:
        ticks = b
    else:
        end = int(b[-1])
        ticks = np.arange(start, end + step, step, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xlim(left=float(start))


def save_method_curve_plot(
    df: pd.DataFrame,
    label: str,
    out_path: Path,
    with_variance: bool,
    with_markers: bool,
    marker: str,
    budget_step: int | None,
) -> None:
    valid = df.dropna(subset=["mean_accuracy"]).sort_values("budget")
    if valid.empty:
        return
    x = valid["budget"].to_numpy()
    y = valid["mean_accuracy"].to_numpy()
    std = valid["std_accuracy"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        y,
        label=label,
        linewidth=2.0,
        marker=marker if with_markers else None,
        markersize=4 if with_markers else None,
    )
    if with_variance:
        plt.fill_between(x, y - std, y + std, alpha=0.2)
    set_budget_ticks(plt.gca(), x, budget_step)
    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.title(f"{label}: Accuracy vs Budget")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_multi_plot(
    summaries: Dict[str, pd.DataFrame],
    labels: List[str],
    out_path: Path,
    with_variance: bool,
    with_markers: bool,
    budget_step: int | None,
) -> None:
    marker_map = {"random": "x", "al": "^", "switch": "o"}

    plt.figure(figsize=(12, 7))
    all_x_for_ticks = []
    for label in labels:
        df = summaries[label].dropna(subset=["mean_accuracy"]).sort_values("budget")
        if df.empty:
            continue
        x = df["budget"].to_numpy()
        y = df["mean_accuracy"].to_numpy()
        std = df["std_accuracy"].to_numpy()
        if label.startswith("mode_switch_"):
            marker = marker_map["switch"]
        elif label == "mode_all_random":
            marker = marker_map["random"]
        elif label == "mode_all_al":
            marker = marker_map["al"]
        else:
            marker = "o"
        plt.plot(
            x,
            y,
            label=label,
            linewidth=1.8,
            marker=marker if with_markers else None,
            markersize=3.8 if with_markers else None,
        )
        if with_variance:
            plt.fill_between(x, y - std, y + std, alpha=0.12)
        all_x_for_ticks.extend(x[x > 0].tolist())

    if all_x_for_ticks:
        set_budget_ticks(plt.gca(), np.array(all_x_for_ticks), budget_step)

    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Budget")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=".",
        help="Directory containing mode_all_random/mode_all_al/mode_switch_*",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_outputs",
        help="Output directory for plots and CSV files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_dirs = [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("mode_")]
    random_dir = next((p for p in mode_dirs if p.name == "mode_all_random"), None)
    al_dir = next((p for p in mode_dirs if p.name == "mode_all_al"), None)
    switch_dirs = sorted([p for p in mode_dirs if MODE_SWITCH_RE.match(p.name)])

    if random_dir is None or al_dir is None or not switch_dirs:
        raise SystemExit("Cannot find required mode directories (random/al/switch).")

    summaries: Dict[str, pd.DataFrame] = {}
    auc_rows = []

    method_dirs = [random_dir, al_dir] + switch_dirs
    for mode_dir in method_dirs:
        summary_df, n_runs = aggregate_method(mode_dir)
        summaries[mode_dir.name] = summary_df

        summary_csv = output_dir / f"{mode_dir.name}_budget_mean_var.csv"
        summary_df.to_csv(summary_csv, index=False)

        auc = compute_auc(summary_df)
        auc_rows.append({"method": mode_dir.name, "auc": auc, "num_runs": n_runs})

    auc_df = pd.DataFrame(auc_rows).sort_values("auc", ascending=False)
    auc_df.to_csv(output_dir / "method_auc.csv", index=False)

    all_budgets = []
    for df in summaries.values():
        valid_b = df.loc[df["num_runs_with_value"] > 0, "budget"].to_numpy()
        if valid_b.size > 0:
            all_budgets.append(valid_b)
    budget_step = infer_budget_step(np.concatenate(all_budgets)) if all_budgets else None

    # 1) Individual plots per method: with variance / without variance; line-only / markers.
    for method_name, df in summaries.items():
        method_type = "switch" if method_name.startswith("mode_switch_") else ("random" if method_name == "mode_all_random" else "al")
        marker = {"random": "x", "al": "^", "switch": "o"}[method_type]
        for with_variance in (False, True):
            for with_markers in (False, True):
                tag_var = "with_variance" if with_variance else "no_variance"
                tag_style = "markers" if with_markers else "line"
                out_path = output_dir / f"{method_name}_{tag_var}_{tag_style}.png"
                save_method_curve_plot(
                    df=df,
                    label=method_name,
                    out_path=out_path,
                    with_variance=with_variance,
                    with_markers=with_markers,
                    marker=marker,
                    budget_step=budget_step,
                )

    # 2) All methods in one figure: with variance / without variance; line-only / markers.
    all_labels = [random_dir.name, al_dir.name] + [p.name for p in switch_dirs]
    for with_variance in (False, True):
        for with_markers in (False, True):
            tag_var = "with_variance" if with_variance else "no_variance"
            tag_style = "markers" if with_markers else "line"
            out_path = output_dir / f"all_methods_{tag_var}_{tag_style}.png"
            save_multi_plot(
                summaries=summaries,
                labels=all_labels,
                out_path=out_path,
                with_variance=with_variance,
                with_markers=with_markers,
                budget_step=budget_step,
            )

    # 3) Best switch (by AUC) + random + al in one figure (with/without variance; line/markers).
    best_switch_row = auc_df[auc_df["method"].str.startswith("mode_switch_")].head(1)
    if not best_switch_row.empty:
        best_switch = best_switch_row.iloc[0]["method"]
        subset_labels = [random_dir.name, al_dir.name, best_switch]
        for with_variance in (False, True):
            for with_markers in (False, True):
                tag_var = "with_variance" if with_variance else "no_variance"
                tag_style = "markers" if with_markers else "line"
                out_path = output_dir / f"best_switch_vs_random_al_{tag_var}_{tag_style}.png"
                save_multi_plot(
                    summaries=summaries,
                    labels=subset_labels,
                    out_path=out_path,
                    with_variance=with_variance,
                    with_markers=with_markers,
                    budget_step=budget_step,
                )

    print(f"Saved analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
