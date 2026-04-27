#!/usr/bin/env python3
"""
TranOne: compare (1) all-random, (2) first query random then pure AL,
(3) random then AL: either by **budget fraction** (default) or by **round index** (for size-based
    queries; switch grid over first-AL round derived from simulating rounds to exhaust budget).

Round convention: **round 0 is the first round** (第一轮); round index is 0-based.

Query batch size: choose one of (1) uniform `--query_frac`, (2) fixed `--query_size`,
(3) split fractions `--query_frac_first` + `--query_frac_rest`,
or (4) split integers `--query_size_first` + `--query_size_rest`.
See TranOne/README.md for details.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import List, Literal, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
for _p in (REPO_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_util as util
import core
from core.helper_functions import collect_results, get_agent_by_name, get_dataset_by_name, save_meta_data

from tranone_al_game import TranOneALGame


def configure_determinism(seed: int, strict: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if strict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

Mode = Literal["all_random", "all_al", "switch"]


def switch_fractions_for_step(step_pct: int) -> List[float]:
    """Grid of cumulative-budget fractions where random→AL switch happens (exclusive of 0 and 1)."""
    if step_pct == 1:
        return [i / 100.0 for i in range(1, 100)]
    if step_pct == 5:
        return [i / 100.0 for i in range(5, 100, 5)]
    if step_pct == 10:
        return [i / 100.0 for i in range(10, 100, 10)]
    raise ValueError(f"Unsupported switch step {step_pct}%")


def read_budget_from_yaml(config_path: str, *, encoded: bool, override: Optional[int]) -> int:
    with open(config_path, "r") as f:
        cfg = yaml.load(f, yaml.Loader)
    if override is not None:
        return int(override)
    key = "dataset_embedded" if encoded else "dataset"
    return int(cfg[key]["budget"])


def count_acquisition_rounds(
    budget: int,
    unlabeled_pool: int,
    *,
    query_frac: float,
    query_size: Optional[int],
    query_size_first: Optional[int],
    query_size_rest: Optional[int],
    query_frac_first: Optional[float],
    query_frac_rest: Optional[float],
) -> int:
    """
    Simulate the same per-round cap min(q, remaining, unlabeled) as the main driver and return
    how many acquisition steps occur before the budget (or unlabeled) is exhausted.
    """
    added = 0
    u = int(unlabeled_pool)
    r = 0
    while added < int(budget) and u > 0:
        qs_b, _ = compute_round_query_budget(
            r,
            budget,
            query_size=query_size,
            query_size_first=query_size_first,
            query_size_rest=query_size_rest,
            query_frac=query_frac,
            query_frac_first=query_frac_first,
            query_frac_rest=query_frac_rest,
        )
        rem = int(budget) - added
        step = int(min(int(qs_b), rem, u))
        if step <= 0:
            break
        added += step
        u -= step
        r += 1
    return r


def switch_first_al_rounds_for_step(step: int, num_rounds: int) -> List[int]:
    """
    0-based round index of the first acquisition round that uses AL (round 0 is always random):
    {r | 1 <= r < num_rounds}, optionally subsampled in steps of 1, 5, 10 to mirror the fraction sweeps.
    """
    if num_rounds < 2:
        return []
    if step not in (1, 5, 10):
        raise ValueError(f"Unsupported switch round step {step}")
    if step == 1:
        return list(range(1, num_rounds))
    out: List[int] = []
    r = int(step)
    while r < num_rounds:
        out.append(int(r))
        r += int(step)
    return out


def frac_to_tag(f: float) -> str:
    s = f"{f:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def query_results_dir_tag(
    *,
    query_size: Optional[int],
    query_size_first: Optional[int],
    query_size_rest: Optional[int],
    query_frac: float,
    query_frac_first: Optional[float],
    query_frac_rest: Optional[float],
) -> str:
    """Folder / manifest token describing how per-round query batch size is chosen."""
    if query_size_first is not None and query_size_rest is not None:
        return f"qsf{int(query_size_first)}_qsr{int(query_size_rest)}"
    if query_size is not None:
        return f"qsize{int(query_size)}"
    if query_frac_first is not None and query_frac_rest is not None:
        return f"qff{frac_to_tag(query_frac_first)}_qfr{frac_to_tag(query_frac_rest)}"
    return f"qfrac{frac_to_tag(query_frac)}"


def compute_round_query_budget(
    round_idx: int,
    budget: int,
    *,
    query_size: Optional[int],
    query_size_first: Optional[int],
    query_size_rest: Optional[int],
    query_frac: float,
    query_frac_first: Optional[float],
    query_frac_rest: Optional[float],
) -> tuple[int, dict]:
    """
    Desired batch size before capping by remaining budget / pool.
    Returns (qs_budget, meta) where meta describes the rule used this round.
    """
    if query_size_first is not None and query_size_rest is not None:
        size = int(query_size_first) if round_idx == 0 else int(query_size_rest)
        meta = {
            "query_mode": "split_size",
            "query_size_first": int(query_size_first),
            "query_size_rest": int(query_size_rest),
            "query_size_applied": int(size),
        }
        return max(1, int(size)), meta
    if query_size is not None:
        meta = {"query_mode": "fixed_size", "query_size_config": int(query_size)}
        return max(1, int(query_size)), meta
    if query_frac_first is not None and query_frac_rest is not None:
        frac = float(query_frac_first) if round_idx == 0 else float(query_frac_rest)
        meta = {
            "query_mode": "split_frac",
            "query_frac_first": float(query_frac_first),
            "query_frac_rest": float(query_frac_rest),
            "query_frac_applied": frac,
        }
        return max(1, int(round(frac * budget))), meta
    meta = {"query_mode": "uniform_frac", "query_frac": float(query_frac)}
    return max(1, int(round(float(query_frac) * budget))), meta


def run_name(
    mode: Mode,
    switch_step: Optional[int],
    switch_frac: Optional[float],
    *,
    switch_policy: str = "frac",
    first_al_round_0based: Optional[int] = None,
) -> str:
    if mode == "all_random":
        return "mode_all_random"
    if mode == "all_al":
        return "mode_all_al"
    assert mode == "switch" and switch_step is not None
    if switch_policy == "round":
        assert first_al_round_0based is not None
        return f"mode_switch_rstep{switch_step}_r{int(first_al_round_0based)}"
    assert switch_frac is not None
    return f"mode_switch_step{switch_step}pct_at{frac_to_tag(switch_frac)}"


def acquisition_mode(
    mode: Mode,
    round_idx: int,
    added_before_round: int,
    budget: int,
    switch_policy: str,
    switch_frac: Optional[float],
    first_al_round_0based: Optional[int],
) -> str:
    """
    Return 'random' or 'al'.

    Round indexing: round_idx is 0-based; by convention **round 0 is the first round** (第一轮).
    - First round (round_idx == 0): always random for every mode.
    - all_random: every round stays random.
    - all_al: from the second round onward (round_idx >= 1), always AL.
    - switch, policy frac: from the second round onward, random while added_before_round < switch_frac * budget, else AL.
    - switch, policy round: from the second round onward, random while round_idx < first_al_round_0based, else AL.
    """
    if round_idx == 0:
        return "random"
    if mode == "all_random":
        return "random"
    if mode == "all_al":
        return "al"
    if switch_policy == "round":
        assert first_al_round_0based is not None
        if round_idx < int(first_al_round_0based):
            return "random"
        return "al"
    # switch, budget fraction
    assert switch_frac is not None
    threshold = max(1, int(round(switch_frac * budget)))
    if added_before_round < threshold:
        return "random"
    return "al"


def run_single(
    *,
    mode: Mode,
    switch_policy: str = "frac",
    switch_step: Optional[int],
    switch_frac: Optional[float],
    first_al_round_0based: Optional[int] = None,
    data_folder: str,
    config_path: str,
    dataset_name: str,
    encoded: bool,
    al_agent_name: str,
    query_frac: float,
    query_size: Optional[int],
    query_size_first: Optional[int],
    query_size_rest: Optional[int],
    query_frac_first: Optional[float],
    query_frac_rest: Optional[float],
    run_id: int,
    pool_seed: int,
    agent_seed: int,
    model_seed: int,
    fitting_mode: Optional[str],
    strict_deterministic: bool,
    results_root: str,
    save_checkpoints: bool,
    override_budget: Optional[int] = None,
) -> str:
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.Loader)

    config_key = "dataset_embedded" if encoded else "dataset"
    config[config_key]["initial_points_per_class"] = 0
    if "dataset_embedded" in config and not encoded:
        # keep embedded block consistent if present
        config["dataset_embedded"]["initial_points_per_class"] = 0

    if fitting_mode is not None:
        config[config_key]["classifier_fitting_mode"] = fitting_mode

    if override_budget is not None:
        config[config_key]["budget"] = int(override_budget)

    _budget = int(config[config_key]["budget"])

    pool_rng = np.random.default_rng(pool_seed + run_id)
    configure_determinism(model_seed + run_id, strict_deterministic)

    RandomAgent = get_agent_by_name("random")
    AlAgent = get_agent_by_name(al_agent_name)
    DatasetClass = get_dataset_by_name(dataset_name)
    RandomAgent.inject_config(config)
    AlAgent.inject_config(config)
    DatasetClass.inject_config(config)

    dataset = DatasetClass(data_folder, config, pool_rng, encoded)
    dataset = dataset.to(util.device)

    pool_n = int(len(getattr(dataset, "x_unlabeled", [])) or 0) or 10**9
    n_rounds_cfg = count_acquisition_rounds(
        int(dataset.budget),
        pool_n,
        query_frac=query_frac,
        query_size=query_size,
        query_size_first=query_size_first,
        query_size_rest=query_size_rest,
        query_frac_first=query_frac_first,
        query_frac_rest=query_frac_rest,
    )
    config["tranone"] = {
        "mode": mode,
        "switch_policy": switch_policy,
        "switch_step_switch_grid": switch_step,
        "switch_frac": switch_frac,
        "first_al_round_0based": first_al_round_0based,
        "unlabeled_pool_size_for_round_estimate": pool_n,
        "estimated_acquisition_rounds": n_rounds_cfg,
        "al_agent": al_agent_name,
        "query_frac": query_frac,
        "query_size": query_size,
        "query_size_first": query_size_first,
        "query_size_rest": query_size_rest,
        "query_frac_first": query_frac_first,
        "query_frac_rest": query_frac_rest,
        "first_query_random": True,
    }

    if mode == "switch" and switch_policy == "round":
        if first_al_round_0based is None:
            raise SystemExit("Internal error: first_al_round_0based missing for switch/round")
        r0 = int(first_al_round_0based)
        if n_rounds_cfg < 2 or not (1 <= r0 < n_rounds_cfg):
            raise SystemExit(
                f"Invalid first AL round {r0} for {n_rounds_cfg} estimated acquisition "
                f"rounds (require 1 <= r < n_rounds; round 0 is always random)"
            )

    env = TranOneALGame(
        dataset,
        pool_rng,
        model_seed=model_seed + run_id,
        data_loader_seed=1,
        device=util.device,
    )
    # Agents require an initial query_size; updated each round before predict.
    _initial_qs, _ = compute_round_query_budget(
        0,
        dataset.budget,
        query_size=query_size,
        query_size_first=query_size_first,
        query_size_rest=query_size_rest,
        query_frac=query_frac,
        query_frac_first=query_frac_first,
        query_frac_rest=query_frac_rest,
    )
    random_agent = RandomAgent(agent_seed, config, _initial_qs)
    al_agent = AlAgent(agent_seed, config, _initial_qs)

    name = run_name(
        mode,
        switch_step,
        switch_frac,
        switch_policy=switch_policy,
        first_al_round_0based=first_al_round_0based,
    )
    qtag = query_results_dir_tag(
        query_size=query_size,
        query_size_first=query_size_first,
        query_size_rest=query_size_rest,
        query_frac=query_frac,
        query_frac_first=query_frac_first,
        query_frac_rest=query_frac_rest,
    )
    qdir = f"tranone_{qtag}_B{dataset.budget}"
    base_path = os.path.join(results_root, dataset.name, qdir, name)
    log_path = os.path.join(base_path, f"run_{run_id}")
    os.makedirs(log_path, exist_ok=True)

    trace_path = os.path.join(log_path, "tranone_trace.jsonl")
    trace_f = open(trace_path, "w", encoding="utf-8")

    with core.EnvironmentLogger(env, log_path, util.is_cluster, save_checkpoints=save_checkpoints) as logger:
        dataset.reset()
        state = logger.reset()
        round_idx = 0
        pbar = tqdm(desc=name, miniters=1)
        while env.added_images < env.budget:
            x_u = state[0]
            remaining = env.budget - env.added_images
            qs_budget, qmeta = compute_round_query_budget(
                round_idx,
                env.budget,
                query_size=query_size,
                query_size_first=query_size_first,
                query_size_rest=query_size_rest,
                query_frac=query_frac,
                query_frac_first=query_frac_first,
                query_frac_rest=query_frac_rest,
            )
            qs = min(qs_budget, remaining, int(x_u.shape[0]))
            if qs <= 0:
                break

            random_agent.query_size = qs
            al_agent.query_size = qs

            use = acquisition_mode(
                mode,
                round_idx,
                env.added_images,
                env.budget,
                switch_policy,
                switch_frac,
                first_al_round_0based,
            )
            if use == "random":
                action = random_agent.predict(*state)
            else:
                action = al_agent.predict(*state)

            # Different agents may return scalar / list / ndarray / tensor.
            # Normalize all of them to a flat python int list for env.step().
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if isinstance(action, np.ndarray):
                action = action.reshape(-1).tolist()
            elif isinstance(action, tuple):
                action = list(action)
            elif not isinstance(action, list):
                action = [action]
            action = [int(a) for a in action]

            state, reward, done, truncated, info = logger.step(action)
            rec = {
                "round": round_idx,
                "round_1based": round_idx + 1,
                "mode": use,
                "query_size": qs,
                "query_budget_before_cap": qs_budget,
                **qmeta,
                "added_images": env.added_images,
                "test_accuracy": float(env.current_test_accuracy),
            }
            if hasattr(env, "current_test_loss"):
                rec["test_loss"] = float(env.current_test_loss) if math.isfinite(env.current_test_loss) else None
            trace_f.write(json.dumps(rec) + "\n")
            trace_f.flush()

            pbar.set_postfix({"acc": f"{env.current_test_accuracy:.4f}", "n": env.added_images})
            pbar.update(1)
            round_idx += 1
            if done or truncated:
                break
        pbar.close()

    trace_f.close()

    collect_results(base_path, "run_")
    meta_agent = al_agent if mode != "all_random" else random_agent
    save_meta_data(log_path, meta_agent, env, dataset, config)
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--config", type=str, default=None, help="YAML path (default: TranOne/configs/tranone_cifar10.yaml)")
    parser.add_argument("--encoded", type=int, default=0)
    parser.add_argument(
        "--query_frac",
        type=float,
        default=0.05,
        help="Uniform mode: every round uses max(1, round(query_frac * budget)) (ignored if fixed/split size or split-frac is set).",
    )
    parser.add_argument(
        "--query_size",
        type=int,
        default=None,
        help="Fixed mode: same integer batch size every round (>=1), capped by remainder/pool. Mutually exclusive with split-frac and split-size pairs.",
    )
    parser.add_argument(
        "--query_size_first",
        type=int,
        default=None,
        help="Split-size mode (round 0 / 第一轮): fixed batch size (>=1). Requires --query_size_rest.",
    )
    parser.add_argument(
        "--query_size_rest",
        type=int,
        default=None,
        help="Split-size mode (round >= 1): fixed batch size (>=1). Requires --query_size_first.",
    )
    parser.add_argument(
        "--query_frac_first",
        type=float,
        default=None,
        help="Split mode (round 0 / 第一轮): batch = max(1, round(this * budget)). Requires --query_frac_rest.",
    )
    parser.add_argument(
        "--query_frac_rest",
        type=float,
        default=None,
        help="Split mode (round >= 1): batch = max(1, round(this * budget)). Requires --query_frac_first.",
    )
    parser.add_argument("--al_agent", type=str, default="margin", help="AL method for all_al and switch phases (after random)")
    parser.add_argument("--fitting_mode", type=str, default="from_scratch", choices=["from_scratch", "finetuning", "shrinking"])
    parser.add_argument("--restarts", type=int, default=1, help="Number of runs (seeds) per experiment cell")
    parser.add_argument("--pool_seed", type=int, default=1)
    parser.add_argument("--agent_seed", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--strict_deterministic", type=int, default=1)
    parser.add_argument("--save_checkpoints", type=int, default=0)
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Where to write runs (default: <repo>/TranOne/results)",
    )
    parser.add_argument(
        "--override_budget",
        type=int,
        default=None,
        help="If set, overrides dataset budget in YAML (for quick smoke tests)",
    )
    parser.add_argument(
        "--sweep-steps",
        type=str,
        default="1,5,10",
        help="In --mode sweep: comma-separated switch grids to include (subset of 1,5,10)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sweep",
        choices=["sweep", "all_random", "all_al", "switch"],
        help="sweep runs baseline + all_al + full switch grids; or a single mode",
    )
    parser.add_argument("--switch-step", type=int, default=None, choices=[1, 5, 10], help="Switch grid: (frac) 1/5/10%% steps; (round) same step on first-AL round index")
    parser.add_argument(
        "--switch-at",
        type=float,
        default=None,
        help="With --mode switch and --switch-policy frac: budget fraction in (0,1), e.g. 0.15",
    )
    parser.add_argument(
        "--switch-policy",
        type=str,
        default="frac",
        choices=["frac", "round"],
        help="How to decide random→AL in switch: frac = label budget threshold; round = first AL at 0-based round (needs size query)",
    )
    parser.add_argument(
        "--first-al-round",
        type=int,
        default=None,
        help="With --mode switch and --switch-policy round: 0-based round index of first AL (round 0 is always random). E.g. 3 => rounds 0–2 random, 3+ AL",
    )
    parser.add_argument(
        "--sweep-switch-policy",
        type=str,
        default="auto",
        choices=["auto", "frac", "round"],
        help="--mode sweep: which switch grid. auto: round grid if query is size-based, else budget-fraction grid",
    )
    args = parser.parse_args()

    fx = args.query_size is not None
    fxs = args.query_size_first is not None
    rxs = args.query_size_rest is not None
    fs = args.query_frac_first is not None
    rs = args.query_frac_rest is not None
    if fx and (fxs or rxs or fs or rs):
        raise SystemExit("Do not combine --query_size with split-size or split-frac options.")
    if fxs != rxs:
        raise SystemExit("Set both --query_size_first and --query_size_rest together, or neither.")
    if fs != rs:
        raise SystemExit("Set both --query_frac_first and --query_frac_rest together, or neither.")
    if fxs and rxs and fs and rs:
        raise SystemExit("Do not combine split-size (--query_size_first/--query_size_rest) with split-frac.")
    if fxs and rxs:
        for label, v in [("first", args.query_size_first), ("rest", args.query_size_rest)]:
            if int(v) < 1:
                raise SystemExit(f"--query_size_{label} must be >= 1, got {v}")
    elif fs and rs:
        for label, v in [("first", args.query_frac_first), ("rest", args.query_frac_rest)]:
            if not (0.0 < v <= 1.0):
                raise SystemExit(f"--query_frac_{label} must be in (0, 1], got {v}")
    elif fx:
        if int(args.query_size) < 1:
            raise SystemExit("--query_size must be >= 1")
    else:
        if not (0.0 < args.query_frac <= 1.0):
            raise SystemExit("--query_frac must be in (0, 1], e.g. 0.05 for 5%% of budget per round.")

    is_size_mode = (args.query_size is not None) or (
        args.query_size_first is not None and args.query_size_rest is not None
    )
    if args.first_al_round is not None and args.switch_policy != "round":
        raise SystemExit("--first-al-round requires --switch-policy round")
    if args.mode == "switch":
        if args.switch_policy == "round":
            if not is_size_mode:
                raise SystemExit('--mode switch with --switch-policy round requires size query (--query_size or --query_size_first+--query_size_rest)')
            if args.switch_at is not None:
                raise SystemExit("Use --first-al-round with --switch-policy round, not --switch-at")
            if args.switch_step is None or args.first_al_round is None:
                raise SystemExit("--mode switch (round) requires --switch-step and --first-al-round")
        else:
            if args.first_al_round is not None:
                raise SystemExit("Use --switch-policy round with --first-al-round")
            if args.switch_step is None or args.switch_at is None:
                raise SystemExit("--mode switch (frac) requires --switch-step and --switch-at (budget fraction)")

    encoded = bool(args.encoded)
    strict = bool(args.strict_deterministic)
    save_ckpt = bool(args.save_checkpoints)
    results_root = args.results_root or os.path.join(THIS_DIR, "results")
    os.makedirs(results_root, exist_ok=True)

    default_cfg = os.path.join(THIS_DIR, "configs", "tranone_cifar10.yaml")
    config_path = args.config or default_cfg
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config not found: {config_path}")

    manifest = []

    def enqueue_and_run(
        mode: Mode,
        *,
        switch_step: Optional[int] = None,
        switch_frac: Optional[float] = None,
        switch_policy: str = "frac",
        first_al_round_0based: Optional[int] = None,
    ):
        for rid in range(args.restarts):
            log_path = run_single(
                mode=mode,
                switch_policy=switch_policy,
                switch_step=switch_step,
                switch_frac=switch_frac,
                first_al_round_0based=first_al_round_0based,
                data_folder=args.data_folder,
                config_path=config_path,
                dataset_name=args.dataset,
                encoded=encoded,
                al_agent_name=args.al_agent,
                query_frac=args.query_frac,
                query_size=args.query_size,
                query_size_first=args.query_size_first,
                query_size_rest=args.query_size_rest,
                query_frac_first=args.query_frac_first,
                query_frac_rest=args.query_frac_rest,
                run_id=rid,
                pool_seed=args.pool_seed,
                agent_seed=args.agent_seed,
                model_seed=args.model_seed,
                fitting_mode=args.fitting_mode,
                strict_deterministic=strict,
                results_root=results_root,
                save_checkpoints=save_ckpt,
                override_budget=args.override_budget,
            )
            manifest.append(
                {
                    "log_path": log_path,
                    "mode": mode,
                    "switch_step": switch_step,
                    "switch_frac": switch_frac,
                    "switch_policy": switch_policy,
                    "first_al_round_0based": first_al_round_0based,
                    "run_id": rid,
                }
            )

    if args.mode == "all_random":
        enqueue_and_run("all_random")
    elif args.mode == "all_al":
        enqueue_and_run("all_al")
    elif args.mode == "switch":
        if args.switch_policy == "round":
            assert args.first_al_round is not None and args.switch_step is not None
            enqueue_and_run(
                "switch",
                switch_step=args.switch_step,
                switch_policy="round",
                first_al_round_0based=int(args.first_al_round),
            )
        else:
            assert args.switch_at is not None and args.switch_step is not None
            enqueue_and_run("switch", switch_step=args.switch_step, switch_frac=float(args.switch_at))
    else:
        # sweep
        steps = [int(x.strip()) for x in args.sweep_steps.split(",") if x.strip()]
        for s in steps:
            if s not in (1, 5, 10):
                raise SystemExit(f"Invalid sweep step {s}; allowed 1, 5, 10")
        _budget = read_budget_from_yaml(config_path, encoded=encoded, override=args.override_budget)
        n_rounds_est = count_acquisition_rounds(
            _budget,
            10**9,
            query_frac=args.query_frac,
            query_size=args.query_size,
            query_size_first=args.query_size_first,
            query_size_rest=args.query_size_rest,
            query_frac_first=args.query_frac_first,
            query_frac_rest=args.query_frac_rest,
        )
        use_round_grid = (args.sweep_switch_policy == "round") or (
            args.sweep_switch_policy == "auto" and is_size_mode
        )
        use_frac_grid = (args.sweep_switch_policy == "frac") or (
            args.sweep_switch_policy == "auto" and not is_size_mode
        )
        if use_round_grid and not is_size_mode:
            raise SystemExit(
                "--sweep round switch: use size query (--query_size or split-size) or set --sweep-switch-policy frac"
            )
        if use_round_grid and n_rounds_est < 2:
            print(
                "Warning: estimated <2 acquisition rounds; skip switch-by-round grid in sweep",
                file=sys.stderr,
            )
        enqueue_and_run("all_random")
        enqueue_and_run("all_al")
        for step in steps:
            if use_frac_grid:
                for sf in switch_fractions_for_step(step):
                    enqueue_and_run("switch", switch_step=step, switch_frac=sf)
            if use_round_grid and n_rounds_est >= 2:
                for r0 in switch_first_al_rounds_for_step(step, n_rounds_est):
                    enqueue_and_run(
                        "switch",
                        switch_step=step,
                        switch_policy="round",
                        first_al_round_0based=int(r0),
                    )

    _mq = query_results_dir_tag(
        query_size=args.query_size,
        query_size_first=args.query_size_first,
        query_size_rest=args.query_size_rest,
        query_frac=args.query_frac,
        query_frac_first=args.query_frac_first,
        query_frac_rest=args.query_frac_rest,
    )
    manifest_path = os.path.join(
        results_root,
        args.dataset,
        f"tranone_{args.al_agent}_{_mq}_manifest.json",
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": config_path,
                "al_agent": args.al_agent,
                "mode": args.mode,
                "switch_policy": args.switch_policy,
                "first_al_round": args.first_al_round,
                "sweep_switch_policy": args.sweep_switch_policy,
                "query_frac": args.query_frac,
                "query_size": args.query_size,
                "query_size_first": args.query_size_first,
                "query_size_rest": args.query_size_rest,
                "query_frac_first": args.query_frac_first,
                "query_frac_rest": args.query_frac_rest,
                "restarts": args.restarts,
                "runs": manifest,
            },
            f,
            indent=2,
        )
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
