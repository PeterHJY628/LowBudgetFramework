# TranOne — random vs. pure AL vs. random→AL switch

This folder adds an experiment driver on top of the TranTest active-learning stack (`ALGame`, agents, datasets). All runs write under `TranOne/results/` (configurable with `--results_root`).

**Per-round batch size** is chosen in one of three ways (then always capped by remaining budget and pool size):

1. **Uniform fraction (default):** `--query_frac` — every round uses `max(1, round(query_frac * budget))`. Example: budget 200, `query_frac=0.05` → 10 labels per round until the last partial batch.
2. **Fixed integer:** `--query_size N` — every round uses `min(N, remaining, pool)` (same `N`, at least 1). Mutually exclusive with split-frac.
3. **Split fractions:** `--query_frac_first F` **and** `--query_frac_rest R` — **round 0 (第一轮)** uses `max(1, round(F * budget))`; **round ≥ 1** uses `max(1, round(R * budget))`. Mutually exclusive with `--query_size`.

## What is compared

1. **All random** (`mode_all_random`): every acquisition round uses `RandomAgent` (**round 0 = first round**, always random; later rounds also random).
2. **All AL** (`mode_all_al`): **round 0 = first round** is always random; from **round 1** (second round / 第二轮) onward the chosen AL agent (e.g. `margin`) is used every time (“first query random, then always AL”).
3. **Switch** (`mode_switch_step{pct}pct_at{frac}`): same **round 0 = first round** random rule; from round 1 onward, **random** until cumulative labeled samples reach `switch_frac * budget`, then **only the AL agent** until the budget is exhausted.

There is **no initial labeled seed set**: `initial_points_per_class` is forced to `0` for each run. The first labeled data appear only after the first query.

### Round indexing（轮次约定）

- 代码与日志里 `round` 为 **从 0 起计** 的轮次索引。
- **默认约定：第 0 轮就是第一轮**（第一轮查询一律随机采样；与「第 1 轮、第 2 轮…」口语对应时：`round_1based = round + 1`）。
- `tranone_trace.jsonl` 每条记录同时包含 `round`（0-based）和 `round_1based`（1-based，便于对齐「第 k 轮」表述）。

## Switch grids (budget fractions)

For the switch family, three grids of `switch_frac` are used (each value is one separate experiment):

| Grid name | Step | Switch fractions `switch_frac` |
|-----------|------|----------------------------------|
| 1% step   | 1    | 1%, 2%, …, 99% of total budget   |
| 5% step   | 5    | 5%, 10%, …, 95%                  |
| 10% step  | 10   | 10%, 20%, …, 90%                 |

Implementation detail: at the **start** of each round, if `added_images < max(1, round(switch_frac * budget))` the policy is random (**第 0 轮 / 第一轮** 恒为随机，与阈值无关)；自第二轮起再按该不等式在 random 与 AL 间切换。每轮标注条数为 `max(1, round(query_frac * budget))`（再与剩余预算、未标注池大小取 min）；若 `query_frac` 较大，少数几轮就会跨过很小的切换阈值，可酌情调小 `query_frac`。

## How to run

From the **TranTest** repository root:

```bash
# Full sweep: all_random + all_al + all switch cells (128 conditions × restarts)
# Each round labels max(1, round(query_frac * budget)) points (default query_frac=0.05 → 5% of budget per round if budget=200 → 10).
python TranOne/run_tranone.py --data_folder /path/to/data_lib --dataset cifar10 \
  --al_agent margin --query_frac 0.05 --restarts 3
# default --mode sweep

# Partial sweep (only 5% and 10% grids → fewer switch runs)
python TranOne/run_tranone.py --data_folder /path/to/data_lib --dataset cifar10 \
  --al_agent margin --query_frac 0.05 --restarts 1 --sweep-steps 5,10

# Single conditions
python TranOne/run_tranone.py --data_folder /path/to/data_lib --mode all_random --restarts 1
python TranOne/run_tranone.py --data_folder /path/to/data_lib --mode all_al --al_agent entropy --restarts 1
python TranOne/run_tranone.py --data_folder /path/to/data_lib --mode switch --switch-step 5 --switch-at 0.25 \
  --al_agent margin --query_frac 0.01 --restarts 1

# Fixed batch size (20 labels per round when budget allows)
python TranOne/run_tranone.py --data_folder /path/to/data_lib --mode all_al --query_size 20 --restarts 1

# Split: 第一轮用总预算的 1% 取整，之后每轮用总预算的 5% 取整
python TranOne/run_tranone.py --data_folder /path/to/data_lib --mode all_al \
  --query_frac_first 0.01 --query_frac_rest 0.05 --restarts 1
```

Default config: `TranOne/configs/tranone_cifar10.yaml` (supervised **ResNet-18**, `from_scratch`, no DINOv3). Pass `--config` to use another YAML.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--query_frac` | Uniform: each round `max(1, round(frac * budget))` (default `0.05`). Unused if `--query_size` or split-frac pair is set. |
| `--query_size` | Fixed: integer batch size every round (≥1). |
| `--query_frac_first` / `--query_frac_rest` | Split: first round vs later rounds each use `max(1, round(frac * budget))`; **both** required together. |
| `--al_agent` | AL method after random phases (`margin`, `entropy`, `leastconfident`, …) |
| `--fitting_mode` | `from_scratch` (default), `finetuning`, or `shrinking` |
| `--override_budget` | Override YAML budget (smoke tests) |
| `--results_root` | Output root (default: `TranOne/results`) |
| `--encoded 1` | Use embedded / SimCLR pathway from YAML |

## Outputs

Per condition and repetition:

- `runs`-style layout: `TranOne/results/<Dataset>/tranone_<qtag>_B<budget>/<run_name>/run_<id>/` where `<qtag>` is e.g. `qfrac0p05`, `qsize10`, or `qff0p01_qfr0p05`
  - `accuracies.csv`, `losses.csv` — same schema as `EnvironmentLogger` in the main codebase
  - `tranone_trace.jsonl` — per round: `round`, `round_1based`, `mode`, **`query_size`** (actual batch), **`query_budget_before_cap`**, **`query_mode`** (`uniform_frac` / `fixed_size` / `split_frac`) and related fields (`query_frac`, `query_frac_applied`, …), `added_images`, `test_accuracy`, `test_loss`
  - `meta.txt` — dataset + agent metadata

After each condition, `collect_results` aggregates `accuracies.csv` / `losses.csv` across `run_*` in that folder.

A **manifest** JSON is written at the end of a driver invocation:

`TranOne/results/<dataset>/tranone_<al_agent>_<qtag>_manifest.json`

## Code files

| File | Role |
|------|------|
| `tranone_al_game.py` | `TranOneALGame`: skips the first classifier fit when the labeled pool is empty (required for `initial_points_per_class = 0`). |
| `run_tranone.py` | CLI, mode logic, sweep over switch grids, manifest. |
| `configs/tranone_cifar10.yaml` | Default CIFAR-10 settings for TranOne. |

## Sweep size

Full `--mode sweep` with default `--sweep-steps 1,5,10` runs **128** acquisition experiments per restart count:

- 1 × all_random  
- 1 × all_al  
- 99 + 19 + 9 = **127** switch configurations  

Use `--sweep-steps` to reduce this, or run `--mode switch` for individual `(step, switch_at)` cells.
