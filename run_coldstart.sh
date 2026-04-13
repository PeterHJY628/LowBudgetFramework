#!/bin/bash -l
#SBATCH --output=/users/%u/logs/%x_%j.out
#SBATCH --error=/users/%u/logs/%x_%j.err
#SBATCH --job-name=coldstart
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

# ── Cold-start experiment worker (mirrors run_agents.sh) ──────────────────
#
# Same interface as run_agents.sh, with two additions:
#   - always saves per-round checkpoints  (--save_checkpoints 1)
#   - supports --experiment_postfix via $8
#
# Usage:
#   sbatch run_coldstart.sh <config> <seed_start> <seeds_per_job> \
#          [dataset] [query_size] [fit_mode] [data_folder] [postfix] [agents...]
#
# Examples:
#   # Phase 1: scratch baseline, seeds 1-5
#   sbatch run_coldstart.sh configs/cifar10_coldstart.yaml 1 5 cifar10 20 from_scratch data_lib coldstart random entropy
#
#   # Phase 3: frozen pretrained, seeds 1-5
#   sbatch run_coldstart.sh configs/cifar10_coldstart_pretrained.yaml 1 5 cifar10 20 from_scratch data_lib coldstart_pretrained random entropy
#
#   # More agents, seeds 6-10
#   sbatch run_coldstart.sh configs/cifar10_coldstart.yaml 6 5 cifar10 20 from_scratch data_lib coldstart random entropy margin badge
#
#   # Local (no sbatch), 3 seeds
#   bash run_coldstart.sh configs/cifar10_coldstart.yaml 1 3 cifar10 20 from_scratch data_lib coldstart random entropy

export HF_TOKEN="${HF_TOKEN:-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}"

CONFIG="${1:?Error: config yaml path required}"
SEED_START="${2:?Error: seed_start required (e.g. 1)}"
SEEDS_PER_JOB="${3:?Error: seeds_per_job required (e.g. 5)}"
DATASET="${4:-cifar10}"
QUERY_SIZE="${5:-20}"
FIT_MODE="${6:-from_scratch}"
DATA_FOLDER="${7:-data_lib}"
POSTFIX="${8:-}"

# Remaining positional args ($9, $10, ...) are agent names.
# Default to random + entropy if none given.
shift $(( $# < 8 ? $# : 8 ))
if [[ $# -gt 0 ]]; then
  AGENTS=("$@")
else
  AGENTS=(random entropy)
fi

echo "========================================"
echo "Config:       ${CONFIG}"
echo "Seeds:        ${SEED_START} .. $((SEED_START + SEEDS_PER_JOB - 1))"
echo "Dataset:      ${DATASET}"
echo "Query size:   ${QUERY_SIZE}"
echo "Fit mode:     ${FIT_MODE}"
echo "Data folder:  ${DATA_FOLDER}"
echo "Postfix:      ${POSTFIX:-<none>}"
echo "Agents:       ${AGENTS[*]}"
echo "Checkpoints:  ON"
echo "========================================"

POSTFIX_ARG=""
if [[ -n "${POSTFIX}" ]]; then
  POSTFIX_ARG="--experiment_postfix ${POSTFIX}"
fi

for agent in "${AGENTS[@]}"; do
  echo "========================================"
  echo "Running agent: ${agent}  seeds ${SEED_START}..$(( SEED_START + SEEDS_PER_JOB - 1 ))"

  python evaluate.py \
    --data_folder "${DATA_FOLDER}" \
    --config "${CONFIG}" \
    --agent "${agent}" \
    --dataset "${DATASET}" \
    --query_size "${QUERY_SIZE}" \
    --run_id "${SEED_START}" \
    --restarts "${SEEDS_PER_JOB}" \
    --fitting_mode "${FIT_MODE}" \
    --save_checkpoints 1 \
    ${POSTFIX_ARG}
done

echo "All agents finished."
