#!/bin/bash
set -euo pipefail

# ── Submit multiple SLURM jobs for cold-start experiments ─────────────────
#
# Splits seeds across jobs so each job fits within the 48-hour wall time.
# Same pattern as submit_jobs.sh, calls run_coldstart.sh instead of run_agents.sh.
#
# Usage:
#   bash submit_coldstart_jobs.sh <config> [dataset] [query_size] \
#        [num_jobs] [seeds_per_job] [fit_mode] [data_folder] [postfix] [agents...]
#
# Examples:
#   # 5 jobs × 1 seed = 5 seeds (safe for 48h wall time per seed)
#   bash submit_coldstart_jobs.sh configs/cifar10_coldstart.yaml cifar10 20 5 1 from_scratch data_lib coldstart random entropy
#
#   # 3 jobs × 2 seeds = 6 seeds
#   bash submit_coldstart_jobs.sh configs/cifar10_coldstart.yaml cifar10 20 3 2 from_scratch data_lib coldstart
#
#   # Phase 3: pretrained backbone
#   bash submit_coldstart_jobs.sh configs/cifar10_coldstart_pretrained.yaml cifar10 20 5 1 from_scratch data_lib coldstart_pretrained random entropy

CONFIG="${1:?Error: config yaml path required}"
DATASET="${2:-cifar10}"
QUERY_SIZE="${3:-20}"
NUM_JOBS="${4:-5}"
SEEDS_PER_JOB="${5:-1}"
FIT_MODE="${6:-from_scratch}"
DATA_FOLDER="${7:-data_lib}"
POSTFIX="${8:-coldstart}"

# Agents: $9 onwards, or default
shift $(( $# < 8 ? $# : 8 ))
if [[ $# -gt 0 ]]; then
  AGENTS=("$@")
else
  AGENTS=(random entropy)
fi

CONFIG_BASENAME="$(basename "${CONFIG}" .yaml)"
TOTAL_SEEDS=$(( NUM_JOBS * SEEDS_PER_JOB ))

echo "========================================"
echo "Submitting ${NUM_JOBS} jobs × ${SEEDS_PER_JOB} seeds = ${TOTAL_SEEDS} total seeds"
echo "Config:      ${CONFIG}"
echo "Dataset:     ${DATASET}"
echo "Query size:  ${QUERY_SIZE}"
echo "Fit mode:    ${FIT_MODE}"
echo "Data folder: ${DATA_FOLDER}"
echo "Postfix:     ${POSTFIX}"
echo "Agents:      ${AGENTS[*]}"
echo "========================================"

for (( i=0; i<NUM_JOBS; i++ )); do
  SEED_START=$(( i * SEEDS_PER_JOB + 1 ))
  SEED_END=$(( SEED_START + SEEDS_PER_JOB - 1 ))
  JOB_NAME="${CONFIG_BASENAME}_s${SEED_START}-${SEED_END}"

  echo "  Job $((i+1))/${NUM_JOBS}: seeds ${SEED_START}..${SEED_END}  name=${JOB_NAME}"

  sbatch --job-name="${JOB_NAME}" \
    run_coldstart.sh \
    "${CONFIG}" "${SEED_START}" "${SEEDS_PER_JOB}" \
    "${DATASET}" "${QUERY_SIZE}" "${FIT_MODE}" "${DATA_FOLDER}" \
    "${POSTFIX}" "${AGENTS[@]}"
done

echo "========================================"
echo "All ${NUM_JOBS} jobs submitted."
