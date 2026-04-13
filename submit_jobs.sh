#!/bin/bash
set -euo pipefail

# Submit multiple SLURM jobs, each running a different seed range.
#
# Usage:
#   bash submit_jobs.sh <config_yaml> [dataset] [query_size] [num_jobs] [seeds_per_job] [data_folder]
#
# Examples:
#   bash submit_jobs.sh configs/cifar10.yaml cifar10 20
#   bash submit_jobs.sh configs/cifar10.yaml cifar10 20 5 10 data_lib
#   bash submit_jobs.sh configs/cifar10_init4_q10.yaml cifar10 10 5 10 data_lib
#
# Defaults: 5 jobs x 10 seeds = 50 seeds total (seeds 1-50)

CONFIG="${1:?Error: config yaml path required (e.g. configs/cifar10.yaml)}"
DATASET="${2:-cifar10}"
QUERY_SIZE="${3:-20}"
NUM_JOBS="${4:-5}"
SEEDS_PER_JOB="${5:-10}"
DATA_FOLDER="${6:-data_lib}"

CONFIG_BASENAME="$(basename "${CONFIG}" .yaml)"

echo "========================================"
echo "Submitting ${NUM_JOBS} jobs x ${SEEDS_PER_JOB} seeds = $(( NUM_JOBS * SEEDS_PER_JOB )) total seeds"
echo "Config:      ${CONFIG}"
echo "Dataset:     ${DATASET}"
echo "Query size:  ${QUERY_SIZE}"
echo "Data folder: ${DATA_FOLDER}"
echo "========================================"

for (( i=0; i<NUM_JOBS; i++ )); do
  SEED_START=$(( i * SEEDS_PER_JOB + 1 ))
  JOB_NAME="${CONFIG_BASENAME}_q${QUERY_SIZE}_s${SEED_START}"

  echo "  Job $((i+1))/${NUM_JOBS}: seeds ${SEED_START}..$(( SEED_START + SEEDS_PER_JOB - 1 )), job-name=${JOB_NAME}"

  sbatch --job-name="${JOB_NAME}" \
    run_agents.sh "${CONFIG}" "${SEED_START}" "${SEEDS_PER_JOB}" "${DATASET}" "${QUERY_SIZE}" "${DATA_FOLDER}"
done

echo "========================================"
echo "All ${NUM_JOBS} jobs submitted."
