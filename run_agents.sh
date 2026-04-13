#!/bin/bash -l
#SBATCH --output=/users/%u/logs/%x_%j.out
#SBATCH --error=/users/%u/logs/%x_%j.err
#SBATCH --job-name=compare
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# Usage:
#   sbatch run_agents.sh <config_yaml> <seed_start> <seeds_per_job> [dataset] [query_size] [data_folder]
#
# Examples:
#   sbatch run_agents.sh configs/cifar10.yaml 1 10 cifar10 20 data_lib
#   sbatch run_agents.sh configs/cifar10_init4_q10.yaml 11 10 cifar10 10 data_lib

# cd /users/k25130670/CSA-DINOv3

# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

# export HF_HOME="/scratch/users/k25130670/cache/huggingface/"
export HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN environment variable not set}"

CONFIG="${1:?Error: config yaml path required (e.g. configs/cifar10.yaml)}"
SEED_START="${2:?Error: seed_start required (e.g. 1)}"
SEEDS_PER_JOB="${3:?Error: seeds_per_job required (e.g. 10)}"
DATASET="${4:-cifar10}"
QUERY_SIZE="${5:-20}"
FIT_MODE="${6:-False}"
DATA_FOLDER="${7:-data_lib}"

AGENTS=(
  random
  leastconfident
  entropy
  coreset
  typiclust
  badge
)

echo "========================================"
echo "Config:       ${CONFIG}"
echo "Seeds:        ${SEED_START} .. $((SEED_START + SEEDS_PER_JOB - 1))"
echo "Dataset:      ${DATASET}"
echo "Query size:   ${QUERY_SIZE}"
echo "Fit mode:     ${FIT_MODE}"
echo "Data folder:  ${DATA_FOLDER}"
echo "========================================"

for agent in "${AGENTS[@]}"; do
  echo "========================================"
  echo "Running agent: ${agent}"
  echo "dataset=${DATASET}, config=${CONFIG}, query_size=${QUERY_SIZE}"
  echo "seed_start=${SEED_START}, seeds_per_job=${SEEDS_PER_JOB}"

  python evaluate.py \
    --data_folder "${DATA_FOLDER}" \
    --config "${CONFIG}" \
    --agent "${agent}" \
    --dataset "${DATASET}" \
    --query_size "${QUERY_SIZE}" \
    --run_id "${SEED_START}" \
    --restarts "${SEEDS_PER_JOB}" \
    --fitting_mode "${FIT_MODE}"
done

echo "All agents finished."
