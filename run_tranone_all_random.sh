#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/tranone_all_random_%j.out
#SBATCH --error=/scratch/users/%u/logs/tranone_all_random_%j.err
#SBATCH --job-name=tranone_rand
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# TranOne: single condition — all acquisition rounds use random sampling.
#
# Usage:
#   sbatch run_tranone_all_random.sh [data_folder] [restarts] [query_frac] [dataset]
#
# query_frac: each round labels max(1, round(query_frac * total_budget)) samples (e.g. 0.05 = 5%% of budget).
#
# Examples:
#   sbatch run_tranone_all_random.sh /scratch/users/${USER}/data_lib 1 0.05 cifar10

REPO_ROOT="/scratch/users/k25130670/TranTest"
cd /scratch/users/k25130670/TranTest/TranOne

# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

DATA_FOLDER="${1:-/scratch/users/k25130670/TranTest/data_lib}"
RESTARTS="${2:-1}"
QUERY_FRAC="${3:-0.05}"
DATASET="${4:-cifar10}"

echo "========================================"
echo "TranOne: mode=all_random"
echo "Repo:           ${REPO_ROOT}"
echo "Data folder:    ${DATA_FOLDER}"
echo "Dataset:        ${DATASET}"
echo "Query frac:     ${QUERY_FRAC} (batch per round ~ this fraction of budget)"
echo "Restarts:       ${RESTARTS}"
echo "========================================"

python TranOne/run_tranone.py \
  --data_folder "${DATA_FOLDER}" \
  --dataset "${DATASET}" \
  --mode all_random \
  --query_frac "${QUERY_FRAC}" \
  --restarts "${RESTARTS}"

echo "TranOne all_random finished."
