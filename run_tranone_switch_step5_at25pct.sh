#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/tranone_sw5_25pct_%j.out
#SBATCH --error=/scratch/users/%u/logs/tranone_sw5_25pct_%j.err
#SBATCH --job-name=tranone_sw25
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# TranOne: switch at 25% of total budget (5% grid family), random before threshold then margin.
# Round 0 is always random; from round 1 onward random while added_images < 0.25*budget, else margin.
#
# Usage:
#   sbatch run_tranone_switch_step5_at25pct.sh [data_folder] [restarts] [query_frac] [dataset]
#
# query_frac: each round labels max(1, round(query_frac * total_budget)) samples.
#
# Examples:
#   sbatch run_tranone_switch_step5_at25pct.sh /scratch/users/${USER}/data_lib 1 0.05 cifar10

REPO_ROOT="/scratch/users/k25130670/TranTest"
cd /scratch/users/k25130670/TranTest/TranOne

# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

DATA_FOLDER="${1:-/scratch/users/k25130670/TranTest/data_lib}"
RESTARTS="${2:-1}"
QUERY_FRAC="${3:-0.05}"
DATASET="${4:-cifar10}"

echo "========================================"
echo "TranOne: mode=switch step=5% grid switch_at=0.25 (25% budget)"
echo "Repo:           ${REPO_ROOT}"
echo "Data folder:    ${DATA_FOLDER}"
echo "Dataset:        ${DATASET}"
echo "Query frac:     ${QUERY_FRAC}"
echo "Restarts:       ${RESTARTS}"
echo "========================================"

python TranOne/run_tranone.py \
  --data_folder "${DATA_FOLDER}" \
  --dataset "${DATASET}" \
  --mode switch \
  --switch-step 5 \
  --switch-at 0.25 \
  --al_agent margin \
  --query_frac "${QUERY_FRAC}" \
  --restarts "${RESTARTS}"

echo "TranOne switch (5% grid @ 25%) finished."
