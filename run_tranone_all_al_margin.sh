#!/bin/bash -l
#SBATCH --output=/users/%u/logs/tranone_all_al_margin_%j.out
#SBATCH --error=/users/%u/logs/tranone_all_al_margin_%j.err
#SBATCH --job-name=tranone_alm
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# TranOne: round 0 random, then margin for every later round (mode_all_al).
#
# Usage:
#   sbatch run_tranone_all_al_margin.sh [data_folder] [restarts] [query_frac] [dataset]
#
# query_frac: each round labels max(1, round(query_frac * total_budget)) samples.
#
# Examples:
#   sbatch run_tranone_all_al_margin.sh /scratch/users/${USER}/data_lib 1 0.05 cifar10

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

DATA_FOLDER="${1:-data_lib}"
RESTARTS="${2:-1}"
QUERY_FRAC="${3:-0.05}"
DATASET="${4:-cifar10}"

echo "========================================"
echo "TranOne: mode=all_al (margin after round 0)"
echo "Repo:           ${REPO_ROOT}"
echo "Data folder:    ${DATA_FOLDER}"
echo "Dataset:        ${DATASET}"
echo "Query frac:     ${QUERY_FRAC}"
echo "Restarts:       ${RESTARTS}"
echo "========================================"

python TranOne/run_tranone.py \
  --data_folder "${DATA_FOLDER}" \
  --dataset "${DATASET}" \
  --mode all_al \
  --al_agent margin \
  --query_frac "${QUERY_FRAC}" \
  --restarts "${RESTARTS}"

echo "TranOne all_al (margin) finished."
