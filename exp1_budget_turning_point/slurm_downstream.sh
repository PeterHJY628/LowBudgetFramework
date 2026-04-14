#!/bin/bash -l
#SBATCH --output=/users/%u/logs/exp1_downstream_%j.out
#SBATCH --error=/users/%u/logs/exp1_downstream_%j.err
#SBATCH --job-name=exp1_downstream
#SBATCH --partition=gpu
#SBATCH --constraint=h200   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail


# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

# Usage: sbatch slurm_downstream.sh <seed> [data_folder] [results_dir]

SEED="${1:?Error: seed required}"
DATA_FOLDER="${2:-../data_lib}"
RESULTS_DIR="${3:-results}"

cd "$(dirname "$0")"

echo "========================================"
echo "Exp1 Downstream Evaluation"
echo "  Seed:       ${SEED}"
echo "========================================"

python evaluate_downstream.py \
    --data_folder "${DATA_FOLDER}" \
    --seed "${SEED}" \
    --results_dir "${RESULTS_DIR}"
