#!/bin/bash -l
#SBATCH --output=/users/%u/logs/exp1_simclr_%x_%j.out
#SBATCH --error=/users/%u/logs/exp1_simclr_%x_%j.err
#SBATCH --job-name=exp1_simclr
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail


# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

# Usage: sbatch slurm_simclr.sh <seed> <fraction> [data_folder] [results_dir] [max_epochs] [patience]
#
# Example: sbatch slurm_simclr.sh 1 0.05 ../data_lib results

SEED="${1:?Error: seed required}"
FRACTION="${2:?Error: fraction required (e.g. 0.05)}"
DATA_FOLDER="${3:-../data_lib}"
RESULTS_DIR="${4:-results}"
MAX_EPOCHS="${5:-500}"
PATIENCE="${6:-30}"

cd "$(dirname "$0")"

echo "========================================"
echo "Exp1 SimCLR Training"
echo "  Seed:       ${SEED}"
echo "  Fraction:   ${FRACTION}"
echo "  Data:       ${DATA_FOLDER}"
echo "  Results:    ${RESULTS_DIR}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Patience:   ${PATIENCE}"
echo "========================================"

python train_simclr_fraction.py \
    --data_folder "${DATA_FOLDER}" \
    --seed "${SEED}" \
    --fraction "${FRACTION}" \
    --results_dir "${RESULTS_DIR}" \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}"
