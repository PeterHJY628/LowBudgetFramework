#!/bin/bash -l
#SBATCH --output=/users/%u/logs/exp1_al_%x_%j.out
#SBATCH --error=/users/%u/logs/exp1_al_%x_%j.err
#SBATCH --job-name=exp1_al
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail


# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv
# Usage: sbatch slurm_al.sh <seed> <agent> [data_folder] [results_dir]
#
# Example: sbatch slurm_al.sh 1 margin ../data_lib results

SEED="${1:?Error: seed required}"
AGENT="${2:?Error: agent required (random/margin/entropy/leastconfident)}"
DATA_FOLDER="${3:-../data_lib}"
RESULTS_DIR="${4:-results}"

cd "$(dirname "$0")"

echo "========================================"
echo "Exp1 Active Learning"
echo "  Seed:       ${SEED}"
echo "  Agent:      ${AGENT}"
echo "========================================"

python run_al.py \
    --data_folder "${DATA_FOLDER}" \
    --seed "${SEED}" \
    --agent "${AGENT}" \
    --results_dir "${RESULTS_DIR}"
