#!/bin/bash -l
#SBATCH --output=/users/%u/logs/tranone_all_random_%j.out
#SBATCH --error=/users/%u/logs/tranone_all_random_%j.err
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

# Resolve driver: works when this .sh lives in repo root OR inside TranOne/ (avoids TranOne/TranOne/...).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/run_tranone.py" ]]; then
  cd "${_SCRIPT_DIR}"
  _TRANONE_PY="${_SCRIPT_DIR}/run_tranone.py"
elif [[ -f "${_SCRIPT_DIR}/TranOne/run_tranone.py" ]]; then
  cd "${_SCRIPT_DIR}"
  _TRANONE_PY="${_SCRIPT_DIR}/TranOne/run_tranone.py"
else
  echo "Cannot find TranOne/run_tranone.py (place script in TranTest root or TranOne/)." >&2
  exit 1
fi

# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

DATA_FOLDER="${1:-data_lib}"
RESTARTS="${2:-1}"
QUERY_FRAC="${3:-0.05}"
DATASET="${4:-cifar10}"

echo "========================================"
echo "TranOne: mode=all_random"
echo "Repo cwd:       $(pwd)"
echo "Driver:         ${_TRANONE_PY}"
echo "Data folder:    ${DATA_FOLDER}"
echo "Dataset:        ${DATASET}"
echo "Query frac:     ${QUERY_FRAC} (batch per round ~ this fraction of budget)"
echo "Restarts:       ${RESTARTS}"
echo "========================================"

python "${_TRANONE_PY}" \
  --data_folder "${DATA_FOLDER}" \
  --dataset "${DATASET}" \
  --mode all_random \
  --query_frac "${QUERY_FRAC}" \
  --restarts "${RESTARTS}"

echo "TranOne all_random finished."
