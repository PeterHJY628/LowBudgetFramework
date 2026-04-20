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
echo "TranOne: mode=all_al (margin after round 0)"
echo "Repo cwd:       $(pwd)"
echo "Driver:         ${_TRANONE_PY}"
echo "Data folder:    ${DATA_FOLDER}"
echo "Dataset:        ${DATASET}"
echo "Query frac:     ${QUERY_FRAC}"
echo "Restarts:       ${RESTARTS}"
echo "========================================"

python "${_TRANONE_PY}" \
  --data_folder "${DATA_FOLDER}" \
  --dataset "${DATASET}" \
  --mode all_al \
  --al_agent margin \
  --query_frac "${QUERY_FRAC}" \
  --restarts "${RESTARTS}"

echo "TranOne all_al (margin) finished."
