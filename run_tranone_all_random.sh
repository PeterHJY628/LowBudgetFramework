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
CONFIG="${5:-${_SCRIPT_DIR}/TranOne/configs/tranone_${DATASET}.yaml}"
QUERY_FRAC_FIRST="${6:-}"
QUERY_FRAC_REST="${7:-}"

if [[ -n "${QUERY_FRAC_FIRST}" || -n "${QUERY_FRAC_REST}" ]]; then
  if [[ -z "${QUERY_FRAC_FIRST}" || -z "${QUERY_FRAC_REST}" ]]; then
    echo "Error: --query_frac_first and --query_frac_rest must be provided together." >&2
    exit 1
  fi
  if [[ -n "${QUERY_FRAC}" && "${QUERY_FRAC}" != "0.05" ]]; then
    echo "Error: provide either QUERY_FRAC or (QUERY_FRAC_FIRST and QUERY_FRAC_REST), not both." >&2
    exit 1
  fi
  QUERY_MODE_MSG="split first=${QUERY_FRAC_FIRST}, rest=${QUERY_FRAC_REST}"
else
  QUERY_MODE_MSG="uniform frac=${QUERY_FRAC}"
fi

echo "========================================"
echo "TranOne: mode=all_random"
echo "Repo cwd:       $(pwd)"
echo "Driver:         ${_TRANONE_PY}"
echo "Data folder:    ${DATA_FOLDER}"
echo "Dataset:        ${DATASET}"
echo "Query setting:  ${QUERY_MODE_MSG}"
echo "Restarts:       ${RESTARTS}"
echo "========================================"

PY_ARGS=(
  --data_folder "${DATA_FOLDER}"
  --dataset "${DATASET}"
  --config "${CONFIG}"
  --mode all_random
  --restarts "${RESTARTS}"
)

if [[ -n "${QUERY_FRAC_FIRST}" && -n "${QUERY_FRAC_REST}" ]]; then
  PY_ARGS+=(--query_frac_first "${QUERY_FRAC_FIRST}" --query_frac_rest "${QUERY_FRAC_REST}")
else
  PY_ARGS+=(--query_frac "${QUERY_FRAC}")
fi

python "${_TRANONE_PY}" \
  "${PY_ARGS[@]}"

echo "TranOne all_random finished."
