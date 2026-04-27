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
#   sbatch run_tranone_all_random.sh [data_folder] [restarts] [query_frac] [dataset] [config]
#     [query_frac_first] [query_frac_rest] [query_size] [query_size_first] [query_size_rest]
#
# Query mode (mutually exclusive): uniform frac (arg3), split-frac (6+7), fixed size (8), split-size (9+10). See all_al margin script.
#
# Examples:
#   sbatch run_tranone_all_random.sh /scratch/users/${USER}/data_lib 1 0.05 cifar10
#   sbatch run_tranone_all_random.sh /data/lib 1 0.05 cifar10 "" "" "" 32

# Resolve driver: works when this .sh lives in repo root OR inside TranOne/ (avoids TranOne/TranOne/...).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/run_tranone.py" ]]; then
  _REPO_ROOT="${_SCRIPT_DIR}"
elif [[ -f "${_SCRIPT_DIR}/TranOne/run_tranone.py" ]]; then
  _REPO_ROOT="${_SCRIPT_DIR}"
elif [[ -f "${_SCRIPT_DIR}/../TranOne/run_tranone.py" ]]; then
  _REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
else
  echo "Cannot find TranOne/run_tranone.py (run from repo root, or from bash_files/ with ../TranOne/)." >&2
  exit 1
fi
cd "${_REPO_ROOT}"
_TRANONE_PY="${_REPO_ROOT}/TranOne/run_tranone.py"

# module load anaconda3/2022.10-gcc-13.2.0
# source activate /scratch/users/${USER}/conda/myenv

DATA_FOLDER="${1:-data_lib}"
RESTARTS="${2:-1}"
QUERY_FRAC="${3:-0.05}"
DATASET="${4:-cifar10}"
CONFIG="${5:-${_REPO_ROOT}/TranOne/configs/tranone_${DATASET}.yaml}"
QUERY_FRAC_FIRST="${6:-}"
QUERY_FRAC_REST="${7:-}"
QUERY_SIZE="${8:-}"
QUERY_SIZE_FIRST="${9:-}"
QUERY_SIZE_REST="${10:-}"

if [[ ( -n "${QUERY_FRAC_FIRST}" && -z "${QUERY_FRAC_REST}" ) || ( -z "${QUERY_FRAC_FIRST}" && -n "${QUERY_FRAC_REST}" ) ]]; then
  echo "Error: --query_frac_first and --query_frac_rest must be provided together." >&2
  exit 1
fi
if [[ ( -n "${QUERY_SIZE_FIRST}" && -z "${QUERY_SIZE_REST}" ) || ( -z "${QUERY_SIZE_FIRST}" && -n "${QUERY_SIZE_REST}" ) ]]; then
  echo "Error: --query_size_first and --query_size_rest must be provided together." >&2
  exit 1
fi

_has_fs=0
[[ -n "${QUERY_FRAC_FIRST}" && -n "${QUERY_FRAC_REST}" ]] && _has_fs=1
_has_fxs=0
[[ -n "${QUERY_SIZE_FIRST}" && -n "${QUERY_SIZE_REST}" ]] && _has_fxs=1
_has_sfx=0
[[ -n "${QUERY_SIZE}" ]] && _has_sfx=1

_modes=0
[[ "${_has_fs}" -eq 1 ]] && _modes=$((_modes+1))
[[ "${_has_fxs}" -eq 1 ]] && _modes=$((_modes+1))
[[ "${_has_sfx}" -eq 1 ]] && _modes=$((_modes+1))
if [[ "${_modes}" -gt 1 ]]; then
  echo "Error: use only one of: split-frac, --query_size, or (--query_size_first and --query_size_rest)." >&2
  exit 1
fi
if [[ "${_has_fs}" -eq 1 ]]; then
  if [[ -n "${QUERY_FRAC}" && "${QUERY_FRAC}" != "0.05" ]]; then
    echo "Error: with split-frac, do not set a custom QUERY_FRAC in arg3 (or use 0.05 and rely on 6+7 only)." >&2
    exit 1
  fi
  QUERY_MODE_MSG="split frac: first=${QUERY_FRAC_FIRST}, rest=${QUERY_FRAC_REST}"
elif [[ "${_has_fxs}" -eq 1 ]]; then
  QUERY_MODE_MSG="split size: first=${QUERY_SIZE_FIRST}, rest=${QUERY_SIZE_REST}"
elif [[ "${_has_sfx}" -eq 1 ]]; then
  QUERY_MODE_MSG="fixed size: ${QUERY_SIZE}"
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

if [[ "${_has_fxs}" -eq 1 ]]; then
  PY_ARGS+=(--query_size_first "${QUERY_SIZE_FIRST}" --query_size_rest "${QUERY_SIZE_REST}")
elif [[ "${_has_sfx}" -eq 1 ]]; then
  PY_ARGS+=(--query_size "${QUERY_SIZE}")
elif [[ "${_has_fs}" -eq 1 ]]; then
  PY_ARGS+=(--query_frac_first "${QUERY_FRAC_FIRST}" --query_frac_rest "${QUERY_FRAC_REST}")
else
  PY_ARGS+=(--query_frac "${QUERY_FRAC}")
fi

python "${_TRANONE_PY}" \
  "${PY_ARGS[@]}"

echo "TranOne all_random finished."
