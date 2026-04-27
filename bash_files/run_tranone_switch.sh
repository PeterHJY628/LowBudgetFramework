#!/bin/bash -l
#SBATCH --output=/users/%u/logs/tranone_sw5_25pct_%j.out
#SBATCH --error=/users/%u/logs/tranone_sw5_25pct_%j.err
#SBATCH --job-name=tranone_sw25
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# TranOne: switch — random first, then margin (AL) after a threshold.
# (A) Budget-fraction: after round 0, random while labeled < switch_at * total_budget, else margin.
# (B) By round (fixed or split *size* only): after round 0, random for rounds 0..(first-1), margin from
#     `FIRST_AL_ROUND` (0-based first AL round). Set env SWITCH_POLICY=round and FIRST_AL_ROUND.
#     `switch_at` (arg 6) is not used; `switch_step` (arg 5) still names the same 1/5/10 run family.
#
# Usage:
#   sbatch run_tranone_switch.sh [data_folder] [restarts] [query_frac] [dataset] [switch_step] [switch_at] [config]
#     [query_frac_first] [query_frac_rest] [query_size] [query_size_first] [query_size_rest]
#
# Env (round mode, size query only) — use SWITCH_POLICY, not POLICY:
#   SWITCH_POLICY=round  FIRST_AL_ROUND=5
#
# Query mode (mutually exclusive): same as all_al / all_random (uniform, split-frac, fixed size, split-size).
#
# Examples:
#   sbatch run_tranone_switch.sh /scratch/users/${USER}/data_lib 1 0.05 cifar10
#   sbatch run_tranone_switch.sh /data/lib 1 0.05 cifar10 5 0.25
#   sbatch run_tranone_switch.sh /data/lib 1 0.05 cifar10 5 0.25 "" "" "" 32
#   # split-size: 40 then 10 per round
#   sbatch run_tranone_switch.sh /data/lib 1 0.05 cifar10 5 0.25 "" "" "" "" 40 10
#   # by round: first AL at 0-based round 5; needs query size
#   SWITCH_POLICY=round FIRST_AL_ROUND=5 sbatch run_tranone_switch.sh /data/lib 1 0.05 cifar10 5 0.50 "" "" "" 2000

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
SWITCH_STEP="${5:-5}"
SWITCH_AT="${6:-0.50}"
CONFIG="${7:-${_REPO_ROOT}/TranOne/configs/tranone_${DATASET}.yaml}"
QUERY_FRAC_FIRST="${8:-}"
QUERY_FRAC_REST="${9:-}"
QUERY_SIZE="${10:-}"
QUERY_SIZE_FIRST="${11:-}"
QUERY_SIZE_REST="${12:-}"

SWITCH_POLICY="${SWITCH_POLICY:-frac}"
FIRST_AL_ROUND="${FIRST_AL_ROUND:-}"
AL_AGENT="${AL_AGENT:-margin}"

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
    echo "Error: with split-frac, do not set a custom QUERY_FRAC in arg3 (or use 0.05 and rely on 8+9 only)." >&2
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

if [[ "${SWITCH_POLICY}" == "round" ]]; then
  if [[ "${_has_sfx}" -ne 1 && "${_has_fxs}" -ne 1 ]]; then
    echo "Error: SWITCH_POLICY=round requires --query_size or (query_size_first+rest)." >&2
    exit 1
  fi
  if [[ -z "${FIRST_AL_ROUND}" ]]; then
    echo "Error: set FIRST_AL_ROUND (0-based first AL round) with SWITCH_POLICY=round." >&2
    exit 1
  fi
fi

echo "========================================"
if [[ "${SWITCH_POLICY}" == "round" ]]; then
  echo "TranOne: mode=switch (by round) step=${SWITCH_STEP} first_al_0based=${FIRST_AL_ROUND}"
else
  echo "TranOne: mode=switch (by budget fraction) step=${SWITCH_STEP}% family switch_at=${SWITCH_AT}"
fi
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
  --mode switch
  --switch-step "${SWITCH_STEP}"
  --al_agent "${AL_AGENT}"
  --restarts "${RESTARTS}"
)

if [[ "${SWITCH_POLICY}" == "round" ]]; then
  PY_ARGS+=(--switch-policy round --first-al-round "${FIRST_AL_ROUND}")
else
  PY_ARGS+=(--switch-at "${SWITCH_AT}")
fi

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

if [[ "${SWITCH_POLICY}" == "round" ]]; then
  echo "TranOne switch (rstep ${SWITCH_STEP} first_al ${FIRST_AL_ROUND}) finished."
else
  echo "TranOne switch (frac grid step ${SWITCH_STEP} @ ${SWITCH_AT}) finished."
fi
