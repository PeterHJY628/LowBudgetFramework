#!/bin/bash
set -euo pipefail

# =================================================================
# Master submission script for Experiment 1.
#
# Submits all SLURM jobs for a given seed:
#   Phase 1: 20 SimCLR training jobs (5%, 10%, ..., 100%)
#   Phase 2: 1 downstream evaluation job (depends on Phase 1)
#   Phase 3: 4 AL experiment jobs (depends on Phase 2)
#   Phase 4: 1 visualization job (depends on Phase 3)
#
# Usage:
#   bash submit_jobs.sh <seed> [data_folder] [results_dir] [epochs]
#
# Examples:
#   bash submit_jobs.sh 1
#   bash submit_jobs.sh 1 ../data_lib results 500
#   bash submit_jobs.sh 42 ../data_lib results 300
# =================================================================

SEED="${1:?Error: seed required (e.g. 1)}"
DATA_FOLDER="${2:-../data_lib}"
RESULTS_DIR="${3:-results}"
EPOCHS="${4:-500}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Ensure log directory exists
mkdir -p "/users/${USER}/logs"

FRACTIONS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
           0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
AGENTS=(random margin entropy leastconfident)

echo "================================================================"
echo "Experiment 1: Budget Turning Point"
echo "  Seed:        ${SEED}"
echo "  Data folder: ${DATA_FOLDER}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Epochs:      ${EPOCHS}"
echo "  Fractions:   ${#FRACTIONS[@]}"
echo "  AL Agents:   ${AGENTS[*]}"
echo "================================================================"

# ---- Phase 1: SimCLR training for each fraction ----
echo ""
echo "Phase 1: Submitting ${#FRACTIONS[@]} SimCLR training jobs..."
SIMCLR_JOBIDS=()

for frac in "${FRACTIONS[@]}"; do
    frac_pct=$(printf "%.0f" "$(echo "${frac} * 100" | bc)")
    job_name="exp1_s${SEED}_f${frac_pct}"

    JOBID=$(sbatch --parsable \
        --job-name="${job_name}" \
        slurm_simclr.sh "${SEED}" "${frac}" "${DATA_FOLDER}" "${RESULTS_DIR}" "${EPOCHS}")

    SIMCLR_JOBIDS+=("${JOBID}")
    echo "  Submitted fraction=${frac} -> JobID=${JOBID}"
done

# Build dependency string: all SimCLR jobs must finish
SIMCLR_DEP=$(IFS=:; echo "${SIMCLR_JOBIDS[*]}")

# ---- Phase 2: Downstream evaluation (after all SimCLR jobs) ----
echo ""
echo "Phase 2: Submitting downstream evaluation job..."
DOWNSTREAM_JOBID=$(sbatch --parsable \
    --dependency=afterok:${SIMCLR_DEP} \
    --job-name="exp1_s${SEED}_downstream" \
    slurm_downstream.sh "${SEED}" "${DATA_FOLDER}" "${RESULTS_DIR}")
echo "  Submitted downstream eval -> JobID=${DOWNSTREAM_JOBID} (depends on Phase 1)"

# ---- Phase 3: AL experiments (after downstream evaluation) ----
echo ""
echo "Phase 3: Submitting ${#AGENTS[@]} AL experiment jobs..."
AL_JOBIDS=()

for agent in "${AGENTS[@]}"; do
    JOBID=$(sbatch --parsable \
        --dependency=afterok:${DOWNSTREAM_JOBID} \
        --job-name="exp1_s${SEED}_al_${agent}" \
        slurm_al.sh "${SEED}" "${agent}" "${DATA_FOLDER}" "${RESULTS_DIR}")

    AL_JOBIDS+=("${JOBID}")
    echo "  Submitted agent=${agent} -> JobID=${JOBID}"
done

AL_DEP=$(IFS=:; echo "${AL_JOBIDS[*]}")

# ---- Phase 4: Visualization (after all AL jobs) ----
echo ""
echo "Phase 4: Submitting visualization job..."
CONDA_ENV="/cephfs/volumes/hpc_data_usr/${USER}/0cbdf2aa-8024-47aa-a2e5-c4fb57faf553/conda/myenv"
VIZ_JOBID=$(sbatch --parsable \
    --dependency=afterok:${AL_DEP} \
    --job-name="exp1_s${SEED}_viz" \
    --partition=gpu \
    --gpus=1 \
    --mem=16G \
    --wrap="source activate ${CONDA_ENV} 2>/dev/null || conda activate ${CONDA_ENV}; cd ${SCRIPT_DIR} && python visualize.py --seed ${SEED} --results_dir ${RESULTS_DIR} --data_folder ${DATA_FOLDER}")
echo "  Submitted visualization -> JobID=${VIZ_JOBID}"

# ---- Summary ----
echo ""
echo "================================================================"
echo "All jobs submitted for seed=${SEED}!"
echo ""
echo "  Phase 1 (SimCLR x${#FRACTIONS[@]}): ${SIMCLR_JOBIDS[*]}"
echo "  Phase 2 (Downstream):     ${DOWNSTREAM_JOBID}"
echo "  Phase 3 (AL x${#AGENTS[@]}):         ${AL_JOBIDS[*]}"
echo "  Phase 4 (Visualization):  ${VIZ_JOBID}"
echo ""
echo "Monitor with: squeue -u \${USER}"
echo "================================================================"
