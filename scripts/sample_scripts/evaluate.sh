#!/bin/bash
# Evaluate relaxed structures against reference set.
# Computes: coverage, energy efficiency, sample budget.
# Usage: composition=si config=mi bash scripts/sample_scripts/evaluate.sh

PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0}

# --- Required ---
composition=${composition:?"composition is required (e.g. si, al2o3)"}
config=${config:?"config is required (e.g. mi, mn, diffusion)"}

composition_lower=$(echo "$composition" | tr '[:upper:]' '[:lower:]')
config_lower=$(echo "$config" | tr '[:upper:]' '[:lower:]')

sample_dir="${PROJECT_ROOT}/exp_results/${composition_lower}/relaxed/${config_lower}"
reference_dir="${PROJECT_ROOT}/exp_results/${composition_lower}/deduped"
# --- Force field for energy calculation ---
potential_path=${potential_path:-"${PROJECT_ROOT}/pretrained_mattersim/mattersim-v1.0.0-5M.pth"}

# --- Evaluation parameters (hardcoded for consistency) ---
WORKERS=32
BIN_SIZE=10.0
ENERGY_THRESHOLD=0.1
BUDGET_THRESHOLD=1.0

echo "=== Evaluation ==="
echo "Composition: ${composition_lower}, Config: ${config_lower}"
echo "Samples: ${sample_dir}"
echo "Reference: ${reference_dir}"

source "${PROJECT_ROOT}/.venv/bin/activate"

# 1. Coverage
echo "--- Computing coverage ---"
python sample_eval_utils/calc_coverage.py \
    --potential_path="${potential_path}" \
    --sample_dir="${sample_dir}" \
    --reference_dir="${reference_dir}" \
    --workers="${WORKERS}" \
    --bin_size="${BIN_SIZE}"

# 2. Energy efficiency
echo "--- Computing energy efficiency ---"
python sample_eval_utils/calc_efficiency.py \
    --potential_path="${potential_path}" \
    --sample_dir="${sample_dir}"

# 3. Sample budget
echo "--- Computing sample budget ---"
python sample_eval_utils/evaluate_efficiency.py \
    --potential_path="${potential_path}" \
    --sample_dir="${sample_dir}" \
    --reference_dir="${reference_dir}" \
    --energy_threshold="${ENERGY_THRESHOLD}" \
    --budget_threshold="${BUDGET_THRESHOLD}"

echo "Evaluation complete."
