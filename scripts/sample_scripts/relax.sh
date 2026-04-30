#!/bin/bash
# Post-relaxation of sampled or RSS structures using MatterSim or UMA force field.
# Usage: composition=si config=mi bash scripts/sample_scripts/relax.sh
#        composition=si config=rss bash scripts/sample_scripts/relax.sh

PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0}

# --- Required ---
composition=${composition:?"composition is required (e.g. si, al2o3)"}
config=${config:?"config is required (e.g. rss, mi, mn)"}

composition_lower=$(echo "$composition" | tr '[:upper:]' '[:lower:]')
config_lower=$(echo "$config" | tr '[:upper:]' '[:lower:]')

# Input: rss_proposal/ for config=rss, sampled/{config}/ otherwise
if [[ "$config_lower" == "rss" ]]; then
    source_dir="${PROJECT_ROOT}/exp_results/${composition_lower}/rss_proposal"
else
    source_dir="${PROJECT_ROOT}/exp_results/${composition_lower}/sampled/${config_lower}"
fi

output_dir="${PROJECT_ROOT}/exp_results/${composition_lower}/relaxed/${config_lower}"
unconverged_dir="${PROJECT_ROOT}/exp_results/${composition_lower}/unconverged/${config_lower}"
log_path="${output_dir}/dyn.log"

# --- Force field backend ---
potential_backend=${potential_backend:-"matsim"}
matsim_path=${matsim_path:-"${PROJECT_ROOT}/pretrained_mattersim/mattersim-v1.0.0-5M.pth"}
uma_model_path=${uma_model_path:-"${PROJECT_ROOT}/pretrained_uma"}
uma_checkpoint=${uma_checkpoint:-"uma-s-1p1.pt"}
uma_atom_refs=${uma_atom_refs:-"iso_atom_elem_refs.yaml"}

# --- Relaxation parameters (hardcoded for consistency) ---
FMAX=0.001
NUM_STEPS=1000
METHOD="fire"

mkdir -p "${output_dir}"
mkdir -p "${unconverged_dir}"

echo "=== Post-Relaxation ==="
echo "Composition: ${composition_lower}, Config: ${config_lower}"
echo "Source: ${source_dir}"
echo "Output (converged): ${output_dir}"
echo "Output (unconverged): ${unconverged_dir}"
echo "Backend: ${potential_backend}"

source "${PROJECT_ROOT}/.venv/bin/activate"

backend_args=()
if [[ "${potential_backend,,}" == "uma" ]]; then
    backend_args+=("--potential_backend=uma")
    backend_args+=("--uma_model_path=${uma_model_path}")
    backend_args+=("--uma_checkpoint=${uma_checkpoint}")
    backend_args+=("--uma_atom_refs=${uma_atom_refs}")
else
    backend_args+=("--potential_backend=matsim")
    backend_args+=("--model_path=${matsim_path}")
fi

python rss_utils/rss_relaxation.py \
    --source_dir="${source_dir}" \
    --output_dir="${output_dir}" \
    --unconverged_dir="${unconverged_dir}" \
    --log_path="${log_path}" \
    --fmax="${FMAX}" \
    --num_steps="${NUM_STEPS}" \
    --method="${METHOD}" \
    "${backend_args[@]}"

echo "Relaxation complete. Converged: ${output_dir}"
