#!/bin/bash
# Guided crystal structure generation for a given composition.
# Usage: composition=si guidance_method=Mi bash scripts/sample_scripts/generate.sh

PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0}

# --- Required ---
composition=${composition:?"composition is required (e.g. si, al2o3)"}

# --- Experiment layout ---
composition_lower=$(echo "$composition" | tr '[:upper:]' '[:lower:]')
guidance_method=${guidance_method:-"Mi"}
config=${config:-"${guidance_method,,}"}

rss_proposal_path="${PROJECT_ROOT}/exp_results/${composition_lower}/rss_proposal"
output_path="${PROJECT_ROOT}/exp_results/${composition_lower}/sampled/${config}"

model_path="${PROJECT_ROOT}/outputs/csp_model/checkpoints/"
guidance_config_path="${PROJECT_ROOT}/sampling_conf/guidance.yaml"

# --- Sampling parameters ---
batch_size=${batch_size:-512}
max_sampling_num=${max_sampling_num:-1024}

# --- Force field backend ---
potential_backend=${potential_backend:-"matsim"}
matsim_path=${matsim_path:-"${PROJECT_ROOT}/pretrained_mattersim/mattersim-v1.0.0-5M.pth"}
uma_model_path=${uma_model_path:-"${PROJECT_ROOT}/pretrained_uma"}
uma_checkpoint=${uma_checkpoint:-"uma-s-1p1.pt"}
uma_atom_refs=${uma_atom_refs:-"iso_atom_elem_refs.yaml"}
uma_device=${uma_device:-"cuda"}
uma_inference_settings=${uma_inference_settings:-"default"}
uma_workers=${uma_workers:-1}

# --- Guidance parameters (all optional, generate.py has defaults) ---
pos_guide_coef=${pos_guide_coef:-""}
cell_guide_coef=${cell_guide_coef:-""}
t_mid=${t_mid:-""}
t_scale=${t_scale:-""}
guidance_steps=${guidance_steps:-""}

# --- Build potential args ---
potential_args=()
if [[ "${potential_backend,,}" == "uma" ]]; then
    potential_args+=("--uma_model_path=${uma_model_path}")
    potential_args+=("--uma_checkpoint=${uma_checkpoint}")
    potential_args+=("--uma_atom_refs=${uma_atom_refs}")
    potential_args+=("--uma_device=${uma_device}")
    potential_args+=("--uma_inference_settings=${uma_inference_settings}")
    potential_args+=("--uma_workers=${uma_workers}")
else
    potential_args+=("--matsim_path=${matsim_path}")
fi

mkdir -p "${output_path}"

echo "=== Guided CSP Generation ==="
echo "Composition: ${composition_lower}"
echo "Guidance method: ${guidance_method}"
echo "Config: ${config}"
echo "Potential backend: ${potential_backend}"
echo "Output: ${output_path}"

source "${PROJECT_ROOT}/.venv/bin/activate"
python mattergen/scripts/generate.py "${output_path}" \
    --model-path="${model_path}" \
    --checkpoint-epoch=last \
    --batch-size="${batch_size}" \
    --sampling-config-name=csp \
    --rss_proposal_path="${rss_proposal_path}" \
    --max_sampling_num="${max_sampling_num}" \
    "${potential_args[@]}" \
    --guidance_config_path="${guidance_config_path}" \
    --guidance_method="${guidance_method}" \
    ${pos_guide_coef:+--pos_guide_coef="${pos_guide_coef}"} \
    ${cell_guide_coef:+--cell_guide_coef="${cell_guide_coef}"} \
    ${t_mid:+--t_mid="${t_mid}"} \
    ${t_scale:+--t_scale="${t_scale}"} \
    ${guidance_steps:+--guidance_steps="${guidance_steps}"}

echo "Generation complete. Output: ${output_path}"
