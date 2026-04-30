#!/bin/bash

# Relax RSS proposal structures using MatterSim for a single composition.
#
# Usage:
#   composition=al2o3 bash scripts/rss_scripts/gen_rss_result.sh
#
# Environment variables:
#   composition      - Composition name, lowercase (e.g., al2o3, catio3)
#   CUDA_VISIBLE_DEVICES - GPU device index (default: 0)
#   checkpoint_path  - Path to MatterSim checkpoint (default: pretrained_mattersim/mattersim-v1.0.0-5M.pth)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

composition=${composition:?"Error: composition is required (e.g., composition=al2o3)"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source_dir="$PROJECT_ROOT/exp_results/$composition/rss_proposal"
output_dir="$PROJECT_ROOT/exp_results/$composition/relaxed/rss"
unconverged_dir="$PROJECT_ROOT/exp_results/$composition/unconverged/rss"
log_file="$output_dir/dyn.log"
checkpoint_path="${checkpoint_path:-$PROJECT_ROOT/pretrained_mattersim/mattersim-v1.0.0-5M.pth}"

mkdir -p "$output_dir"
mkdir -p "$unconverged_dir"

source "$PROJECT_ROOT/.venv/bin/activate"
python "$PROJECT_ROOT/rss_utils/rss_relaxation.py" \
    --source_dir "$source_dir" \
    --output_dir "$output_dir" \
    --unconverged_dir "$unconverged_dir" \
    --log_path "$log_file" \
    --model_path "$checkpoint_path" \
    --model_type m3gnet

echo "Done: $output_dir"
