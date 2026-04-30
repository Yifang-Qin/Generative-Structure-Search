#!/bin/bash

# Deduplicate relaxed RSS structures for a single composition.
#
# Usage:
#   composition=al2o3 bash scripts/rss_scripts/dedup_structures.sh
#
# Environment variables:
#   composition  - Composition name, lowercase (e.g., al2o3, catio3)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

composition=${composition:?"Error: composition is required (e.g., composition=al2o3)"}

sample_dir="$PROJECT_ROOT/exp_results/$composition/relaxed/rss"
target_dir="$PROJECT_ROOT/exp_results/$composition/deduped"

if [ ! -d "$sample_dir" ] || [ -z "$(ls -A "$sample_dir" 2>/dev/null)" ]; then
    echo "Error: relaxed/rss directory not found or empty: $sample_dir"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

echo "Deduplicating: $sample_dir -> $target_dir"

python "$PROJECT_ROOT/sample_eval_utils/dedup_samples.py" \
    --sample_path "$sample_dir" \
    --target_path "$target_dir"

echo "Done: $target_dir"
