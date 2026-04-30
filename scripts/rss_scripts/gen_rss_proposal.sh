#!/bin/bash

# Generate RSS structure proposals for a single composition.
#
# Usage:
#   composition=Al2O3 multiplier=2 num_structures=1024 bash scripts/rss_scripts/gen_rss_proposal.sh
#
# Environment variables:
#   composition      - Base chemical formula (e.g., Al2O3, CaTiO3, C)
#   multiplier       - Supercell multiplier (default: 1)
#   num_structures   - Number of structures to generate (default: 512)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

composition=${composition:?"Error: composition is required (e.g., composition=Al2O3)"}
multiplier=${multiplier:-1}
num_structures=${num_structures:-512}

PYTHON_SCRIPT_PATH="$PROJECT_ROOT/rss_utils/gen_rss_comp.py"

source "$PROJECT_ROOT/.venv/bin/activate"
PYTHON_CMD=$(command -v python3 || command -v python)

# Multiply chemical formula by supercell multiplier.
# e.g., Al2O3 * 2 -> Al4O6, C * 2 -> C2
generate_multiplied_formula() {
    local base_formula=$1
    local mult=$2
    local result=""

    local parts=$(echo "$base_formula" | sed -r 's/([A-Z][a-z]?)([0-9]*)/\1 \2\n/g' | sed '/^[[:space:]]*$/d')

    while IFS= read -r line; do
        local element=$(echo "$line" | awk '{print $1}')
        local count_str=$(echo "$line" | awk '{print $2}')
        local count

        if [[ -z "$count_str" ]]; then
            count=1
        else
            count=$((count_str))
        fi

        local new_count=$((count * mult))
        result+="${element}${new_count}"
    done <<< "$parts"

    echo "$result"
}

dir_name=$(echo "$composition" | tr '[:upper:]' '[:lower:]')
output_subdir="$PROJECT_ROOT/exp_results/$dir_name/rss_proposal"
mkdir -p "$output_subdir"

formula=$(generate_multiplied_formula "$composition" "$multiplier")
output_filename="${formula,,}.xyz"
output_file_path="$output_subdir/$output_filename"

echo "Composition: $composition, Multiplier: $multiplier, Num structures: $num_structures"
echo "Formula: $formula -> $output_file_path"

"$PYTHON_CMD" "$PYTHON_SCRIPT_PATH" "$formula" --num "$num_structures" --output "$output_file_path"

echo "Done: $output_file_path"
