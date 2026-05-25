#!/usr/bin/env bash
# attr_generator.sh — produce the attribute JSON for the current cell.
# Reads conf.sh; writes to $dataset_attr_file.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
source ./conf.sh

categorical_attr_max_cardinality="${categorical_attr_max_cardinality:-21}"
numerical_max_attr="${numerical_max_attr:-100000}"

if [ -f "$dataset_attr_file" ]; then
    echo "[attr_generator] $dataset_attr_file exists, skip"
    exit 0
fi

mkdir -p "$(dirname "$dataset_attr_file")"

python -u ../tests/attr_generator.py \
    --output_file                       "$dataset_attr_file" \
    --N                                 "$N" \
    --attr_type_list                    "$attr_type" \
    --categorical_attr_max_cardinality  "$categorical_attr_max_cardinality" \
    --numerical_max_attr                "$numerical_max_attr"
