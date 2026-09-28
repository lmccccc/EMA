#!/usr/bin/env bash
# ground_truth_generator.sh — explicitly selected C++ or NumPy brute-force GT.
# Reads conf.sh; expects dnf_spec / dnf_name + predicate already generated.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
source ./conf.sh

if [ -f "$ground_truth_file" ]; then
    echo "[gt] $ground_truth_file exists, skip"
    exit 0
fi

case "${GT_BACKEND:-cpp}" in
    numpy)
        exec python ../tests/groundtruth_bruteforce.py \
            --dataset_file "$dataset_file" \
            --query_file "$query_file" \
            --attr_file "$dataset_attr_file" \
            --predicate_file "$query_predicate_file" \
            --attr_type_list "$attr_type" \
            --N "$N" --d "$dim" --query_size "$query_size" --K "$K" \
            --metric "$metric" --gt_file "$ground_truth_file"
        ;;
    cpp) ;;
    *)
        echo "[gt] GT_BACKEND must be cpp or numpy" >&2
        exit 1
        ;;
esac

# This is a custom FAISS target, not part of an upstream FAISS checkout.
FAISS_ROOT="${FAISS_ROOT:-/home/mocheng/code/faiss}"
GT_BIN="${FAISS_ROOT}/build/demos/generate_groundtruth_arbi"
if [ ! -x "$GT_BIN" ]; then
    echo "[gt] missing binary: $GT_BIN" >&2
    echo "Provide the matching custom FAISS build, or explicitly select GT_BACKEND=numpy for new workloads." >&2
    exit 1
fi

attr_type_csv=$(echo "$attr_type" | tr -d '[] ')

"$GT_BIN" \
    "$N" \
    "$dataset_file" \
    "$dataset_attr_file" \
    "$query_file" \
    "$query_predicate_file" \
    "$ground_truth_file" \
    "$K" \
    "$dim" \
    "$threads" \
    "$attr_type_csv" \
    "$metric"
