#!/usr/bin/env bash
# ground_truth_generator.sh — brute-force GT via FAISS C++ binary.
# Reads conf.sh; expects dnf_spec / dnf_name + predicate already generated.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
source ./conf.sh

if [ -f "$ground_truth_file" ]; then
    echo "[gt] $ground_truth_file exists, skip"
    exit 0
fi

# Point this at your FAISS build with the demo `generate_groundtruth_arbi`
FAISS_ROOT="${FAISS_ROOT:-/home/mocheng/code/faiss}"
GT_BIN="${FAISS_ROOT}/build/demos/generate_groundtruth_arbi"
if [ ! -x "$GT_BIN" ]; then
    echo "[gt] missing binary: $GT_BIN" >&2
    echo "build: cd $FAISS_ROOT && make -C build generate_groundtruth_arbi" >&2
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
