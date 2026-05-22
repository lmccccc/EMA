#!/usr/bin/env bash
# insert.sh — incremental insertion benchmark (build N0, then +step until Nfinal).
# Defaults match the paper (sift10m, 5M→10M in +1M stages, 10% sel predicate).
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./conf.sh

# Export dataset paths so the python script picks them up.
export EMA_BASE_FVECS="$dataset_file"
export EMA_QUERY_FVECS="$query_file"
export EMA_ATTR_JSON="$dataset_attr_file"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="logs/dynamic_insert_${dataset}_${TS}"
mkdir -p "$OUT_DIR"

INITIAL_N="${INITIAL_N:-$((N / 2))}"
FINAL_N="${FINAL_N:-$N}"
STEP_N="${STEP_N:-1000000}"
EF_SEARCH_LIST_CSV="${EF_SEARCH_LIST_CSV:-10,20,40,80,160,320}"
PRED_VAL="${PRED_VAL:-9}"
N_QUERY="${N_QUERY:-$query_size}"
EDGE_LEVEL_FT_INT=$([ "$edge_level_ft" = "true" ] && echo 1 || echo 0)

python -u dynamic/incremental.py \
    --out_dir         "$OUT_DIR" \
    --initial_n       "$INITIAL_N" \
    --final_n         "$FINAL_N" \
    --step_n          "$STEP_N" \
    --M               "$M" \
    --ef_construction "$ef_construction" \
    --ft_bits         "$ft_bits" \
    --edge_level_ft   "$EDGE_LEVEL_FT_INT" \
    --ef_search_list  "$EF_SEARCH_LIST_CSV" \
    --threads_build   "$threads" \
    --threads_query   1 \
    --K               "$K" \
    --predicate_value "$PRED_VAL" \
    --n_query         "$N_QUERY" \
    2>&1 | tee "$OUT_DIR/run.log"

echo "[insert] DONE → $OUT_DIR"
