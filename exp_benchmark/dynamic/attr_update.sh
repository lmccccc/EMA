#!/usr/bin/env bash
# attr_update.sh — attribute-only update benchmark (no vector change).
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./conf.sh

export EMA_BASE_FVECS="$dataset_file"
export EMA_QUERY_FVECS="$query_file"
export EMA_ATTR_JSON="$dataset_attr_file"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="logs/dynamic_attr_update_${dataset}_${TS}"
mkdir -p "$OUT_DIR"

INDEX_PATH="${INDEX_PATH:-$hashann_index_file}"
if [ ! -f "$INDEX_PATH" ]; then
    echo "[attr_update] missing index: $INDEX_PATH" >&2
    exit 1
fi

UPDATE_STEP="${UPDATE_STEP:-1000000}"
ROUNDS="${ROUNDS:-5}"
EF_SEARCH_LIST_CSV="${EF_SEARCH_LIST_CSV:-40,60,80,100,140,200,300}"
PRED_VAL="${PRED_VAL:-9}"
N_QUERY="${N_QUERY:-$query_size}"

python -u dynamic/attr_update.py \
    --index_path      "$INDEX_PATH" \
    --out_dir         "$OUT_DIR" \
    --N               "$N" \
    --step            "$UPDATE_STEP" \
    --rounds          "$ROUNDS" \
    --ef_search_list  "$EF_SEARCH_LIST_CSV" \
    --threads_update  "$threads" \
    --threads_query   1 \
    --K               "$K" \
    --predicate_value "$PRED_VAL" \
    --n_query         "$N_QUERY" \
    2>&1 | tee "$OUT_DIR/run.log"

echo "[attr_update] DONE → $OUT_DIR"
