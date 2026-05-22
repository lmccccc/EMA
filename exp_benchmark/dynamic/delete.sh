#!/usr/bin/env bash
# delete.sh — mark_delete benchmark (no graph repair).
# Requires an existing index — pass via INDEX_PATH or it falls back to the
# index produced by insert.sh's last stage.
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./conf.sh

export EMA_BASE_FVECS="$dataset_file"
export EMA_QUERY_FVECS="$query_file"
export EMA_ATTR_JSON="$dataset_attr_file"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="logs/dynamic_delete_${dataset}_${TS}"
mkdir -p "$OUT_DIR"

INDEX_PATH="${INDEX_PATH:-$hashann_index_file}"
if [ ! -f "$INDEX_PATH" ]; then
    echo "[delete] missing index: $INDEX_PATH" >&2
    echo "  Build one first via ./build_index.sh or ./dynamic/insert.sh" >&2
    exit 1
fi

DELETE_STEP="${DELETE_STEP:-1000000}"
N_DELETE_STAGES="${N_DELETE_STAGES:-5}"
EF_SEARCH_LIST_CSV="${EF_SEARCH_LIST_CSV:-40,60,80,100,120,160,200,260}"
PRED_VAL="${PRED_VAL:-9}"
N_QUERY="${N_QUERY:-$query_size}"

python -u dynamic/delete.py \
    --index_path      "$INDEX_PATH" \
    --out_dir         "$OUT_DIR" \
    --N               "$N" \
    --delete_step     "$DELETE_STEP" \
    --n_delete_stages "$N_DELETE_STAGES" \
    --ef_search_list  "$EF_SEARCH_LIST_CSV" \
    --threads_query   1 \
    --K               "$K" \
    --predicate_value "$PRED_VAL" \
    --n_query         "$N_QUERY" \
    2>&1 | tee "$OUT_DIR/run.log"

echo "[delete] DONE → $OUT_DIR"
