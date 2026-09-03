#!/usr/bin/env bash
# point_update.sh — vector+attribute (point) update benchmark.
#
# For each round, mark_deletes 1M old internal ids and inserts 1M new ones
# at fresh internal ids with new vec + new attr (drawn from base[N..2N)).
#
# Requirements:
#   - An index built with max_elements >= N + rounds*step (so the index has
#     headroom for the new internal ids). The current EMA build_index.sh
#     creates an index sized exactly N, which is NOT enough for this case.
#     You must rebuild a "5M-with-10M-capacity" index first (see comment
#     below in this script).
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./conf.sh

export EMA_BASE_FVECS="$dataset_file"
export EMA_QUERY_FVECS="$query_file"
export EMA_ATTR_JSON="$dataset_attr_file"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="logs/dynamic_point_update_${dataset}_${TS}"
mkdir -p "$OUT_DIR"

# Default index path: prefer the dedicated point-update build, fall back to
# an in-process build inside point_update.py if it doesn't exist (workaround
# for a C++ load_index bug — see point_update.py header comment).
POINT_UPDATE_N_DEFAULT="${UPDATE_N:-5000000}"
POINT_UPDATE_CAP_DEFAULT=$((POINT_UPDATE_N_DEFAULT + ${UPDATE_STEP:-1000000} * ${ROUNDS:-5}))
DEFAULT_INDEX="${hashann_index_file}_N${POINT_UPDATE_N_DEFAULT}_cap${POINT_UPDATE_CAP_DEFAULT}"
INDEX_PATH="${INDEX_PATH:-$DEFAULT_INDEX}"
# Empty string => let point_update.py build in-process.
if [ ! -f "$INDEX_PATH" ]; then
    echo "[point_update] no on-disk index at $INDEX_PATH — point_update.py will build in-process."
    INDEX_PATH=""
fi

UPDATE_STEP="${UPDATE_STEP:-1000000}"
ROUNDS="${ROUNDS:-5}"
UPDATE_N="${UPDATE_N:-5000000}"
EF_SEARCH_LIST_CSV="${EF_SEARCH_LIST_CSV:-40,60,80,100,140,200,300}"
PRED_VAL="${PRED_VAL:-9}"
N_QUERY="${N_QUERY:-$query_size}"

python -u dynamic/point_update.py \
    --index_path      "$INDEX_PATH" \
    --out_dir         "$OUT_DIR" \
    --N               "$UPDATE_N" \
    --step            "$UPDATE_STEP" \
    --rounds          "$ROUNDS" \
    --M               "$M" \
    --ef_search_list  "$EF_SEARCH_LIST_CSV" \
    --threads_update  "$threads" \
    --threads_query   1 \
    --K               "$K" \
    --predicate_value "$PRED_VAL" \
    --n_query         "$N_QUERY" \
    2>&1 | tee "$OUT_DIR/run.log"

echo "[point_update] DONE → $OUT_DIR"
