#!/usr/bin/env bash
# point_update_build.sh — build an EMA index sized for the point_update
# benchmark (vector+attribute update).
#
# Unlike build_index.sh, this script lets the HNSW `max_elements` exceed the
# initial loaded data size `N`, leaving capacity headroom so that the
# benchmark can mark_delete old internal ids and insert new vec+attr rows at
# fresh internal ids. The on-disk file lives next to the regular index but
# with a `_capCAP` suffix so it never clashes with build_index.sh output.
#
# Env vars (with defaults):
#   POINT_UPDATE_N=5000000        # initial rows loaded into the index
#   POINT_UPDATE_CAP=10000000     # max_elements (= N + rounds*step)
#   plus everything conf.sh consumes (dataset, M, ef_construction, ft_bits,
#   edge_level_ft, threads, ...)
#
# Output:
#   $hashann_index_root/index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}${edge_ft_suffix}_N${N}_cap${CAP}
#   (path is exported as POINT_UPDATE_INDEX_PATH and printed to stdout)
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./conf.sh

POINT_UPDATE_N="${POINT_UPDATE_N:-5000000}"
POINT_UPDATE_CAP="${POINT_UPDATE_CAP:-10000000}"

if [ "$POINT_UPDATE_CAP" -lt "$POINT_UPDATE_N" ]; then
    echo "[point_update_build] POINT_UPDATE_CAP ($POINT_UPDATE_CAP) < POINT_UPDATE_N ($POINT_UPDATE_N)" >&2
    exit 1
fi

# Distinct filename so this doesn't collide with the standard build.
POINT_UPDATE_INDEX_PATH="${hashann_index_file}_N${POINT_UPDATE_N}_cap${POINT_UPDATE_CAP}"
export POINT_UPDATE_INDEX_PATH

for f in "$dataset_file" "$dataset_attr_file"; do
    if [ ! -f "$f" ]; then
        echo "[point_update_build] missing: $f" >&2
        exit 1
    fi
done

if [ -f "$POINT_UPDATE_INDEX_PATH" ]; then
    echo "[point_update_build] already exists, skip: $POINT_UPDATE_INDEX_PATH"
    ls -lh "$POINT_UPDATE_INDEX_PATH"
    echo "$POINT_UPDATE_INDEX_PATH"
    exit 0
fi

# Cheap predicate/GT for the in-process verifier inside hashann_build.py.
[ -f "$query_predicate_file" ] || ./predicate_generator.sh
[ -f "$ground_truth_file"      ] || ./ground_truth_generator.sh

mkdir -p logs

python -u ../tests/hashann_build.py \
    --data_path        "$dataset_file" \
    --index_cache_path "$POINT_UPDATE_INDEX_PATH" \
    --ef_search        "$ef_search_list" \
    --k                "$K" \
    --N                "$POINT_UPDATE_N" \
    --max_elements     "$POINT_UPDATE_CAP" \
    --M                "$M" \
    --dim              "$dim" \
    --metric           "$metric" \
    --efConstruction   "$ef_construction" \
    --name             HNSW \
    --query_path       "$query_file" \
    --attr_path        "$dataset_attr_file" \
    --qrange_path      "$query_predicate_file" \
    --gt_path          "$ground_truth_file" \
    --n_query_to_use   "$query_size" \
    --attr_type_list   "$attr_type" \
    --ft_bits          "$ft_bits" \
    $( [ "$edge_level_ft" = "true" ] && echo "--edge_level_ft" ) \
    --threads          "$threads" \
    2>&1 | tee "logs/point_update_build_${dataset}_N${POINT_UPDATE_N}_cap${POINT_UPDATE_CAP}_M${M}_ft${ft_bits}.log"

echo "[point_update_build] done: $POINT_UPDATE_INDEX_PATH"
echo "$POINT_UPDATE_INDEX_PATH"
