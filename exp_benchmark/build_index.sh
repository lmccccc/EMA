#!/usr/bin/env bash
# build_index.sh — build (or skip if existing) one EMA index.
# Env vars: dataset, M, ef_construction, ft_bits, edge_level_ft.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
source ./conf.sh

for f in "$dataset_file" "$dataset_attr_file"; do
    if [ ! -f "$f" ]; then
        echo "[build] missing: $f" >&2
        exit 1
    fi
done

if [ -f "$hashann_index_file" ]; then
    echo "[build] $hashann_index_file already exists, skip"
    ls -lh "$hashann_index_file"
    exit 0
fi

# A predicate + GT are needed only by the in-process verifier inside
# hashann_build.py; pick any cheap selectivity if missing.
[ -f "$query_predicate_file" ] || ./predicate_generator.sh
[ -f "$ground_truth_file"      ] || ./ground_truth_generator.sh

mkdir -p logs

python -u ../tests/hashann_build.py \
    --data_path        "$dataset_file" \
    --index_cache_path "$hashann_index_file" \
    --ef_search        "$ef_search_list" \
    --k                "$K" \
    --N                "$N" \
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
    2>&1 | tee "logs/build_${dataset}_M${M}_ft${ft_bits}.log"

echo "[build] done: $hashann_index_file"
