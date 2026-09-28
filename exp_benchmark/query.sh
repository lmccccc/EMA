#!/usr/bin/env bash
# query.sh — run one EMA query cell.
# Env vars: dataset, M, ft_bits, dnf_spec, dnf_name, FT_ROUTING_MIN_DEG, threads.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
threads="${threads:-1}"
source ./conf.sh

: "${FT_ROUTING_MIN_DEG:=$ft_routing_min_deg}"

if [ ! -f "$hashann_index_file" ]; then
    echo "[query] missing index: $hashann_index_file" >&2
    exit 1
fi

[ -f "$query_predicate_file" ] || ./predicate_generator.sh
[ -f "$ground_truth_file"      ] || ./ground_truth_generator.sh

echo "[query] dataset=$dataset M=$M ft_bits=$ft_bits min_deg=$FT_ROUTING_MIN_DEG"
echo "  index=$hashann_index_file"
echo "  predicate=$query_predicate_file"

python -u ../tests/hashann_query.py \
    --data_path        "$dataset_file" \
    --index_cache_path "$hashann_index_file" \
    --k                "$K" \
    --N                "$N" \
    --M                "$M" \
    --dim              "$dim" \
    --metric           "$metric" \
    --efConstruction   "$ef_construction" \
    --ef_search        "$ef_search_list" \
    --ef_top           1 \
    --name             HNSW \
    --query_path       "$query_file" \
    --attr_path        "$dataset_attr_file" \
    --qrange_path      "$query_predicate_file" \
    --gt_path          "$ground_truth_file" \
    --n_query_to_use   "$query_size" \
    --attr_type_list   "$attr_type" \
    --threads          "$threads" \
    --use_ft           "$use_ft" \
    --augment_cht      "${augment_cht:-false}" \
    --ft_routing_min_deg "$FT_ROUTING_MIN_DEG" \
    --ft_bits          "$ft_bits"
