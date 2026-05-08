#!/bin/bash
# Vanilla HNSW post-filter benchmark across selectivities.

set -e
cd "$(dirname "$0")"

declare -a SEL_NAMES=("10pct" "60pct" "100pct")
declare -a SEL_VALS=("[0.333,7]" "[0.75,2]" "[1.0,0]")

EF_LIST="[10,50,100,200,500,1000,1500,2000]"

for i in "${!SEL_NAMES[@]}"; do
    name="${SEL_NAMES[$i]}"
    qsel="${SEL_VALS[$i]}"
    export query_sel="$qsel"
    source ./conf.sh

    if [ "$metric" = "IP" ] || [ "$metric" = "ip" ]; then
        vanilla_index_file="${index_root}/hnswlib_vanilla/index_${M}_${ef_construction}_ip"
    else
        vanilla_index_file="${index_root}/hnswlib_vanilla/index_${M}_${ef_construction}_l2"
    fi

    LOG="logs/vanilla_${dataset}_${name}.log"

    echo ""
    echo "================================================"
    echo "  Selectivity ${name}  query_sel=${qsel}"
    echo "  predicate: ${query_predicate_file}"
    echo "  log:       ${LOG}"
    echo "================================================"

    python -u ../tests/vanilla_postfilter_query.py \
        --data_path "$dataset_file" \
        --index_path "$vanilla_index_file" \
        --query_path "$query_file" \
        --attr_path "$dataset_attr_file" \
        --qrange_path "$query_predicate_file" \
        --gt_path "$ground_truth_file" \
        --ef_search "$EF_LIST" \
        --k $K \
        --n_query $query_size \
        --dim $dim \
        --metric "$metric" \
        --attr_type_list "$attr_type" \
        2>&1 | tee "$LOG"
done

echo "=== Done ==="
