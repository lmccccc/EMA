#!/bin/bash
# Query benchmark of K=1 colored edge-FT-32 index across selectivities.
# Note: no exact 50% predicate exists; using 60% [0.75,2] as proxy.

set -e
cd "$(dirname "$0")"

declare -a SEL_NAMES=("10pct" "60pct" "100pct")
declare -a SEL_VALS=("[0.333,7]" "[0.75,2]" "[1.0,0]")

EF_LIST="[1,5,10,20,50,100,200,500]"

for i in "${!SEL_NAMES[@]}"; do
    name="${SEL_NAMES[$i]}"
    qsel="${SEL_VALS[$i]}"
    export query_sel="$qsel"
    source ./conf.sh

    INDEX="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_edgeft32_colored"
    LOG="logs/color_ft_K1_${dataset}_${name}.log"

    echo ""
    echo "================================================"
    echo "  Selectivity ${name}  query_sel=${qsel}"
    echo "  index: ${INDEX}"
    echo "  log:   ${LOG}"
    echo "================================================"

    python -u ../tests/hashann_query.py \
        --data_path "$dataset_file" \
        --index_cache_path "$INDEX" \
        --k $K --N $N --M $M --dim $dim \
        --metric ${metric} --efConstruction $ef_construction \
        --ef_search "$EF_LIST" --ef_top $ef_top \
        --name $algo \
        --query_path "$query_file" \
        --attr_path "$dataset_attr_file" \
        --qrange_path "$query_predicate_file" \
        --gt_path "$ground_truth_file" \
        --n_query_to_use $query_size \
        --attr_type_list "$attr_type" \
        --ft_bits 32 \
        --threads 1 \
        --use_ft true \
        --ft_routing_min_deg 0 \
        2>&1 | tee "$LOG"
done

echo ""
echo "=== All done ==="
for name in "${SEL_NAMES[@]}"; do
    echo "  logs/color_ft_K1_Redcaps_4M_${name}.log"
done
