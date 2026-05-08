#! /bin/bash

source ./conf.sh

threads=1

# Vanilla HNSW index path — generalized for any dataset
if [ "$metric" = "IP" ] || [ "$metric" = "ip" ]; then
    vanilla_index_file="${index_root}/hnswlib_vanilla/index_${M}_${ef_construction}_ip"
else
    vanilla_index_file="${index_root}/hnswlib_vanilla/index_${M}_${ef_construction}_l2"
fi

echo "=== Vanilla HNSW Post-Filter ==="
echo "dataset: $dataset"
echo "vanilla index: $vanilla_index_file"
echo "ef_search: $ef_search_list"
echo "k: $K"
echo "attr_path: $dataset_attr_file"
echo "predicate: $query_predicate_file"
echo "gt: $ground_truth_file"

if [ ! -f "$vanilla_index_file" ]; then
    echo "vanilla index not found: $vanilla_index_file"
    exit 1
fi

python -u ../tests/vanilla_postfilter_query.py \
    --data_path "$dataset_file" \
    --index_path "$vanilla_index_file" \
    --query_path "$query_file" \
    --attr_path "$dataset_attr_file" \
    --qrange_path "$query_predicate_file" \
    --gt_path "$ground_truth_file" \
    --ef_search "$ef_search_list" \
    --k $K \
    --n_query $query_size \
    --dim $dim \
    --metric "$metric" \
    --attr_type_list "$attr_type" \
    2>&1 | tee logs/vanilla_postfilter.log

python ../tests/extract_results.py "logs/vanilla_postfilter.log" $dataset $attr_type $query_sel $M "vanilla_postfilter" $K "$query_predicate_file"
