#! /bin/bash

source ./conf.sh
source ./post_conf.sh

threads=1

if [ ! -f "$hnsw_index_file" ]; then
    echo "index file does not exist: $hnsw_index_file"
    exit 1
fi

echo "dataset: ${dataset}"
echo "N: ${N}"
echo "M: ${M}"
echo "K: ${K}"
echo "threads: ${threads}"
echo "dataset file: ${dataset_file}"
echo "query file: ${query_file}"
echo "dataset attr file: ${dataset_attr_file}"
echo "query predicate file: ${query_predicate_file}"
echo "ground truth file: ${ground_truth_file}"
echo "hnsw index file: ${hnsw_index_file}"
echo "efs: ${ef_search_list}"
echo "dim: ${dim}"
echo "attr type: ${attr_type}"
echo "test query size: ${test_query_size}"
echo "query mode: navix"

python -u ../tests/hnsw_query_navix.py --data_path $dataset_file \
                                       --index_cache_path $hnsw_index_file \
                                       --k $K \
                                       --N $N \
                                       --M $M \
                                       --dim $dim \
                                       --metric ${metric} \
                                       --efConstruction $ef_construction \
                                       --ef_search $ef_search_list \
                                       --name ${algo}_navix \
                                       --query_path $query_file \
                                       --attr_path $dataset_attr_file \
                                       --qrange_path $query_predicate_file \
                                       --gt_path $ground_truth_file \
                                       --n_query_to_use $test_query_size \
                                       --attr_type_list $attr_type \
                                       --threads $threads \
                                       2>&1 | tee logs/hnsw_navix.log

python ../tests/extract_results.py "logs/hnsw_navix.log" $dataset $attr_type $query_sel $M "hnsw_navix"