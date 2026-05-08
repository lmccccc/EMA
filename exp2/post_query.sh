#! /bin/bash


##########################################
# TESTING SIFT1M and PAPER
##########################################

source ./conf.sh


threads=1

echo "dataset: $dataset"
echo "data_path: $dataset_file"
echo "index_cache_path: $hnsw_index_file"
echo "ef_list: $ef_search"
echo "k: $K"
echo "N: $N"
echo "M: $M"
echo "dim: $dim"
echo "metric: ${metric}"
echo "efConstruction: $ef_construction"
echo "ef_search: $ef_search_list"
echo "ef_top: $ef_top"
echo "name: $algo"
echo "query_path: $query_file"
echo "attr_path: $dataset_attr_file"
echo "qrange_path: $query_predicate_file"
echo "gt_path: $ground_truth_file"
echo "n_query_to_use: $query_size"
echo "attr_type_list: $attr_type"
echo "threads: $threads"

if [ ! -f $hashann_index_file ]; then
    echo "index file does not exist $hashann_index_file"
    exit 0
fi

# echo "dataset: $dataset" >> logs.txt
# echo "attr: $attr_type" >> logs.txt
# echo "sel: $query_sel" >> logs.txt
# echo "algo: bfann" >> logs.txt

python -u ../tests/post_query.py --data_path $dataset_file \
                                               --index_cache_path $hashann_index_file \
                                               --k $K \
                                               --N $N \
                                               --M $M \
                                               --dim $dim \
                                               --metric ${metric} \
                                               --efConstruction $ef_construction \
                                               --ef_search $ef_search_list \
                                               --ef_top $ef_top \
                                               --name $algo \
                                               --query_path $query_file \
                                               --attr_path $dataset_attr_file \
                                               --qrange_path $query_predicate_file \
                                               --gt_path $ground_truth_file \
                                               --n_query_to_use $query_size \
                                               --attr_type_list $attr_type \
                                               --threads $threads \
                                               --use_ft "false" \
                                                2>&1 | tee logs/post_query.log

if [ $? -ne 0 ]; then
    echo "Post-query failed to run."
else
    echo "Post-query succeed."
fi

python ../tests/extract_results.py "logs/post_query.log" $dataset $attr_type $query_sel $M "post"

# status=$?
# if [ $status -eq 0 ]; then
#     source ./run_txt2csv.sh
# fi
