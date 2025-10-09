#! /bin/bash


##########################################
# TESTING SIFT1M and PAPER
##########################################

source ./conf.sh

echo "dataset: $dataset"
echo "datasize: $N"
echo "query_size: $query_size"
echo "dataset_file: $dataset_file"
echo "query_file: $query_file"
echo "dataset_attr_file: $dataset_attr_file"
echo "query_predicate_file: $query_predicate_file"
echo "ground_truth_file: $ground_truth_file"
echo "index_file: $index_file"
echo "top_k: $K"
echo "threads: $threads"
echo "ef_search: $ef_search"

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $index_file ]; then
        echo "index file already exist"
        exit 0
    fi
fi

python -u hashann_query.py --data_path $dataset_file \
                                               --index_cache_path $index_file \
                                               --ef_list $ef_search \
                                               --k $K \
                                               --N $N \
                                               --M $M \
                                               --dim $dim \
                                               --metric "l2" \
                                               --efConstruction $ef_construction \
                                               --ef_search $ef_search \
                                               --ef_top $ef_top \
                                               --name $algo \
                                               --query_path $query_file \
                                               --attr_path $dataset_attr_file \
                                               --qrange_path $query_predicate_file \
                                               --gt_path $ground_truth_file \
                                               --n_query_to_use $query_size \
                                               --attr_type_list "[0,1]" \
                                               --threads $threads

if [ $? -ne 0 ]; then
    echo "HashANN failed to run."
else
    echo "HashANN succeed."
fi

# status=$?
# if [ $status -eq 0 ]; then
#     source ./run_txt2csv.sh
# fi