#! /bin/bash


##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

algo=HNSW

# dataset="sift1M"
# nsw_root="/mnt/data/mocheng/dataset/sift/nsw/"
# N=1000000
# query_size=10000
# dataset_file="/mnt/data/mocheng/dataset/sift/sift_base.fvecs"
# query_file="/mnt/data/mocheng/dataset/sift/sift_query.fvecs"
# dataset_attr_file="/mnt/data/mocheng/dataset/sift/label/sel_1_100000_random/attr_sel_1_100000_random.json"
# query_predicate_file="/mnt/data/mocheng/dataset/sift/label/sel_1_100000_random/qrangesel_1_1_100000_random.json"
# ground_truth_file="/mnt/data/mocheng/dataset/sift/label/sel_1_100000_random/sif_gt_sel_1_1_100000_random_10.json"


# dataset="siftsmall"
# N=10000
# query_size=100
# nsw_root="/mnt/data/mocheng/dataset/siftsmall/nsw/"
# dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
# query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
# dataset_attr_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/attr_sel_1_100000_random.json"
# query_predicate_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/qrangesel_1_1_100000_random.json"
# ground_truth_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/sif_gt_sel_1_1_100000_random_10.json"

dataset="siftsmall"
N=10000
query_size=100
nsw_root="/mnt/data/mocheng/dataset/siftsmall/nsw_filter/"
dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
dataset_attr_file="/mnt/data/mocheng/dataset/siftsmall/label/arbi_0_1_random/attr_arbi_0_1_random.json"
query_predicate_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/qrangesel_1_1_100000_random.json"
ground_truth_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/sif_gt_sel_1_1_100000_random_10.json"

nsw_index_root=${nsw_root}index/

M=16
threads=64
ef_construction=1000
ef_search=100
K=10
dim=128

nsw_index_file=${nsw_index_root}index_${M}_${ef_construction}



dir=${now}_${dataset}_${algo}

if [ ! -d "$dir" ]; then
    mkdir ${dir}
fi

if [ ! -d "$nsw_result_root" ]; then
    mkdir ${nsw_root}
    mkdir ${nsw_index_root}
fi

log_file=${dir}/summary_${algo}_${dataset}_${ef_search}.txt
TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file

echo "dataset: $dataset"
echo "datasize: $N"
echo "query_size: $query_size"
echo "dataset_file: $dataset_file"
echo "query_file: $query_file"
echo "dataset_attr_file: $dataset_attr_file"
echo "query_predicate_file: $query_predicate_file"
echo "ground_truth_file: $ground_truth_file"
echo "nsw_index_file: $nsw_index_file"
echo "top_k: $K"
echo "threads: $threads"
echo "ef_search: $ef_search"

if [ "$mode" == "construction" ] || [ "$mode" == "all" ]; then
    if [ -e $nsw_index_file ]; then
        echo "index file already exist"
        exit 0
    fi
fi

# /bin/time -v -p python -u hnswlib/tests/python/nsw_build.py --data_path $dataset_file \
#                                                --index_cache_path $nsw_index_file \
#                                                --ef_list $ef_search \
#                                                --k $K \
#                                                --N $N \
#                                                --M $M \
#                                                --dim $dim \
#                                                --metric "l2" \
#                                                --efConstruction $ef_construction \
#                                                --name $algo \
#                                                --query_path $query_file \
#                                                --attr_path $dataset_attr_file \
#                                                --qrange_path $query_predicate_file \
#                                                --gt_path $ground_truth_file \
#                                                --n_query_to_use $query_size \
#                                                --threads $threads \
#                                                &>> $log_file

python -u hashann_query.py --data_path $dataset_file \
                                               --index_cache_path $nsw_index_file \
                                               --ef_list $ef_search \
                                               --k $K \
                                               --N $N \
                                               --M $M \
                                               --dim $dim \
                                               --metric "l2" \
                                               --efConstruction $ef_construction \
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