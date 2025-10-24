##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

algo=HNSW


attr_type="[0]"  # 0 for numerical, 1 for categorical
distribution_type="random"  # random, normal, zipf

# query_sel="[0.5,[1]]"  # 0.5 for numerical, [1,2] for categorical label(s)
num_query_sel="0.2"
cate_query_sel="[1]"
categorical_attr_max_cardinality=5
numerical_max_attr=100000


attr_index_type="arbi_0_1_random"
if [ "$attr_type" = "[0]" ]; then
    attr_index_type="arbi_0_"${distribution_type}
elif [ "$attr_type" = "[1]" ]; then
    attr_index_type="arbi_1_"${distribution_type}
elif [ "$attr_type" = "[0,1]" ]; then
    attr_index_type="arbi_0_1_"${distribution_type}
fi

query_file_prefix="arbi_"${num_query_sel}"_"${cate_query_sel}
if [ "$attr_type" = "[0]" ]; then
    query_file_prefix="arbi_0_"${num_query_sel}
elif [ "$attr_type" = "[1]" ]; then
    query_file_prefix="arbi_1_"${cate_query_sel}
elif [ "$attr_type" = "[0,1]" ]; then
    query_file_prefix="arbi_0_"${num_query_sel}"_1_"${cate_query_sel}
fi


dataset="sift"
N=1000000
query_size=10000
hashann_root="/mnt/data/mocheng/dataset/sift/hashann/"
dataset_file="/mnt/data/mocheng/dataset/sift/sift_base.fvecs"
query_file="/mnt/data/mocheng/dataset/sift/sift_query.fvecs"
label_root="/mnt/data/mocheng/dataset/sift/label/"${attr_index_type}"/"


# dataset="siftsmall"
# N=10000
# query_size=100
# hashann_root="/mnt/data/mocheng/dataset/siftsmall/hashann/"
# dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
# query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
# label_root="/mnt/data/mocheng/dataset/siftsmall/label/"${attr_index_type}"/"


ground_truth_collection_name=${dataset}_${attr_index_type}
dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"

ori_query_predicate_file=${label_root}"predicate_"${attr_index_type}".json"
ori_ground_truth_file=${label_root}"gt_"${attr_index_type}".json"

query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"



index_root=${hashann_root}index/

M=120
threads=64 # 1
ef_construction=1000
ef_search=100
ef_top=10
K=10
dim=128
ft_bits=128
use_ft=true
ef_search_list="[50,100,200,300]"
# use_ft=false

index_file=${index_root}index_${M}_${ef_construction}_${attr_index_type}

if [ ! -f "$index_file" ]; then
    echo "$index_file does not exist. Please check."
fi

# dir=${now}_${dataset}_${algo}

# if [ ! -d "$dir" ]; then
#     mkdir ${dir}
# fi

if [ ! -d "$index_root" ]; then
    mkdir -p ${index_root}
fi

if [ ! -d "$label_root" ]; then
    mkdir -p ${label_root}
fi

# log_file=${dir}/summary_${algo}_${dataset}_${ef_search}.txt
# TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> $log_file


# if [ ! -f "$query_predicate_file" ]; then
#     if [ ! -f "$ori_query_predicate_file" ]; then
#         echo "$ori_query_predicate_file does not exist. Please check."
#         exit 1
#     else
#         cp $ori_query_predicate_file $query_predicate_file
#         echo "copying $ori_query_predicate_file to $query_predicate_file"
#         exit 1
#     fi
#     echo "$query_predicate_file does not exist. Please check."
#     exit 1
# fi

# if [ ! -f "$ground_truth_file" ]; then
#     if [ ! -f "$ori_ground_truth_file" ]; then
#         echo "$ori_ground_truth_file does not exist. Please check."
#         exit 1
#     else
#         cp $ori_ground_truth_file $ground_truth_file
#         echo "copying $ori_ground_truth_file to $ground_truth_file"
#         exit 1
#     fi
#     echo "$ground_truth_file does not exist. Please check."
#     exit 1
# fi