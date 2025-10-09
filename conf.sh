##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

algo=HNSW

# dataset="siftsmall"
# N=10000
# query_size=100
# nsw_root="/mnt/data/mocheng/dataset/siftsmall/nsw_filter/"
# dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
# query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
# dataset_attr_file="/mnt/data/mocheng/dataset/siftsmall/label/arbi_0_1_random/attr_arbi_0_1_random.json"
# query_predicate_file="/mnt/data/mocheng/dataset/siftsmall/label/arbi_0_1_random/predicate_arbi_0_1_random.json"
# ground_truth_file="/mnt/data/mocheng/dataset/siftsmall/label/arbi_0_1_random/gt_arbi_0_1_random.json"

attr_type="[0,1]"  # 0 for numerical, 1 for categorical
distribution_type="random"  # random, normal, zipf

query_sel="[0.5,[1]]"  # 0.5 for numerical, [1,2] for categorical label(s)
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

label_root="/mnt/data/mocheng/dataset/sift/label/"${attr_index_type}"/"
ground_truth_collection_name=${dataset}_${attr_index_type}

dataset="sift"
N=1000000
query_size=10000
hashann_root="/mnt/data/mocheng/dataset/sift/hashann/"
dataset_file="/mnt/data/mocheng/dataset/sift/sift_base.fvecs"
query_file="/mnt/data/mocheng/dataset/sift/sift_query.fvecs"
dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
query_predicate_file=${label_root}"predicate_"${attr_index_type}".json"
ground_truth_file=${label_root}"gt_"${attr_index_type}".json"

index_root=${hashann_root}index/

M=80
threads=64 # 1
ef_construction=1000
ef_search=20
ef_top=10
K=10
dim=128
ft_bits=128

index_file=${index_root}index_${M}_${ef_construction}



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