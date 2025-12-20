##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

algo=HNSW


attr_type="[0,0]"  # 0 for numerical, 1 for categorical
distribution_type="random"  # random, normal, zipf

# 0.5 for numerical, [1,2] for categorical label(s)
query_sel="[0.1,0.1]"


M=180
threads=64 # 1
ef_construction=1000
ef_top=10
K=10
ft_bits=128
use_ft=true

ef_search_list=[10,12,15,18,20]
# ef_search_list="[10,12,15,18,20,30,40,50,80,100,120,150,180,200]"
# ef_search_list="[80,100,120,150,180,200,300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"
# ef_search_list="[10,12,15,18,20,30,35,40,45,50]"
# ef_search_list="[80,100,120,150,180,200]"
# ef_search_list="[80,100,120,150,180,200,300]"
# ef_search_list="[300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"

# num_query_sel="1"
# cate_query_sel="[2,3]"
categorical_attr_max_cardinality=21
numerical_max_attr=100000


attr_index_type="arbi_0_1_random"
if [ "$attr_type" = "[0]" ]; then
    attr_index_type="arbi_0_"${distribution_type}
elif [ "$attr_type" = "[1]" ]; then
    attr_index_type="arbi_1_"${distribution_type}
elif [ "$attr_type" = "[0,1]" ]; then
    attr_index_type="arbi_0_1_"${distribution_type}
elif [ "$attr_type" = "[0,0]" ]; then
    attr_index_type="arbi_0_0_"${distribution_type}
fi

query_file_prefix="arbi_"${query_sel}
if [ "$attr_type" = "[0]" ]; then
    query_file_prefix="arbi_0_"${query_sel}
elif [ "$attr_type" = "[1]" ]; then
    query_file_prefix="arbi_1_"${query_sel}
elif [ "$attr_type" = "[0,1]" ]; then
    query_file_prefix="arbi_0_1_"${query_sel}
elif [ "$attr_type" = "[0,0]" ]; then
    query_file_prefix="arbi_0_0_"${query_sel}
fi


# dataset="sift"
# N=1000000
# dim=128
# query_size=10000
# index_root="/mnt/data/mocheng/dataset/sift"
# dataset_file="/mnt/data/mocheng/dataset/sift/sift_base.fvecs"
# query_file="/mnt/data/mocheng/dataset/sift/sift_query.fvecs"
# label_root="/mnt/data/mocheng/dataset/sift/label/"${attr_index_type}"/"
# dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
# query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
# ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
# ground_truth_collection_name=${dataset}_${attr_index_type}
# metric="L2"
# test_query_size=100

# dataset="sift10m"
# dim=128
# N=10000000
# query_size=1000
# index_root="/mnt/data/mocheng/dataset/sift10m/" 
# dataset_file=${index_root}sift10m.fvecs
# query_file=${index_root}sift10m_query.fvecs
# label_root="/mnt/data/mocheng/dataset/sift10m/label/"${attr_index_type}"/"
# dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
# query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
# ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
# ground_truth_collection_name=${dataset}_${attr_index_type}
# metric="L2"
# test_query_size=1000

# dataset="siftsmall"
# N=10000
# dim=128
# query_size=100
# index_root="/mnt/data/mocheng/dataset/siftsmall"
# dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
# query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
# label_root="/mnt/data/mocheng/dataset/siftsmall/label/"${attr_index_type}"/"
# query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
# ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
# ground_truth_collection_name=${dataset}_${attr_index_type}
# dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
# metric="L2"
# test_query_size=100

# ---------------- youtube RGB -----------------
attr_type="[0,0]"
attr_index_type="arbi_0_0_"${distribution_type}
dataset="youtube_rgb"
dim=1024
N=1000000
query_size=1000
index_root="/mnt/data/mocheng/dataset/youtube1m"
dataset_file="/mnt/data/mocheng/dataset/youtube1m/rgb.fvecs"
query_file="/mnt/data/mocheng/dataset/youtube1m/rgb_query.fvecs"
dataset_attr_file="/mnt/data/mocheng/dataset/youtube1m/dates_views.json"
label_root="/mnt/data/mocheng/dataset/youtube1m/label/"${attr_index_type}"/"
query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
ground_truth_collection_name=${dataset}_${attr_index_type}
metric="IP"
test_query_size=1000


# ---------------- navix uncorr 1.01 -----------------
# attr_type="[0,0]"
# attr_index_type="arbi_0_0_"${distribution_type}
# dataset="navix_uncorr_1_01"
# # dataset="navix_uncorr_5_10"
# # dataset="navix_uncorr_9_96"
# # dataset="navix_uncorr_15_02"
# # dataset="navix_uncorr_22_93"
# N=15435516
# dim=1024
# query_size=50
# source ./wiki_conf.sh   # set files according to dataset
# metric="IP"
# test_query_size=50

# -----------------------------------------------

hashann_root="${index_root}/hashann/"
hashann_index_root=${hashann_root}index/

# use_ft=false

hashann_index_file=${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}

if [ ! -f "$hashann_index_file" ]; then
    echo "$hashann_index_file does not exist. Please check."
fi

# dir=${now}_${dataset}_${algo}

# if [ ! -d "$dir" ]; then
#     mkdir ${dir}
# fi

if [ ! -d "$hashann_index_root" ]; then
    mkdir -p ${hashann_index_root}
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