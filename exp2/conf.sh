##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")

algo=HNSW

 # youtube_rgb wiki_negcorr_1_01 wiki_15_4M sift10m Redcaps_4M
dataset="wiki_negcorr_1_01"
wiki_neg_query="wiki_negcorr_15_02"
# wiki_neg_query="wiki_negcorr_5_10" 
# wiki_neg_query="wiki_negcorr_9_96"
# wiki_neg_query="wiki_negcorr_15_02" 
# wiki_neg_query="wiki_negcorr_22_93" 

attr_type="[0]"
distribution_type="random"  # random, normal, zipf

# 0.5 for numerical, [1,2] for categorical label(s)
query_sel="[0.1]"


M=40
threads=32 # 1
ef_construction=300
ef_top=100
K=10
ft_bits=256
use_ft=true

# ef_search_list=[70]
# ef_search_list=[150,180,200,250,300]
# ef_search_list="[80,100,120,150,180,200,300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"
ef_search_list="[10,15,20,30,40,50,80,100,150,200,250,300]"
# ef_search_list="[80,100,120,150,180,200]"
# ef_search_list="[80,100,120,150,180,200,300]"
# ef_search_list="[300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"
# L_list="600 800 1000 1200 1500 2000 2500 3000 4000"
L_list="300 350"
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

# if dataset = sift
if [ "$dataset" = "sift" ]; then
# dataset="sift"
    N=1000000
    dim=128
    query_size=10000
    index_root="/mnt/data/mocheng/dataset/sift"
    dataset_file="/mnt/data/mocheng/dataset/sift/sift_base.fvecs"
    query_file="/mnt/data/mocheng/dataset/sift/sift_query.fvecs"
    label_root="/mnt/data/mocheng/dataset/sift/label/"${attr_index_type}"/"
    dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
    query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
    ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
    ground_truth_collection_name=${dataset}_${attr_index_type}
    metric="L2"
    test_query_size=100

elif [ "$dataset" = "sift10m" ]; then
    # dataset="sift10m"
    dim=128
    N=10000000
    query_size=1000
    index_root="/mnt/data/mocheng/dataset/sift10m/" 
    dataset_file=${index_root}sift10m.fvecs
    query_file=${index_root}sift10m_query.fvecs
    label_root="/mnt/data/mocheng/dataset/sift10m/label/"${attr_index_type}"/"
    dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
    query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
    ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
    ground_truth_collection_name=${dataset}_${attr_index_type}
    metric="L2"
    test_query_size=1000

elif [ "$dataset" = "siftsmall" ]; then
    # dataset="siftsmall"
    N=10000
    dim=128
    query_size=100
    index_root="/mnt/data/mocheng/dataset/siftsmall"
    dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
    query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
    label_root="/mnt/data/mocheng/dataset/siftsmall/label/"${attr_index_type}"/"
    query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
    ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
    ground_truth_collection_name=${dataset}_${attr_index_type}
    dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
    metric="L2"
    test_query_size=100

# ---------------- youtube RGB -----------------
elif [ "$dataset" = "youtube_rgb" ]; then
    # dataset="youtube_rgb"
    dim=1024
    N=1000000
    query_size=1000
    index_root="/mnt/data/mocheng/dataset/youtube1m"
    dataset_file="/mnt/data/mocheng/dataset/youtube1m/rgb.fvecs"
    query_file="/mnt/data/mocheng/dataset/youtube1m/rgb_query.fvecs"
    label_root="/mnt/data/mocheng/dataset/youtube1m/label/"${attr_index_type}"/"
    if [ "$attr_type" = "[0,0]" ]; then
        dataset_attr_file="/mnt/data/mocheng/dataset/youtube1m/dates_views.json"
    else
        dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
    fi
    query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
    ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
    ground_truth_collection_name=${dataset}_${attr_index_type}
    metric="IP"
    test_query_size=1000

# ---------------- Redcaps 4M -----------------
elif [ "$dataset" = "Redcaps_4M" ]; then
    # dataset="Redcaps_4M"
    dim=512
    N=4000000
    query_size=1000
    index_root="/mnt/data/mocheng/dataset/redcaps4m"
    dataset_file="${index_root}/image_embeddings.fvecs"
    query_file="${index_root}/query.fvecs"
    label_root="${index_root}/label/"${attr_index_type}"/"
    dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
    query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
    ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
    ground_truth_collection_name=${dataset}_${attr_index_type}
    metric="IP"
    test_query_size=1000

# ---------------- wiki uncorr -----------------
elif [ "$dataset" = "wiki_15_4M" ]; then
    # attr_type="[0,0]"
    # attr_index_type="arbi_0_0_"${distribution_type}
    dataset="wiki_15_4M"
    N=15435516
    dim=1024
    index_root="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M"
    dataset_file="${index_root}/fvecs/wiki_15.4M.fvecs"
    query_file="${index_root}/fvecs/wiki_15.4M_query.fvecs"
    label_root="${index_root}/label/"${attr_index_type}"/"
    dataset_attr_file=${label_root}"attr_"${attr_index_type}".json"
    query_predicate_file=${label_root}"predicate_"${query_file_prefix}".json"
    ground_truth_file=${label_root}"gt_"${query_file_prefix}".json"
    ground_truth_collection_name=${dataset}_${attr_index_type}
    query_size=1000
    metric="IP"
    test_query_size=1000


# ---------------- wiki negcorr 1.01 -----------------
elif [ "$dataset" = "wiki_negcorr_1_01" ]; then
    attr_type="[0]"
    attr_index_type="arbi_0_"${distribution_type}
    # dataset="wiki_negcorr_1_01"
    # wiki_neg_query="wiki_negcorr_1_01"
    # wiki_neg_query="wiki_negcorr_5_10"
    # wiki_neg_query="wiki_negcorr_9_96"
    # wiki_neg_query="wiki_negcorr_15_02"
    # wiki_neg_query="wiki_negcorr_22_93"
    N=15435516
    dim=1024
    query_size=50
    source ./wiki_conf.sh   # set files according to dataset
    metric="IP"
    test_query_size=50

else 
    echo "Dataset not recognized. Please check."
    exit 1
fi
# -----------------------------------------------

hashann_root="${index_root}/hashann/"
hashann_index_root=${hashann_root}index/

# use_ft=false

# hashann_index_file=${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}
hashann_index_file=${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}

# if hashann_index_file not exist, copy from ori_hashann_index_file and ft_bits=256
ori_hashann_index_file=${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}
if [ ! -f "$hashann_index_file" ] && [ -f "$ori_hashann_index_file" ] && [ "$ft_bits" -eq 128 ]; then
    echo "moving $ori_hashann_index_file to $hashann_index_file"
    mv $ori_hashann_index_file $hashann_index_file
    exit 1
fi

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