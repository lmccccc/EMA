#!/bin/bash

# pred1 = [0.32, 0.45, 0.55, 0.63, 0.71, 0.775, 0.84, 0.9, 0.95, 1.0]

num_high_sel=(
    "[0.2]" # 20%
    "[0.4]" # 40%
    "[0.6]" # 60%
    "[0.8]"  # 80%
    "[1.0]" # 100%
    )



num_sel=(
    "[0.01]"   # 1%
    "[0.02]"   # 2%
    "[0.03]"   # 3%
    "[0.05]"   # 5%
    "[0.07]"   # 7%
    "[0.1]"    # 10%
)

num_num_sel=(
    "[0.1,0.1]"   # 1%
    "[0.141,0.141]" # 2%
    "[0.173,0.173]" # 3%
    "[0.2236,0.2236]" # 5%
    "[0.2646,0.2646]" # 7%
    "[0.316,0.316]" # 10%
    )

label_sel=(
    "[18]"   # 1%
    "[17]"   # 2%
    "[16]"   # 3%
    "[14]"   # 5%
    "[12]"   # 7%
    "[9]"    # 10%
)

num_label_sel=(
    "[0.1,9]"  # 0.1*0.1=1%
    "[0.1,8]"  # 0.1*0.2=2%
    "[0.15,8]" # 0.15*0.2=3%
    "[0.177,7]"  # 0.177*0.3=5%
    "[0.233,7]"  # 0.233*0.3=7%
    "[0.333,7]"   # 0.333*0.3=10%
)



datasets=(
    # "youtube_rgb"
    "wiki_15_4M"
    # "sift10m"
    # "Redcaps_4M"
)

range_datasets=(
    "youtube_rgb"
    "Redcaps_4M"
)

label_datasets=(
    "sift10m"
    "wiki_15_4M"
)

# num_label="[0]"
# sed -i "s|^attr_type=.*|attr_type=\"$num_label\"|" conf.sh
# tmp_num_sel="[1.0]"
# sed -i "s|^query_sel=.*|query_sel=\"$tmp_num_sel\"|" conf.sh
# # num high sel attr_and_gt_generate
# for dataset in "${range_datasets[@]}"
# do
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     # ./attr_generator.sh
#     for sel in "${num_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         # ./query.sh
#         # ./navix_query.sh
#         ./acorn_query.sh
#         # ./milvus_hnsw_query.sh
#         # ./msvbase_hnsw_query.sh
#         # ./irange_query.sh
#     done
# done


# cate_label="[1]"
# sed -i "s|^attr_type=.*|attr_type=\"$cate_label\"|" conf.sh
# tmp_cate_sel="[18]"
# sed -i "s|^query_sel=.*|query_sel=\"$tmp_cate_sel\"|" conf.sh
# # num high sel attr_and_gt_generate
# echo "query for [1]"
# for dataset in "${label_datasets[@]}"
# do
#     echo "dataset: $dataset"
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     for sel in "${label_sel[@]}"
#     do
#         echo "sel: $sel"
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         ./query.sh
#         ./navix_query.sh
#         ./acorn_query.sh
#         ./milvus_hnsw_query.sh
#         ./msvbase_hnsw_query.sh
#         ./diskann_query.sh
#     done
# done

num_label_high_sel=(
    "[0.4,5]" # 0.4 * 0.5 = 20%
    "[0.667,4]" # 0.667 * 0.6 = 40%
    "[0.75,2]" # 0.75 * 0.8 = 60%
    "[0.9,1]"  # 0.9 * 0.9 = 81%
    "[1.0,0]" # 1 * 1 = 100%
)

num_cate_label="[0,1]"
sed -i "s|^attr_type=.*|attr_type=\"$num_cate_label\"|" conf.sh
tmp_num_cate_sel="[0.1,9]"
sed -i "s|^query_sel=.*|query_sel=\"$tmp_num_cate_sel\"|" conf.sh
# num high sel attr_and_gt_generate
echo "query for [0,1]"
for dataset in "${datasets[@]}"
do
    echo "dataset: $dataset"
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # for sel in "${num_label_sel[@]}"
    # do
    #     # modify dataset in conf.sh
    #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
    #     echo "sel: $sel"
    #     # ./query.sh
    #     # ./navix_query.sh
    #     # ./acorn_query.sh
    #     # ./milvus_hnsw_query.sh
    #     ./msvbase_hnsw_query.sh
    # done

    for sel in "${num_label_high_sel[@]}"
    do
        # modify dataset in conf.sh
        sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
        echo "sel: $sel"
        # ./query.sh
        # ./navix_query.sh
        # ./acorn_query.sh
        # ./milvus_hnsw_query.sh
        # ./msvbase_hnsw_query.sh
        ./ground_truth_generator.sh
    done
done


# num_num_label="[0,0]"
# sed -i "s|^attr_type=.*|attr_type=\"$num_num_label\"|" conf.sh
# tmp_num_num_sel="[0.1,0.1]"
# sed -i "s|^query_sel=.*|query_sel=\"$tmp_num_num_sel\"|" conf.sh
# # num high sel attr_and_gt_generate
# for dataset in "${datasets[@]}"
# do
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     for sel in "${num_num_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         ./query.sh
#         # ./navix_query.sh
#         # ./acorn_query.sh
#         # ./milvus_hnsw_query.sh
#         # ./msvbase_hnsw_query.sh
#     done
# done


