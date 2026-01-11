#!/bin/bash

# index construction
# 1. [0,0] all
# 2. [0,1] all
# 3. [0]  all+irange, redcaps, youtube
# 4. [1]  all+diskann, sift, wiki
# 5. [0] neg corr wiki, all

# sel=(
#     "[0]"
#     "[1]"
#     "[2]"
#     "[3]"
#     "[4]"
#     "[5]"
#     "[6]"
#     "[7]"
#     "[8]"
#     "[9]"
#     "[18]"
# )

# sel=(
#     "[0.01]"
#     "[0.03]"
#     "[0.05]"
#     "[0.08]"
#     "[0.1]"
#     "[0.2]"
#     "[0.5]"
#     "[0.8]"
#     "[1.0]"
# )

sel=(
    "[0.18,0.18]"
    "[0.23,0.23]"
)

datasets=(
    "youtube_rgb"
    "Redcaps_4M"
    "sift10m"
    "wiki_15_4M"
)

num_num_label="[0,0]"
sed -i "s|^attr_type=.*|attr_type=\"$num_num_label\"|" conf.sh
tmp_num_num_sel="[0.1,0.1]"
sed -i "s|^query_sel=.*|query_sel=\"$tmp_num_num_sel\"|" conf.sh

for dataset in "${datasets[@]}"
do
    # # modify dataset in conf.sh
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # # construct diskann and hashann index
    ./attr_generator.sh
    ./predicate_generator.sh
    ./ground_truth_generator.sh
    # ./diskann_index.sh
    # ./irange_index.sh
    ./hashann.sh
    ./acorn_index.sh
    ./navix_index.sh
    ./milvus_hnsw_index.sh
    ./msvbase_hnsw_index.sh


    # for sel_value in "${sel[@]}"
    # do
    #     # modify sel in conf.sh
    #     sed -i "s|^query_sel=.*|query_sel=$sel_value|" conf.sh
    #     # generate query predicate and ground truth
    #     ./predicate_generator.sh
    #     ./ground_truth_generator.sh
    #     # run query
    #     # ./diskann_query.sh
    #     # ./irange_query.sh
    #     ./query.sh
    # done
done

num_cate_label="[0,1]"
sed -i "s|^attr_type=.*|attr_type=\"$num_cate_label\"|" conf.sh
num_cate_sel="[0.1,9]"  # 0.1*0.1=1%
sed -i "s|^query_sel=.*|query_sel=\"$num_cate_sel\"|" conf.sh

for dataset in "${datasets[@]}"
do
    # # modify dataset in conf.sh
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # # construct diskann and hashann index
    ./attr_generator.sh
    ./predicate_generator.sh
    ./ground_truth_generator.sh
    # ./diskann_index.sh
    # ./irange_index.sh
    ./hashann.sh
    ./acorn_index.sh
    ./navix_index.sh
    ./milvus_hnsw_index.sh
    ./msvbase_hnsw_index.sh
done


range_datasets=(
    "youtube_rgb"
    "Redcaps_4M"
)

num_label="[0]"
sed -i "s|^attr_type=.*|attr_type=\"$num_label\"|" conf.sh
num_sel="[0.1]"
sed -i "s|^query_sel=.*|query_sel=\"$num_sel\"|" conf.sh

for dataset in "${range_datasets[@]}"
do
    # # modify dataset in conf.sh
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # # construct diskann and hashann index
    ./attr_generator.sh
    ./predicate_generator.sh
    ./ground_truth_generator.sh

    ./hashann.sh
    ./acorn_index.sh
    ./navix_index.sh
    ./milvus_hnsw_index.sh
    ./msvbase_hnsw_index.sh
    ./irange_index.sh
done

label_datasets=(
    "sift10m"
    "wiki_15_4M"
)

cate_label="[1]"
sed -i "s|^attr_type=.*|attr_type=\"$cate_label\"|" conf.sh
cate_sel="[18]"
sed -i "s|^query_sel=.*|query_sel=\"$cate_sel\"|" conf.sh

for dataset in "${label_datasets[@]}"
do
    # # modify dataset in conf.sh
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # # construct diskann and hashann index
    ./attr_generator.sh
    ./predicate_generator.sh
    ./ground_truth_generator.sh

    ./hashann.sh
    ./acorn_index.sh
    ./navix_index.sh
    ./milvus_hnsw_index.sh
    ./diskann_index.sh
    ./msvbase_hnsw_index.sh
done


neg_corr_datasets="wiki_15_4M"

neg_predicates=(
    "wiki_negcorr_1_01"
    "wiki_negcorr_5_10"
    "wiki_negcorr_9_96"
    "wiki_negcorr_15_02"
    "wiki_negcorr_22_93"
)
cate_label="[0]"
sed -i "s|^attr_type=.*|attr_type=\"$cate_label\"|" conf.sh
cate_sel="[0.01]"
sed -i "s|^query_sel=.*|query_sel=\"$cate_sel\"|" conf.sh

sed -i "s|^dataset=.*|dataset=\"wiki_negcorr_1_01\"|" conf.sh
./hashann.sh
./acorn_index.sh
./navix_index.sh
./milvus_hnsw_index.sh
./msvbase_hnsw_index.sh