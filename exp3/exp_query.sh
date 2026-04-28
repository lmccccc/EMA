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

# num_num_sel=(
#     "[0.1,0.1]"   # 1%
#     "[0.141,0.141]" # 2%
#     "[0.173,0.173]" # 3%
#     "[0.2236,0.2236]" # 5%
#     "[0.2646,0.2646]" # 7%
#     "[0.316,0.316]" # 10%
#     )

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
    # "wiki_15_4M"
    # "sift10m"
    "Redcaps_4M"
)

range_datasets=(
    "youtube_rgb"
    "Redcaps_4M"
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
#         ./query.sh
#         # ./navix_query.sh
#         # ./acorn_query.sh
#         # ./milvus_hnsw_query.sh
#         # ./msvbase_hnsw_query.sh
#         # ./irange_query.sh
#     done
# done

label_datasets=(
    "youtube_rgb"
    # "wiki_15_4M"
    # "sift10m"
    # "Redcaps_4M"
)

# efs1="[10,12,15,18,20,30,40,50,80,100,120,150,180,200]"
# efs2="[100,150,200,300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"
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

#         # sed -i "s|^ef_search_list=.*|ef_search_list=$efs1|" conf.sh
#         # ./query.sh

#         # sed -i "s|^ef_search_list=.*|ef_search_list=$efs2|" conf.sh
#         # ./navix_query.sh
#         # ./acorn_query.sh
#         # sed -i "s|^ef_search_list=.*|ef_search_list=$efs1|" conf.sh
#         ./milvus_hnsw_query.sh
#         # ./msvbase_hnsw_query.sh
#         # ./diskann_query.sh
#     done
# done

# num_label_high_sel=(
#     "[0.4,5]" # 0.4 * 0.5 = 20%
#     "[0.667,4]" # 0.667 * 0.6 = 40%
#     "[0.75,2]" # 0.75 * 0.8 = 60%
#     "[0.9,1]"  # 0.9 * 0.9 = 81%
#     "[1.0,0]" # 1 * 1 = 100%
# )

datasets=(
    "youtube_rgb"
    # "wiki_15_4M"
    # "sift10m"
    # "Redcaps_4M"
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
    for sel in "${num_label_sel[@]}"
    do
        # modify dataset in conf.sh
        sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
        echo "sel: $sel"
        # ./ground_truth_generator.sh
        # ./query.sh
        ./navix_query.sh
        ./acorn_query.sh
        # ./milvus_hnsw_query.sh
        # ./msvbase_hnsw_query.sh
    done

    # for sel in "${num_label_high_sel[@]}"
    # do
    #     # modify dataset in conf.sh
    #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
    #     echo "sel: $sel"
    #     ./ground_truth_generator.sh
    #     # ./query.sh
    #     # ./navix_query.sh
    #     # ./acorn_query.sh
    #     # ./milvus_hnsw_query.sh
    #     # ./msvbase_hnsw_query.sh
    # done
done

# for dataset in "${datasets[@]}"
# do
#     echo "dataset: $dataset"
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     for sel in "${num_label_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         echo "sel: $sel"
#         ./query.sh
#         ./navix_query.sh
#         ./acorn_query.sh
#         # ./milvus_hnsw_query.sh
#         # ./msvbase_hnsw_query.sh
#     done

# done

efs1="[10,12,15,18,20,30,40,50,80,100,120,150,180,200]"
# efs2="[100,150,200,300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"

datasets=(
    # "youtube_rgb"
    # "wiki_15_4M"
    "sift10m"
    # "Redcaps_4M"
)


num_num_sel=(
    "[0.55,0.01818]"    # ~1%
    "[0.60,0.03333]"    # ~2%
    "[0.65,0.04615]"    # ~3%
    "[0.70,0.07143]"    # ~5%
    "[0.75,0.09333]"    # ~7%
    "[0.80,0.125]"      # ~10%
)
# num_num_label="[0,0]"
# sed -i "s|^attr_type=.*|attr_type=\"$num_num_label\"|" conf.sh
# tmp_num_num_sel="[0.1,0.1]"
# sed -i "s|^query_sel=.*|query_sel=\"$tmp_num_num_sel\"|" conf.sh
# # num high sel attr_and_gt_generate
# for dataset in "${datasets[@]}"
# do
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     echo "dataset: $dataset"
#     for sel in "${num_num_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         echo "sel: $sel"
#         sed -i "s|^ef_search_list=.*|ef_search_list=$efs1|" conf.sh
#         ./irange_multi_query.sh
#     done 
#     # sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     # for sel in "${num_num_sel[@]}"
#     # do
#     #     # modify dataset in conf.sh
#     #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#     #     echo "sel: $sel"
#     #     sed -i "s|^ef_search_list=.*|ef_search_list=$efs1|" conf.sh
#     #     ./query.sh
#     # done    

#     # for sel in "${num_num_sel[@]}"
#     # do
#     #     # modify dataset in conf.sh
#     #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#     #     echo "sel: $sel"
#     #     sed -i "s|^ef_search_list=.*|ef_search_list=$efs2|" conf.sh
#     #     ./navix_query.sh
#     # done 


#     # for sel in "${num_num_sel[@]}"
#     # do
#     #     # modify dataset in conf.sh
#     #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#     #     echo "sel: $sel"
#     #     sed -i "s|^ef_search_list=.*|ef_search_list=$efs2|" conf.sh
#     #     ./acorn_query.sh
#     # done 


#     # for sel in "${num_num_sel[@]}"
#     # do
#     #     # modify dataset in conf.sh
#     #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#     #     echo "sel: $sel"
#     #     ./msvbase_hnsw_query.sh

#     # done 


#     # for sel in "${num_num_sel[@]}"
#     # do
#     #     # modify dataset in conf.sh
#     #     sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#     #     echo "sel: $sel"
#     #     sed -i "s|^ef_search_list=.*|ef_search_list=$efs1|" conf.sh
#     #     # skip milvus if dataset is wiki
#     #     if [ "$dataset" == "wiki_15_4M" ]; then
#     #         echo "skip milvus for wiki_15_4M"
#     #         continue
#     #     else
#     #         ./milvus_hnsw_query.sh
#     #     fi
#     # done 
# done


# ==========================================
# DNF (OR) predicate experiments
# ==========================================
# Uncomment and configure to run DNF experiments.
# Each dnf_spec is a JSON array of terms (OR'd).
# Each term is {attr_idx: selectivity_or_labels, ...}.
# Missing attrs = don't care.
#
# Example dnf_specs and names:
#   (range sel=0.5 AND label=1) OR (label in [2,3])
#     dnf_spec='[{"0":0.5,"1":[1]},{"1":[2,3]}]'
#     dnf_name="or_0_0.5_1_[1]__1_[2,3]"
#
#   range sel=0.2 OR label=1
#     dnf_spec='[{"0":0.2},{"1":[1]}]'
#     dnf_name="or_0_0.2__1_[1]"

# dnf_experiments=(
#     '[{"0":0.5,"1":[1]},{"1":[2,3]}]|or_0_0.5_1_[1]__1_[2,3]'
#     '[{"0":0.2},{"1":[1]}]|or_0_0.2__1_[1]'
#     '[{"0":0.1},{"1":[5]}]|or_0_0.1__1_[5]'
# )
#
# dnf_datasets=(
#     "sift"
# )
#
# num_cate_label="[0,1]"
# sed -i "s|^attr_type=.*|attr_type=\"$num_cate_label\"|" conf.sh
#
# for dataset in "${dnf_datasets[@]}"
# do
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     for entry in "${dnf_experiments[@]}"
#     do
#         IFS='|' read -r spec name <<< "$entry"
#         sed -i "s|^dnf_spec=.*|dnf_spec='$spec'|" conf.sh
#         sed -i "s|^dnf_name=.*|dnf_name=\"$name\"|" conf.sh
#         echo "=== DNF: $name ==="
#         ./predicate_generator_dnf.sh
#         ./ground_truth_generator_dnf.sh
#         ./query.sh
#     done
#     # reset DNF mode
#     sed -i 's|^dnf_spec=.*|dnf_spec=""|' conf.sh
#     sed -i 's|^dnf_name=.*|dnf_name=""|' conf.sh
# done

