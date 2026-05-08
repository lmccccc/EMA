#!/bin/bash
set -euo pipefail

ensure_inputs() {
    source ./conf.sh

    if [ ! -f "$dataset_attr_file" ]; then
        ./attr_generator.sh
    fi

    if [ ! -f "$query_predicate_file" ]; then
        ./predicate_generator.sh
    fi

    if [ ! -f "$ground_truth_file" ]; then
        ./ground_truth_generator.sh
    fi
}

# run_case() {
#     local dataset="$1"
#     local attr_type="$2"
#     local query_sel="$3"

#     set_conf "dataset" "\"${dataset}\""
#     set_conf "attr_type" "\"${attr_type}\""
#     set_conf "query_sel" "\"${query_sel}\""
#     set_conf "use_ft" "false"

#     source ./conf.sh
#     echo "===================================================="
#     echo "[Baseline] dataset=${dataset}, attr_type=${attr_type}, query_sel=${query_sel}, use_ft=${use_ft}"

#     ensure_inputs
#     ./hashann.sh
#     ./query.sh
# }

num_high_sel=(
    "[0.2]"
    "[0.4]"
    "[0.6]"
    "[0.8]"
    "[1.0]"
)

num_sel=(
    "[0.01]"
    "[0.02]"
    "[0.03]"
    "[0.05]"
    "[0.07]"
    "[0.1]"
)

num_num_sel=(
    "[0.1,0.1]"
    "[0.141,0.141]"
    "[0.173,0.173]"
    "[0.2236,0.2236]"
    "[0.2646,0.2646]"
    "[0.316,0.316]"
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
    "youtube_rgb"
    # "wiki_15_4M"
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
#     ./attr_generator.sh
#     for sel in "${num_high_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         ./predicate_generator.sh
#         ./ground_truth_generator.sh
#     done

#     for sel in "${num_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         ./predicate_generator.sh
#         ./ground_truth_generator.sh
#     done
# done


# cate_label="[1]"
# sed -i "s|^attr_type=.*|attr_type=\"$cate_label\"|" conf.sh
# tmp_cate_sel="[18]"
# sed -i "s|^query_sel=.*|query_sel=\"$tmp_cate_sel\"|" conf.sh
# # num high sel attr_and_gt_generate
# for dataset in "${label_datasets[@]}"
# do
#     sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
#     ./attr_generator.sh
#     for sel in "${label_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         ./predicate_generator.sh
#         ./ground_truth_generator.sh
# done

num_cate_label="[0,1]"
sed -i "s|^attr_type=.*|attr_type=\"$num_cate_label\"|" conf.sh
tmp_num_cate_sel="[0.1,9]"
sed -i "s|^query_sel=.*|query_sel=\"$tmp_num_cate_sel\"|" conf.sh
# num high sel attr_and_gt_generate
for dataset in "${datasets[@]}"
do
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # ./attr_generator.sh
    for sel in "${num_label_sel[@]}"
    do
        # modify dataset in conf.sh
        sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
        # ./predicate_generator.sh
        # ./ground_truth_generator.sh
        ./hnsw_query.sh
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
#     # ./attr_generator.sh
#     for sel in "${num_num_sel[@]}"
#     do
#         # modify dataset in conf.sh
#         sed -i "s|^query_sel=.*|query_sel=\"$sel\"|" conf.sh
#         # ./predicate_generator.sh
#         # ./ground_truth_generator.sh
#         ./post_query.sh
#     done
# done


