#!/bin/bash

sel=(
    "[0]"
    "[2]"
    "[5]"
    "[8]"
    "[9]"
    "[11]"
    "[14]"
    "[16]"
    "[18]"
)

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

datasets=(
    "youtube_rgb"
    "Redcaps_4M"
    "wiki_15_4M"
    "sift10m"
)

for dataset in "${datasets[@]}"
do
    # # modify dataset in conf.sh
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    # # construct diskann and hashann index
    ./attr_generator.sh
    ./predicate_generator.sh
    ./ground_truth_generator.sh
    ./diskann_index.sh
    # ./irange_index.sh
    ./hashann.sh


    # for sel_value in "${sel[@]}"
    # do
    #     # modify sel in conf.sh
    #     sed -i "s|^query_sel=.*|query_sel=$sel_value|" conf2.sh
    #     # generate query predicate and ground truth
    #     ./predicate_generator.sh
    #     ./ground_truth_generator.sh
    #     # run query
    #     # ./diskann_query.sh
    #     ./irange_query.sh
    #     ./query.sh
    # done
done
