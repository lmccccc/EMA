#!/bin/bash

# pred1 = [0.32, 0.45, 0.55, 0.63, 0.71, 0.775, 0.84, 0.9, 0.95, 1.0]

# generate attr, predicate and ground truth files

# [0,0]

sel=(
    "[0.1,0.1]" 
    "[0.32,0.32]" 
    "[0.45,0.45]" 
    "[0.55,0.55]" 
    "[0.63,0.63]" 
    "[0.71,0.71]" 
    "[0.775,0.775]" 
    "[0.84,0.84]" 
    "[0.9,0.9]" 
    "[0.95,0.95]" 
    "[1.0,1.0]"
    )

# sel=("[0.3,0.3]" 
#     "[0.282,0.282]" 
#     "[0.265,0.265]" 
#     "[0.245,0.245]" 
#     "[0.223,0.223]" 
#     "[0.2,0.2]" 
#     "[0.173,0.173]" 
#     "[0.141,0.141]" 
#     "[0.1,0.1]" 
#     )

# sel=(
#     "[0.032,0.032]" # 0.1%
#     # "[0.07,0.07]"   # 0.5%
#     # "[0.22,0.23]"   # 5%
#     )
#     "[0.1,0.1]"    # 1%
#     "[0.32,0.32]"  # 10%

wiki_dataset=(
    "wiki_negcorr_1_01" 
    "wiki_negcorr_5_10" 
    "wiki_negcorr_9_96" 
    "wiki_negcorr_15_02" 
    "wiki_negcorr_22_93" 
)

ef_search_list1="[10,12,15,18,20]"
ef_search_list2="[10,12,15,18,20,30,40,50,80,100,120,150,180,200]"
ef_search_list3="[80,100,120,150,180,200,300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"
ef_search_list4="[300,400,600,800,1000,1200,1500,2000,2500,3000,4000]"

# for datasets with various query selectivities
for query_sel in "${sel[@]}"
do
    # modify sel in conf.sh
    sed -i "s|^query_sel=.*|query_sel=\"$query_sel\"|" conf.sh
    # construct predicate and ground truth files
    # ./predicate_generator.sh
    # ./ground_truth_generator.sh

    # run exps
    sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list2|" conf.sh
    # sed -i 's/ft_bits=256/ft_bits=128/g' conf.sh
    ./query.sh  # bfann
    # sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list2|" conf.sh
    # sed -i "s/ft_bits=128/ft_bits=256/g" conf.sh
    # ./256query.sh  # bfann_256 bf bits
    # sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list3|" conf.sh
    # ./navix_query.sh # navix
    # sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list4|" conf.sh
    # ./acorn_query.sh # acorn
    # sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list1|" conf.sh
    # ./milvus_hnsw_query.sh # milvus
    # ./msvbase_hnsw_query.sh # msvbase
done


# for wiki neg corr datasets
# for dataset in "${wiki_dataset[@]}"
#     # modify sel in conf.sh
#     sed -i "/^[[:space:]]*#/! s|^[[:space:]]*wiki_neg_query=.*|wiki_neg_query=\"$dataset\"|" conf.sh

#     # construct predicate and ground truth files
#     ./predicate_generator.sh
#     ./ground_truth_generator.sh

#     # run exps
#     # sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list2|" conf.sh
#     # sed -i 's/ft_bits=256/ft_bits=128/g' conf.sh
#     # ./query.sh  # bfann
#     sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list2|" conf.sh
#     sed -i "s/ft_bits=128/ft_bits=256/g" conf.sh
#     ./256query.sh  # bfann_256 bf bits
#     sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list3|" conf.sh
#     ./navix_query.sh # navix
#     sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list4|" conf.sh
#     ./acorn_query.sh # acorn
#     sed -i "s|^ef_search_list=.*|ef_search_list=$ef_search_list1|" conf.sh
#     ./milvus_hnsw_query.sh # milvus
#     ./msvbase_hnsw_query.sh # msvbase
# done