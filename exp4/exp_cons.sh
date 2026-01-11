#!/bin/bash




# ./predicate_generator.sh
./ground_truth_index.sh
./ground_truth_generator.sh

# run exps
./msvbase_hnsw_index.sh # msvbase
./hashann.sh  # bfann
# ./navix_index.sh # navix
# ./acorn_index.sh # acorn
./milvus_hnsw_index.sh # milvus