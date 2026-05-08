#! /bin/bash

source ./conf.sh
source ./post_conf.sh

# check hnsw_index_file exists
if [ -f "$hnsw_index_file" ]; then
    echo "HNSW index file $hnsw_index_file exists. Skipping construction."
    exit 0
fi

echo "====================================================" >> logs/hnsw_construction.log
echo "dataset: $dataset" >> logs/hnsw_construction.log
echo "attr: $attr_type" >> logs/hnsw_construction.log
echo "algo: hnsw" >> logs/hnsw_construction.log

python -u ../tests/hnsw_build.py --data_path $dataset_file \
                                 --index_cache_path $hnsw_index_file \
                                 --N $N \
                                 --M $M \
                                 --dim $dim \
                                 --metric ${metric} \
                                 --efConstruction $ef_construction \
                                 --name $algo \
                                 --threads $threads \
                                 2>&1 | tee -a logs/hnsw_construction.log

echo "\n" >> logs/hnsw_construction.log