
algo=HNSWlib
# test_query_size=${query_size}

hnsw_index_root="${index_root}/hnsw/index"
# M=40

if [ ! -d "$hnsw_index_root" ]; then
    mkdir -p ${hnsw_index_root}
fi

hnsw_index_file=${hnsw_index_root}/index_hnsw_random_M${M}_efc${ef_construction}