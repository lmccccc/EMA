
algo=Navix
# test_query_size=${query_size}

navix_index_root="${index_root}/navix/index"
# M=40

if [ ! -d "$navix_index_root" ]; then
    mkdir -p ${navix_index_root}
fi

navix_index_file=${navix_index_root}/index_navix_random_M${M}_efc${ef_construction}