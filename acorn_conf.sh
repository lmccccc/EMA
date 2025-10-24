
algo=ACORN
M=40
gamma=25
M_beta=64
test_query_size=100
# test_query_size=${query_size}

acorn_root="/mnt/data/mocheng/dataset/sift/acorn"
acorn_index_root="/mnt/data/mocheng/dataset/sift/acorn/index"

if [ ! -d "$acorn_index_root" ]; then
    mkdir -p ${acorn_root}
    mkdir -p ${acorn_index_root}
fi

acorn_index_file=${acorn_index_root}/index_acorn_sel_1_100000_random_M${M}_ga${gamma}_Mb${M_beta}