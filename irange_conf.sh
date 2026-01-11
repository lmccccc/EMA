# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn




algo=iRangeGraph

# test_query_size=${query_size}

irange_index_root="${index_root}/irange/index"

dataset_bin_file="${index_root}/bin/dataset_base.bin"
query_bin_file="${index_root}/bin/query_base.bin"
attr_bin_file="${index_root}/bin/attr_${attr_index_type}.bin"
irange_id2od_file="${irange_index_root}/id2od_irange_M${M}_efc${ef_construction}.bin"
predicate_bin_file="${index_root}/bin/predicate_${query_file_prefix}.bin"
irange_result_root="${index_root}/irange/result/"
irange_result_file="${index_root}/irange/result/result_irange_M${M}_efc${ef_construction}_efs${ef_search}"
ground_truth_bin_file="${index_root}/bin/groundtruth_${query_file_prefix}.bin"

if [ ! -d "$irange_index_root" ]; then
    mkdir -p ${irange_index_root}
fi
irange_index_file=${irange_index_root}/index_irange_M${M}_efc${ef_construction}

if [ ! -d "${index_root}/bin" ]; then
    mkdir -p ${index_root}/bin
fi

if [ ! -d "$irange_result_root" ]; then
    mkdir -p ${irange_result_root}
fi

if [ -e $dataset_bin_file ]; then
    echo "dataset bin already exist"
else
    echo "convert fvecs to bin"
    # same file format used with DiskANN, so use it
    ./../code/DiskANN/build/apps/utils/fvecs_to_bin float $dataset_file $dataset_bin_file
fi

if [ -e $query_bin_file ]; then
    echo "query bin already exist"
else
    echo "convert query vecs to bin"
    # same file format used with DiskANN, so use it
    ./../code/DiskANN/build/apps/utils/fvecs_to_bin float $query_file $query_bin_file
fi

if [ -e $attr_bin_file ]; then
    echo "query range bin file already exist"
else
    echo "convert json attr to bin, N inetger."
    python tests/qrange_json2bin.py $dataset_attr_file $attr_bin_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi
if [ -e $predicate_bin_file ]; then
    echo "query range bin file already exist"
else
    echo "convert json query range to bin, N*2 inetger formated as left, right, left, right..."
    python tests/qrange_json2bin.py $query_predicate_file $predicate_bin_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
    else
        echo "Python script ran successfully"
    fi
fi

if [ -e $ground_truth_bin_file ]; then
    echo "groundtruth bin file already exist"
else
    echo "convert json range query label to keyword txt"
    python tests/gt_json2bin.py $ground_truth_file $ground_truth_bin_file $K

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
    else
        echo "Python script ran successfully"
    fi
fi