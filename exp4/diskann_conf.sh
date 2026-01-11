
algo=DiskANN_Stitched

alpha=1.2
Stitched_R=80

diskann_root="${index_root}/diskann/"
diskann_index_root="${diskann_root}/index/index_diskann_${attr_index_type}_M${M}_efc${ef_construction}/"
diskann_index_prefix="${diskann_index_root}index_"
diskann_result_root="${diskann_root}/result/"
diskann_result_path=${diskann_result_root}result_${attr_index_type}_efs${ef_search}
label_file=${label_root}data_attr.txt
predicate_txt_file=${label_root}query_predicate_${query_sel}.txt
# keyword_query_range_file=${label_root}query_keyword_${attr_index_type}.txt
ground_truth_bin_file=${label_root}groundtruth_${query_sel}.bin

if [ ! -d "$diskann_index_root" ]; then
    mkdir -p ${diskann_index_root}
fi
if [ ! -d "$diskann_result_root" ]; then
    mkdir -p ${diskann_result_root}
fi


dataset_bin_file="${index_root}/bin/dataset_base.bin"
query_bin_file="${index_root}/bin/query_base.bin"
attr_bin_file="${index_root}/bin/attr_${attr_index_type}.bin"


if [ -e $dataset_bin_file ]; then
    echo "dataset bin already exist at $dataset_bin_file"
else
    echo "convert base vecs to bin"
    ./../../code/DiskANN/build/apps/utils/fvecs_to_bin float $dataset_file $dataset_bin_file
fi

if [ -e $query_bin_file ]; then
    echo "query bin already exist at $query_bin_file"
else
    echo "convert query vecs to bin"
    ./../../code/DiskANN/build/apps/utils/fvecs_to_bin float $query_file $query_bin_file
fi

if [ -e $label_file ]; then
    echo "label file already exist at $label_file"
else
    echo "convert json label to txt"
    python ../tests/json2txt.py $dataset_attr_file $label_file

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script json2txt.py failed with exit status $status"
        exit $status
    else
        echo "Python script json2txt.py ran successfully"
    fi
fi

if [ -e $predicate_txt_file ]; then
    echo "query keyword file already exist at $predicate_txt_file"
else
    echo "convert json range query predicate to keyword txt"
    python ../tests/range2keyword.py $query_predicate_file $predicate_txt_file
    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script range2keyword.py failed with exit status $status"
        exit $status
    else
        echo "Python script range2keyword.py ran successfully"
    fi
fi


if [ -e $ground_truth_bin_file ]; then
    echo "groundtruth bin file already exist at $ground_truth_bin_file"
else
    echo "convert json range query label to keyword txt"
    python ../tests/gt_json2bin.py $ground_truth_file $ground_truth_bin_file $K

    status=$?
    if [ $status -ne 0 ]; then
        echo "Python script failed with exit status $status"
        exit $status
    else
        echo "Python script ran successfully"
    fi
fi