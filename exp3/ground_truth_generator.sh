source ./conf.sh


if [ -f ${ground_truth_file} ]; then
    echo "${ground_truth_file} exists, skip predicate generation."
    exit 0
fi


# python ../tests/groundtruth_generator.py --dataset_file ${dataset_file} \
#                              --d ${dim} \
#                              --attr_file ${dataset_attr_file} \
#                              --query_file ${query_file} \
#                              --attr_type_list ${attr_type} \
#                              --N ${N} \
#                              --query_size ${query_size} \
#                              --predicate_file ${query_predicate_file} \
#                              --c_name ${ground_truth_collection_name} \
#                              --mode query \
#                              --max_cate_val ${categorical_attr_max_cardinality} \
#                              --K ${K} \
#                              --gt_file ${ground_truth_file} \
#                              --metric ${metric}

# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

source ./conf.sh
source ./navix_conf.sh

threads=1

echo "dataset: ${dataset}"
echo "N: ${N}"
echo "M: ${M}"
echo "K: ${K}"
echo "threads: ${threads}"
echo "dataset file: ${dataset_file}"
echo "query file: ${query_file}"
echo "dataset attr file: ${dataset_attr_file}"
echo "query predicate file: ${query_predicate_file}"
echo "ground truth file: ${ground_truth_file}"
echo "ground truth index file: ${ground_truth_index_file}"
echo "efs: $ef_search_list"
echo "dim: ${dim}"
echo "attr type: ${attr_type}"
echo "test query size: ${test_query_size}"

# echo "dataset: $dataset" >> logs.txt
# echo "attr: $attr_type" >> logs.txt
# echo "sel: $query_sel" >> logs.txt
# echo "algo: navix" >> logs.txt

../../code/faiss-navix/build/demos/groundtruth_generator $dataset \
                                $N \
                                $M \
                                $K \
                                $threads \
                                $dataset_file \
                                $query_file \
                                $dataset_attr_file \
                                $query_predicate_file \
                                $ground_truth_file \
                                $ground_truth_index_file \
                                $ef_search_list \
                                $dim \
                                $attr_type \
                                $test_query_size \
                                2>&1 | tee logs/navix.log


# python ../tests/extract_results.py "logs/groundtruth_generator.log" $dataset $attr_type $query_sel $M "groundtruth_generator"
