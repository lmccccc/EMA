# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

source ./conf.sh
source ./acorn_conf.sh

threads=1

echo "dataset: ${dataset}"
echo "N: ${N}"
echo "gamma: ${gamma}"
echo "M: ${M}"
echo "M_beta: ${M_beta}"
echo "K: ${K}"
echo "threads: ${threads}"
echo "dataset file: ${dataset_file}"
echo "query file: ${query_file}"
echo "dataset attr file: ${dataset_attr_file}"
echo "query predicate file: ${query_predicate_file}"
echo "ground truth file: ${ground_truth_file}"
echo "acorn index file: ${acorn_index_file}"
echo "efs: $ef_search_list"
echo "dim: ${dim}"
echo "attr type: ${attr_type}"
echo "test query size: ${test_query_size}"

../code/ACORN/build/demos/acorn_query_arbi $dataset \
                                $N \
                                $gamma \
                                $M \
                                $M_beta \
                                $K \
                                $threads \
                                $dataset_file \
                                $query_file \
                                $dataset_attr_file \
                                $query_predicate_file \
                                $ground_truth_file \
                                $acorn_index_file \
                                $ef_search_list \
                                $dim \
                                $attr_type \
                                $test_query_size