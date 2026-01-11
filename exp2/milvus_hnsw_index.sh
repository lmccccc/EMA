source ./conf.sh
source ./milvus_conf.sh

echo "====================================================" >> logs/bfann_construction.log
echo "dataset: $dataset" >> logs/milvus_construction.log
echo "attr: $attr_type" >> logs/milvus_construction.log
echo "algo: milvus" >> logs/milvus_construction.log


python -u ../tests/milvus_hnsw_index.py --dataset_file ${dataset_file} \
                             --d ${dim} \
                             --attr_file ${dataset_attr_file} \
                             --query_file ${query_file} \
                             --attr_type_list ${attr_type} \
                             --N ${N} \
                             --query_size ${query_size} \
                             --predicate_file ${query_predicate_file} \
                             --c_name ${milvus_collection_name} \
                             --mode "construction" \
                             --max_cate_val ${categorical_attr_max_cardinality} \
                             --K ${K} \
                             --gt_file ${ground_truth_file} \
                             --metric ${metric} \
                             --M ${M} \
                             --ef_construction ${ef_construction} \
                             --ef_search ${ef_search_list} \
                            2>&1 | tee -a logs/milvus_construction.log

echo "\n" >> logs/milvus_construction.log