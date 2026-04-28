source ./conf.sh
source ./milvus_conf.sh

echo "dataset: $dataset" >> logs.txt
echo "attr: $attr_type" >> logs.txt
echo "sel: $query_sel" >> logs.txt
echo "algo: milvus" >> logs.txt

python -u ../tests/milvus_hnsw_index.py --dataset_file ${dataset_file} \
                             --d ${dim} \
                             --attr_file ${dataset_attr_file} \
                             --query_file ${query_file} \
                             --attr_type_list ${attr_type} \
                             --N ${N} \
                             --query_size ${query_size} \
                             --predicate_file ${query_predicate_file} \
                             --c_name ${milvus_collection_name} \
                             --mode "query" \
                             --max_cate_val ${categorical_attr_max_cardinality} \
                             --K ${K} \
                             --gt_file ${ground_truth_file} \
                             --metric ${metric} \
                             --M ${M} \
                             --ef_construction ${ef_construction} \
                             --ef_search ${ef_search_list} \
                             2>&1 | tee logs/milvus.log

python ../tests/extract_results.py "logs/milvus.log" $dataset $attr_type $query_sel $M "milvus" $K