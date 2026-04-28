source ./conf.sh



# python tests/groundtruth_generator.py --dataset_file ${dataset_file} \
#                              --d ${dim} \
#                              --attr_file ${dataset_attr_file} \
#                              --query_file ${query_file} \
#                              --attr_type_list ${attr_type} \
#                              --N ${N} \
#                              --query_size ${query_size} \
#                              --predicate_file ${query_predicate_file} \
#                              --c_name ${ground_truth_collection_name} \
#                              --mode "construction" \
#                              --max_cate_val ${categorical_attr_max_cardinality} \
#                              --K ${K} \
#                              --gt_file ${ground_truth_file} \
#                              --metric ${metric}

if [ -f $ground_truth_index_file ]; then
    echo "ground truth index file already exist $ground_truth_index_file"
    exit 0
fi

echo "dataset: $dataset" 
echo "N: $N" 
echo "M: $M" 
echo "K: $K" 
echo "threads: $threads" 
echo "dataset_file: $dataset_file" 
echo "dataset_attr_file: $dataset_attr_file" 
echo "ground_truth_index_file: $ground_truth_index_file" 
echo "dim: $dim" 
echo "metric: $metric" 

../../code/faiss-navix/build/demos/groundtruth_build $dataset \
                            $N \
                            $M \
                            $K \
                            $threads \
                            $dataset_file \
                            $dataset_attr_file \
                            $ground_truth_index_file \
                            $dim \
                            $metric  \
                            2>&1 | tee -a logs/groundtruth_construction.log   
