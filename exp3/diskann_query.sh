source ./conf.sh
source ./diskann_conf.sh

# --universal_label $universal_label \
echo "index file: $diskann_index_prefix" 
echo "query file: $query_bin_file"
echo "ground truth file: $ground_truth_bin_file"
echo "query label file: $predicate_txt_file"
echo "result save path: $diskann_result_path"
../../DiskANN/build/apps/search_memory_index  --data_type float \
                                        --dist_fn l2 \
                                        --index_path_prefix $diskann_index_prefix \
                                        --query_file $query_bin_file \
                                        --gt_file $ground_truth_bin_file \
                                        --query_filters_file $predicate_txt_file \
                                        -K $K \
                                        -L $L \
                                        --result_path $diskann_result_path \
                                        --num_threads $threads \
                                        --dist_fn $metric \
                                        2>&1 | tee logs/diskann.log



python tests/extract_results.py "logs/diskann.log" $dataset $attr_type $query_sel $M "diskann"
