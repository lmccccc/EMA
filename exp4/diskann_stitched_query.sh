source ./conf.sh
source ./diskann_stitched_conf.sh

threads=1
# --universal_label $universal_label \
echo "metric: $metric"
echo "index file: $diskann_index_prefix" 
echo "query file: $query_bin_file"
echo "ground truth file: $ground_truth_bin_file"
echo "query label file: $predicate_txt_file"
echo "K:" $K
echo "L:" $L_list
echo "result save path: $diskann_result_path"
echo "threads: $threads"
../../code/DiskANN/build/apps/search_memory_index  --data_type float \
                                        --dist_fn $metric \
                                        --index_path_prefix $diskann_index_prefix \
                                        --query_file $query_bin_file \
                                        --gt_file $ground_truth_bin_file \
                                        --query_filters_file $predicate_txt_file \
                                        -K $K \
                                        -L $L_list \
                                        --result_path $diskann_result_path \
                                        --num_threads $threads \
                                        --query_size $query_size \
                                        2>&1 | tee logs/diskann_stitched.log



python ../tests/extract_results.py "logs/diskann_stitched.log" $dataset $attr_type $query_sel $M "diskann_stitched"
