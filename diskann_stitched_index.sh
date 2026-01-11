#! /bin/bash

source ./conf.sh
source ./diskann_stitched_conf.sh


        
echo "dataset file: $dataset_bin_file" 
echo "index path prefix: $diskann_index_prefix"
echo "label file: $label_file"
../code/DiskANN/build/apps/build_stitched_index  --data_type float \
                                        --data_path $dataset_bin_file \
                                        --index_path_prefix $diskann_index_prefix \
                                        -R $M \
                                        --alpha $alpha \
                                        --Lbuild $ef_construction \
                                        --label_file $label_file \
                                        --num_threads $threads \
                                        --stitched_R $Stitched_R \
                                        --dist_fn $metric \
                                        2>&1 | tee -a logs/diskann_stitched_construction.log

echo "${diskann_index_prefix} construction done" >> logs/diskann_stitched_construction.log
echo "\n" >> logs/diskann_stitched_construction.log