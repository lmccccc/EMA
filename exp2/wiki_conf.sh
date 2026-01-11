
dataset_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/fvecs/wiki_15.4M.fvecs"
query_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_22.93_embedding.fvecs"
index_root="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated"
# index_root="/home/mocheng/temp_index/wiki15_4"
dataset_attr_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/fvecs/wiki_15.4M_attr.json"
ground_truth_collection_name=navix_uncorr_1_01_join_1_01
if [ "$wiki_neg_query" = "wiki_negcorr_1_01" ]; then

    # ---------------- wiki negcorr 1.01 -----------------

    query_predicate_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_1.01_predicate.json"
    ground_truth_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_1.01_gt.json"
elif [ "$wiki_neg_query" = "wiki_negcorr_5_10" ]; then
    # ----------------- wiki negcorr 5.10

    query_predicate_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_5.10_predicate.json"
    ground_truth_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_5.10_gt.json"

elif [ "$wiki_neg_query" = "wiki_negcorr_9_96" ]; then
    # ----------------- wiki negcorr 9.96

    query_predicate_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_9.96_predicate.json"
    ground_truth_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_9.96_gt.json"

elif [ "$wiki_neg_query" = "wiki_negcorr_15_02" ]; then
    # ----------------- wiki negcorr 15.02

    query_predicate_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_15.02_predicate.json"
    ground_truth_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_15.02_gt.json"

elif [ "$wiki_neg_query" = "wiki_negcorr_22_93" ]; then
    # ----------------- wiki negcorr 22.93

    query_predicate_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_22.93_predicate.json"
    ground_truth_file="/mnt/data/mocheng/dataset/navix_dataset/wiki_15.4M/neg_correlated/join_22.93_gt.json"

fi