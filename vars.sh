attr_type="[0,1]"  # 0 for numerical, 1 for categorical


dataset="siftsmall"
N=10000
query_size=100
nsw_root="/mnt/data/mocheng/dataset/siftsmall/nsw_filter/"
dataset_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_base.fvecs"
query_file="/mnt/data/mocheng/dataset/siftsmall/siftsmall_query.fvecs"
dataset_attr_file="/mnt/data/mocheng/dataset/siftsmall/label/arbi_0_1_random/attr_arbi_0_1_random.json"
query_predicate_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/qrangesel_1_1_100000_random.json"
ground_truth_file="/mnt/data/mocheng/dataset/siftsmall/label/sel_1_100000_random/sif_gt_sel_1_1_100000_random_10.json"




python attr_generator.py --output_file ${dataset_attr_file} --N ${N} --attr_type_list ${attr_type} --categorical_attr_max_cardinality 5 --numerical_max_attr 100000