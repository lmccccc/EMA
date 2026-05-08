"""Analyze FT false positive rate using built-in stats API."""
import sys, json, numpy as np
sys.path.insert(0, 'tests')

from hashann import HashANN

def read_fvecs(fname):
    with open(fname, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, d+1)[:, 1:]
    return data

label_root = '/mnt/data/mocheng/dataset/redcaps4m/label/arbi_0_1_random/'

params = {
    'M': 40, 'N': 4000000, 'dim': 512, 'ef_construction': 300,
    'metric': 'ip', 'K': 10, 'ft_bits': 128
}
hash_ann = HashANN()
hash_ann.init_params(params)
index = hash_ann.load_index(
    params, [0, 1],
    '/mnt/data/mocheng/dataset/redcaps4m/hashann/index/index_40_300_arbi_0_1_random_128',
    1, 'bfann'
)

queries = read_fvecs('/mnt/data/mocheng/dataset/redcaps4m/query.fvecs')
print(f"Queries: {queries.shape}")

index.set_num_threads(1)
index.set_ft_routing_flag(True)
index.set_ft_routing_min_deg(16)

for sel_name, sel_file in [('[0.1,9]', '1%'), ('[0.177,7]', '5%'), ('[0.75,2]', '60%')]:
    preds = json.load(open(label_root + f'predicate_arbi_0_1_{sel_name}.json'))
    gt = json.load(open(label_root + f'gt_arbi_0_1_{sel_name}.json'))
    
    print(f"\n{'='*60}")
    print(f"Selectivity: {sel_file} ({sel_name})")
    print(f"{'='*60}")
    
    for ef in [10, 50, 200]:
        index.set_ef(ef)
        index.set_ft_flag(True)
        index.reset_ft_stats()
        
        test_q = queries[:1000]
        test_pred = preds[:1000]
        ids, dists = index.hybrid_knn_query(test_q, test_pred, k=10)
        
        stats = index.get_ft_stats()
        ft_total = stats['ft_passed_total']
        ft_fp = stats['ft_false_positives']
        ft_tp = stats['ft_true_positives']
        fp_rate = stats['ft_fp_rate']
        
        # Compute recall
        correct = 0
        for qi in range(len(gt)):
            correct += len(set(gt[qi]) & set(int(x) for x in ids[qi]))
        recall = correct / (len(gt) * 10)
        
        print(f"  ef={ef:3d}: recall={recall:.4f}, ft_passed={ft_total}, ft_fp={ft_fp}, ft_tp={ft_tp}, FP_rate={fp_rate:.4f} ({fp_rate*100:.1f}%)")
        print(f"          avg ft_passed/query={ft_total/1000:.1f}, avg fp/query={ft_fp/1000:.1f}, avg tp/query={ft_tp/1000:.1f}")

