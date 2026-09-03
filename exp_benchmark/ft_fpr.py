"""Measure EMA / FT false-positive rate at several selectivities.

Loads a pre-built HashANN index and counts, per query, how many nodes were
admitted by the FT (Edge Marker) check but then failed the actual predicate.
The C++ side maintains atomic counters (`metric_ft_passed_total`,
`metric_ft_false_positives`); we expose them via `get_ft_stats()` /
`reset_ft_stats()`.

Reproducible recipe — see exp_benchmark/README.md.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np

# tests/ contains hashann.py thin wrapper around hashannlib.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
from hashann import HashANN  # noqa: E402


def fvecs_read(path, max_n=None):
    with open(path, "rb") as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, d + 1)[:, 1:]
    return data if max_n is None else data[:max_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Redcaps_4M",
                    help="Used only for default paths (see env vars).")
    ap.add_argument("--data_root", default=os.environ.get(
        "DATA_ROOT", "/mnt/data/mocheng/dataset"))
    ap.add_argument("--index_path", default="",
                    help="Override the index file path. If empty, derived "
                         "from dataset/M/ef_construction/ft_bits.")
    ap.add_argument("--label_subdir", default="label/arbi_0_1_random",
                    help="Predicate/GT directory under data_root/<dataset>/.")
    ap.add_argument("--query_file", default="",
                    help="fvecs query file (default: data_root/<dataset>/query.fvecs).")
    ap.add_argument("--attr_type", default="0,1",
                    help="Comma-separated attribute indices the index was "
                         "built with (e.g. '0,1').")
    ap.add_argument("--M", type=int, default=40)
    ap.add_argument("--ef_construction", type=int, default=300)
    ap.add_argument("--ft_bits", type=int, default=128)
    ap.add_argument("--N", type=int, default=4_000_000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--ef_search_list", default="10,50,200")
    ap.add_argument("--selectivities", default="[0.1,9]:1%;[0.177,7]:5%;[0.75,2]:60%",
                    help="Semicolon-separated predicate_str:label pairs.")
    ap.add_argument("--ft_routing_min_deg", type=int, default=16)
    ap.add_argument("--out_json", default="",
                    help="If set, dump all measurements to this JSON file.")
    args = ap.parse_args()

    # ---- resolve paths -------------------------------------------------------
    ds_root = Path(args.data_root) / {
        "Redcaps_4M": "redcaps4m",
        "sift10m": "sift10m",
        "wiki_15_4M": "navix_dataset/wiki_15.4M",
        "youtube_rgb": "youtube1m",
    }.get(args.dataset, args.dataset)
    label_root = ds_root / args.label_subdir
    attr_tag = "arbi_" + "_".join(args.attr_type.split(",")) + "_random"

    if args.index_path:
        index_path = args.index_path
    else:
        index_path = str(
            ds_root / "hashann" / "index"
            / f"index_{args.M}_{args.ef_construction}_{attr_tag}_{args.ft_bits}"
        )
    query_file = args.query_file or str(ds_root / "query.fvecs")
    print(f"[FPR] dataset       = {args.dataset}")
    print(f"[FPR] index_path    = {index_path}")
    print(f"[FPR] label_root    = {label_root}")
    print(f"[FPR] query_file    = {query_file}")
    assert Path(index_path).is_file(), f"index file missing: {index_path}"

    attr_idx_list = [int(x) for x in args.attr_type.split(",")]

    params = {
        "M": args.M, "N": args.N, "dim": args.dim,
        "ef_construction": args.ef_construction,
        "metric": args.metric, "K": args.K, "ft_bits": args.ft_bits,
    }
    hash_ann = HashANN()
    hash_ann.init_params(params)
    index = hash_ann.load_index(params, attr_idx_list, index_path, 1, "bfann")

    queries = fvecs_read(query_file, max_n=args.n_query)
    print(f"[FPR] queries       = {queries.shape}")

    index.set_num_threads(1)
    index.set_ft_routing_flag(True)
    index.set_ft_routing_min_deg(args.ft_routing_min_deg)

    sel_pairs = []
    for token in args.selectivities.split(";"):
        if ":" not in token:
            raise ValueError(f"--selectivities token '{token}' must be 'predicate:label'")
        pred, label = token.rsplit(":", 1)
        sel_pairs.append((pred.strip(), label.strip()))
    ef_list = [int(x) for x in args.ef_search_list.split(",") if x]

    all_results = []
    for predicate_str, sel_label in sel_pairs:
        pred_file = label_root / f"predicate_arbi_0_1_{predicate_str}.json"
        gt_file = label_root / f"gt_arbi_0_1_{predicate_str}.json"
        preds = json.load(open(pred_file))
        gt = json.load(open(gt_file))

        print(f"\n{'='*64}")
        print(f"Selectivity {sel_label}  (predicate={predicate_str})")
        print(f"  pred_file = {pred_file}")
        print(f"  gt_file   = {gt_file}")
        print(f"{'='*64}")

        test_q = queries[:args.n_query]
        test_pred = preds[:args.n_query]
        for ef in ef_list:
            index.set_ef(ef)
            index.set_ft_flag(True)
            index.reset_ft_stats()
            ids, _ = index.hybrid_knn_query(test_q, test_pred, k=args.K)
            stats = index.get_ft_stats()
            ft_total = int(stats["ft_passed_total"])
            ft_fp = int(stats["ft_false_positives"])
            ft_tp = int(stats["ft_true_positives"])
            fp_rate = float(stats["ft_fp_rate"])
            Q = len(gt)
            correct = sum(
                len(set(gt[qi]) & set(int(x) for x in ids[qi]))
                for qi in range(Q)
            )
            recall = correct / (Q * args.K)
            print(f"  ef={ef:3d}: recall={recall:.4f}  "
                  f"ft_passed={ft_total}  ft_fp={ft_fp}  ft_tp={ft_tp}  "
                  f"FP_rate={fp_rate:.4f} ({fp_rate*100:.1f}%)")
            print(f"          avg ft_passed/query={ft_total/Q:.1f}  "
                  f"avg fp/q={ft_fp/Q:.1f}  avg tp/q={ft_tp/Q:.1f}")
            all_results.append({
                "selectivity_label": sel_label,
                "predicate_str": predicate_str,
                "ef": ef, "recall": recall,
                "ft_passed_total": ft_total,
                "ft_false_positives": ft_fp,
                "ft_true_positives": ft_tp,
                "ft_fp_rate": fp_rate,
                "queries": Q,
            })

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)
        print(f"\n[FPR] wrote {args.out_json}")


if __name__ == "__main__":
    main()
