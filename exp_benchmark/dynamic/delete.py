import os
"""
Deletion workflow on sift10m.

Loads the 10M index built by incremental_sift10m.py, then in 5 stages
mark_deletes 1M random ids each (cumulative 1M, 2M, 3M, 4M, 5M deleted).
At every stage:
  - rebuild filtered mask = (attr matches predicate) AND (id NOT deleted)
  - recompute filtered ground truth
  - run hybrid_knn_query sweep at multiple ef values
No graph rebuild / repair is invoked.

Mirrors incremental_sift10m.py conventions.
"""

import os, sys, time, json, gc, argparse, datetime
from pathlib import Path
import numpy as np

import hashannlib

try:
    import faiss
except ImportError:
    faiss = None


BASE_FVECS  = Path(os.environ.get("EMA_BASE_FVECS",  "/mnt/data/mocheng/dataset/sift10m/sift10m.fvecs"))
QUERY_FVECS = Path(os.environ.get("EMA_QUERY_FVECS", "/mnt/data/mocheng/dataset/sift10m/sift10m_query.fvecs"))
ATTR_JSON   = Path(os.environ.get("EMA_ATTR_JSON",   "/mnt/data/mocheng/dataset/sift10m/label/arbi_1_random/attr_arbi_1_random.json"))


def fvecs_read(path, max_n=None):
    a = np.fromfile(str(path), dtype=np.int32)
    d = int(a[0])
    rows = a.reshape(-1, d + 1)
    n = rows.shape[0] if max_n is None else min(max_n, rows.shape[0])
    return rows[:n, 1:].view(np.float32).copy()


def load_attrs_as_value_array(path, target_value):
    print(f"[attr] loading {path}")
    t0 = time.time()
    with open(path) as f:
        attrs_all = json.load(f)
    n = len(attrs_all)
    print(f"[attr] loaded {n} entries in {time.time()-t0:.1f}s")
    mask = np.zeros(n, dtype=np.uint8)
    t1 = time.time()
    for i, a in enumerate(attrs_all):
        if target_value in a[0]:
            mask[i] = 1
    print(f"[attr] base mask built in {time.time()-t1:.1f}s, "
          f"hits = {mask.sum()} ({mask.sum()/n*100:.2f}%)")
    return attrs_all, mask


def compute_gt_filtered(base, queries, mask, k, batch=500):
    keep_idx = np.nonzero(mask)[0].astype(np.int64)
    sub = base[keep_idx]
    print(f"[gt] subset {sub.shape}, queries {queries.shape}")
    if faiss is not None:
        index = faiss.IndexFlatL2(sub.shape[1])
        index.add(sub)
        Q = queries.shape[0]
        out = np.empty((Q, k), dtype=np.int64)
        t0 = time.time()
        for s in range(0, Q, batch):
            e = min(s + batch, Q)
            _, I = index.search(queries[s:e], k)
            out[s:e] = keep_idx[I]
        print(f"[gt] faiss search done in {time.time()-t0:.1f}s")
        return out
    Q = queries.shape[0]
    out = np.empty((Q, k), dtype=np.int64)
    for i in range(Q):
        d = np.sum((sub - queries[i]) ** 2, axis=1)
        topk_local = np.argpartition(d, k)[:k]
        topk_local = topk_local[np.argsort(d[topk_local])]
        out[i] = keep_idx[topk_local]
    return out


def recall_at_k(pred_ids, gt_ids, k):
    Q = pred_ids.shape[0]
    hits = 0
    for i in range(Q):
        hits += len(set(int(x) for x in pred_ids[i, :k]) & set(int(x) for x in gt_ids[i, :k]))
    return hits / (Q * k)


def run_query_sweep(idx, queries, predicate_per_q, gt, k, ef_list,
                    num_threads=1, repeats=3):
    results = []
    for ef in ef_list:
        idx.set_ef(int(ef))
        idx.hybrid_knn_query(queries[:min(50, len(queries))],
                             predicate_per_q[:min(50, len(queries))],
                             k=k, num_threads=num_threads)
        timings = []
        last_pred = None
        for _ in range(repeats):
            t0 = time.time()
            pred_ids, _ = idx.hybrid_knn_query(queries, predicate_per_q,
                                               k=k, num_threads=num_threads)
            timings.append(time.time() - t0)
            last_pred = pred_ids
        elapsed = min(timings)
        Q = queries.shape[0]
        qps = Q / elapsed
        r = recall_at_k(last_pred, gt, k)
        print(f"  [query] ef={ef:5d}  recall@{k}={r:.4f}  qps={qps:.1f}  "
              f"elapsed={elapsed:.2f}s (best of {repeats})")
        results.append({"ef": int(ef), "recall": float(r),
                        "qps": float(qps), "elapsed_s": float(elapsed),
                        "queries": int(Q), "threads": int(num_threads)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",    type=str, required=True)
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--N",          type=int, default=10_000_000)
    ap.add_argument("--delete_step", type=int, default=1_000_000)
    ap.add_argument("--n_delete_stages", type=int, default=5)
    ap.add_argument("--ef_search_list", type=str,
                    default="40,60,70,80,90,100,110,120,140,160,200,260")
    ap.add_argument("--threads_query", type=int, default=1)
    ap.add_argument("--query_repeats", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--predicate_value", type=int, default=9)
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--include_stage0", type=int, default=1,
                    help="Also run a stage 0 with no deletions for reference.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] out_dir = {out_dir}")
    print(f"[main] args = {vars(args)}")

    base    = fvecs_read(BASE_FVECS, max_n=args.N)
    queries = fvecs_read(QUERY_FVECS, max_n=args.n_query)
    print(f"[main] base = {base.shape}, queries = {queries.shape}")
    dim = base.shape[1]

    attrs_all, base_mask = load_attrs_as_value_array(
        str(ATTR_JSON), args.predicate_value)
    assert len(attrs_all) >= args.N
    base_mask = base_mask[:args.N]

    predicate_per_q = [[[args.predicate_value]]] * args.n_query

    ef_list = [int(x) for x in args.ef_search_list.split(",") if x]

    # Load index
    idx = hashannlib.Index(space="l2", dim=dim)
    print(f"[idx] loading {args.index_path}")
    t0 = time.time()
    idx.load_index(args.index_path, max_elements=args.N, top_elements=1,
                   allow_replace_deleted=False, dynamic=True)
    print(f"[idx] loaded in {time.time()-t0:.1f}s, "
          f"current_count={idx.get_current_count()}")

    # Random deletion order, fixed seed
    rng = np.random.default_rng(args.seed)
    delete_order = rng.permutation(args.N).astype(np.int64)

    deleted_mask = np.zeros(args.N, dtype=bool)

    all_stages = []
    stages_to_run = list(range(0 if args.include_stage0 else 1,
                               args.n_delete_stages + 1))

    for stage in stages_to_run:
        n_deleted = stage * args.delete_step
        # delete the next chunk of ids relative to previous stage
        prev_deleted = int(deleted_mask.sum())
        if n_deleted > prev_deleted:
            chunk = delete_order[prev_deleted:n_deleted]
            t0 = time.time()
            for lid in chunk:
                idx.mark_deleted(int(lid))
            deleted_mask[chunk] = True
            print(f"[stage {stage}] deleted {len(chunk)} ids in {time.time()-t0:.1f}s "
                  f"(total deleted = {n_deleted}, remaining = {args.N - n_deleted})")
        else:
            print(f"[stage 0] no deletions; baseline run on full {args.N} index")

        # Rebuild filtered GT mask: attr matches AND not deleted
        eff_mask = base_mask.copy()
        eff_mask[deleted_mask] = 0
        sel = eff_mask.sum() / args.N
        print(f"[stage {stage}] filtered subset = {int(eff_mask.sum())} "
              f"({sel*100:.2f}% of {args.N})")

        t0 = time.time()
        gt = compute_gt_filtered(base, queries, eff_mask, args.K)
        gt_time = time.time() - t0
        np.save(out_dir / f"gt_del{n_deleted}.npy", gt)

        print(f"[stage {stage}] running query sweep")
        runs = run_query_sweep(idx, queries, predicate_per_q, gt, args.K,
                               ef_list, num_threads=args.threads_query,
                               repeats=args.query_repeats)

        all_stages.append({
            "stage": stage,
            "n_deleted": int(n_deleted),
            "n_remaining": int(args.N - n_deleted),
            "filtered_subset": int(eff_mask.sum()),
            "selectivity": float(sel),
            "gt_time_s": float(gt_time),
            "ef_results": runs,
        })

        # Save intermediate
        with open(out_dir / "all_results.json", "w") as f:
            json.dump({"config": vars(args), "stages": all_stages}, f, indent=2)

    # Summary
    lines = [f"# delete_sift10m  N={args.N}  predicate=[[{args.predicate_value}]]\n"]
    for s in all_stages:
        lines.append(f"\n## stage {s['stage']}  deleted={s['n_deleted']}  "
                     f"remaining={s['n_remaining']}  "
                     f"filtered_subset={s['filtered_subset']} "
                     f"({s['selectivity']*100:.2f}%)\n")
        lines.append(f"{'ef':>6} {'recall@'+str(args.K):>10} {'qps':>10}\n")
        for r in s['ef_results']:
            lines.append(f"{r['ef']:>6} {r['recall']:>10.4f} {r['qps']:>10.1f}\n")
    with open(out_dir / "summary.txt", "w") as f:
        f.writelines(lines)
    print(f"[main] wrote {out_dir/'summary.txt'}")
    print(f"[main] DONE")


if __name__ == "__main__":
    main()
