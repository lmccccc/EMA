import os
"""
Incremental insert workflow on sift10m.

Stages:
  - Stage 0: build index with first 5M base points.
  - Stage 1..5: add 1M more points each (=> 6M, 7M, 8M, 9M, 10M).
  - At every stage: compute filtered ground-truth on the current N-point base
    for predicate=[[9]] (10% selectivity, attr_type=[1]), then run
    hybrid_knn_query at multiple ef values and record (recall@10, qps).
  - Save the index after every stage.
  - All numbers + per-stage details written to a single JSON file under
    logs/incremental_sift10m_<ts>/.

This script is intentionally self-contained (no Django/airflow). It mirrors
exp3 conventions:
  - SIFT10M paths come from /mnt/data/mocheng/dataset/sift10m/.
  - attr file: arbi_1_random / 21 cardinalities; predicate=[[9]] ~10% sel.
  - Index params: M=40 ef_construction=300 ft_bits=128 edge_level_ft=True.
"""

import os, sys, time, json, gc, argparse, datetime
from pathlib import Path
import numpy as np

import hashannlib

try:
    import faiss
except ImportError:
    faiss = None


# ---------- Paths ----------
BASE_FVECS  = Path(os.environ.get("EMA_BASE_FVECS",  "/mnt/data/mocheng/dataset/sift10m/sift10m.fvecs"))
QUERY_FVECS = Path(os.environ.get("EMA_QUERY_FVECS", "/mnt/data/mocheng/dataset/sift10m/sift10m_query.fvecs"))
ATTR_JSON   = Path(os.environ.get("EMA_ATTR_JSON",   "/mnt/data/mocheng/dataset/sift10m/label/arbi_1_random/attr_arbi_1_random.json"))


# ---------- IO helpers ----------
def fvecs_read(path, max_n=None):
    a = np.fromfile(str(path), dtype=np.int32)
    d = int(a[0])
    rows = a.reshape(-1, d + 1)
    n = rows.shape[0] if max_n is None else min(max_n, rows.shape[0])
    return rows[:n, 1:].view(np.float32).copy()


def load_attrs_as_value_array(path, target_value):
    """attr file format: list of [[v1,v2,...]] per point (single cate attr).
    For 10%-sel predicate=[[9]] we only need to know which points contain 9.
    Returns a uint8 mask of length N (1 where attr contains target_value).
    Also returns the raw attrs (list-of-lists) needed by the index initAttrMapping
    + add_items.
    """
    print(f"[attr] loading {path}")
    t0 = time.time()
    with open(path, "r") as f:
        attrs_all = json.load(f)
    n = len(attrs_all)
    print(f"[attr] loaded {n} entries in {time.time()-t0:.1f}s")
    mask = np.zeros(n, dtype=np.uint8)
    t1 = time.time()
    for i, a in enumerate(attrs_all):
        # a == [[v1, v2, ...]]
        if target_value in a[0]:
            mask[i] = 1
    print(f"[attr] mask built in {time.time()-t1:.1f}s, "
          f"hits = {mask.sum()} ({mask.sum()/n*100:.2f}%)")
    return attrs_all, mask


# ---------- GT computation ----------
def compute_gt_filtered(base_n, queries, mask_n, k, gpu=False, batch=500):
    """Filtered brute-force GT: for each query, top-k nearest among
    base_n[mask_n==1]. Returns (Q, k) of original-id ground truth labels.

    Uses faiss IndexFlatL2 if available, else numpy. base_n shape: (N, D).
    """
    keep_idx = np.nonzero(mask_n)[0].astype(np.int64)
    sub = base_n[keep_idx]
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
    # numpy fallback
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


# ---------- Index build / add ----------
def build_initial_index(N_total, dim, attrs_all_for_mapping, M, ef_construction,
                        ft_bits, edge_level_ft, threads):
    """Init a fresh index sized for N_total points; populate FT codebook from
    attrs_all_for_mapping (must cover all values that will ever appear).
    Returns the bare index (no points yet)."""
    idx = hashannlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=N_total,
        top_elements=1,
        M=M,
        ef_construction=ef_construction,
        ft_bits=ft_bits,
        attr_type=[1],
        max_cate_size=21,           # arbi_1_random uses 0..20
        allow_replace_deleted=False,
        edge_level_ft=edge_level_ft,
    )
    idx.set_num_threads(threads)
    # Codebook must see the full value distribution -> use all attrs.
    print(f"[idx] initAttrMapping with {len(attrs_all_for_mapping)} attrs")
    t0 = time.time()
    idx.initAttrMapping(attrs_all_for_mapping)
    print(f"[idx] mapping done in {time.time()-t0:.1f}s")
    return idx


def add_chunk(idx, base, attrs, ids_list, chunk=100000, threads=32, levels=None):
    """Insert base[ids_list] (must be a contiguous slice in our usage) into idx.
    levels: optional np.int32 array same length as base; uses 0 for the slice if None."""
    n = len(ids_list)
    print(f"[add] inserting {n} points, chunk={chunk}, threads={threads}")
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sub_ids = ids_list[s:e]
        first, last = sub_ids[0], sub_ids[-1] + 1
        assert sub_ids == list(range(first, last)), "non-contiguous chunk"
        sub_levels = (levels[first:last].tolist() if levels is not None
                      else [0] * (last - first))
        t0 = time.time()
        idx.add_items(
            base[first:last],
            attrs[first:last],
            ids=sub_ids,
            levels=sub_levels,
            num_threads=threads,
        )
        print(f"  [add] {first}..{last} done in {time.time()-t0:.1f}s")


# ---------- Query benchmark ----------
def run_query_sweep(idx, queries, predicate_per_q, gt, k, ef_list, num_threads=1, repeats=1):
    """For each ef in ef_list, run hybrid_knn_query and record recall@k + qps."""
    results = []
    for ef in ef_list:
        idx.set_ef(int(ef))
        # warm up once
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


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output dir for results + indexes")
    ap.add_argument("--initial_n", type=int, default=5_000_000)
    ap.add_argument("--final_n",   type=int, default=10_000_000)
    ap.add_argument("--step_n",    type=int, default=1_000_000)
    ap.add_argument("--M",         type=int, default=40)
    ap.add_argument("--ef_construction", type=int, default=300)
    ap.add_argument("--ft_bits",   type=int, default=128)
    ap.add_argument("--edge_level_ft", type=int, default=1)
    ap.add_argument("--ef_search_list", type=str, default="10,20,40,80,160")
    ap.add_argument("--threads_build", type=int, default=32)
    ap.add_argument("--threads_query", type=int, default=1)
    ap.add_argument("--query_repeats", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--predicate_value", type=int, default=9)
    ap.add_argument("--n_query", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] out_dir = {out_dir}")
    print(f"[main] args = {vars(args)}")

    ef_list = [int(x) for x in args.ef_search_list.split(",")]

    # ---- Load data ----
    base = fvecs_read(BASE_FVECS, max_n=args.final_n)
    queries = fvecs_read(QUERY_FVECS, max_n=args.n_query)
    print(f"[main] base = {base.shape}, queries = {queries.shape}")
    assert base.shape[0] >= args.final_n, "base shorter than --final_n"

    attrs_all, mask_all = load_attrs_as_value_array(ATTR_JSON, args.predicate_value)
    attrs_all = attrs_all[:args.final_n]
    mask_all  = mask_all[:args.final_n]
    print(f"[main] attrs sliced to {len(attrs_all)}, hits = {int(mask_all.sum())}")

    # Pre-generate levels for the full set (Bernoulli(1/M) -> level 1, else 0)
    rng_lvl = np.random.RandomState(42)
    p_upper = 1.0 / args.M
    levels_all = (rng_lvl.rand(args.final_n) < p_upper).astype(np.int32)
    print(f"[main] levels: {int(levels_all.sum())} at level 1, "
          f"{args.final_n - int(levels_all.sum())} at level 0")

    # Single predicate per query: [[predicate_value]]
    predicate_per_q = [[[args.predicate_value]] for _ in range(args.n_query)]

    # ---- Build initial index ----
    dim = base.shape[1]
    idx = build_initial_index(
        N_total=args.final_n, dim=dim,
        attrs_all_for_mapping=attrs_all,  # cover full value range
        M=args.M, ef_construction=args.ef_construction,
        ft_bits=args.ft_bits, edge_level_ft=bool(args.edge_level_ft),
        threads=args.threads_build,
    )

    all_results = {
        "config": {
            "dataset": "sift10m",
            "predicate": [[args.predicate_value]],
            "K": args.K,
            "M": args.M,
            "ef_construction": args.ef_construction,
            "ft_bits": args.ft_bits,
            "edge_level_ft": bool(args.edge_level_ft),
            "ef_search_list": ef_list,
            "n_query": args.n_query,
            "threads_build": args.threads_build,
            "threads_query": args.threads_query,
            "query_repeats": args.query_repeats,
            "initial_n": args.initial_n,
            "final_n": args.final_n,
            "step_n": args.step_n,
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "stages": [],
    }

    def save_all_results():
        p = out_dir / "all_results.json"
        with open(p, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[main] saved {p}")

    # ---- Stage loop ----
    cur_n = 0
    stage_targets = list(range(args.initial_n, args.final_n + 1, args.step_n))
    if stage_targets[-1] != args.final_n:
        stage_targets.append(args.final_n)

    for stage_idx, target_n in enumerate(stage_targets):
        print(f"\n========== Stage {stage_idx}: grow to {target_n} ==========")
        ids_to_add = list(range(cur_n, target_n))
        t_build0 = time.time()
        add_chunk(idx, base, attrs_all, ids_to_add,
                  chunk=100_000, threads=args.threads_build,
                  levels=levels_all)
        t_build = time.time() - t_build0
        cur_n = target_n
        print(f"[stage] grow time = {t_build:.1f}s, cur_n = {cur_n}")

        # GT for current scale
        t_gt0 = time.time()
        gt = compute_gt_filtered(base[:cur_n], queries, mask_all[:cur_n], args.K)
        t_gt = time.time() - t_gt0
        gt_subset_size = int(mask_all[:cur_n].sum())
        sel = gt_subset_size / cur_n
        print(f"[stage] gt time = {t_gt:.1f}s, subset = {gt_subset_size} "
              f"({sel*100:.2f}% selectivity)")

        # Save GT (so re-running queries doesn't repeat the work)
        gt_path = out_dir / f"gt_N{cur_n}.npy"
        np.save(gt_path, gt)

        # Query sweep
        print(f"[stage] running query sweep at cur_n = {cur_n}")
        q_results = run_query_sweep(
            idx, queries, predicate_per_q, gt, args.K, ef_list,
            num_threads=args.threads_query, repeats=args.query_repeats,
        )

        stage_record = {
            "stage_idx": stage_idx,
            "cur_n": cur_n,
            "grow_time_s": float(t_build),
            "gt_time_s": float(t_gt),
            "gt_subset_size": gt_subset_size,
            "selectivity": float(sel),
            "ef_results": q_results,
        }

        # Per user spec: save index only at the final stage.
        is_final = (stage_idx == len(stage_targets) - 1)
        if is_final:
            idx_path = out_dir / (
                f"sift10m_{cur_n//1_000_000}M_M{args.M}_ef{args.ef_construction}"
                f"_ftb{args.ft_bits}_edge{int(args.edge_level_ft)}.index"
            )
            t_save0 = time.time()
            idx.save_index(str(idx_path))
            stage_record["index_path"] = str(idx_path)
            stage_record["index_save_s"] = float(time.time() - t_save0)
            stage_record["index_size_bytes"] = int(idx_path.stat().st_size)
            print(f"[stage] saved final index -> {idx_path} "
                  f"({stage_record['index_size_bytes']/1e9:.2f} GB)")

        all_results["stages"].append(stage_record)
        save_all_results()  # update after every stage

    all_results["config"]["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_all_results()

    # Compact human-readable summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Incremental sift10m workflow\n")
        f.write(f"Config: {json.dumps(all_results['config'], indent=2)}\n\n")
        for s in all_results["stages"]:
            f.write(f"\n--- N={s['cur_n']} (grow {s['grow_time_s']:.1f}s, "
                    f"gt {s['gt_time_s']:.1f}s, "
                    f"sel {s['selectivity']*100:.2f}%, "
                    f"subset {s['gt_subset_size']}) ---\n")
            for r in s["ef_results"]:
                f.write(f"  ef={r['ef']:>5d}  recall@{args.K}={r['recall']:.4f}  qps={r['qps']:.1f}\n")
    print(f"[main] wrote {summary_path}")
    print("[main] DONE")


if __name__ == "__main__":
    main()
