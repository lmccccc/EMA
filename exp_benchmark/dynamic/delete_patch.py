import os
"""
Delete + FreshDiskANN-style patching benchmark on sift10m.

Workflow:
  Load 10M index.
  For stage in 1..5 (each = 1M random deletes):
    - mark_deleted current chunk
    - recompute filtered GT
    - run query sweep "deleted only" (this also populates dirty_bitmap_)
    - if ratio in [20%, 50%): batched_patch_deletes → sweep "after_patch"
    - if ratio >= 50%:        rebuildGraph (parallel from-scratch) → sweep "after_rebuild"
"""

import os, sys, time, json, argparse
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


def load_attrs(path):
    print(f"[attr] loading {path}")
    t0 = time.time()
    with open(path) as f:
        attrs_all = json.load(f)
    print(f"[attr] loaded {len(attrs_all)} entries in {time.time()-t0:.1f}s")
    return attrs_all


def build_attr_mask(attrs_all, target_value, n):
    mask = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if target_value in attrs_all[i][0]:
            mask[i] = 1
    return mask


def compute_gt_filtered(base, queries, mask, k, batch=500):
    keep_idx = np.nonzero(mask)[0].astype(np.int64)
    sub = base[keep_idx]
    print(f"[gt] subset {sub.shape}, queries {queries.shape}")
    index = faiss.IndexFlatL2(sub.shape[1])
    index.add(sub)
    Q = queries.shape[0]
    out = np.empty((Q, k), dtype=np.int64)
    t0 = time.time()
    for s in range(0, Q, batch):
        e = min(s + batch, Q)
        _, I = index.search(queries[s:e], k)
        out[s:e] = keep_idx[I]
    print(f"[gt] faiss done in {time.time()-t0:.1f}s")
    return out


def recall_at_k(pred_ids, gt_ids, k):
    Q = pred_ids.shape[0]
    hits = 0
    for i in range(Q):
        hits += len(set(int(x) for x in pred_ids[i, :k]) & set(int(x) for x in gt_ids[i, :k]))
    return hits / (Q * k)


def run_query_sweep(idx, queries, predicate_per_q, gt, k, ef_list,
                    num_threads=1, repeats=3, tag=""):
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
        # Count deleted hits as sanity check
        print(f"  [{tag}] ef={ef:5d}  recall@{k}={r:.4f}  qps={qps:.1f}  "
              f"elapsed={elapsed:.2f}s")
        results.append({"ef": int(ef), "recall": float(r), "qps": float(qps),
                        "elapsed_s": float(elapsed), "queries": int(Q),
                        "threads": int(num_threads)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",    type=str, required=True)
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--N",          type=int, default=10_000_000)
    ap.add_argument("--delete_step", type=int, default=1_000_000)
    ap.add_argument("--n_delete_stages", type=int, default=5)
    ap.add_argument("--patch_at", type=str, default="2,3,4",
                    help="Stage indices to run batched_patch_deletes (after sweep)")
    ap.add_argument("--rebuild_at", type=str, default="5",
                    help="Stage indices to run full rebuild")
    ap.add_argument("--ef_search_list", type=str,
                    default="40,60,70,80,90,100,110,120,140,160")
    ap.add_argument("--threads_patch", type=int, default=32)
    ap.add_argument("--threads_query", type=int, default=1)
    ap.add_argument("--query_repeats", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--predicate_value", type=int, default=9)
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] out_dir = {out_dir}")
    print(f"[main] args = {vars(args)}")

    patch_stages = set(int(x) for x in args.patch_at.split(",") if x.strip())
    rebuild_stages = set(int(x) for x in args.rebuild_at.split(",") if x.strip())

    base    = fvecs_read(BASE_FVECS, max_n=args.N)
    queries = fvecs_read(QUERY_FVECS, max_n=args.n_query)
    print(f"[main] base = {base.shape}, queries = {queries.shape}")
    dim = base.shape[1]

    attrs_all = load_attrs(str(ATTR_JSON))
    base_mask = build_attr_mask(attrs_all, args.predicate_value, args.N)
    print(f"[main] base mask hits = {int(base_mask.sum())} "
          f"({base_mask.sum()/args.N*100:.2f}%)")
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

    rng = np.random.default_rng(args.seed)
    delete_order = rng.permutation(args.N).astype(np.int64)
    deleted_mask = np.zeros(args.N, dtype=bool)

    all_records = []

    for stage in range(args.n_delete_stages + 1):
        n_deleted = stage * args.delete_step

        if stage > 0:
            prev = int(deleted_mask.sum())
            chunk = delete_order[prev:n_deleted]
            t0 = time.time()
            for lid in chunk:
                idx.mark_deleted(int(lid))
            deleted_mask[chunk] = True
            print(f"[stage {stage}] deleted {len(chunk)} ids in "
                  f"{time.time()-t0:.1f}s (total {n_deleted}, "
                  f"ratio={idx.get_deleted_ratio()*100:.1f}%)")

        eff_mask = base_mask.copy()
        eff_mask[deleted_mask] = 0
        print(f"[stage {stage}] filtered subset = {int(eff_mask.sum())} "
              f"({eff_mask.sum()/args.N*100:.2f}%)")

        t0 = time.time()
        gt = compute_gt_filtered(base, queries, eff_mask, args.K)
        gt_time = time.time() - t0
        np.save(out_dir / f"gt_del{n_deleted}.npy", gt)

        # Measurement BEFORE maintenance
        print(f"[stage {stage}] query sweep (deleted only)")
        runs_before = run_query_sweep(idx, queries, predicate_per_q, gt, args.K,
                                      ef_list, num_threads=args.threads_query,
                                      repeats=args.query_repeats,
                                      tag=f"s{stage}/del")
        print(f"[stage {stage}] dirty_count after sweep = {idx.get_dirty_count()}")

        record = {
            "stage": stage,
            "n_deleted": int(n_deleted),
            "n_remaining": int(args.N - n_deleted),
            "filtered_subset": int(eff_mask.sum()),
            "deleted_ratio": float(idx.get_deleted_ratio()),
            "dirty_count_before_maint": int(idx.get_dirty_count()),
            "gt_time_s": float(gt_time),
            "before_maint": runs_before,
        }

        # Batched patching (cheap maintenance)
        if stage in patch_stages and n_deleted > 0:
            print(f"[stage {stage}] *** batched_patch_deletes ***")
            t0 = time.time()
            n_patched = idx.batched_patch_deletes(num_threads=args.threads_patch)
            patch_time = time.time() - t0
            print(f"[patch] patched {n_patched} nodes in {patch_time:.1f}s")
            record["patch_time_s"] = float(patch_time)
            record["patched_count"] = int(n_patched)

            print(f"[stage {stage}] query sweep (after patch)")
            runs_after_patch = run_query_sweep(idx, queries, predicate_per_q,
                                               gt, args.K, ef_list,
                                               num_threads=args.threads_query,
                                               repeats=args.query_repeats,
                                               tag=f"s{stage}/pt")
            record["after_patch"] = runs_after_patch

        # Full rebuild (heavy maintenance) — from-scratch parallel rebuild
        # mirroring delete_rebuild_sift10m.py since C++ rebuild_graph is serial
        if stage in rebuild_stages and n_deleted > 0:
            print(f"[stage {stage}] *** from-scratch parallel rebuild ***")
            keep_ids = np.where(~deleted_mask)[0].astype(np.int64)
            n_keep = len(keep_ids)
            print(f"[rebuild] surviving = {n_keep}")
            del idx
            import gc; gc.collect()
            t0 = time.time()
            new_idx = hashannlib.Index(space="l2", dim=dim)
            new_idx.init_index(
                max_elements=n_keep, top_elements=1, M=40,
                ef_construction=300, ft_bits=128,
                attr_type=[1], max_cate_size=21,
                allow_replace_deleted=False, edge_level_ft=True,
            )
            new_idx.set_num_threads(args.threads_patch)
            new_idx.initAttrMapping(attrs_all)
            keep_attrs = [attrs_all[int(i)] for i in keep_ids]
            keep_levels_rng = np.random.default_rng(args.seed + stage)
            keep_levels = keep_levels_rng.geometric(p=1.0/40, size=n_keep).astype(np.int32) - 1
            keep_levels = np.clip(keep_levels, 0, 5)
            chunk = 100_000
            for s in range(0, n_keep, chunk):
                e = min(s + chunk, n_keep)
                new_idx.add_items(
                    base[keep_ids[s:e]],
                    keep_attrs[s:e],
                    ids=keep_ids[s:e].tolist(),
                    levels=keep_levels[s:e].tolist(),
                    num_threads=args.threads_patch,
                )
                if s % 1_000_000 == 0:
                    print(f"[rebuild] added {e}/{n_keep} elapsed={time.time()-t0:.1f}s")
            rebuild_time = time.time() - t0
            print(f"[rebuild] DONE in {rebuild_time:.1f}s, count={new_idx.get_current_count()}")
            record["rebuild_time_s"] = float(rebuild_time)
            record["rebuilt_count"] = int(new_idx.get_current_count())
            idx = new_idx
            print(f"[stage {stage}] query sweep (after rebuild)")
            runs_after_rb = run_query_sweep(idx, queries, predicate_per_q,
                                            gt, args.K, ef_list,
                                            num_threads=args.threads_query,
                                            repeats=args.query_repeats,
                                            tag=f"s{stage}/rb")
            record["after_rebuild"] = runs_after_rb

        all_records.append(record)
        with open(out_dir / "all_results.json", "w") as f:
            json.dump({"config": vars(args), "stages": all_records}, f, indent=2)

    # Summary
    lines = [f"# delete+patch sift10m  N={args.N}  predicate=[[{args.predicate_value}]]\n"]
    for r in all_records:
        lines.append(f"\n## stage {r['stage']}  deleted={r['n_deleted']}  "
                     f"remaining={r['n_remaining']}  filtered={r['filtered_subset']}\n")
        lines.append("### before maintenance\n")
        lines.append(f"{'ef':>6} {'recall':>10} {'qps':>10}\n")
        for row in r['before_maint']:
            lines.append(f"{row['ef']:>6} {row['recall']:>10.4f} {row['qps']:>10.1f}\n")
        if 'after_patch' in r:
            lines.append(f"### after patch  (time={r['patch_time_s']:.1f}s, "
                         f"patched={r['patched_count']})\n")
            lines.append(f"{'ef':>6} {'recall':>10} {'qps':>10}\n")
            for row in r['after_patch']:
                lines.append(f"{row['ef']:>6} {row['recall']:>10.4f} {row['qps']:>10.1f}\n")
        if 'after_rebuild' in r:
            lines.append(f"### after rebuild  (time={r['rebuild_time_s']:.1f}s)\n")
            lines.append(f"{'ef':>6} {'recall':>10} {'qps':>10}\n")
            for row in r['after_rebuild']:
                lines.append(f"{row['ef']:>6} {row['recall']:>10.4f} {row['qps']:>10.1f}\n")
    with open(out_dir / "summary.txt", "w") as f:
        f.writelines(lines)
    print(f"[main] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
