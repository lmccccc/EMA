import os
"""
Update experiment shared helpers.

Common dataset loading, GT computation, query sweep, and update planning.
"""
import json, time
from pathlib import Path
import numpy as np
import faiss


BASE_FVECS  = Path(os.environ.get("EMA_BASE_FVECS",  "/mnt/data/mocheng/dataset/sift10m/sift10m.fvecs"))
QUERY_FVECS = Path(os.environ.get("EMA_QUERY_FVECS", "/mnt/data/mocheng/dataset/sift10m/sift10m_query.fvecs"))
ATTR_JSON   = Path(os.environ.get("EMA_ATTR_JSON",   "/mnt/data/mocheng/dataset/sift10m/label/arbi_1_random/attr_arbi_1_random.json"))


def fvecs_read(path, max_n=None):
    a = np.fromfile(str(path), dtype=np.int32)
    d = int(a[0])
    rows = a.reshape(-1, d + 1)
    n = rows.shape[0] if max_n is None else min(max_n, rows.shape[0])
    return rows[:n, 1:].view(np.float32).copy()


def load_attrs(path, n=None):
    print(f"[attr] loading {path}")
    t0 = time.time()
    with open(path) as f:
        attrs_all = json.load(f)
    if n is not None:
        attrs_all = attrs_all[:n]
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
                    num_threads=1, repeats=3, tag="",
                    internal_to_logical=None):
    """Run an ef-sweep and measure recall+qps.

    If `internal_to_logical` is provided (dict: internal_id -> logical_id),
    returned ids are translated before recall computation. This is required
    when the index has been modified by add_items at NEW internal ids (e.g.
    vector+attr updates) so that ground truth (which uses logical ids) is
    comparable.
    """
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
        if internal_to_logical is not None:
            translated = np.empty_like(last_pred)
            for i in range(Q):
                for j in range(k):
                    translated[i, j] = internal_to_logical.get(int(last_pred[i, j]), -1)
            r = recall_at_k(translated, gt, k)
        else:
            r = recall_at_k(last_pred, gt, k)
        print(f"  [{tag}] ef={ef:5d}  recall@{k}={r:.4f}  qps={qps:.1f}  elapsed={elapsed:.2f}s")
        results.append({"ef": int(ef), "recall": float(r), "qps": float(qps),
                        "elapsed_s": float(elapsed), "queries": int(Q),
                        "threads": int(num_threads)})
    return results


def generate_new_attr_for_id(current_attr, max_cate=21, rng=None):
    """
    Generate a fresh attr that differs from current_attr.
    Attr layout: [[category_value]] (1 attr column, 1 value each).
    Picks a random value in [0, max_cate) not equal to current.
    """
    if rng is None:
        rng = np.random.default_rng()
    cur = current_attr[0][0]
    while True:
        new_v = int(rng.integers(0, max_cate))
        if new_v != cur:
            return [[new_v]]


def plan_update_ids(N, round_idx, step):
    """Return ids in [(round_idx-1)*step, round_idx*step) (1-based round)."""
    s = (round_idx - 1) * step
    e = min(round_idx * step, N)
    return np.arange(s, e, dtype=np.int64)
