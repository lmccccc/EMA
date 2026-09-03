"""
Experiment: vector+attr (point) updates on sift10m 5M.

Starts from a saved 5M index. For each round 1..rounds, for logical ids
[(r-1)*step, r*step):
  - mark_delete the CURRENT internal id holding that logical slot
  - add_items at a NEW internal id with a new vector (from base[N+id])
    and a new attribute that differs from the current one.

Recomputes filtered GT and runs ef sweep each round. No patch / no rebuild
inside this script -- this measures the pure (mark_delete + add) update cost.

The index file MUST have been built on an N+rounds*step element capacity
(headroom for new internal slots) and must be loadable with dynamic=True.
"""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import hashannlib

sys.path.insert(0, str(Path(__file__).parent))
from update_common import (
    BASE_FVECS, QUERY_FVECS, ATTR_JSON,
    fvecs_read, load_attrs, build_attr_mask, compute_gt_filtered,
    run_query_sweep, generate_new_attr_for_id, plan_update_ids,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", default="",
                    help="(optional) saved 5M index with max_elements>=N+rounds*step. "
                         "If empty, the index is built in-process. NOTE: the C++ "
                         "load_index path has a known bug where add_items at a new "
                         "internal id segfaults on a loaded index built by "
                         "hashann_build.py — so by default we always build "
                         "in-process. Set --index_path to a known-good cache "
                         "produced by this script itself (via --save_cache) to "
                         "skip the rebuild.")
    ap.add_argument("--save_cache", default="",
                    help="If non-empty, save the freshly built index to this path "
                         "after build (so subsequent runs can pass it as --index_path).")
    ap.add_argument("--ef_construction", type=int, default=300)
    ap.add_argument("--ft_bits", type=int, default=128)
    ap.add_argument("--threads_build", type=int, default=64)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=5_000_000)
    ap.add_argument("--step", type=int, default=1_000_000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--M", type=int, default=40,
                    help="must match the index that was built")
    ap.add_argument("--ef_search_list", type=str,
                    default="40,60,80,100,140,200,300")
    ap.add_argument("--threads_update", type=int, default=64)
    ap.add_argument("--threads_query", type=int, default=1)
    ap.add_argument("--query_repeats", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--predicate_value", type=int, default=9)
    ap.add_argument("--max_cate", type=int, default=21)
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[P] out_dir = {out_dir}")
    print(f"[P] args = {vars(args)}")

    # Need base[0..2N) -- first N for the loaded index, [N..2N) as the
    # replacement-vector pool for the rounds.
    base_full = fvecs_read(BASE_FVECS, max_n=2 * args.N)
    base_eff = base_full[:args.N].copy()
    new_vec_pool = base_full[args.N:2 * args.N]
    queries = fvecs_read(QUERY_FVECS, max_n=args.n_query)
    dim = base_full.shape[1]
    print(f"[P] base_eff={base_eff.shape} pool={new_vec_pool.shape} queries={queries.shape}")

    attrs_all = load_attrs(str(ATTR_JSON), n=args.N)
    attrs_eff = list(attrs_all)
    predicate_per_q = [[[args.predicate_value]]] * args.n_query
    ef_list = [int(x) for x in args.ef_search_list.split(",") if x]

    max_total = args.N + args.rounds * args.step
    rng_build = np.random.default_rng(args.seed + 99)
    idx = hashannlib.Index(space="l2", dim=dim)
    if args.index_path and os.path.isfile(args.index_path):
        print(f"[P] loading {args.index_path} (max_elements={max_total})")
        t0 = time.time()
        idx.load_index(args.index_path, max_elements=max_total, top_elements=1,
                       allow_replace_deleted=False, dynamic=True)
        print(f"[P] loaded in {time.time()-t0:.1f}s, count={idx.get_current_count()}")
    else:
        # In-process build (works around a C++ load_index bug where add_items at
        # new internal ids segfaults on indexes saved by hashann_build.py).
        print(f"[P] building 5M index in-process (max_elements={max_total})")
        idx.init_index(
            max_elements=max_total, top_elements=1, M=args.M,
            ef_construction=args.ef_construction, ft_bits=args.ft_bits,
            attr_type=[1], max_cate_size=args.max_cate,
            allow_replace_deleted=False, edge_level_ft=True,
        )
        idx.set_num_threads(args.threads_build)
        # init_attr_mapping iterates [0, max_elements_); pad with first attr.
        if len(attrs_eff) < max_total:
            pad_attr = [attrs_eff[0]] * (max_total - len(attrs_eff))
            idx.initAttrMapping(list(attrs_eff) + pad_attr)
        else:
            idx.initAttrMapping(attrs_eff)
        # Bernoulli(1/M) levels to match the canonical build distribution.
        levels = (rng_build.random(args.N) < 1.0 / args.M).astype(np.int32)
        chunk = 100_000
        tb = time.time()
        for s in range(0, args.N, chunk):
            e = min(s + chunk, args.N)
            idx.add_items(base_eff[s:e], attrs_eff[s:e],
                          ids=list(range(s, e)),
                          levels=levels[s:e].tolist(),
                          num_threads=args.threads_build)
        print(f"[P] inserted {args.N} in {time.time()-tb:.1f}s")
        if args.save_cache:
            try:
                Path(args.save_cache).parent.mkdir(parents=True, exist_ok=True)
                idx.save_index(args.save_cache)
                print(f"[P] saved index cache -> {args.save_cache}")
            except Exception as e:
                print(f"[P] WARN: save_index failed: {e}")
    idx.set_num_threads(args.threads_update)

    rng = np.random.default_rng(args.seed)
    logical_to_internal = {i: i for i in range(args.N)}
    new_id_cursor = args.N

    all_records = []

    # round 0 = baseline (no updates yet)
    for r in range(0, args.rounds + 1):
        if r > 0:
            old_ids = plan_update_ids(args.N, r, args.step)
            new_attrs = [generate_new_attr_for_id(attrs_eff[int(i)],
                                                  max_cate=args.max_cate,
                                                  rng=rng)
                         for i in old_ids]
            new_vecs = new_vec_pool[old_ids].copy()
            new_internal_ids = list(range(new_id_cursor,
                                          new_id_cursor + len(old_ids)))
            new_id_cursor += len(old_ids)

            old_internals = np.array(
                [logical_to_internal[int(slot)] for slot in old_ids],
                dtype=np.uint64,
            )
            print(f"[P r{r}] update {len(old_ids)} pts: mark_delete logical "
                  f"[{old_ids[0]}..{old_ids[-1]}] then add at internal "
                  f"[{new_internal_ids[0]}..{new_internal_ids[-1]}]")

            t0 = time.time()
            n_marked = idx.batch_mark_deleted(old_internals,
                                              num_threads=args.threads_update)
            t_md = time.time() - t0
            if n_marked != len(old_internals):
                print(f"[P r{r}] WARN: batch_mark_deleted marked {n_marked}/"
                      f"{len(old_internals)}")

            # Match initial build distribution: Bernoulli(1/M) -> {0,1}.
            new_levels = (rng.random(len(new_internal_ids))
                          < 1.0 / args.M).astype(np.int32)
            t0 = time.time()
            idx.add_items(new_vecs, new_attrs,
                          ids=new_internal_ids,
                          levels=new_levels.tolist(),
                          num_threads=args.threads_update)
            t_add = time.time() - t0
            up_time = t_md + t_add
            print(f"[P r{r}] mark_delete {t_md:.1f}s + add {t_add:.1f}s "
                  f"= {up_time:.1f}s  deleted_ratio={idx.get_deleted_ratio():.3f}")

            # Maintain effective base/attrs.
            for j, slot in enumerate(old_ids):
                base_eff[int(slot)] = new_vecs[j]
                attrs_eff[int(slot)] = new_attrs[j]
                logical_to_internal[int(slot)] = int(new_internal_ids[j])
        else:
            up_time = 0.0
            t_md = 0.0
            t_add = 0.0

        # Inverse mapping for translating returned internal ids back to logical
        internal_to_logical = {iv: slot for slot, iv in logical_to_internal.items()}

        eff_mask = build_attr_mask(attrs_eff, args.predicate_value, args.N)
        print(f"[P r{r}] filtered subset = {int(eff_mask.sum())} "
              f"({eff_mask.sum()/args.N*100:.2f}%)")

        gt_t0 = time.time()
        gt = compute_gt_filtered(base_eff, queries, eff_mask, args.K)
        gt_time = time.time() - gt_t0
        np.save(out_dir / f"gt_r{r}.npy", gt)

        runs = run_query_sweep(idx, queries, predicate_per_q, gt, args.K,
                               ef_list, num_threads=args.threads_query,
                               repeats=args.query_repeats, tag=f"P/r{r}",
                               internal_to_logical=internal_to_logical)

        rec = {
            "round": r,
            "n_updated_this_round": int(0 if r == 0 else len(old_ids)),
            "n_updated_cumulative": int(min(r, args.rounds) * args.step),
            "mark_delete_time_s": float(t_md),
            "add_time_s": float(t_add),
            "update_time_s": float(up_time),
            "gt_time_s": float(gt_time),
            "filtered_count": int(eff_mask.sum()),
            "deleted_ratio": float(idx.get_deleted_ratio()),
            "current_count": int(idx.get_current_count()),
            "runs": runs,
        }
        all_records.append(rec)
        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "records": all_records}, f, indent=2)
        print(f"[P r{r}] saved checkpoint")

    # ---- summary ----
    lines = [f"# Experiment P: vector+attr update sift10m N={args.N}\n"]
    lines.append(f"predicate=[[{args.predicate_value}]]  step={args.step}  "
                 f"rounds={args.rounds}  threads_update={args.threads_update}\n\n")
    lines.append("round | n_updated_cum | mark_delete | add | total_update | "
                 "filtered | best_ef@>=0.95 | qps\n")
    md_times = []; ad_times = []; tot_times = []
    for r in all_records:
        runs = r["runs"]
        best = None
        for q in runs:
            if q["recall"] >= 0.95 and (best is None or q["qps"] > best["qps"]):
                best = q
        if best is None:
            best = max(runs, key=lambda x: x["recall"])
        lines.append(f"{r['round']} | {r['n_updated_cumulative']} | "
                     f"{r['mark_delete_time_s']:.2f}s | "
                     f"{r['add_time_s']:.2f}s | "
                     f"{r['update_time_s']:.2f}s | "
                     f"{r['filtered_count']} | "
                     f"ef={best['ef']} r={best['recall']:.4f} | {best['qps']:.1f}\n")
        if r["round"] > 0:
            md_times.append(r["mark_delete_time_s"])
            ad_times.append(r["add_time_s"])
            tot_times.append(r["update_time_s"])
    if md_times:
        lines.append("\n## Per-round averages (excluding round 0)\n")
        lines.append(f"  avg mark_delete  = {sum(md_times)/len(md_times):.2f}s\n")
        lines.append(f"  avg add          = {sum(ad_times)/len(ad_times):.2f}s\n")
        lines.append(f"  avg total update = {sum(tot_times)/len(tot_times):.2f}s "
                     f"(per {args.step} ops)\n")
    (out_dir / "summary.txt").write_text("".join(lines))
    print(f"[P] DONE")


if __name__ == "__main__":
    main()
