"""
Experiment A: attr-only updates on sift10m 5M.

Starts from a saved 5M index. For each round 1..5, replaces attrs of ids
[(r-1)*1M, r*1M). No vector changes, no patch/rebuild. Recomputes filtered GT
and runs ef sweep each round.
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
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=5_000_000)
    ap.add_argument("--step", type=int, default=1_000_000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ef_search_list", type=str,
                    default="40,60,80,100,140,200,300")
    ap.add_argument("--threads_update", type=int, default=64)
    ap.add_argument("--threads_query", type=int, default=1)
    ap.add_argument("--query_repeats", type=int, default=3)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--predicate_value", type=int, default=9)
    ap.add_argument("--max_cate", type=int, default=21)
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260514)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[A] out_dir = {out_dir}")
    print(f"[A] args = {vars(args)}")

    base    = fvecs_read(BASE_FVECS, max_n=args.N)
    queries = fvecs_read(QUERY_FVECS, max_n=args.n_query)
    dim = base.shape[1]
    print(f"[A] base={base.shape} queries={queries.shape}")

    attrs_all = load_attrs(str(ATTR_JSON), n=args.N)
    predicate_per_q = [[[args.predicate_value]]] * args.n_query
    ef_list = [int(x) for x in args.ef_search_list.split(",") if x]

    idx = hashannlib.Index(space="l2", dim=dim)
    print(f"[A] loading {args.index_path}")
    t0 = time.time()
    idx.load_index(args.index_path, max_elements=args.N, top_elements=1,
                   allow_replace_deleted=False, dynamic=True)
    print(f"[A] loaded in {time.time()-t0:.1f}s, count={idx.get_current_count()}")

    rng = np.random.default_rng(args.seed)
    attrs_eff = [a for a in attrs_all]  # mutable per-id record

    all_records = []

    # round 0 = baseline (no updates yet)
    for r in range(0, args.rounds + 1):
        if r > 0:
            ids = plan_update_ids(args.N, r, args.step)
            new_attrs = [generate_new_attr_for_id(attrs_eff[i],
                                                  max_cate=args.max_cate,
                                                  rng=rng)
                         for i in ids]
            print(f"[A r{r}] updating attrs for {len(ids)} ids "
                  f"[{ids[0]}..{ids[-1]}]")
            t0 = time.time()
            cnt = idx.batch_update_attr(ids, new_attrs,
                                        num_threads=args.threads_update)
            up_time = time.time() - t0
            if cnt != len(ids):
                raise RuntimeError(
                    f"batch_update_attr updated {cnt}/{len(ids)} points")
            print(f"[A r{r}] batch_update_attr ok={cnt} in {up_time:.1f}s")
            for j, lid in enumerate(ids):
                attrs_eff[int(lid)] = new_attrs[j]
        else:
            up_time = 0.0

        eff_mask = build_attr_mask(attrs_eff, args.predicate_value, args.N)
        print(f"[A r{r}] filtered subset = {int(eff_mask.sum())} "
              f"({eff_mask.sum()/args.N*100:.2f}%)")

        gt_t0 = time.time()
        gt = compute_gt_filtered(base, queries, eff_mask, args.K)
        gt_time = time.time() - gt_t0
        np.save(out_dir / f"gt_r{r}.npy", gt)

        runs = run_query_sweep(idx, queries, predicate_per_q, gt, args.K,
                               ef_list, num_threads=args.threads_query,
                               repeats=args.query_repeats, tag=f"A/r{r}")

        rec = {
            "round": r,
            "n_updated_this_round": int(0 if r == 0 else len(ids)),
            "n_updated_cumulative": int(min(r, args.rounds) * args.step),
            "update_time_s": float(up_time),
            "gt_time_s": float(gt_time),
            "filtered_count": int(eff_mask.sum()),
            "runs": runs,
        }
        all_records.append(rec)
        with open(out_dir / "results.json", "w") as f:
            json.dump({"args": vars(args), "records": all_records}, f, indent=2)
        print(f"[A r{r}] saved checkpoint")

    # write summary
    lines = [f"# Experiment A: attr-only update sift10m N={args.N}\n"]
    lines.append(f"predicate=[[{args.predicate_value}]]  step={args.step}  rounds={args.rounds}\n\n")
    lines.append("round | n_updated_cum | update_time | filtered | best_ef@>=0.95 | qps\n")
    for r in all_records:
        runs = r["runs"]
        best = None
        for q in runs:
            if q["recall"] >= 0.95 and (best is None or q["qps"] > best["qps"]):
                best = q
        if best is None:
            best = max(runs, key=lambda x: x["recall"])
        lines.append(f"{r['round']} | {r['n_updated_cumulative']} | "
                     f"{r['update_time_s']:.1f}s | {r['filtered_count']} | "
                     f"ef={best['ef']} r={best['recall']:.4f} | {best['qps']:.1f}\n")
    (out_dir / "summary.txt").write_text("".join(lines))
    print(f"[A] DONE")


if __name__ == "__main__":
    main()
