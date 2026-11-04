# Same-graph strict in-filtering results

Completed on 2026-11-02. This extends, rather than replaces, the
[five-variant component study](RESULTS.md).

## What was compared

All three policies use the **same saved CHT-on graph**, all 1M YouTube-RGB
vectors, the first 1,000 original queries, and the original eleven mixed-AND
predicate/GT cells. The metric is raw IP without normalization; the target is
95% **mean recall@10**. M40, ef-construction300, 128-bit edge Markers,
ef-top1, and the original row order remain unchanged.

| Policy | Base-layer routing | Recovery |
|---|---|---|
| Strict | Every newly visited neighbor must pass the exact predicate before distance evaluation or queue insertion | off |
| Marker only | Passing edge Markers can admit nonmatching intermediate points | off |
| EMA | Marker routing plus the existing routing-only backfill | on |

Upper-layer navigation is unfiltered and identical. The initial base-layer
seed is a common bootstrap exception: it may be nonmatching. Strict filtering
does not receive a free matching seed, precomputed matching bitmap, or a new
graph. Existing prefetches are retained; rejecting a neighbor does not imply
zero vector-memory traffic.

All three policies were freshly timed with the same new timing binary.
The original study's QPS samples were not reused. Each cell has seven complete
CPU-clean paired rounds on one core; ef selection uses recall only, with a
maximum of 4096. Separate diagnostic replays preserve the timed outputs.
Reproduction commands are in the [ablation README](README.md#same-graph-strict-in-filtering-comparison).

## Complete result matrix

QPS95 uses an exact target or interpolation inside the measured recall bracket.
Values below are rounded to one decimal.

| Nominal selectivity | Strict QPS95 | Marker-only QPS95 | EMA QPS95 |
|---|---:|---:|---:|
| 1% | NA | NA | 116.1 |
| 2% | NA | NA | 230.1 |
| 3% | NA | NA | 294.2 |
| 5% | NA | NA | 427.3 |
| 7% | NA | NA | 470.0 |
| 10% | NA | NA | 657.4 |
| 20% | NA | 930.1 | 879.6 |
| 40% | 457.3 | 1072.7 | 1040.9 |
| 60% | 609.3 | 976.2 | 964.0 |
| 80% | 808.2 | 935.6 | 934.5 |
| 100% | 895.9 | 882.9 | 883.4 |

Here every NA means **no complete 1,000-query top-10 batch at any sampled ef**,
not zero recall or zero throughput. In particular, this does not prove that a
different API returning partial results would have mean recall below 95%.
The failed observations and native exceptions are retained without padding
outputs or switching search policies.

Actual mean selectivities are 0.9953, 1.9931, 2.9881, 5.3144, 6.9943, 9.9947,
20.0070, 40.0416, 60.0190, 81.0364, and 100.0000%. Different cells retain their
original numeric ranges and categorical labels.

The following are **medians of within-round QPS95 ratios**, not ratios of
the independently summarized QPS columns above. Bracketed values are observed
paired-round minima/maxima, not confidence intervals.

| Selectivity | Marker / strict | EMA / strict |
|---|---:|---:|
| 40% | 2.337 [2.260, 2.380] | 2.271 [2.231, 2.299] |
| 60% | 1.604 [1.594, 1.619] | 1.597 [1.583, 1.636] |
| 80% | 1.159 [1.154, 1.162] | 1.154 [1.148, 1.173] |
| 100% | 0.983 [0.978, 0.992] | 0.987 [0.977, 0.996] |

Thus Marker routing has a substantial advantage over strict in-filtering at
40% and 60%, a smaller advantage at 80%, and a small loss when every point
matches. At 20%, Marker-only completes the workload and reaches the target,
while strict filtering still has an incomplete batch. Below 20%, Marker-only
does not eliminate the incomplete-batch problem; EMA's recovery remains needed
for this workload.

## Interpretation

Strict filtering restricts new traversal to predicate-matching points.
Marker routing instead allows some nonmatching points to serve as intermediate
steps. This changes reachable paths; it is not merely a cheaper way to evaluate
the same exact predicate.

At 40%, the selected ef is 247 for strict, 57 for Marker-only, and 53 for EMA.
At those qualifying observations, the mean number of expanded base rows is
249.6, 101.3, and 99.5, respectively. Newly scored targets are 2876.1, 1808.0,
and 1812.2. These observations support the usefulness of less restrictive
routing at the target recall. They do not establish that distance arithmetic
is the runtime bottleneck.

At 100%, all three select ef63 and return **identical labels and distances**.
All have 65.556 expanded base rows and 2332.223 newly scored targets per query,
with no recovery work. Marker pruning has nothing to remove here; the measured
Marker-only loss is about 1.7%. Strict performs 2332.223 neighbor predicate
checks versus 264.3 later-stage checks for Marker-only, yet is slightly faster.
Neither predicate-count nor distance-count reductions alone explain runtime.

Recovery has a separate, conditional cost. In this newly timed run, EMA /
Marker-only paired QPS95 ratios are 0.948, 0.966, 1.001, 0.995, and 1.003 at
20, 40, 60, 80, and 100%, respectively. Recovery costs about 5.2% at 20% and
3.4% at 40%, while restoring complete top-10 batches at 1-10%. The new timing
observations do not overwrite the original component study's measurements.

### Strict-filter failure witnesses

The untimed diagnostic scans original queries in order and stops at the
first failure at ef4096. IDs are zero-based. These are witnesses, **not failure
rates**, and they do not supply a mean recall for an incomplete batch.

| Selectivity | First failed query | Expanded base rows | Newly scored neighbors | Neighbor predicate checks | Rejected neighbors |
|---|---:|---:|---:|---:|---:|
| 1% | 0 | 1 | 0 | 17 | 17 |
| 2% | 0 | 2 | 1 | 20 | 19 |
| 3% | 0 | 1 | 0 | 17 | 17 |
| 5% | 1 | 4 | 3 | 94 | 91 |
| 7% | 0 | 1 | 0 | 17 | 17 |
| 10% | 0 | 1 | 0 | 17 | 17 |
| 20% | 573 | 1 | 0 | 10 | 10 |

The 20% witness is particularly clear: all ten neighbors of its sole expanded
row fail the exact predicate, so strict traversal cannot leave the seed.
Increasing ef cannot open a rejected edge. For the **same query 573**, the
saved Marker-only ef67 and EMA ef54 results each contain ten valid neighbors
and have recall 1.0. This single-query recall is not the workload's mean.

The 16-query pilot completed strict filtering at 20%, because it never reached
query 573. Its outcome is therefore not substituted for the full workload.
The six Marker-only failure witnesses reproduce the parent study's
queries 0, 22, 16, 102, 76, and 741 at 1, 2, 3, 5, 7, and 10%, respectively.

Strict `predicate_checked` counts newly visited neighbor checks, excluding
the common seed and upper-layer work. Its legacy `ft_false_positives` field
counts exact predicate rejections, not Marker false positives. Edge-position,
unique-target, and result-admission counters have different populations.

## Scope and artifacts

This is one dataset, one graph, one common-entry policy, and the original
mixed-AND workload. It is not a comparison with an optimized external
in-filtering system, an alternate matching-entry policy, or a construction
seed study. It isolates routing strategies on the CHT-on graph and adds no
new evidence for or against C1's construction benefit. No new geometric
skipping rule, connectivity proof, paper edit, or historical-figure replacement
is included.

Full results:

```text
/mnt/data/mocheng/dataset/youtube1m/hashann/benchmarks/component_ablation_20261101_FDIj854s/strict_filter_20261102_S7WPaGKv/full
```

`summary.csv`, `curves.csv`, `contrasts.csv`, and `results.json` contain the
unrounded observations. `measurements/T*/` retains every sampled ef, raw
successful result array, incomplete point, accepted/rejected timing sample,
and diagnostic witness. `plots/` contains PDF and TikZ outputs for QPS95,
observed qualifying QPS, paired ratios, and recall-QPS curves.

The parent graph construction provenance remains unchanged. The new query
native is in the sibling `native-v2/` directory. Its 159 successful control
calibration points match the original labels/distances bitwise, and its 60
incomplete control points retain the same batch-failure classification.

| Identity | SHA256 |
|---|---|
| Original CHT-on graph | `cbad22f76020a59dfe6a26e4e0c43881a7e9765b8447865ed367699c58f18feb` |
| Strict-only patch | `d4b84d2b3376810f5de801ce526a123854f38402632abd185f5a69ce654d0d4f` |
| Timing native | `722e61f555f8f699adb55907c1aae56ecf5d04c5dfd5fee63a1d57de2980c381` |
| Diagnostic native | `a35036d8754b3fcca1d2e715124a3d94572a8c8180ac90d98351788de812c8f8` |
| Full request | `a92cc649dec0f90f198e6d080911b9e24932486c9509e0f3ba7d18b73f7dcbc4` |

The native source base is commit
`d95fb49973f4fe53f5a2c1ac444dd5fdb879f551` plus the original component patch,
then the strict-only patch. The request, source snapshots, original control
plans, native provenance, and unchanged construction record are pinned
separately. The incomplete first native-build attempt is preserved under
`native/`; it failed during provenance collection, not during compilation or
querying, and was not used for the reported full study.
