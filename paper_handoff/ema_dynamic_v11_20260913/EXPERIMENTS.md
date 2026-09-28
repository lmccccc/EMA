# Experimental protocol and interpretation

This document describes the completed V11 run, not a proposed benchmark.
The corresponding algorithm is in `DELETE_STRATEGY.md`; numerical tables are
in `RESULTS.md` and `data/stage_metrics.csv`.

## 1. Research scope

The experiment measures EMA / HashANN under insertion, synchronous structural
deletion, and complete-point replacement. It follows one fixed SIFT10M source
and one fixed set of mixed-DNF queries through controlled graph histories.
The performance questions are mutation maintenance time and single-thread
query throughput at 90% ID recall.

The same V11 native is used for construction, every mutation, and every query.
There is no tombstone-only substitution, query-triggered deletion patching,
attribute-only benchmark, reuse of an old-format graph, or private
reserve/ranking-cache optimization in the measured implementation.

This is one complete graph/trace realization. It does not provide
independent-build replications or a benchmark against another ANN system.

## 2. Dataset and fixed queries

| Item | Setting |
|---|---|
| Source | Original SIFT10M vectors and original complete mixed attributes |
| Source rows | 10,000,000 |
| Vector dimension / metric | 128 / squared L2 |
| Stored vector type | float32 fvecs; SIFT coordinates are integral in [0,255] |
| Attributes | Two attributes: numerical attribute 0 and categorical label-set attribute 1 |
| Queries | First 1000 original query vectors, in their original order |
| Returned neighbors | k=10 |
| Predicate family | A query-specific inclusive numerical range AND label9, OR label12 |
| Workload selectivity | Approximately 10%; measured on each current live set |

For query i, the unchanged predicate is

```text
P_i(x) =
    ((lo_i <= numeric(x) <= hi_i) AND (9 in labels(x)))
    OR (12 in labels(x)).
```

The numerical bounds may differ between queries, but the same query i keeps
exactly the same predicate in every stage and every operation. Predicates are
not regenerated to maintain a selected selectivity after mutation.
The exact predicate file is included as `data/workload/predicates.original.json`.

The canonical parsed-payload identities are identical in all three manifests:

```text
query_payload_sha256:
5fc3ea9c506f7ca2b250879f15c1674a8e70b1e7afb7c2cea60fa4396c61cb75

predicates_payload_sha256:
92c807088481aa571ab7df491a5fe9ac710daa871907056731b9c4ea13986ab4
```

These payload hashes differ from raw-file hashes because the query file
contains 10,000 rows while the experiment uses its first 1000, and predicates
are also fingerprinted in their canonical parsed representation.
Raw file names, byte sizes, and SHA256s are in
`provenance/pipeline-manifest.json`.

On the full 10M live source, the 1000 query matching counts range from
978,433 to 980,599, with mean 979,325.947. The mean live selectivity is
9.79325947%. Other stages have their own measured live matching counts.
Do not label every stage as having exactly 10% selectivity.

## 3. Three operation traces and graph lineage

Let m=1,000,000 and s=0,...,5 denote the stage. Stage 0 is the baseline;
each later stage applies m record-level operations.

| Operation | Initial graph | Transition at stage s>0 | Occupied / deleted / live after stage s |
|---|---|---|---|
| insert | Fresh original first 5M source rows | Append the next 1M complete source records | 5M+s*m / 0 / 5M+s*m |
| delete | Exact completed insertion-stage-5 10M checkpoint | Delete the next 1M entries of the fixed deletion permutation | 10M / s*m / 10M-s*m |
| point_update | Same unchanged original 5M prefix used by insertion | Delete 1M consecutive original logical points, then add their complete replacement records at fresh labels | 5M+s*m / s*m / 5M |

The deletion order is:

```python
order = numpy.random.default_rng(12345).permutation(10_000_000)
```

Its recorded SHA256 is
`ad71165820148868ea655be3fe91efea9d9e9184542373fdfc8d5c280b0e3baa`.
At deletion stage s, the native call processes
`order[(s-1)*m:s*m]`.

For point update, logical point i in the original prefix is retired and
replaced by source row `5M+i`, including its vector and **both complete original
attributes**. The new external label is also `5M+i`.
The delete call finishes before the add call begins, and
`replace_deleted=False` prevents retired-slot reuse. This operation maintains
5M live points while occupied slots rise to 10M. It is neither an
attribute-only update nor native in-place `updatePoint`.

Insertion stage 5 and deletion stage 0 use the same 10M checkpoint. Their
same-parameter query labels, distances, recalls, and exact GT are required
to agree before deletion begins. Point update starts from the original prefix,
not from the random 5M survivors at the end of deletion.

The initial graph is shared byte-for-byte where specified. Its separately
timed appearances can have slightly different QPS values: this is not evidence
of different predicates or different initial graph bytes.

## 4. Index construction and query configuration

| Parameter | Value |
|---|---|
| M | 40 |
| Maximum bottom-layer adjacency capacity | 80 |
| Construction ef | 300 |
| Deletion search width | The loaded native `ef_construction`, i.e. 300 |
| Edge Marker layout | Edge FT, 128 bits per attribute |
| Attribute types | [0,1] |
| CHT thresholds | [0.0001,0.0001,0.0001] |
| Candidate ordering | `vector-distance-only-v1` |
| Query ef_top | 1 |
| Query FT / FT routing | Enabled / enabled |
| Routing min_deg / tail backfill | 8 / disabled |
| Query ef | Calibrated from k=10 upward to first recall >=0.90; ceiling 3000 |
| Build / update / exact-GT threads | 32 / 32 / 32 |
| Query threads | 1 |
| Addition chunk size | 100,000 source rows |
| Native construction seed | 100 |

The full 10M source is available offline to construct the attribute codebook
and the representative level plan. Only the first 5M vertices are initially
inserted into the graph. This is not an experiment in which future source
vectors and attributes are unknown to the preparation stage.

One whole-source level plan is frozen and sliced for the initial prefix and
all later insertions:

- FAISS clustering with `floor(sqrt(10M))*4 = 12,648` centroids.
- Up to 2,560,000 training rows sampled without replacement.
- 25 clustering iterations; sampling and FAISS seed 1234.
- For each centroid, use its nearest assigned source row as the level-1
  representative; equal distances choose the lowest source-row ID.
- All remaining source rows are level 0.

This is the production representative-level convention in this codebase,
not conventional Bernoulli/exponentially sampled HNSW levels.
The codebook and level plan are not rebuilt separately for each operation.
`provenance/levels.json` contains training-ID, centroid, representative, and
level-array digests.

The current mixed index uses edge format10, numerical Marker version2,
owner version2, distance-order version1, and serialization version4.
The source supports node/edge formats9/10. Older attribute-bearing formats
are not silently retagged or migrated into this experiment.

## 5. Exact GT and recall definition

At every stage, the oracle computes an independent exact filtered top-10
over the **current live set**, not a cached baseline GT or a query subset.
It uses FAISS `IndexFlatL2` with disjoint branches:

```text
B = live points containing label12
A = live points containing label9 but not label12,
    restricted to the query's inclusive numerical interval
eligible = A union B
```

Numerically sorted branch A and a verified `IDSelectorRange` implement the
inclusive range. The two exact candidate lists are merged by distance and
external ID. Every query in this run has at least ten matching live records.

Returned ANN IDs must be live, unique, predicate-valid source-row labels.
Returned distances are checked against direct float64 squared-L2 computation
for those IDs. The published ID recall is

```text
recall =
    sum_i |ANN_i intersect GT_i| / (1000 * 10).
```

Point update also checks that logical-slot recall and external-label recall
agree. The numeric QPS90 tables use this ID recall; do not substitute the
separate tie-aware diagnostic metric used in earlier small-graph investigations.

When all 10M source rows are live, the existing canonical GT IDs are retained
only after independent exact-distance validation. Any differing IDs must be
proved to lie on the same exact k-th-distance tie boundary. The oracle
produces valid exact top-k results; this is not a claim that FAISS globally
selects the smallest external ID among every tied point.

Each raw stage directory includes `ground-truth.npz`,
`ground-truth.json`, and `calibration.npz`. The first stores the used GT,
independently computed exact labels, distances, matching counts, and logical
labels. The metadata describes any canonical tie reconciliation.

## 6. QPS90: recall-only calibration and median paired timing

Calibration queries every integer ef starting at 10 until recall first reaches
0.90. It uses recall alone, not QPS. The final adjacent recall bracket is
frozen before timing. Result hashes must remain identical at that ef during
all timed evaluations.

The immutable initial graph and current graph are timed as paired cases on
one physical core. There are warmup queries, including an untimed query before
each measured query batch. This is a warmed-query protocol, not cold-cache
service latency. A timed batch issues the complete fixed set of 1000 queries.

For a graph g and ef e, a timed batch in round r has

```text
Q[g,e,r] = 1000 / wall_seconds[g,e,r].
```

A round is accepted only if **every paired case** is clean. Use the first
seven accepted complete rounds on one selected core, with cyclic rotation of
case order. The QPS at each bracket endpoint is the **median of the seven
per-round QPS values**, not an arithmetic mean or pooled queries/sum-time ratio.

For adjacent measured endpoints `(R0,Q0)` and `(R1,Q1)` bracketing 0.90:

```text
QPS90 = Q0 + (Q1-Q0) * (0.90-R0) / (R1-R0).
```

If the minimum legal ef=10 is already above 0.90, there is no legal lower
endpoint. The pipeline leaves QPS90 null and records the measured QPS,
recall, and ef. If recall is exactly 0.90 at that minimum, its observed QPS
can be used directly. No extrapolation below k is allowed.

For each stage, the initial graph is independently paired with the current
graph. Use `same_recall_qps90_change` for the already paired relative change;
do not silently replace its baseline with a different stage's baseline QPS.

### Query noise admission

The process/update affinity is physical CPUs0-31 on NUMA node0. Query candidates
are physical cores8,16,24 with SMT siblings72,80,88. Candidate choice is based on
observed physical-core-plus-sibling load, not measured QPS.

| Condition for each accepted timed query batch | Requirement |
|---|---|
| Calling-thread CPU time / wall time | >=0.98 |
| SMT sibling busy fraction | <=0.05 |
| Query-core frequency sampled before and after | 3,267,000 through 3,333,000 kHz |
| Thread/process major page faults | 0 |
| Query CPU / affinity | Stay on the selected single physical core |
| Accepted paired rounds | 7 |
| Maximum attempted rounds per core / cores | 21 / 3 |

The frequency rule is sampled admission, not a claim that the global CPU
governor was changed or every instantaneous clock was fixed. No governor,
swap, cache, or other users' process settings were changed.

There is no fastest-sample selection or cross-core pooling. The folder retains
all 556 timed query batches, including cases not used in the final medians.
Of these, 483 belong to accepted paired rounds. The completed run needed no
whole-stage resume or superseded transaction, although individual timing
samples/rounds could be rejected.

These rules apply to official query QPS. Mutation calls run with 32 workers
on the shared host and are reported as observed elapsed maintenance time;
they are not independently replicated CPU-isolated microbenchmarks.

## 7. Maintenance time versus end-to-end runtime

The authoritative paper value is `maintenance_wall_s`, originating from
`mutation.wall_s` in the committed stage result. It is the sum of the timed
delete/add phase call boundaries:

```text
insert_time       = add_phase.wall_s
delete_time       = delete_phase.wall_s
point_update_time = delete_phase.wall_s + add_phase.wall_s
```

The delete phase includes Python binding conversion/return overhead and native
synchronous structural repair, Marker work, and final scrub. The add phase
also includes source-row preparation/chunk dispatch inside `add_source_rows`.
These values should not be described as isolated C++ kernel CPU time.

They exclude initial graph/level preparation, exact GT, recall calibration,
query timing, checkpoint I/O/hashing, and most coordinator checks outside the
timed calls. `mutation.transaction_wall_s` has a broader inter-phase boundary.
`process_cpu_s` is aggregated CPU time, not wall time.

The point-update denominator is the number of logical full-point replacements,
not twice that number because each replacement contains a delete and an add.
Baseline stages contain no mutation and are excluded from per-million means.

The total run from launch to paper publication took about 3h2m27s. This is not
the sum of the mutation-only columns: it also includes about 26 minutes of
whole-source level preparation, the initial prefix, and all measurement,
checkpoint, and publication work.

## 8. Hardware and software

The experiment ran on a shared x86_64 server. Configuration recorded at
execution and same-host inspection at handoff give:

| Item | Setting |
|---|---|
| CPU model | Intel Xeon Platinum 8358, nominal 2.60GHz |
| Sockets / physical cores / logical CPUs | 2 / 64 / 128 |
| Cores per socket / SMT threads per core | 32 / 2 |
| NUMA nodes | 2 |
| Memory | Approximately 2TiB |
| L3 cache | 96MiB total across two instances |
| Experiment mutation resources | One NUMA node, physical CPUs0-31, 32 workers |
| Linux kernel | 6.8.0-111-generic |
| Python | 3.12.3 |
| NumPy | 2.5.3 |
| FAISS | 1.15.0 |
| Compiler available on the same host at handoff | GCC/g++13.3.0 |

The exact frozen native hash is the binary identity. The same-host compiler
observation is not a promise of bit-for-bit recompilation with an arbitrary
toolchain or of a complete toolchain archive in this folder.

Environment controls include `OMP_NUM_THREADS=32`, `OMP_PROC_BIND=false`,
`OMP_WAIT_POLICY=PASSIVE`, `OMP_DYNAMIC=FALSE`, and
`OPENBLAS_NUM_THREADS=1`. Native publication uses the C++14 standard shared
gate. No claim of 32 continuously busy physical cores is implied by the
configured worker count.

## 9. Evidence and manuscript boundaries

The pipeline entrypoint is
`python -m exp_benchmark.dynamic.pipeline run`. It ran in official mode with
`--operations insert,delete,point_update`, capacity10M, initial5M, and step1M.
The complete immutable configuration is preserved in
`provenance/pipeline-manifest.json`; original operation manifests and final
journals are under `data/operations/`.

Publication required all 18 compatible complete stages on the same native
and the required shared-prefix/insertion-final graph lineage. The final
paper JSON/TSV and current-pointer files are included byte-for-byte.

Source reference locations:

| Subject | Source reference |
|---|---|
| Inputs, fixed parameters, traces, and levels | `source_reference/exp_benchmark/dynamic/protocol.py` |
| Record validity, recall, exact GT use, mutation time boundaries | `source_reference/exp_benchmark/dynamic/runner.py`, especially `validate_result`, `ground_truth`, `mutate` |
| Exact mixed-DNF oracle and tie handling | `source_reference/exp_benchmark/dynamic/gt.py` |
| Median paired timing and ef calibration | `source_reference/exp_benchmark/dynamic/measurement.py` |
| CPU/NUMA controls and sampled noise checks | `source_reference/exp_benchmark/dynamic/runtime.py` |
| QPS interpolation and compatible-stage publication | `source_reference/exp_benchmark/aggregate.py` and `source_reference/exp_benchmark/dynamic/aggregate.py` |

Safe descriptive conclusions include the recorded mutation costs, the actual
stage QPS90 values where a legal bracket exists, and the observed minimum-ef
points where it does not. Important limits are:

- All numbers are from one source/trace/graph realization. Do not use the five
  evolving stages as independent statistical replications.
- Query timing rounds estimate timing variation on those fixed indexes, not
  cross-build or cross-workload generalization.
- Deletion changes corpus size and graph history jointly. QPS trends cannot
  be attributed solely to Marker repair or called a recall-preservation theorem.
- Insert, delete, and point-update mutation means come from their defined
  graph histories; they are not paired identical-graph microbenchmarks.
- There is no other-system comparison, end-to-end online-arrival workload,
  disk-serving benchmark, concurrent mutation/query service test, or
  attribute-only update experiment in these data.
- Do not infer broader selectivity coverage, statistical significance,
  original IP-DiskANN equivalence, or new algorithmic complexity bounds.
- The reference source is uncommitted V11 worktree code. A public release or
  reproduction commit must be supplied separately before claiming one.
