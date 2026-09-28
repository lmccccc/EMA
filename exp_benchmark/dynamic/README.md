# Canonical dynamic reproduction (recall95)

Only **insert**, **delete**, and **point_update** belong to this pipeline. The
attribute-only and legacy delete/patch experiment entrypoints were removed;
native attribute-update and maintenance APIs remain available separately.
This measurement pipeline does not plot; the current
[R/TikZ publication sources](../../paper/README.md) are packaged separately.

The fresh-run policy remains **`fix_rebuild_rerun_all`**: correct numeric mapping,
Marker ownership and vector-distance-only ordering, fresh indexes and all three
experiments rerun on one fixed native. Mixed indexes must use **edge format10**, bounded FLOOR bucket mapping
and inclusive query endpoints, **numeric_marker_version=2**,
**marker_owner_version=2**, **distance_order_version=1**, and pickle
**ser_version=4**. The canonical complete-Codebook node/edge formats are9/10,
for both numeric and categorical attributes. The current native also writes
formats11/12 for partial categorical mappings; those are not canonical inputs.
Formats1–8
require rebuilding; retagging, graph migration and query retiming cannot repair
incompatible graph evolution. In particular, formats7/8 have no persisted
ranking-policy identity and cannot be reused as pure-distance graph evidence.
The fresh pipeline still rejects legacy deletion reuse. A separate, explicitly
authorized recall remeasurement of **already compatible, unchanged checkpoints**
is described below; it is not a graph migration or a fresh mutation replay.

The canonical target is restored to **95% recall**, using protocol
`sift10m-mixed-dnf-dynamic-rebuild-v5`, operation schema
`canonical-dynamic-operation-v4`, and fresh paper/current schema
`canonical-dynamic-paper-current-v6`. The user-authorized recall95 revision
reuses the completed V11 checkpoint chain with the same native and workload.
It does not authorize a new native, full graph builds/replays, or old-output
cleanup. Historical recall90 values must never be relabeled as QPS95.

The raw vectors, complete attributes, original queries, predicates and exact GT
retain their canonical checksums. Existing base-index files are protected but
are **not** reproduction inputs. Historical recall90 results, source snapshots,
manifests and journals remain preserved as original provenance. Artifact cleanup
is a separate, explicitly approved action, never a consequence of changing the
recall target.

Run from the repository root:

```bash
python -m exp_benchmark.dynamic.pipeline run --help
```

`reproduce.sh` forwards to that command. `insert.sh`, `delete.sh`,
`point_update.sh`, and their Python aliases select a single operation; a
single-operation output is **not** an official three-operation paper result.
Old categorical flags/configuration are deliberately not accepted.

## Inputs and traces

Official runs enforce the canonical SIFT10M input checksums, original mixed
`[0,1]` attributes, all 1,000 query-specific T10 DNFs and exactly the first 1,000
query vectors in their original order. The predicate is
`(inclusive numeric range AND label9) OR label12`, not categorical `[[9]]`.
Queries use L2/128 dimensions, k10, one thread, ef_top1, FT/routing enabled,
min_deg8 and no tail backfill. Index parameters are M40, construction ef300 and
edge-FT with 128 bits **per attribute**.
Dynamic routing **min_deg8 remains different from static routing min_deg16**;
restoring the shared 95% recall target does not change either routing setting.

Candidate ordering is **`vector-distance-only-v1`**: predicate/Marker checks may
filter admission; otherwise candidates are predicate-independent. Attribute
distance is never a ranking term. Existing CHT coverage quotas and query
eligibility/routing remain unchanged. The native compatibility API
`set_attr_sort_alpha` accepts only0 and rejects nonzero values; its getter is0.

Candidate coverage is intentionally bounded, not exhaustive. Construction/
deletion width300, the nearest50 window, three-replacement budget and one-owner
cleanup scope remain unchanged. This pipeline neither expands those budgets
nor adds exhaustive support scans or claims candidate-pool completeness.

| Operation | Initial state | Each of five stages |
|---|---|---|
| insert | first 5M source rows, capacity10M | append the next1M original complete records |
| delete | completed fresh insert stage5 checkpoint (full10M) | synchronously delete the next1M labels from `default_rng(12345).permutation(10M)` |
| point_update | same first5M graph as insertion | retire consecutive1M logical slots, then append source records `5M+i` at fresh external labels `5M+i` |

Point update is explicitly versioned
`complete-record-consecutive-fresh-label-v1`: both the vector **and all original
mixed attributes** come from the replacement source row. It does not reproduce
the old categorical random-new-label trace. `delete_items` is synchronous;
`add_items(..., replace_deleted=False)` never reuses retired slots. External
query/GT IDs equal source-row IDs. The logical-slot mapping and active source
mask are saved and independently checked, including logical/external recall.
Maintenance time counts native delete/add calls (including conversion/chunk
dispatch); GT, calibration, checkpoint IO and reporting are separate.

### Per-batch deletion timing

For both `delete` and the deletion phase of `point_update`, the runner retains
the complete delete-call wall time in `phases.delete.wall_s` and the native
final-scrub wall time in `phases.delete.scrub_wall_s`. The latter covers one
`scrubDeletedEdges()` call after all point workers and deferred repairs finish:
full-graph traversal, residual live-to-deleted reference removal, retired
outgoing-row clearing, adjacency/Marker compaction and scrub bookkeeping.
It is monotonic elapsed wall time, not a sum of worker CPU times.

`mutation.json`, stage output and paper JSON/TSV expose a `timing_breakdown`
(flattened in the paper rows) with:

| Field | Meaning |
|---|---|
| `maintenance_wall_s` | Delete plus add call time; unchanged throughput denominator |
| `delete_wall_s` / `add_wall_s` | Complete deletion / insertion phase time |
| `delete_scrub_wall_s` | Final whole-graph scrub time, already included in deletion |
| `delete_non_scrub_wall_s` | Delete total minus scrub; includes local/deferred repair, conversion, dispatch and other deletion overhead |
| `delete_scrub_fraction` | Scrub time divided by deletion time |
| `maintenance_scrub_fraction` | Scrub time divided by total delete-plus-add time |

The stage console names this object `mutation_timing`. Fractions are in `[0,1]`;
the optional LaTeX table displays the maintenance fraction as a percentage.
The recall-revision CSV uses `delete_scrub_seconds` and
`delete_non_scrub_seconds` for the corresponding duration columns and retains
both fraction columns. The scrub is **not** an extra top-level phase: adding it
again would double-count deletion cost. Non-scrub time is not claimed to be
pure local graph-repair time.

Timing values are separated from integer native work counters. Uninstrumented
native binaries and historical mutation logs retain `null`/`NA`/empty timing
cells, not fabricated zeros; insertion-only and initial stages have no
deletion breakdown. Recall-only remeasurement cannot recover an unrecorded
scrub duration. Existing frozen natives, mutation records and published
timings are not replaced to populate these columns.

### Initial graph and deterministic levels

There is no fallback from a missing/bad saved graph to an undocumented rebuild.
An explicit `--initial-prefix` must name a complete compatible prefix cache.
Otherwise the pipeline builds one in its own fresh output and shares it between
insert and point_update. It is **not** a random surviving5M deletion checkpoint.

The codebook sees all10M original records. Levels use the production
`tests/hashann.py` convention: FAISS clustering, `floor(sqrt(N))*4` centroids,
25 iterations, at most2.56M training rows, and the nearest assigned source row
per centroid at level1; all others are level0. The dynamic pipeline freezes
**one whole-source level plan** and slices it for the first5M and all subsequent
additions. This policy is versioned in the manifest; it is not the obsolete
Bernoulli(1/M) convention. `--level-seed` (default1234) seeds both explicit
training-row sampling and FAISS. Level, centroid, representative and training
ID digests are recorded. The native construction seed is100. Parallel builds
and mutation implementations can produce different graph bytes or work counts
on an uncommitted replay; the committed prefix/checkpoint is immutable and
reused byte-for-byte. Checkpoint reload/query consistency is still mandatory.
Python waits for each native mutation to finish before checking state, saving
or querying; point-update deletion finishes before fresh insertion starts.
Additional nonnegative integer work counters are retained without interpreting
them as unique edges or requiring serial/parallel count equality.

In the default `insert -> delete -> point_update` run, deletion loads the
**completed insert stage5 file itself**, preserving it unchanged. It does not
build another10M graph. Before any deletion, its freshly computed initial GT
and every calibration result must match insertion's final GT/query results at
identical ef and query parameters. The full-index origin and comparison
reference are recorded in deletion's manifest. Point update still loads the
same unmodified first5M prefix as insertion.

## Select and pin the native build

For a separately approved fresh run, build the **current production source** using the repository's
[installation instructions](../../README.md) and the same interpreter that will
run the pipeline. This build step is **not** part of the compatible checkpoint
recall revision, which must retain its existing pinned V11 binary.
With the build/runtime dependencies already installed:

```bash
PY=/path/to/selected/python
mkdir -p .native-build
TMPDIR="$PWD/.native-build" "$PY" -m pip install --no-build-isolation --no-deps ./hnswlib
NATIVE_FILE=$("$PY" -c 'from pathlib import Path; import hashannlib; print(Path(hashannlib.__file__).resolve())')
NATIVE_DIR=$(dirname "$NATIVE_FILE")
NATIVE_SHA=$("$PY" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$NATIVE_FILE")
OUT=/mnt/data/mocheng/dataset/sift10m/hashann/benchmarks/dynamic_compare_20260910
```

Alternatively set `NATIVE_DIR` to an explicit frozen extension directory and
`NATIVE_SHA` to that artifact's verified SHA256. The extension must match `$PY`'s
ABI. The pipeline hashes the selected file **before import** and verifies the
loaded module's origin. Do not rebuild or replace a native artifact referenced
by an active or resumable run.
If the selected frozen directory provides `provenance.json`, its native,
`hnswalg.h`, binding-source and `setup.py` hashes and version fields are also verified and
included in the immutable source inventory.
Every declared `auxiliary_source_sha256` entry is likewise hashed from the
actual frozen relative path and included in cache/run provenance, including
`hnswlib/parallel_for.h` and `hnswlib/deletion_cache.h`. Missing or changed
helpers, malformed paths and symlinked auxiliary sources fail without fallback.

Both **inclusive numeric endpoints and bounded FLOOR bucket mapping**, plus
**correct witness-to-owner alignment**, are required. An endpoint-only fix is
insufficient: old quantile boundaries plus unclamped ceiling lookup can spill
numeric bits into the next attribute bitmap. A numeric-only fix is also
insufficient: witness rows selected in heuristic order must remain attached to
the same retained owners after adjacency reordering. This ownership defect
affects categorical indexes too.
Vector-only candidate ranking is separately versioned; correct Marker ownership
alone is not sufficient. The pipeline requires the selected native's lifecycle
round-trip to emit format10 and its pickle metadata to report
numeric2/owner2/distance-order1/state4. It rejects legacy
attribute-bearing checkpoints before loading them. Versioned attribute-index
caches use `_mo2_do1`, with `_nb2_mo2_do1` for numerical mapping.
Each run/resume first runs a tiny isolated native
save/load/add/delete/fresh-add/query canary with core dumps disabled. Failure
stops the run; there is no rebuild or update-semantic workaround.
The canary seeds its complete20-label codebook without changing the numeric
samples or inserted records, so incremental label registration still produces
the required complete-Codebook format10. This changes only synthetic fixture
coverage; canonical inputs and the rejection of partial-Codebook formats11/12
remain unchanged.

The canary gates **lifecycle integrity**, not exact ANN quality: all occupied
records (including retired slots), complete attributes/vectors/labels, live
predicate-valid results, exact squared-L2 distances for returned labels, state
counts, and save/load result identity must pass. Its report separately records
`ann.recall`, raw neighbors, exact GT and missed neighbors. Fixture-only
upper-bound63/64 and FT-disabled diagnostics preserve evidence of FT sensitivity
when the eligible set is unchanged. These diagnostic variants never replace the
formal query results, predicates, FT settings or benchmark calibration. A
lifecycle pass is not a claim of 100% ANN recall. Separately, the numeric
upper-slot omission in pre-correction Marker construction is a **confirmed
false-negative bug**, reproduced on a fully connected equal-distance fixture;
it must not be dismissed as normal ANN approximation. The fixed native changes
bucket mapping, not the workload's bounds, ef policy or FT flags.
The separate maximum-bin overflow can corrupt stored Markers and graph evolution;
it cannot be repaired by query retiming. `marker_fidelity_certified=false` in the
lifecycle report is intentional: its padded128-slot codebook does not cover this
maximum-bin failure, and metadata equality does not prove witness-to-owner
alignment or complete Marker fidelity. Dedicated native regressions remain
required; a lifecycle pass must not certify a full-scale native build.
`candidate_completeness_certified=false` likewise preserves the distinction
between valid lifecycle behavior and intentionally bounded candidate coverage.

## Small real-data pilot

```bash
export OMP_NUM_THREADS=32 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
numactl --cpunodebind=0 --membind=0 "$PY" -B -m exp_benchmark.dynamic.pipeline run \
  --data-root /mnt/data/mocheng/dataset/sift10m \
  --output "$OUT/canonical_dynamic_pilot_NEW" \
  --native-dir "$NATIVE_DIR" --expected-sha256 "$NATIVE_SHA" \
  --numa-node 0 --threads-update 32 --threads-build 32 --threads-gt 32 \
  --pilot-size 1000
```

This pilot uses only the first1,000 actual source records (500 initial,
five100-record stages) and the original1,000 queries/DNFs. It builds no5M/10M
graph and does not read/hash the full vector or attribute payload. Deletion
reuses the fresh1,000-record insertion checkpoint. Its prefix
payloads and full query/predicate identities are recorded. It exercises all
three operations, exact GT, real query calibration/timing observations,
saved-graph loading and journals. Pilot QPS95 is always null; timing noise does
not prevent correctness mutations. Pilots cannot be reused as paper evidence.
Use `--through-stage 1`, followed by identical arguments plus `--resume` and
without the stage limit, to exercise crash-safe saved-prefix continuation.

## Full fresh rebuild and all-three rerun

Fresh builds and mutation replays are separate from the recall-only revision;
the commands here are not steps for that revision. Do not start merely because
a native passes lifecycle tests. Qualifying a new native requires an explicitly
approved frozen native SHA,
successful fresh-process pipeline integration, and at least three stable,
balanced, same-graph32-thread observations with deletion/insertion wall ratio
at most3 and no recall regression. This is an approval gate, not a reason to
change candidate budgets, ranking, repair scope, data or measurement rules.
Historical timings from another native do not qualify the selected artifact.

After correctness and performance qualification, choose a fresh versioned
output directory and a validated format10/distance-order1 native:

```bash
export OMP_NUM_THREADS=32 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
numactl --cpunodebind=0 --membind=0 "$PY" -B -m exp_benchmark.dynamic.pipeline run \
  --data-root /mnt/data/mocheng/dataset/sift10m \
  --output "$OUT/canonical_dynamic_format10_NEW" \
  --native-dir "$NATIVE_DIR" --expected-sha256 "$NATIVE_SHA" \
  --operations insert,delete,point_update \
  --numa-node 0 --threads-update 32 --threads-build 32 --threads-gt 32 \
  --frequency-khz 3267000,3333000 \
  --publish-current "$OUT/dynamic-paper-current.json"
```

Neither `--reuse-delete`, `--retime-reused-delete`, nor `validate-reuse` is part
of this reproduction surface. Old4e6 deletion evidence cannot enter the paper
table, even with newly measured query curves.

For deletion without insertion, select `--operations delete`. The pipeline
constructs a fresh full-source index in `initial-full/` using the same whole-source
level plan and full attribute codebook. Alternatively `--initial-full` may name
a complete compatible **full-cache directory** containing `full-cache.json`.
The cache must match the dataset, native, levels, construction parameters and
format10/numeric2/owner2/distance-order1/state4 fingerprint; a missing, legacy or
incompatible cache fails without fallback. `--initial-full` cannot override
insert stage5 in a combined run.
A standalone deletion run does not publish a three-operation table.

## Compatible checkpoint recall remeasurement

The user-authorized recall90-to-recall95 revision keeps the completed V11 native
SHA256 **`608e9ae692fa2fd9d5ec160a370a63b90419356c6b64d67285f10d6bd1e4f84e`**,
M40, construction ef300, k10, routing min_deg8, the fixed SIFT10M mixed DNFs,
query order and all three activation/replacement traces unchanged. All18
completed stage checkpoints and their exact GT are retained; no new graph
construction, insertion, deletion or point-update replay is needed for this
recall-target change.

Use the separate `exp_benchmark.dynamic.remeasure` entrypoint; consult
`python -m exp_benchmark.dynamic.remeasure --help` for its explicit inputs and
output controls. This is a **read-only checkpoint recall revision**, not the
removed unsafe `retime.py`/legacy categorical deletion-reuse path:

- Accept only the explicitly selected, unchanged format10 graphs with
  numeric2/owner2/distance-order1/state4 semantics and the same pinned native,
  dataset, query configuration and traces.
- Preserve original manifests, journals, calibration results, checkpoint/GT
  identities and executed-source snapshots. An old source hash must not be
  rewritten to match the new recall95 Python source.
- Recalibrate each graph from k to its first recall>=95%, then collect new
  frozen-query and CPU-clean paired timing evidence in a **new result tree**.
  Old QPS90 timings or 90%-crossing calibration points are not QPS95 evidence,
  even when an old minimum-ef observation happened to exceed95%.
- Record the original maintenance wall times as **reused mutation timing
  provenance**, separately from the new query measurements. This evidence must
  not claim newly executed mutations or enter the unmodified fresh-run
  aggregator as `reused=False` / `queries_retimed=False`.

The fresh pipeline's `run --resume` and `aggregate` reject old recall90
protocol/operation schemas. The separate recall-revision output must identify
its distinct provenance rather than bypass those guards. Incompatible old
Marker/ranking graphs still require rebuilding; neither this authorization nor
a lifecycle canary generically certifies arbitrary native builds.

The completed V11 recall95 revision is packaged in
[`paper_handoff/ema_dynamic_v11_recall95_20260914`](../../paper_handoff/ema_dynamic_v11_recall95_20260914/README.md).
It contains all18 supported QPS95 values, the unchanged original maintenance
times, and504 accepted paired-round samples out of696 recorded samples.
The earlier dated recall90 handoff remains historical evidence and is not
overwritten or relabeled.

## Measurement, retention and publication

- Each operation calibrates its immutable baseline once. Each current state
  scans every integer ef from k to the **first** recall>=95%, without timing-based
  tuning, then freezes adjacent points. The existing static
  `exp_benchmark.aggregate.qps_at_recall` interpolation is used explicitly at95%;
  the static CLI default remains95%. Failure to reach95% by `--max-query-ef`
  stops calibration with no timing fallback.
- Calibration states, timing reports and summaries explicitly record
  `target_recall=0.95`. The query metric is `qps95`, never a renamed `qps90`.
- If ef=k already exceeds95%, `qps95=null`. `observed_qps`, selected recall/ef
  and explicit `>=95%` status are retained—never extrapolated. An observation
  exactly at95% has a supported QPS95 value. Pilots always keep QPS95 null.
- Timing reports use `qps95_change_fraction`; paper rows use
  `paired_initial_qps95` and `same_recall_qps95_change`. Changes are
  current/paired-initial minus1, and stay null unless both QPS95 values exist.
- Official paired initial/current measurements share one physical local core:
  two warmups/case, one before each sample, rotating case order, first7 complete
  clean rounds, max21/core and3 cores. Selection uses noise only, never highest
  QPS or cross-core pooling. All raw rejected samples remain available.
- Public CPU/node lists and frequency bounds are explicit; no host CPU IDs or
  3.3GHz default are hardcoded. `--update-cpus`, `--query-cores`,
  `--frequency-khz` and noise controls are configurable. The current official
  command above uses the same >=98% threadCPU/wall, <=5% sibling busy,
  3.267–3.333GHz, zero-major-fault limits as the established measurement protocol.
- NUMA validation includes actual anonymous residency, all worker affinities,
  and default MPOL_BIND. FAISS `local` arenas are allowed only when pages and
  every worker remain on the selected node.
- Checkpoints, GT, source/activation traces and results are fsynced before the
  journal commit. A partial mutation cannot be completed by metadata alone.
  Uncommitted attempts are retained and replay from the previous checkpoint.
  Noise pauses return exit2 with the update/checkpoint committed; resume re-times
  that exact graph/GT without repeating mutation.
- Resume requires matching source/native/input/config/level/trace identities.
  If initialization stopped after publishing a manifest but before its empty
  journal/state, `--resume` validates that immutable identity and completes the
  bootstrap. Existing compatible prefix caches and operation journals are
  verified and retained, not rebuilt. A missing operation journal alongside
  mutation attempts is not treated as an empty bootstrap.
  `--through-stage` and `--resume` are operational controls. There are no broad
  data deletes or mutation/query concurrency.

In the fresh-run pipeline, only all three compatible **complete official**
operations produce immutable
`paper.json`, `paper.tsv` (and optional `--latex` `paper.tex`). Only then is the
chosen current-manifest pointer replaced atomically. Partial, pilot, categorical,
attribute-only and stale results cannot enter that paper output. Rows include
live/occupied/retired counts, maintenance wall/throughput, actual live selectivity,
observed recall/ef/QPS, nullable QPS95 and paired baseline comparisons.
Version6 paper/current manifests require recall95 evidence and fresh format10 graphs with numeric2/
owner2/distance-order1/state4 and `candidate_order=vector-distance-only-v1`, one fixed native,
the shared insertion-final/deletion-initial checkpoint and shared5M prefix.
Old graph evolution, partial runs and mixed-native evidence cannot be published.
The separate compatible recall-revision output is not a fresh-run version6
mutation replay; reused maintenance provenance must remain explicit.

Revalidate/export a completed fresh pipeline:

```bash
"$PY" -B -m exp_benchmark.dynamic.pipeline aggregate \
  --run "$OUT/canonical_dynamic_format10_NEW" \
  --publish-current "$OUT/dynamic-paper-current.json" --latex
```

Only the authoritative published manifest—not exit0 from a pilot, a scoped run,
or a clean stage pause—identifies the current paper data. Physical artifact
cleanup requires explicit user approval and successful replacement publication.
The recall95 revision does not authorize removing historical recall90 results or
their frozen handoff. Reused checkpoints, GT and original mutation provenance
remain referenced evidence, not obsolete data. Raw vectors/attributes/queries/GT,
base-index files, and every artifact referenced by the new current manifest
remain protected. The pipeline itself never deletes data.

## Targeted regression tests

```bash
HASHANN_DYNAMIC_NATIVE_DIR="$NATIVE_DIR" HASHANN_DYNAMIC_NATIVE_SHA="$NATIVE_SHA" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE \
  "$PY" -B -m unittest discover -s exp_benchmark/tests -p 'test_dynamic*.py'
```

Run this suite in a **fresh process**, separately from native test suites: the
SHA-pinned import guard intentionally rejects an already imported extension.
Static-native integration also needs its own process; do not enable both
static and dynamic native integration in one full-suite discovery process.
Omit the two `HASHANN_DYNAMIC_NATIVE_*` variables for unit fixtures only. Native
tests disable core dumps and reject superseded natives, including owner-correct
format8 builds without ranking-policy identity. With no native selected, native
integration is skipped; select only a distance-order1 artifact for that integration.
Fixtures use only local test artifacts; no full graph or full-size experiment
is started.
