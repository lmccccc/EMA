# Controlled component ablation

`python -m exp_benchmark.component_ablation` runs a new, source-pinned study
of the **current** implementation on YouTube-RGB 1M. It does not regenerate
the historical appendix observations, replace published figures,
or use the frozen V11 dynamic implementation.

The [technical report](../../technical_report.pdf) includes the completed study's figures.
Its gains, losses, and incomplete-batch cells are distinct from the pilot.

## Components and fixed workload

| Variant | C1: CHT construction admission | C2: query Markers | C3: routing recovery |
|---|---|---|---|
| `baseline` | off | off | off |
| `c1` | on | off | off |
| `c1_c2` | on | on | off |
| `full` | on | on | on |
| `without_c1` | off | on | on |

C1-off bypasses only CHT admission. RNG candidate ordering/pruning and Marker
witness collection remain enabled. It is **not** equivalent to changing
`set_min_deg`, and the baseline is not a separately built upstream HNSW
post-filter implementation. C3 is the current routing-only backfill:
`set_ft_routing_min_deg(16)` with tail backfill disabled, not an old two-hop scan.

Two new graphs share the vectors, complete attribute Codebook, explicit level
plan, source-row IDs, and native seed. Each graph is built once, using 32
threads; parallel insertion interleavings are not deterministic. The
construction switch is not serialized, so these are **read-only ablation
checkpoints**, not indexes for subsequent update experiments.

The fixed parameters are M40, ef-construction300, edge-level 128-bit Markers
per attribute, ef-top1, and recall@10. Queries use one physical core.
YouTube uses **raw inner product, without normalization**.

The full study uses all 1M base vectors, the first 1,000 original queries, and
the original mixed range-AND-label predicates and top-10 GT at nominal
selectivities **1, 2, 3, 5, 7, 10, 20, 40, 60, 80, 100%**. Actual selectivity
is audited separately. Numeric ranges and categorical labels follow the
original cells; this is not a fixed-label, numeric-range-only sweep.
Inputs must retain their original row order:

```text
youtube1m/
  rgb.fvecs
  rgb_query.fvecs
  label/arbi_0_1_random/
    attr_arbi_0_1_random.json
    predicate_arbi_0_1_<selector>.json
    gt_arbi_0_1_<selector>.json
```

Exact selectors are defined in `static_mixed_inputs.py`. Full-study GT is
checked for predicate validity and spot-checked by exact raw-IP search; it
is not regenerated or silently replaced.

## Build, pilot, and full run

Run from the repository root. Use an existing Python environment with NumPy,
FAISS, pybind11, and setuptools, plus a C++ compiler and `numactl`. The example
CPU allocation is NUMA node 0, physical cores 0-31; candidate query cores are
8, 16, and 24. Do not alter the machine's governor or interrupt other workloads.

```bash
PY=/absolute/path/to/python
DATA=/absolute/path/to/youtube1m
RUN=/absolute/path/to/new/component-study

bash exp_benchmark/ablation/build_component_native.sh "$PY" "$RUN/native"

"$PY" -B -m exp_benchmark.component_ablation prepare \
  --data-root "$DATA" --output "$RUN/pilot" \
  --native-provenance "$RUN/native/provenance.json" --pilot-size 10240

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=32 OMP_PROC_BIND=false
export OMP_WAIT_POLICY=PASSIVE OMP_DYNAMIC=FALSE
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -B -m exp_benchmark.component_ablation run \
  --request "$RUN/pilot/request.json"

# Only after the pilot completes:
"$PY" -B -m exp_benchmark.component_ablation prepare \
  --data-root "$DATA" --output "$RUN/full" \
  --native-provenance "$RUN/native/provenance.json"
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -B -m exp_benchmark.component_ablation run \
  --request "$RUN/full/request.json"
```

The pilot uses a base prefix and 32 queries with prefix-only exact GT.
Its `profile=pilot` results are not 1M measurements or paper evidence.
The builder refuses existing output and uncommitted production-native changes.
It archives the selected commit and applies `component_native.patch` only
inside the isolated build directory; nothing is installed into a shared
Python environment.

The **timing** extension has no new query-counter instrumentation. A separate
**diagnostic** extension uses a compile-time macro to count work. Both binaries,
the patch, source files, inputs, levels, and saved graphs are fingerprinted.
Diagnostic replays must return the same labels and distances as the timing
extension.

## Measurement and resumption

`run` launches `build`, `measure`, `diagnose`, and `report` in separate
processes so native imports cannot leak between phases. Repeating the same
`run --request ...` command reuses completed graphs and cells. Existing
per-ef calibration records are validated and reused; orphan raw files and
rejected timing attempts are retained rather than overwritten. A saved graph
without its final build record requires investigation, not silent reuse.
Do not edit frozen source files or input artifacts before resuming a request.

Calibration selects/refines ef by **recall only**, up to ef4096. The first
sampled crossing is refined inside its bracket; global recall monotonicity is
not assumed. Timing order is randomized within paired rounds. The first seven
complete CPU-clean rounds on one core are retained. After an unsuccessful
21-round attempt, another candidate core is selected by utilization, not QPS,
up to three distinct cores per invocation. Insufficient clean rounds then
stop the phase with `needs_timing`, without discarding its evidence.
Frequency and CPU telemetry are
recorded, not used to select the fastest samples.

The native fewer-than-k exception is an explicit failed point, not zero recall,
a padded result, or a reason to substitute another search implementation.
`unreached` means no successful sampled point achieved 95% recall.
When the first successful point already exceeds 95% without a lower bracket,
`qps95` is absent and actual observed recall/QPS remain available.

For incomplete batches, an optional **untimed** diagnostic finds the first
failing original query at the largest failed ef. It replays that same query
with C2 disabled or C3 enabled, keeping the graph and ef fixed, and also
records the full method's selected ef. This supplies a reachability witness,
not a failure-frequency, mean-recall, or throughput estimate:

```bash
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -B -m exp_benchmark.ablation.diagnose_failures \
  --request "$RUN/full/request.json"
```

The command uses the same environment as above and stores its own source,
provenance, scanned observations, and probes in `failure_diagnostics/`.
It never overwrites the primary calibration or measurements.

## Outputs and interpretation

| Artifact | Meaning |
|---|---|
| `request.json`, `source/` | Frozen protocol and source snapshots |
| `input-audit.json` | Actual selectivity, payload hashes, GT checks |
| `graphs/cht_{off,on}/build.json` | Shared-level identity, graph identity, build costs and degree statistics |
| `measurements/T*/` | Calibration arrays, successful/failed ef points, accepted/rejected timing attempts, separate diagnostics |
| `summary.csv` | Per-variant recall status, interpolated QPS95, actual qualifying observation, work counts |
| `curves.csv` | All timed recall-QPS observations |
| `contrasts.csv` | Paired component speedups and observed min/max across rounds |
| `results.json` | Consolidated results and limitations |

The component contrasts are `c1/baseline`, `c1_c2/c1`, `full/c1_c2`, and
`full/without_c1`; the last is the complementary C1 leave-one-out comparison.
`full/baseline` describes the total method. These effects are conditional on
component order and are **not additive**.

Primary `paired_qps95_speedup` uses the exact target or linear interpolation
inside a measured recall bracket. `paired_at_least95_speedup` compares
observations meeting the threshold, not necessarily identical achieved recall.
Their meanings must not be interchanged. Round minima/maxima are descriptive
variability, not confidence intervals or construction-seed uncertainty.

Counters are per query at the selected observed ef. `base_expanded` counts
expanded base-layer rows; `empty_marker_rows` counts nonempty rows with no
passing Marker before recovery; `backfill_edges` counts selected positions
before visited deduplication; `backfill_visits` counts newly visited routing-only
targets, not later expansions. `base_distance_evaluations` excludes entry
and upper-layer work. Existing edge-position and predicate counters have
different populations: do not call their ratio a clean Marker FPR, or label
ordinary predicate failures with Markers disabled as Marker false positives.

Explain measured gains, losses, and unreachable recall separately at each
selectivity. Reduced distance counts alone do not establish a runtime benefit;
these counters also do not prove a cache or memory bottleneck. Small C1 effects
from one parallel-built graph pair do not establish multi-seed robustness.

The runner exports CSV summaries, contrasts and recall-QPS curves for external
analysis. Plotting scripts are not distributed. Missing supported points
remain missing.

## Same-graph strict in-filtering comparison

`python -m exp_benchmark.strict_filter_ablation` adds a **separate** three-policy
comparison using the completed study's CHT-on graph. It does not rebuild a graph,
change the original component study's source pins, or reuse its QPS values.
The completed 1,000-query comparison is described in the technical report.

| Policy | Newly reached base-layer neighbors | Recovery |
|---|---|---|
| `strict` | Exact predicate must pass before distance calculation or queue insertion | off |
| `c1_c2` | Marker-gated routing, including nonmatching intermediate points | off |
| `full` | Marker-gated routing plus the existing routing-only backfill | on |

Upper-layer navigation and the initial base-layer seed remain common. The seed
is an explicit **bootstrap exception** to strict filtering: it may be
nonmatching. There is no GT-informed or freely selected matching entry point.
Rejected strict neighbors are marked visited but never scored, enqueued, or
expanded. Passing neighbors are not predicate-checked a second time for result
admission. Existing vector prefetches remain, so rejected neighbors are not
claimed to incur zero vector-memory traffic.

The additional `strict_native.patch` is applied **after** the original
component patch in an isolated source tree. Its builder uses the parent native's
recorded commit, patch hash, source hashes and compiler environment rather than
the current HEAD. Construction provenance remains the original build record;
the different query-native identities are recorded separately. Production
native sources and installed packages are not modified.

The completed original request, CHT-on checkpoint, original native provenance
and binaries, calibration plans and raw control arrays must remain available.
Use the same Python environment and NUMA/environment configuration as above:

```bash
PARENT=/absolute/path/to/completed/component-study
STRICT=/absolute/path/to/new/strict-filter-study

"$PY" -B -m exp_benchmark.ablation.build_strict_native \
  --base-native-provenance "$PARENT/native/provenance.json" \
  --output "$STRICT/native"

"$PY" -B -m exp_benchmark.strict_filter_ablation prepare \
  --parent-request "$PARENT/full/request.json" \
  --native-provenance "$STRICT/native/provenance.json" \
  --output "$STRICT/pilot" --query-count 16
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -B -m exp_benchmark.strict_filter_ablation run \
  --request "$STRICT/pilot/request.json"

# After the pilot succeeds, retain all 1,000 original queries:
"$PY" -B -m exp_benchmark.strict_filter_ablation prepare \
  --parent-request "$PARENT/full/request.json" \
  --native-provenance "$STRICT/native/provenance.json" \
  --output "$STRICT/full"
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -B -m exp_benchmark.strict_filter_ablation run \
  --request "$STRICT/full/request.json"

```

Unlike the construction pilot above, this pilot keeps the **full 1M graph and
original GT** and prefixes only queries, predicates and GT rows. No clustering
or prefix-GT recomputation is performed. It is still a pilot, not the full
1,000-query observation.

`run` launches `measure`, `diagnose`, and `report` in fresh processes. All three
policies are freshly timed together using the new timing binary, the unchanged
recall-only ef calibration and CPU-clean paired-round policy. When a control
has an original successful sample at the same ef, its labels and distances must
match that original query prefix bitwise. A failed original full batch may
have a successful prefix; that is not treated as an equivalence failure.

The output tables retain the component study's format, with contrasts
`Marker_vs_strict`, `full_vs_strict`, and `recovery_vs_Marker`. Summary columns
also distinguish no complete batch from complete batches below the sampled
recall target. Neither case is assigned a fabricated QPS95. In particular,
failure of the fixed-size batch API does **not** by itself prove mean recall
below 95% under a hypothetical partial-result API.

Untimed diagnostics replay every timed point and check identical outputs.
They also retain the first failing original query at the largest failed ef for
an incomplete policy, without turning that witness into a failure rate.
For strict mode, `predicate_checked` counts newly visited neighbor predicate
checks; the legacy `ft_false_positives` field counts strict predicate rejections,
**not** Marker false positives. Diagnostic counters for a below-target policy
are labeled with their actual `diagnostic_ef` and `diagnostic_recall`, not
presented as 95%-recall work.
