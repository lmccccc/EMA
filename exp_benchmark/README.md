# EMA (HashANN) Benchmark Scripts

End-to-end build / query / ablation scripts for **EMA** (Edge Marker;
the method described in the HashANN paper) on four datasets:

| dataset       | n          | dim  | metric |
|---------------|-----------:|-----:|--------|
| sift10m       | 10 000 000 | 128  | L2     |
| youtube_rgb   | 1 000 000  | 1024 | IP     |
| wiki_15_4M    | 15 435 516 | 1024 | IP     |
| Redcaps_4M    | 4 000 000  | 512  | IP     |

The scripts wrap the C++/Python bindings in `hnswlib/` and the helper
tools under `tests/`. They are intentionally self-contained and do **not**
depend on any of the historical `exp*/` folders.

> The paper's **Edge Marker** is called **`ft`** throughout the code
> (legacy name "fingerprint"). When you see `ft_bits`, `edge_level_ft`,
> `ft_routing_min_deg`, etc., read them as Edge-Marker parameters.

## Paper-to-code map

The table refers to the current IEEE manuscript and the accompanying appendix.
**An available runner, a re-rendered figure, and a reproduction of the original
measurements are different things.** The repository does not contain every
original data file, checkpoint, modified baseline, or historical experiment
driver. Do not report a generated-input run as the original paper experiment.

| Paper material | Available entrypoint / artifact | Remaining requirement or limitation |
|---|---|---|
| Main Figure 5: high/low selectivity and composed predicates | `replay_existing prepare/collect`; SIFT-only `figure4a`; `paper/figures/range_label_combined_plot.R` | Replay requires the original queries, attributes, predicates, GT and matching saved indexes/native. The R files contain plotted values, not baseline executables. |
| Main Figure 6: range, label and OCQ | Same existing-index collector; separate current-format `static_inputs` / `static_paper suite`; `paper/figures/single_attribute_combined_plot.R` | Wiki OCQ needs the original 50 `join_*` queries and birthdate attributes. A fresh current-format graph is not a legacy-index replay. |
| Main Table III: baseline-relative speedups | Figure 5 plotted values and EMA result collectors | `main_experiment.sh` runs only EMA; it does not generate the all-system table. Matching baseline observations are required. |
| Main Table V: construction time and size | `static_inputs` / `static_paper suite` build and record EMA indexes | Competing-system builds/services and their original logs are external. Do not combine fresh EMA costs with old results as a newly repeated all-system experiment. |
| Main Figure 7: insert, delete, complete-record replacement | `dynamic.pipeline`, `dynamic.remeasure`, [frozen V11 source/evidence](../paper_handoff/README.md), `paper/figures/ema_dynamic_v11_plot.R` | Large vectors, original inputs/checkpoints and the exact original native binary are not bundled. The current native is not the frozen V11 binary. |
| Appendix: biased range+range selectivity | Predicate generators and the generic build/query entrypoints | No complete frozen per-cell request or dedicated biased-selectivity experiment driver is included. The figure PDF is included for document compilation. |
| Appendix: comparison with HNSWLib and NaviX | `tests/vanilla_postfilter_query.py`, `tests/hnsw_build.py`, legacy NaviX adapters | Requires upstream `hnswlib`, the matching modified NaviX build, and original indexes/inputs; there is no complete all-baseline launcher here. |
| Appendix: parameter sensitivity | `ablation/M_sweep.sh`, `min_deg_sweep.sh`, `ft_bits_sweep.sh` | These generate the current default DNF cells. Their default grid/input files are not claimed to reproduce every historical appendix table or figure. |
| Appendix: Marker FPR | `ft_fpr.py` / `ft_fpr.sh` | The CLI measures an existing index using its corresponding query/GT prefix. The historical Redcaps1M fixture is not bundled; the default Redcaps4M diagnostic is not that table. |
| Appendix: recall@100 | Generic query entrypoint with `K=100` and a suitable `ef_search_list` | Requires matching top100 GT and the original predicates/index; no dedicated frozen recall@100 collection is included. |
| Appendix: component ablation | Published figure PDF; `tests/alblation.py` contains stored observations | `tests/alblation.py` is a formatter, not an executable component ablation. A pinned driver for the four original variants and their original inputs is still missing. |
| Appendix: theoretical analysis | `paper/appendix/APP/app.tex` | Compiles with the bundled driver and figure dependencies; no experiment run is needed. |

Main Tables I/II contain a method comparison and notation, not benchmark runs.
Table IV lists dataset statistics; the exact prepared subsets remain external
inputs, as described below.
The old appendix Dynamic Support table and attribute-only/mark-delete/patch
experiments are retired. Native compatibility APIs still exist, but only the
three operations in `dynamic/README.md` define the current dynamic experiment.

### External inputs and baseline dependencies

Original vectors and their exact row ordering, attribute JSON, query vectors,
predicate JSON, GT, and saved graph files are separate inputs. `DATA_ROOT`
changes paths, not their contents. The source-pinned runners fingerprint these
files and fail on missing or incompatible inputs; they do not silently regenerate
them. SIFT comes from the TexMex corpus, RedCaps from redcaps.xyz, YouTube
features from YouTube-8M, and Wiki inputs from the NaviX dataset, but downloading
an upstream dataset alone does not recover the exact prepared paper files.

The repository ships EMA's native source, not the modified `faiss-navix`,
ACORN, Filtered DiskANN or iRangeGraph source trees used by legacy adapters.
For example, `navix_query.sh` expects
`../code/faiss-navix/build/demos/navix_query_arbi`, and `irange_query.sh`
expects `../code/iRangeGraph/build/tests/search`. These are not ordinary
upstream install commands. Root-level adapters are historical references, not
validated portable entrypoints; notably `acorn_query.sh` currently invokes a
DiskANN search binary and must not be treated as an ACORN reproduction command.
Milvus and VBase additionally require independently deployed services and their
client dependencies. No commit-locked all-baseline installer is supplied.

Do not run large fresh builds just to regenerate a figure. The
[publication bundle](../paper/README.md) documents R/TikZ rendering from the
recorded values, and building the cleaned appendix PDF separately.

## Layout

```
exp_benchmark/
├── conf.sh                       # dataset + index path resolver
├── env.sh                        # keep active Python; optional explicit conda activation
├── attr_generator.sh             # generate attribute JSON (per dataset)
├── predicate_generator.sh        # generate DNF predicate JSON
├── ground_truth_generator.sh     # explicit custom C++ or bundled NumPy GT
├── build_index.sh                # build a single EMA index
├── query.sh                      # run a single EMA query cell
├── main_experiment.sh            # EMA-only generated-workload selectivity grid
├── aggregate.py                  # parse logs → QPS @ recall=0.95
├── replay_existing.py            # original-index load/query/collection; no builds
├── screen_youtube.py             # bounded low-AND screening against original QPS95
├── build_legacy_query_native.sh   # isolated legacy layout/query compatibility extension
├── legacy_query_compat.patch     # original native's legacy layout correction
├── legacy_query_mask_compat.patch # inclusive upper bucket in legacy query Markers
├── index_metadata.py             # read-only serialized layout/sample inspection
├── ablation/
│   ├── M_sweep.sh                # graph degree ablation
│   ├── min_deg_sweep.sh          # FT routing min_deg ablation
│   ├── ft_bits_sweep.sh          # bloom width ablation
│   └── selectivity_specs.sh      # generation defaults, not frozen paper inputs
├── static_inputs.py              # fingerprint original build/Figure6 inputs
├── static_paper.py               # current-format static build/query protocol
├── static_publish.py             # raw-evidence validation and paper CSV export
├── dynamic/                      # canonical three-operation dynamic pipeline
│   ├── reproduce.sh             # one entrypoint for the complete paper run
│   ├── pipeline.py              # preflight, operation selection and publication
│   ├── runner.py                # transactional stages and resume
│   ├── protocol.py              # mixed-DNF inputs, traces and level plan
│   ├── gt.py                    # exact live-set mixed-DNF ground truth
│   ├── measurement.py           # recall-only calibration and paired timing
│   ├── runtime.py               # native identity, checkpoint and NUMA guards
│   ├── aggregate.py             # compatible three-operation paper data
│   ├── canary.py                # isolated native load/add capability probe
│   ├── insert.sh                # single-operation convenience wrapper
│   ├── delete.sh                # single-operation convenience wrapper
│   ├── point_update.sh          # full-record delete + fresh-label insertion
│   └── README.md                # complete dynamic protocol and command reference
├── ft_fpr.sh                     # Edge-Marker FP-rate diagnostic
├── ft_fpr.py
└── logs/                         # all run logs land here (git-ignored)
```

## Prerequisites

1. Use Python 3.10+ and install `hashannlib` and Python FAISS
   (see the top-level README).
2. For new workloads, explicitly choose `GT_BACKEND=numpy` to use
   `tests/groundtruth_bruteforce.py`. The default `cpp` backend instead requires
   the external custom `generate_groundtruth_arbi` binary; upstream FAISS does
   not provide this target. Neither backend is invoked by a read-only replay.
3. Download each dataset to `/mnt/data/mocheng/dataset/<dataset_name>/`
   (paths are configurable via env vars in `conf.sh`).
4. Activate the desired venv/conda environment before running the shell scripts.
   They preserve it by default; `EMA_CONDA_ENV=myenv` explicitly requests conda
   activation and fails if activation is unavailable.

The dynamic pipeline uses the chosen Python interpreter directly and also needs
the Python FAISS package and Linux `numactl`. It does not require the standalone
C++ GT generator: each live-set GT is computed with exact FAISS flat search.

## Data preparation (one-time per dataset)

Before any experiment, materialise the attribute JSON, predicate JSON
and brute-force ground truth for the chosen DNF cell:

```bash
# 1. Generate per-vector synthetic attributes (idempotent; skips if file exists).
dataset=Redcaps_4M attr_type=[0,1] \
    categorical_attr_max_cardinality=21 numerical_max_attr=100000 \
    ./attr_generator.sh

# 2. Generate the DNF predicate file for a selectivity cell.
dataset=Redcaps_4M attr_type=[0,1] \
    dnf_spec='[{"0":0.3,"1":[9]},{"1":[12]}]' dnf_name=or_T10 \
    ./predicate_generator.sh

# 3. Generate brute-force GT for this new workload with the bundled NumPy backend.
dataset=Redcaps_4M attr_type=[0,1] \
    dnf_spec='[{"0":0.3,"1":[9]},{"1":[12]}]' dnf_name=or_T10 \
    GT_BACKEND=numpy ./ground_truth_generator.sh
```

Predicate / GT scripts re-read `conf.sh`, so `attr_type`, `dnf_spec` and
`dnf_name` must match across the three steps. These are generation defaults,
not replacements for the frozen paper inputs. Some saved composite predicates
differ from `ablation/selectivity_specs.sh`; the static rerun below never
regenerates them.
The NumPy generator computes a new exact-distance ranking; ties may differ from
the original custom FAISS GT. Keep the selected backend and inputs with the new
results. A missing C++ binary never triggers an automatic backend fallback.

## Workflow

### Existing-index Figure 5/6 replay (no construction or clustering)

**Use this path to rerun the saved experiments.** Do not run
`main_experiment.sh`, `build_index.sh`, or `static_paper.py suite` for a
query-only replay: those entrypoints can build indexes. The shell query
entrypoint can also generate missing inputs. The replay below fails on missing
files instead; it never generates vectors, attributes, predicates, GT, level
plans, or indexes, and never calls augmentation or saves an index.

The execution chain remains the repository's original query pipeline:
`replay_existing.py` -> `HashANN.load_index()` in `tests/hashann.py` ->
`load_query_data()` / `query_sweep()` in `tests/hashann_query.py` ->
the existing native query API -> `aggregate.qps_at_recall()`.
One worker loads one existing index once and queries all of its cells; workers
run sequentially. `generateAttrIndexes()` initializes the original in-memory
attribute lookup structures, not FAISS clustering or graph construction.

The existing mixed caches are
`<dataset>/hashann/index/index_40_300_arbi_0_1_random_128_edgeFT`.
YouTube/Redcaps range and label caches end in
`index_40_300_arbi_{0,1}_random_256`. Wiki OCQ uses
`neg_correlated/hashann/index/index_40_300_arbi_0_random_256`.
The manifest records the actual serialized format, offsets, Marker width,
M40, efc300, corpus size and dimensions; filenames alone are not compatibility
evidence. Existing graph files are identified by path, inode/device,
size, timestamps and decoded metadata, not a new full-file graph hash.

#### Why a separate legacy loader is needed

The original native misclassifies some unversioned single-attribute caches as
node-Marker layouts. In the existing YouTube cache, for example, the tail
contains **80 edge Marker blocks**, followed by the attribute. The erroneous
node-layout migration copies only one block and moves the attribute offset
from6988 to4460, so queries read the wrong attributes even though the saved
records are correct.

`legacy_query_compat.patch` recognizes the saved layout geometry and preserves
all graph, vector, label, Marker and attribute offsets. It applies only to
original commit `28e07e33cb2dd4e7c173ca1030e6c5bbd7f4f7bb`; it does not alter
construction, ranking, queries, or index files. The helper builds that small
extension in an isolated directory and records the source, patch and binary
hashes. **Compiling this loader is not rebuilding a benchmark index.**

The helper also applies `legacy_query_mask_compat.patch`. The saved legacy
numeric Markers use ceiling-bucket encoding, but the old AND and DNF query
mask loops excluded the bucket containing the inclusive upper bound. This can
reject valid neighbors even though their exact attributes satisfy the query;
a range lying inside one bucket produced an empty mask. Both loops now include
that bucket, retaining their byte bounds checks. The exact predicate, saved
bucket boundaries, graph and stored Markers are unchanged. Marker filtering
stays enabled; this is not a Marker-off fallback or the V11 floor-bucket format.
Old serialized overflow/ownership limitations are not repaired by this query
patch, so this compatibility path is not a general conversion to V11.

This is a **saved-index replay with layout and query-mask corrections**, not
a V11 experiment. Current V11 requires its own compatible formats9/10 and
different graph/Marker semantics. Neither native's guard is weakened, and no
old index is retagged. Keep this output separate from completed V11 SIFT and
dynamic results; do not relabel it as a current-V11 paper update.

#### Reproduction commands

Run from the repository root. Use an absolute interpreter path, and use the
same interpreter for the native build and replay. Its existing environment
needs the project build dependencies, NumPy and the original helper's FAISS
Python dependency. Linux `numactl` is required by this host-specific launcher.
The native-build and result directories must be new.

```bash
PY=/absolute/path/to/the/chosen/venv/bin/python
AUDIT=/absolute/path/to/new-replay-audit
NATIVE=/absolute/path/to/new-legacy-query-native
RUN=/absolute/path/to/new-existing-index-results
mkdir -p "$AUDIT"

bash exp_benchmark/build_legacy_query_native.sh "$PY" "$NATIVE"

"$PY" -m exp_benchmark.replay_existing prepare \
  --data-root /mnt/data/mocheng/dataset \
  --native-provenance "$NATIVE/provenance.json" \
  --request "$AUDIT/request.json"

OMP_NUM_THREADS=32 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE \
OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -m exp_benchmark.replay_existing collect \
  --request "$AUDIT/request.json" --output "$RUN"
```

The request freezes all paths, input/source/native hashes and query settings
before measurement. Do not edit these sources or inputs during a run. This
entrypoint deliberately refuses to overwrite an existing output directory;
there is no log-text-based automatic resume. Failed runs retain their raw
artifacts and explicit failure status rather than pretending to be complete.

All queries use one thread pinned to physical CPU24, `k=10`, `ef_top=1`,
routing16, FT enabled, thresholds `[0.0001,0.0001,0.0001]`, no augmentation and
no tail backfill. The manifest contains the fixed25-point ef grid from10 to4000.
Each ef has a three-query warmup and three full-batch timing repetitions.
Every repetition is retained and the median is used; no fastest-sample or
CPU-clean-round selection is performed. The sweep stops at the first measured
ef reaching95% recall, not on QPS. AND/single-attribute queries retain the
original `hybrid_knn_query_with_stats` API where available; DNF uses
`hybrid_knn_query_dnf`. The statistics binding is inherently sequential and
does not accept `num_threads`; the other APIs receive `num_threads=1`.
This timed API differs from the separate V11
`figure4a.py` timing protocol and is recorded explicitly.

The initial5-repeat, loader-only replay was stopped on2026-09-16 after low-selectivity
workloads became much slower than the historical results and some failed to
reach95% recall. Its immutable manifest and measurements still declare5
repetitions; they are not rewritten as3-repeat results. Correcting repetition
count reduces total work, but does not explain or fix lower per-batch QPS or
missing recall. Do not publish this stopped run as a completed reproduction.
The collector now rejects a loader-only native provenance rather than silently
restarting that known-bad numerical sweep. Rebuild only the small compatibility
extension with both patches and use a new request/output; preserve the earlier
binary, manifest and measurements.

For a bounded YouTube1% AND smoke run after building the corrected native,
the original query entrypoint can run just `ef=50`, with Marker enabled and
three repetitions. This needs only the YouTube files, not the other datasets:

```bash
YOUTUBE_ROOT=/mnt/data/mocheng/dataset/youtube1m
POINT_JSON=/absolute/path/to/new-youtube-and1-point.json
PYTHONPATH="$NATIVE/native" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=24 --membind=0 \
  "$PY" tests/hashann_query.py \
  --name HNSW --metric IP --N 1000000 --dim 1024 --M 40 --efConstruction 300 \
  --k 10 --threads 1 --attr_type_list '[0,1]' --n_query_to_use 1000 \
  --data_path "$YOUTUBE_ROOT/rgb.fvecs" \
  --query_path "$YOUTUBE_ROOT/rgb_query.fvecs" \
  --attr_path "$YOUTUBE_ROOT/label/arbi_0_1_random/attr_arbi_0_1_random.json" \
  --qrange_path "$YOUTUBE_ROOT/label/arbi_0_1_random/predicate_arbi_0_1_[0.1,9].json" \
  --gt_path "$YOUTUBE_ROOT/label/arbi_0_1_random/gt_arbi_0_1_[0.1,9].json" \
  --index_cache_path "$YOUTUBE_ROOT/hashann/index/index_40_300_arbi_0_1_random_128_edgeFT" \
  --ef_search '[50]' --ef_top 1 --ft_routing_min_deg 16 --use_ft true \
  --query_repeats 3 --result_json "$POINT_JSON"
```

The2026-09-16 one-point run used the source-pinned collector with the same
query settings and original files: recall rose from92.81% to96.07% at `ef=50`,
with median QPS64.50 before and66.05 after the query-mask correction.
Both measurements kept Marker enabled. These are observed fixed-ef points,
not an interpolated QPS95 curve or evidence that every dataset has recovered.
The global rerun remains separate from this bounded regression.

For the subsequent fine refinement, keep `ef_top=1`, Marker enabled and all
other arguments above, replace the ef list with
`[30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]`,
and add `--target_recall 0.95`. Each point still uses three repetitions and
the scan stops at its first qualifying point. The2026-09-16 run stopped at
ef37: ef36 measured94.85% recall/87.02 QPS and ef37 measured95.02%/83.99 QPS.
The original interpolation rule gives84.35 QPS at95% between those points.
This is a YouTube AND1% result only, not completion of the other workloads.

#### Screen for a gain above5% before doing more reruns

`screen_youtube.py` compares only the original YouTube AND1/2/3/5/7/10%
points with their original Figure5(b) plotting values, never with the failed
loader-only replay. It reuses the completed corrected-native1% result after
checking its raw evidence and the matching input/index identities, then loads
the same index once for the other five cells. Marker stays enabled,
`ef_top=1`, routing16, `k=10`, and the original1,000-query prefixes are retained.

Calibration uses one deterministic full batch per sampled ef on the fixed grid
`[10,20,40,80,120,160]`, then recall-only bisection to an adjacent integer-ef
bracket. It does not claim a globally first crossing on a potentially
nonmonotonic curve. Only the final bracket endpoints are timed with **three**
full repetitions each; their output hashes must match calibration. Calibration
throughput is not used to select ef or report improvement. The final medians
use the existing QPS-at-recall interpolation.

Run from the repository root with the interpreter matching the corrected
native recorded in the completed1% reference:

```bash
OMP_NUM_THREADS=32 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE \
OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=0-31 --membind=0 \
  "$PY" -m exp_benchmark.screen_youtube \
  --matrix /path/to/original-input-matrix/manifest.json \
  --reference /path/to/completed-youtube-and1-fine-run \
  --baseline /path/to/paper/CH5/figures/range_label_sel_recall_plot.R \
  --output /path/to/new-screen-output
```

The original matrix supplies paths only; it is not executed. The corrected
native/settings come from the completed fine-run reference. The output retains
the baseline/source fingerprints, raw calibration and timing arrays, all three
timing samples, `summary.csv`, and the per-cell `exceeds_5pct` decision.
An improvement of exactly5% is not enough. Cells with no gain or a gain at most5%
do not need another rerun; the screen never starts a global suite. Unreached
recall at the fixed ceiling fails explicitly rather than extending to ef4000.
The manifest separates the cached `reference_settings` from the fixed screening
settings; each cell's `measurement.points` records its effective timing efs.
For a cell above the gate, an independent `replay_existing collect` confirmation
can retain just that original cell and its selected adjacent efs, with the same
native/settings and three repetitions. Retain the screening evidence and report
the confirmation regardless of whether it is faster; do not pick the better run.

The September16 screen found only AND10% above the gate. Only that cell was
confirmed with the original collector at ef73/74; AND1/2/3/5/7% and the global
queue were not restarted.

The original prefixes and workload families are preserved:

| Workload | Used queries | Saved inputs |
|----------|-------------:|--------------|
| Figure5 AND, all four datasets |1000| original high/low AND predicates |
| Figure5 composite, YouTube |100| original six DNF files |
| Figure5 composite, other datasets |1000| original six DNF files |
| Figure6 YouTube range / label |1000 /100| single-attribute range / label files |
| Figure6 Redcaps range / label |1000 /1000| single-attribute range / label files |
| Figure6 Wiki OCQ |50| `join_*` queries and birthdate predicates/GT |

The six saved composite label pairs are19/18,18/17,17/16,16/14,14/12 and9/12,
not six copies of the current generator's default predicate. The identical
10% AND workload is measured once and used in both Figure5(a)/(b).
YouTube uses raw IP vectors; no normalization is introduced. Wiki OCQ uses
`fvecs/wiki_15.4M_birthdate.json` and the byte-identical saved `join_*` query
files, never the general Wiki query file.

`tests/hashann_query.py` now selects GT **columns** (`truth[:Nq,:k]`), not a
reshape of stored top100 rows. Warmup and batch errors propagate; there is no
silent per-query fallback. Missing/sentinel/duplicate/out-of-range labels,
predicate violations and changes between read-only repetitions fail the cell.

Outputs include the immutable `manifest.json`, source snapshots, per-index
worker/status logs, per-cell original-format logs and JSON, raw returned-label
and distance NPZ files, every timing repetition, and `summary.csv` /
`summary.json`. The summary has **107 plotted rows**:72 Figure5,29 Figure6
primary95%, and6 YouTube-range90% auxiliary points. Interpolation reuses
`aggregate.qps_at_recall()`, with the first appropriate bracket for each target.
If ef10 already exceeds the target, `qps_at_target` is empty and
`reported_qps` is explicitly marked `minimum_above` with its observed recall.
Unreached/failed cells remain explicit and the collection exits nonzero.
A setup failure or worker crash stops the queue before loading more indexes.
Collection neither edits the paper nor replaces any existing result.

### 1. Build the EMA index (per-dataset)

```bash
dataset=Redcaps_4M ./build_index.sh
```

This builds the edge-level FT HNSW index with the recommended defaults
(`M=40`, `efc=300`, `ft_bits=128`, `edgeFT=true`).

### 2. Main 4-dataset experiment

```bash
GT_BACKEND=numpy ./main_experiment.sh
```

Runs EMA only at six selectivity points (1, 2, 3, 5, 7, 10 %) on each
dataset. Logs land in `logs/main/`. Aggregate with:

```bash
python aggregate.py logs/main
```

### 3. Ablations (run on Redcaps_4M by default; override with `dataset=...`)

| script                      | axis swept                              |
|-----------------------------|-----------------------------------------|
| `ablation/M_sweep.sh`       | `M ∈ {16, 32, 40, 64}`                  |
| `ablation/min_deg_sweep.sh` | `min_deg ∈ {0, 5, 8, 15, 20, 30, 40, 80}` |
| `ablation/ft_bits_sweep.sh` | `ft_bits ∈ {32, 64, 128, 256}`          |

Each script writes per-cell logs under `logs/<axis>_sweep/`.
Override lists with space-separated values, for example
`M_VALS="16 40" GT_BACKEND=numpy ./ablation/M_sweep.sh` or
`EMA_DATASETS="sift10m youtube_rgb" GT_BACKEND=numpy ./main_experiment.sh`.
These are parameter sweeps, not the four-variant component ablation.

### Figure 5 SIFT10M existing-input trials

`figure4a.py` is a read-only query runner. Its default configuration remains
panel (a) of current Figure 5 (previously Figure 4): six existing legacy AND
predicate/GT pairs at nominal 10, 20, 40,
60, 80, and 100% selectivity. It does not generate data, GT, or a new graph.

Set `"figure_panel": "b"` in the audited request for the existing low-selectivity
AND files, or `"figure_panel": "c"` for the existing
`((range AND labels) OR labels)` composite files. Both use the six nominal
1, 2, 3, 5, 7, and 10% cells. The saved composite predicates have different
categorical conditions across cells; do not substitute or regenerate them from
the current `main_experiment.sh` defaults. A larger stored GT requires an explicit
`ground_truth_rows` declaration for that cell; the runner uses and validates the
unchanged first 1000 rows corresponding to the query/predicate prefix.

Run `python -m exp_benchmark.figure4a --config <audited-request.json>
--output <new-result-directory>` under the NUMA/CPU environment required by
`dynamic.runtime`. The request pins the current native and compatible full10M
checkpoint, original input hashes, all six predicate/GT paths, historical
reference values, and known comparison limitations. A legacy cache rejected
by the native loader must not be retagged or silently substituted.

Panels (a)/(b) use regular `hybrid_knn_query`; panel (c) uses
`hybrid_knn_query_dnf`. All use no timed statistics or edge augmentation, one
physical core, `k=10`, and the requested routing degree.
Recall-only calibration targets 95%; endpoint QPS is the median of the first
seven CPU-clean complete rounds. The original inputs and all plotted-reference
values remain unchanged. `manifest.json`, calibration/timing records,
`results.json`, and `status.json` retain provenance and distinguish a current
checkpoint trial from an unproven reproduction of an old serialized index.
Actual selectivity is counted separately from nominal cell labels, including
DNF branch overlap. Failed runs keep completed cells explicitly marked partial.

`python -m exp_benchmark.figure4a_api_control --reference <completed-trial>
--output <new-control-directory>` replays that trial's frozen ef brackets.
It pairs regular and `with_stats` queries on the same graph and physical core,
requires identical returned labels/distances, and measures API overhead without
rebuilding an index or regenerating GT.

`figure4a_work_control.py` compares recall and search work without using QPS or
CPU-clock-dependent timing as evidence. Run it separately for each explicitly
identified native and its own compatible index:

```bash
python -m exp_benchmark.figure4a_work_control \
  --reference <completed-SIFT10M-trial> \
  --native-dir <version-specific-native-directory> \
  --native-sha <native-sha256> \
  --source-header <matching-hnswalg.h> \
  --index <version-compatible-index> \
  --output <new-work-control-directory>
```

Use the trial's NUMA-0/CPU-0--31 launcher. This SIFT10M diagnostic pins queries
to CPU 24 and defaults to T10 and T100 (`--cells` selects other reference cells).
Both indexes must contain the full zero-deletion corpus with M40, efc300 and
128-bit edge Markers. It records the first integer ef reaching complete 95%
recall, mean distance computations, expanded nodes, and repeated-result hashes.
Counter definitions must be comparable across the selected source versions.
This can establish a hardware-independent code/index effect, but does not
identify an unrecorded historical figure binary or isolate individual graph
changes. Legacy formats are loaded only by their matching native, never retagged.

For both controls, consult `status.json` before interpreting intermediate
`results.json` files. Failed runs may retain completed cells; unaccepted timing
rounds and remaining cells must not be reported as completed measurements.

### Separate fresh-build study (not the existing-index replay)

The fresh-build study below is **not required for rerunning Figure5/6 on saved
indexes**. Its earlier construction/NN2 queue was stopped and superseded by the
query-only path above. Completed artifacts remain separate; never resume this
queue merely to collect query data. Use it only when a new graph or construction
cost study is specifically requested.

`static_inputs.py` fingerprints the original files and creates an immutable
`initial-request.json`. Its initial queue includes fresh mixed-index construction
costs for all four datasets and the displayed Figure 6 range, label, and OCQ
queries. The already measured SIFT Figure 5 curves remain separate query
evidence; a fresh SIFT cost build is not a replay of dynamic mutation timings.

`static_paper.py suite` executes jobs sequentially so index builds do not overlap
query measurements. Each worker verifies the original file identities, complete
source-label/attribute mapping, serialized format and vector samples. The
protocol fixes `M=40`, `efc=300`, 128 bits **per attribute**, routing degree16,
`ef_top=1`, one query thread and no augmentation or tail backfill.

The original query prefixes are not uniform: the saved YouTube range workload
has1,000 queries, whereas its label workload has100. Redcaps range/label use1,000.
Wiki OCQ uses the50 saved `neg_correlated/join_*_embedding.fvecs` queries and
`fvecs/wiki_15.4M_birthdate.json`, not the regular Wiki query file or synthetic
numeric attributes. The five saved OCQ query files have identical payloads.
Every cell declares both stored predicate/GT dimensions and its used prefix.

Vectors are used exactly as stored. In particular, YouTube is **raw IP**, not
cosine: its vectors are not unit-normalized, and normalizing them changes the
original GT. Redcaps and Wiki embeddings are already normalized. Native IP
distance is `1-dot`, which has the same ranking as negated inner product.

Fresh builds use `tests/hashann.py` with32 threads, seed1234,25 FAISS iterations
and at most2,560,000 training rows. FAISS IP scores are maximized when choosing
representatives; L2 distances are minimized. Empty clusters receive no
representative, never source row-1. New `*_clustering_nn2.npz` caches pin vector
payload, metric, full row count, dimension, seed, threads, FAISS version and
selection policy. Unversioned level caches and old graph edges/Markers are not
reused. Construction cost excludes provenance hashing and save I/O; those times
are recorded separately. Each construction phase has a20-hour limit.

Run from the repository root with the interpreter matching the pinned native:

```bash
python -m exp_benchmark.static_inputs \
  --data-root /path/to/datasets --output /path/to/new-audit \
  --native-dir /path/to/pinned-native --native-sha256 "$NATIVE_SHA256"

OMP_NUM_THREADS=32 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE \
OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=0-31 --membind=0 \
  python -m exp_benchmark.static_paper suite \
  --request /path/to/new-audit/initial-request.json --output /path/to/new-run
```

This timing protocol is specific to the paper host: NUMA0, physical query
cores8/16/24 and3.3GHz within1%. It retains every attempt and uses the first seven
complete CPU-clean rounds, never the fastest samples. Calibration finds the
first integer ef reaching95%; the displayed YouTube range90% auxiliary curve
is measured separately from the same calibration. If minimum legal ef10 already
exceeds a target, only observed throughput at that recall is reported, without
extrapolation. Unsupported recall and noisy timing fail explicitly.

Use `--resume` only with the unchanged request, sources and artifacts. Completed
jobs still enter worker validation, and cached cell measurements must match the
same checkpoint, native, manifest and query payload. Old and failed attempts are
retained. `static_publish.py` checks raw calibration arrays, GT recall, CPU
telemetry and median/interpolation arithmetic before emitting the complete paper
CSV; incomplete suites are not publication input.

## Configuration knobs

The shell generation/ablation scripts accept the following environment knobs.
The source-pinned static request is not changed by these overrides.

| variable               | default            | description                          |
|------------------------|--------------------|--------------------------------------|
| `dataset`              | `Redcaps_4M`       | one of the four supported datasets   |
| `M`                    | `40`               | HNSW max degree                      |
| `ef_construction`      | `300`              |                                      |
| `ef_search_list`       | `[10,20,40,80,150,300]` | comma-separated efSearch values |
| `K`                    | `10`               | top-K                                |
| `threads`              | `32`               |                                      |
| `ft_bits`              | `128`              | FT width per attribute (multiple of 8) |
| `edge_level_ft`        | `true`             | per-edge FT layout (EMA default)     |
| `use_ft`               | `true`             | enable FT-based routing              |
| `ft_routing_min_deg`   | `16`               | static min #edges to expand per node |
| `dnf_spec` / `dnf_name`| see selectivity_specs.sh | predicate JSON + filename tag  |
| `DATA_ROOT`            | `/mnt/data/mocheng/dataset` | base path for all datasets   |
| `GT_BACKEND`           | `cpp`                  | external custom C++ GT or explicit bundled `numpy` generation |
| `EMA_CONDA_ENV`        | unset                  | preserve active Python unless explicitly selecting conda |

`threads=32` is the construction default. `query.sh` defaults its index-loading
argument to one thread before resolving the shared configuration; the canonical
query sweep explicitly uses one query thread.

### Marker false-positive diagnostic

For a new diagnostic on a compatible existing index, pass the index and query
paths explicitly rather than relying on the historical node-Marker default:

```bash
python exp_benchmark/ft_fpr.py \
  --dataset Redcaps_4M --data_root "$DATA_ROOT" \
  --index_path "$FPR_INDEX" --query_file "$FPR_QUERIES" \
  --attr_type 0,1 --N 4000000 --dim 512 --metric ip \
  --K 10 --n_query 1000 --ef_search_list 10,50,200 \
  --out_json /absolute/path/to/new-fpr-results.json
```

Run this command from the repository root. `FPR_INDEX`, `FPR_QUERIES` and the
predicate/GT files must describe the same corpus. The native build must match
the serialized index format. The diagnostic validates the requested query
prefix and uses the first `K` GT columns, even when a file stores top100.
Short inputs, invalid returned IDs and malformed GT fail instead of silently
changing the denominator. These FT counters are empirical query diagnostics,
not estimates of the separate analytical Case-(1)/Case-(2) bounds.

## Output

Each run prints a `Final results (ef_search, recall, QPS, cmps):` block.
`aggregate.py` interpolates these to `QPS @ recall = 0.95`.

## Legacy generation and ablation entrypoints

The following regenerate workloads from current defaults. Use the existing-index
replay above for a query-only Figure5/6 rerun with preserved inputs and graphs.

```bash
# EMA-only generated workload (4 datasets × selectivity grid)
GT_BACKEND=numpy ./main_experiment.sh
python aggregate.py logs/main > main_table.tsv

# ablations
GT_BACKEND=numpy ./ablation/M_sweep.sh        && python aggregate.py logs/M_sweep
GT_BACKEND=numpy ./ablation/min_deg_sweep.sh  && python aggregate.py logs/min_deg_sweep
GT_BACKEND=numpy ./ablation/ft_bits_sweep.sh  && python aggregate.py logs/ft_bits_sweep
```

## Dynamic updates

The paper dynamic result contains exactly **insert**, **delete**, and
**point_update**. Attribute-only and legacy mark-delete/patch experiment
entrypoints are no longer part of this pipeline; their native APIs remain
available. Do not aggregate old categorical experiments as mixed-DNF results.

| Operation | Initial live points | Five stages |
|-----------|--------------------:|-------------|
| insert | first 5M source rows | append the next 1M original records, reaching 10M |
| delete | canonical full 10M index | delete 1M per stage in the seed-12345 permutation, reaching 5M |
| point_update | same first-5M graph as insertion | replace consecutive 1M logical slots per stage, keeping 5M live |

In a combined run, deletion starts from the freshly completed insertion
checkpoint, not the old prebuilt numeric index. This avoids a second 10M build
and makes the insertion endpoint and deletion starting graph identical.
The original first-5M prefix remains the starting graph for point update.

Point replacement takes both the vector and its complete mixed attribute record
from source row `5M+i`, retires the old label `i`, and inserts at fresh label
`5M+i`. The logical slot is still `i`. This is an explicitly versioned trace,
not the former random single-category update experiment. Retired slots are
never reused, and the initial prefix graph is not a random surviving subset
from deletion.

All three operations use the canonical SIFT10M numerical-plus-categorical
layout, the original first 1,000 query vectors and the corresponding
`predicate_dnf_or_T10.json` predicates:
`(numeric in inclusive [lo, hi] AND label9) OR label12`.
The initial full-data mean selectivity is approximately 9.793%; actual live-set
selectivity is recorded at every stage. Index settings are `M=40`,
`ef_construction=300`, edge-level FT and 128 bits per attribute.
Queries always use **one thread**, `k=10`, `ef_top=1`, FT/routing enabled,
`min_deg=8` and no tail backfill. The canonical target is restored to **95%
recall**. Dynamic routing8 still differs from static routing16; this recall-only
revision changes neither routing nor construction, native, data or trace settings.

The user-authorized recall95 revision remeasures the18 saved compatible V11
checkpoints and exact GT with the same frozen native
`608e9ae692fa2fd9d5ec160a370a63b90419356c6b64d67285f10d6bd1e4f84e`.
Use the separate `python -m exp_benchmark.dynamic.remeasure --help` interface,
not a rebuild or mutation replay. It writes new query evidence and explicitly
reuses original maintenance timing provenance. Original QPS90 results and source
snapshots stay frozen; they are not renamed QPS95.

For a **separately approved fresh run**, use the following from the repository
root after selecting/building its pinned compatible native in the chosen
interpreter. These are not steps for the checkpoint recall revision:

```bash
PY="${PY:-python3}"
SIFT_ROOT="${DATA_ROOT:-/mnt/data/mocheng/dataset}/sift10m"
NATIVE_DIR="$("$PY" -c 'from pathlib import Path; import hashannlib; print(Path(hashannlib.__file__).resolve().parent)')"
NATIVE_SHA="$("$PY" -c 'from pathlib import Path; import hashlib, hashannlib; print(hashlib.sha256(Path(hashannlib.__file__).read_bytes()).hexdigest())')"
RUN="$SIFT_ROOT/hashann/benchmarks/canonical_dynamic_$(date +%Y%m%d_%H%M%S)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=32 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE

# Small real-data workflow before a full run; never publishes paper QPS.
numactl --cpunodebind=0 --membind=0 "$PY" -B -m exp_benchmark.dynamic.pipeline run \
    --data-root "$SIFT_ROOT" --output "${RUN}_pilot" \
    --native-dir "$NATIVE_DIR" --expected-sha256 "$NATIVE_SHA" \
    --numa-node 0 --pilot-size 1000

# Fresh full run of all three operations.
numactl --cpunodebind=0 --membind=0 "$PY" -B -m exp_benchmark.dynamic.pipeline run \
    --data-root "$SIFT_ROOT" --output "$RUN" \
    --native-dir "$NATIVE_DIR" --expected-sha256 "$NATIVE_SHA" \
    --numa-node 0 --threads-build 32 --threads-update 32 --threads-gt 32 \
    --frequency-khz 3267000,3333000 \
    --publish-current "$SIFT_ROOT/hashann/benchmarks/dynamic-paper-current.json"
```

Choose a NUMA node and core/frequency controls appropriate to the machine.
Measurements on this host use node0 and `--frequency-khz 3267000,3333000`.
The software does not change host performance settings or choose samples by
their QPS.

Legacy attribute-bearing indexes and the earlier `4e6`/`6fa` results are not inputs
to the corrected paper run. Numeric bucket mapping, inclusive Marker bounds,
and pruning-witness ownership after neighbor reordering were repaired; candidate
ranking now uses vector distance only. Those incompatible graphs require all
three operations to be rebuilt/rerun on the same native; query retiming cannot
repair their graph evolution. Ownership-correct formats7/8 also require rebuilding because they did not
record the construction ordering policy. The canonical complete-Codebook
node/edge formats are9/10. Current native formats11/12 preserve partial
categorical mappings, but are not accepted by this canonical pipeline.
The
numeric and owner semantic versions are both2, distance-order version is1, and
initialized pickle state version is4. Regular benchmark cache names end in
`_mo2_do1`, or `_nb2_mo2_do1` for numeric layouts.
Raw vectors, attributes, query predicates and exact GT remain unchanged.

Deletion uses the saved native `ef_construction`, with no separate deletion-width
override. Maintenance timing includes native graph/Marker repair and the
distance-free scrub, but excludes GT, query measurements, loading and checkpoint
IO. With one worker, point deletions follow input order; multiple workers
execute complete point tasks in parallel, so graph interleavings are not
deterministic. The call waits for all local/deferred repairs and the final
adjacency scrub. Queries never overlap mutations.
The candidate pool is an intentional coverage/runtime budget; larger pools may
improve discovered support, but no exhaustive support scan is required. Unfiltered
maintenance candidates are ranked solely by vector distance; predicate/Marker
checks do not add an attribute-similarity term to that ranking.

Per-stage exact GT, actual query outputs, operation traces, checkpoints and
immutable source/native/input identities accompany the results. Query ef is
calibrated using recall alone, scanning every integer ef to the first recall>=95%.
If minimum legal `ef=k` already exceeds95%, `qps95` is null and the observed
recall/QPS/ef are reported explicitly; no extrapolation is used. Failure to
reach95% stops calibration without a timing fallback. Pilot QPS95 remains null.
A noise failure pauses for remeasurement of the saved
state, not repeated mutation. Resume with the same arguments plus `--resume`.

Only a complete compatible **fresh recall95** three-operation run publishes `paper.json`,
`paper.tsv` and, with `--latex`, `paper.tex`. Publication atomically updates the
chosen current-manifest pointer. Pilot, partial, categorical, attribute-only and
stale outputs cannot enter that table. The active protocol is
`sift10m-mixed-dnf-dynamic-rebuild-v5`, with operation schema
`canonical-dynamic-operation-v4` and fresh paper/current schema
`canonical-dynamic-paper-current-v6`. They explicitly record `target_recall=0.95`
and use `qps95`, `paired_initial_qps95` and `same_recall_qps95_change`.
Old recall90 manifests cannot resume or aggregate as recall95. The separate
compatible checkpoint revision records reused maintenance timing and new query
provenance; it is not accepted as a fresh mutation replay. The static
`aggregate.py` retains its 95% recall default.

See [dynamic/README.md](dynamic/README.md) for the complete CLI, deterministic
level plan, native capability probe, CPU/NUMA requirements and resume protocol.
Benchmark commands do not delete old data. Cleanup requires separate explicit
approval and must preserve raw inputs, original base indexes and every artifact
referenced by current data, including reused checkpoints/GT and original mutation
provenance. The recall95 revision does not authorize deleting frozen recall90
results or their handoff.

## Edge-Marker false-positive rate (FT diagnostic)

`ft_fpr.sh` loads an existing EMA index and, for each (selectivity, ef)
cell, counts how many neighbours were admitted by the Edge-Marker check
but then failed the actual predicate (false positives). This is the
filter quality metric the paper reports as "FT FP rate".

```bash
# Defaults: Redcaps_4M, M=40, efc=300, ft_bits=128, attr_type=[0,1]
# Selectivities: 1% / 5% / 60%, ef ∈ {10, 50, 200}
bash exp_benchmark/ft_fpr.sh
```

Overrides:

```bash
dataset=Redcaps_4M M=40 ft_bits=128 \
    ef_search_list="10,50,200" \
    selectivities="[0.1,9]:1%;[0.177,7]:5%;[0.75,2]:60%" \
    bash exp_benchmark/ft_fpr.sh
```

Note the **semicolon** separator between selectivity tokens (the
predicate strings themselves contain commas, e.g. `[0.1,9]`). Each token
is `<predicate_str>:<short_label>`; the predicate/GT JSON files are
expected at `<label_root>/predicate_arbi_0_1_<predicate_str>.json` and
`gt_arbi_0_1_<predicate_str>.json`.

The script writes both a human-readable log and a `results.json` to
`logs/ft_fpr_<dataset>_<timestamp>/`.
