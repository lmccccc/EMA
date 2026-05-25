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

## Layout

```
exp_benchmark/
├── conf.sh                       # dataset + index path resolver
├── env.sh                        # conda activation
├── attr_generator.sh             # generate attribute JSON (per dataset)
├── predicate_generator.sh        # generate DNF predicate JSON
├── ground_truth_generator.sh     # generate brute-force GT (FAISS)
├── build_index.sh                # build a single EMA index
├── query.sh                      # run a single EMA query cell
├── main_experiment.sh            # 4 datasets × 6 selectivities (main results)
├── aggregate.py                  # parse logs → QPS @ recall=0.95
├── ablation/
│   ├── M_sweep.sh                # graph degree ablation
│   ├── min_deg_sweep.sh          # FT routing min_deg ablation
│   ├── ft_bits_sweep.sh          # bloom width ablation
│   └── selectivity_specs.sh      # shared DNF specs for the 6 sel points
├── dynamic/                      # dynamic update benchmarks
│   ├── insert.sh                 # incremental insertion (build N0, then +step)
│   ├── delete.sh                 # mark_delete only (no graph repair)
│   ├── delete_patch.sh           # mark_delete + selective edge patching
│   ├── attr_update.sh            # attribute-only updates (no vector change)
│   ├── point_update_build.sh     # build a capacity-headroom index for point_update
│   ├── point_update.sh           # vector+attribute updates (mark_delete + add)
│   ├── incremental.py
│   ├── delete.py
│   ├── delete_patch.py
│   ├── attr_update.py
│   ├── point_update.py
│   └── update_common.py          # shared IO / GT helpers
├── ft_fpr.sh                     # Edge-Marker FP-rate diagnostic
├── ft_fpr.py
└── logs/                         # all run logs land here (git-ignored)
```

## Prerequisites

1. Build the project and install `hashannlib` (see top-level README).
2. Build the FAISS-based GT generator
   (`generate_groundtruth_arbi` under `faiss/build/demos`).
3. Download each dataset to `/mnt/data/mocheng/dataset/<dataset_name>/`
   (paths are configurable via env vars in `conf.sh`).
4. Activate a conda env that has `hashannlib`, `numpy`, etc. Default name
   is `py310`; override via `EMA_CONDA_ENV=myenv`.

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

# 3. Generate brute-force ground truth (FAISS).
dataset=Redcaps_4M attr_type=[0,1] \
    dnf_spec='[{"0":0.3,"1":[9]},{"1":[12]}]' dnf_name=or_T10 \
    ./ground_truth_generator.sh
```

Predicate / GT scripts re-read `conf.sh`, so `attr_type`, `dnf_spec` and
`dnf_name` must match across the three steps. See
`ablation/selectivity_specs.sh` for the six DNF cells used in the paper.

## Workflow

### 1. Build the EMA index (per-dataset)

```bash
dataset=Redcaps_4M ./build_index.sh
```

This builds the edge-level FT HNSW index with the recommended defaults
(`M=40`, `efc=300`, `ft_bits=128`, `edgeFT=true`).

### 2. Main 4-dataset experiment

```bash
./main_experiment.sh
```

Runs EMA against six selectivity points (1, 2, 3, 5, 7, 10 %) on each
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

## Configuration knobs

All knobs can be overridden via environment variables:

| variable               | default            | description                          |
|------------------------|--------------------|--------------------------------------|
| `dataset`              | `Redcaps_4M`       | one of the four supported datasets   |
| `M`                    | `40`               | HNSW max degree                      |
| `ef_construction`      | `300`              |                                      |
| `ef_search_list`       | `[10,20,40,80,150,300]` | comma-separated efSearch values |
| `K`                    | `10`               | top-K                                |
| `threads`              | `32`               |                                      |
| `ft_bits`              | `128`              | FT bloom width (multiple of 8)       |
| `edge_level_ft`        | `true`             | per-edge FT layout (EMA default)     |
| `use_ft`               | `true`             | enable FT-based routing              |
| `ft_routing_min_deg`   | `8`                | min #edges to expand per node        |
| `dnf_spec` / `dnf_name`| see selectivity_specs.sh | predicate JSON + filename tag  |
| `DATA_ROOT`            | `/mnt/data/mocheng/dataset` | base path for all datasets   |

## Output

Each run prints a `Final results (ef_search, recall, QPS, cmps):` block.
`aggregate.py` interpolates these to `QPS @ recall = 0.95`.

## Reproducing the paper tables

```bash
# main table (4 datasets × selectivity grid)
./main_experiment.sh
python aggregate.py logs/main > main_table.tsv

# ablations
./ablation/M_sweep.sh        && python aggregate.py logs/M_sweep
./ablation/min_deg_sweep.sh  && python aggregate.py logs/min_deg_sweep
./ablation/ft_bits_sweep.sh  && python aggregate.py logs/ft_bits_sweep
```

## Dynamic updates

All four dynamic operations are exercised on the same EMA index. Run on
the categorical-attribute layout (`attr_type=[1]`); paths come from
`conf.sh` and a 10 % selectivity predicate is used by default.

```bash
# 1. Incremental insertion: build N/2 then add +1M repeatedly up to N.
dataset=sift10m attr_type=[1] ./dynamic/insert.sh

# 2. Mark-delete only (no graph repair) on an existing index.
dataset=sift10m attr_type=[1] \
    INDEX_PATH=<path to existing index> \
    ./dynamic/delete.sh

# 3. Mark-delete + selective edge patching at chosen stages.
dataset=sift10m attr_type=[1] \
    INDEX_PATH=<path to existing index> \
    PATCH_AT=2,3,4 ./dynamic/delete_patch.sh

# 4. Attribute-only updates (no vector change).
dataset=sift10m attr_type=[1] \
    INDEX_PATH=<path to existing index> \
    ROUNDS=5 ./dynamic/attr_update.sh

# 5. Vector+attribute (point) updates: mark_delete old internal id + add a
#    new vec+attr at a fresh internal id.
#
#    point_update.py defaults to an in-process build (working around a
#    pre-existing C++ load_index bug where add_items at a new internal id
#    segfaults on a saved-then-loaded index). It uses attr_update's
#    single-categorical predicate layout, so use `attr_type=[1]` with an
#    all-categorical DNF spec:
#
#    dataset=sift10m attr_type=[1] \
#        dnf_spec='[{"0":[9]},{"0":[12]}]' dnf_name=or_T10_cat \
#        UPDATE_N=5000000 UPDATE_STEP=1000000 ROUNDS=5 \
#        ./dynamic/point_update.sh
#
#    If you have a pre-built capacity-headroom index from a previous
#    point_update.py run (saved via `--save_cache`), pass its path with
#    INDEX_PATH=/path/... to skip the in-process rebuild.
#
#    `dynamic/point_update_build.sh` also exists and builds a 5M-with-10M
#    capacity index via the regular build path, but that path currently
#    triggers the load-then-add bug, so it is NOT used by default.
```

Each run produces a `logs/dynamic_<op>_<dataset>_<timestamp>/run.log` and
a per-stage JSON with recall / QPS / wall-time.

Override `EMA_BASE_FVECS`, `EMA_QUERY_FVECS`, `EMA_ATTR_JSON` directly if
you want to point at non-default files; the shell wrappers populate these
from `conf.sh` automatically.

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
