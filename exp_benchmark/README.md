# EMA Benchmark Scripts

End-to-end build / query / ablation scripts for **EMA** (the HashANN-based
edge-FT method described in the paper) on four datasets:

| dataset       | n          | dim  | metric |
|---------------|-----------:|-----:|--------|
| sift10m       | 10 000 000 | 128  | L2     |
| youtube_rgb   | 1 000 000  | 1024 | IP     |
| wiki_15_4M    | 15 435 516 | 1024 | IP     |
| Redcaps_4M    | 4 000 000  | 512  | IP     |

The scripts wrap the C++/Python bindings in `hnswlib/` and the helper
tools under `tests/`. They are intentionally self-contained and do **not**
depend on any of the historical `exp*/` folders.

## Layout

```
exp_benchmark/
├── conf.sh                       # dataset + index path resolver
├── env.sh                        # conda activation
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
