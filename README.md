# HashANN (EMA)

**EMA** (Edge-FT Mixed ANN) — a hybrid vector-and-attribute approximate
nearest neighbour index. EMA augments an HNSW graph with **per-edge
Fingerprint (FT) bloom filters** that route the search away from neighbours
that cannot satisfy the predicate. The FT bits are bundled into the same
cache lines as the neighbour list, so attribute filtering is essentially
free at query time.

## Features

- **Numerical and categorical attributes** — each item can carry one or more
  attributes of either kind. Numerical attributes are stored as ranges with
  inclusive lower/upper bounds; categorical attributes are stored as label
  sets. Both kinds share the same edge-FT bloom layout.
- **Multi-predicate queries (DNF)** — a query predicate is an OR of ANDs,
  e.g. `(attr0 ∈ [0.1, 0.3] AND attr1 ∈ {9}) OR (attr1 ∈ {12})`. Predicates
  may mix numerical bounds and label lists arbitrarily, on any subset of
  attributes. Pure attribute-AND and single-bound queries are special cases.
- **Dynamic insertion / deletion / update** — the index supports
  `add_items`, `mark_delete`, and in-place attribute updates without
  rebuilding the graph. Deleted nodes are filtered out at query time;
  optional **selective edge patching** can be applied at chosen stages to
  recover navigability after large deletions. Attribute-only updates
  recompute the affected edge FTs in place.
- **Hybrid search** — queries return top-K nearest vectors among the
  attribute-filtered subset. The selectivity-aware FT routing gives large
  QPS gains over post-filtering across the 1 %–100 % selectivity range.

## Repository layout

| path                | purpose                                                         |
|---------------------|-----------------------------------------------------------------|
| `hnswlib/`          | C++ implementation of EMA (HNSW + edge-level FT) and Python bindings |
| `tests/`            | Python entry points: `hashann_build.py`, `hashann_query.py`, GT/predicate generators |
| `exp_benchmark/`    | **Clean reproduction harness for the paper experiments**         |
| `docs/`             | Project map, design notes                                       |
| `tools/`            | Repo-guard and other developer scripts                          |

The historical `exp/`, `exp2/`, `exp3/`, `exp4/` (and their dataset
variants `exp3_redcaps/`, `exp3_wiki/`, `exp3_youtube/`) folders are kept
only as references and are **not the canonical reproduction path** —
please use `exp_benchmark/`.

## Quick start

1. Build and install the C++ extension:

    ```bash
    cd hnswlib && pip install -e .
    ```

2. (Optional) build the FAISS-based brute-force ground-truth generator
    used by `ground_truth_generator.sh`:

    ```bash
    cd /path/to/faiss && cmake -B build -DBUILD_TESTING=ON
    make -C build generate_groundtruth_arbi
    ```

3. Run the EMA benchmarks:

    ```bash
    cd exp_benchmark

    # Main 4-dataset table (sift10m, youtube_rgb, wiki_15.4M, Redcaps_4M).
    ./main_experiment.sh
    python aggregate.py logs/main

    # Parameter ablations (M, min_deg, ft_bits).
    ./ablation/M_sweep.sh        && python aggregate.py logs/M_sweep
    ./ablation/min_deg_sweep.sh  && python aggregate.py logs/min_deg_sweep
    ./ablation/ft_bits_sweep.sh  && python aggregate.py logs/ft_bits_sweep

    # Dynamic update benchmarks.
    dataset=sift10m attr_type=[1] ./dynamic/insert.sh
    dataset=sift10m attr_type=[1] INDEX_PATH=<idx> ./dynamic/delete.sh
    dataset=sift10m attr_type=[1] INDEX_PATH=<idx> ./dynamic/delete_patch.sh
    dataset=sift10m attr_type=[1] INDEX_PATH=<idx> ./dynamic/attr_update.sh
    ```

    See `exp_benchmark/README.md` for the full knob list and per-dataset
    paths.

## Datasets

| name          | n          | dim  | metric | attribute layout                |
|---------------|-----------:|-----:|--------|---------------------------------|
| `sift10m`     | 10 000 000 | 128  | L2     | one categorical (21 cardinalities) |
| `youtube_rgb` |  1 000 000 | 1024 | IP     | one numerical + one categorical |
| `wiki_15_4M`  | 15 435 516 | 1024 | IP     | one numerical + one categorical |
| `Redcaps_4M`  |  4 000 000 | 512  | IP     | one numerical + one categorical |

Each dataset can be configured via the `DATA_ROOT` env var pointing at the
directory that holds the per-dataset sub-folders (defaults to
`/mnt/data/mocheng/dataset/`).

## Developer notes (legacy)

The historical `exp*/` workspaces, `tools/repo_guard.sh`, and the
`docs/PROJECT_MAP.md` map are still available. `repo_guard.sh` performs
basic syntax / sourcing / mutation checks over `*.sh` files; run it
before and after any change to the legacy folders:

```bash
./tools/repo_guard.sh           # full repo
./tools/repo_guard.sh exp3      # scoped to one folder
```
