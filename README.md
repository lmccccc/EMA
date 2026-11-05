# EMA

EMA is a graph-based approximate nearest-neighbor index for vector search with
numerical and categorical filters. It supports compound predicates, insertion,
deletion, and updates using per-edge Markers.

[Technical report](technical_report.pdf) |
[Experiment guide](exp_benchmark/README.md)

## Installation

Requires Python 3.10+ and a C++ compiler with OpenMP. Run from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ./hnswlib
python -m pip install faiss-cpu
```

The Python extension is imported as `hashannlib`. Experiment launchers also
require Linux `numactl`.

## Running experiments

Prepare vectors, attributes, queries, predicates, and ground truth as described
in the [input requirements](exp_benchmark/README.md#prerequisites).
Set `DATA_ROOT` to your dataset directory.

| Experiment | Instructions |
|---|---|
| Static search and parameter sweeps | [Experiment guide](exp_benchmark/README.md) |
| Query-only replay of existing indexes | [Replay guide](exp_benchmark/README.md#existing-index-figure-56-replay-no-construction-or-clustering) |
| Insertion, deletion, and complete-record replacement | [Dynamic experiments](exp_benchmark/dynamic/README.md) |
| Component ablation and same-graph strict in-filtering | [Ablation guide](exp_benchmark/ablation/README.md) |

Datasets, saved indexes, historical binaries, and other systems' baselines are
not bundled. Fresh runs with the current implementation are not exact replays
of historical measurements.

## Usage notes

- `delete_items` performs synchronous graph and Marker repair; `mark_deleted`
  only marks a point as deleted.
- Do not overlap maintenance with queries or other mutations. After in-place
  deletion, insert replacements with fresh labels and sufficient capacity;
  deleted slots cannot be reused until a full rebuild.
- Load indexes with `dynamic=True` when further updates are required. Older
  incompatible index formats must be rebuilt.

## Repository layout

| Path | Contents |
|---|---|
| `hnswlib/` | C++ implementation, Python bindings, and native tests |
| `tests/` | Build/query entrypoints and input-preparation utilities |
| `exp_benchmark/` | Experiment drivers, configurations, and regression tests |
| `technical_report.pdf` | Proofs, experimental results, and parameter settings |
