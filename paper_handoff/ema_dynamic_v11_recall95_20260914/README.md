# EMA / HashANN V11: recall95 paper handoff

This package supersedes the recall metric in the dated recall90 handoff; it does
not overwrite or invalidate that historical evidence. All18 archived graph states
were loaded and queried again at95% recall. No insertion, deletion, replacement,
index rebuilding, or graph migration was performed for this revision.

Read `DELETE_STRATEGY.md`, `EXPERIMENTS.md`, then `RESULTS.md`.
The deletion algorithm is unchanged; its source-grounded description is copied
byte-for-byte from the original V11 package.

| Artifact | Purpose |
|---|---|
| `data/stage_metrics.csv` | Full-precision18-stage QPS95 and original maintenance times |
| `data/paper.json` | Authoritative recall-revision publication |
| `data/raw/` | New95% timing/calibration results and the reused exact live-set GT |
| `data/timing_samples.csv` | Every accepted and rejected timing sample |
| `data/calibration_points.csv` | Recall-only calibration points, not a timed ef sweep |
| `data/stage-index.json` | Portable raw-artifact and original checkpoint mapping |
| `provenance/query-manifest.json` | New measurement code, original input and native identities |
| `provenance/source90/` | Explicitly historical source-run manifests/journals, not new QPS95 |
| `source_reference/query95/` | Executed95% query/validation/aggregation code |
| `source_reference/` outside `query95/` | Original reference snapshot, preserving deletion-strategy citation paths |

Native SHA256: `608e9ae692fa2fd9d5ec160a370a63b90419356c6b64d67285f10d6bd1e4f84e`.
Query protocol: `sift10m-mixed-dnf-dynamic-rebuild-v5`.
Publication schema: `canonical-dynamic-recall-revision-v1`.
Completion timestamp: `2026-09-14T10:54:25.181851+00:00`.

There are 696 recorded timing samples, of which 504 belong
to the accepted complete paired rounds. Use `selected_for_paper`, not merely
`sample_clean`, when reconstructing the reported medians.
The original reference snapshot may include historical query code; the current
recall95 query implementation is explicitly separated in `query95/`.

The current source remains a local working-tree implementation; do not imply
that the V11/revision code was committed or publicly released. This is a
paper-writing evidence package, not a full-data executable distribution:
multi-gigabyte source vectors, graph checkpoints and the native binary are
identified but not included. Embedded absolute paths refer to the original
machine; use `data/stage-index.json` for packaged copies.

Keep implementation facts, measurements and hypotheses distinct. Do not invent
cross-system numbers, independent-build confidence intervals, or per-component
maintenance costs. Five sequential stages are not five independent repetitions.
Use R or LaTeX-native plotting, not Python plotting.
