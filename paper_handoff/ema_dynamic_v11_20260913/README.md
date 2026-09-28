# EMA / HashANN dynamic V11: paper-writing handoff

This folder packages the current deletion strategy, prose and pseudocode,
the experimental protocol, and the completed SIFT10M measurements. It is
intended for another agent to write the method description and experimental
results without guessing implementation details or rerunning the benchmark.

**Read these files in order:**

| File | Purpose |
|---|---|
| `DELETE_STRATEGY.md` | Source-grounded V11 deletion explanation, notation, pseudocode, concurrency rules, and limitations. |
| `EXPERIMENTS.md` | Dataset, predicates, traces, graph lineage, timing definitions, QPS90 calculation, and writing constraints. |
| `RESULTS.md` | Human-readable tables generated from the published data. |
| `data/stage_metrics.csv` | Portable, full-precision table for all 18 official stages. |
| `data/paper.json`, `data/paper.tsv` | Byte-identical original paper outputs. |

## Identity and status

- System: EMA / HashANN; the native module retains the name `hashannlib`.
- Implementation: retained V11 point-parallel deletion with the shared
  publication gate, not the rejected reserve/ranking-cache candidates.
- Native SHA256:
  `608e9ae692fa2fd9d5ec160a370a63b90419356c6b64d67285f10d6bd1e4f84e`.
- Main native header SHA256:
  `7de1d2edcaee93c098712db21f9ccd797cf4620e8386798fa2bef3842e9cf9b2`.
- Protocol: `sift10m-mixed-dnf-dynamic-rebuild-v4`.
- Run: 2026-09-13, started at 18:51:09 and published at 21:53:36, UTC+08:00.
- All three operations and all 18 official stages completed. Each operation
  contains its own baseline stage plus five million-record mutation stages.
- The source was in the local repository worktree but remained uncommitted
  and partly untracked. **Git HEAD `28e07e3` does not identify the V11 code.**
  Do not claim that checking out that commit reproduces these results.
- `provenance/source-status.json` records the branch, HEAD, worktree status,
  and exact source-reference hashes.

The original native and provenance were frozen for the experiment. A later
comparison confirmed that the corresponding repository source files and native
binary matched that frozen implementation. Frozen provenance contains
point-in-time flags such as "review pending at freeze"; these are not the final
run status. The completed publication and completion summary are the final
experimental status.

## Data navigation

| Path | Contents |
|---|---|
| `data/operation-totals.json` | Five batch times, totals, and per-million means for each operation. |
| `data/timing_samples.csv` | All recorded timed query batches, including selection and rejection information. |
| `data/calibration_points.csv` | Recall-only integer-ef calibration points; these are not a full timed QPS-versus-ef curve. |
| `data/mutation_work_counts.csv` | Native work counters in long format; counters are not unique edge/node counts. |
| `data/stage-index.json` | Portable index from operation/stage to the bundled raw result and array files. |
| `data/raw/<operation>/stage-XX/` | Original final-attempt JSON, exact-GT arrays, and calibration arrays. |
| `data/operations/<operation>/` | Original immutable operation manifest and completed journal. |
| `data/workload/` | The exact 1000 predicates and the original full-10M canonical GT JSON. |
| `provenance/` | Pipeline configuration, source/native identity, level-plan metadata, publication pointer, and completion evidence. |
| `source_reference/` | Exact source files supporting the descriptions and citations; a reference subset, not a complete buildable distribution. |
| `MANIFEST.json`, `SHA256SUMS` | Final folder file inventory and integrity hashes. |

The raw results contain all timing attempts and samples. There are 556
individual timed query batches, of which 483 belong to the accepted paired
rounds used for the paper summaries. A sample can be individually clean but
not selected because another case in the same paired round was rejected.
Use `selected_for_paper`, not just `sample_clean`, when reconstructing a result.

Each final `results.json` is authoritative for its stage. The progress JSON
files are retained for inspection but do not supersede the committed result.
The original files are copied without rewriting their contents or hashes.
Consequently, some embedded paths still name the original machine. Resolve
bundled copies using `data/stage-index.json` and
`provenance/data-inventory.json`, rather than assuming those absolute paths
exist on a new machine.

## What is and is not self-contained

The folder is sufficient to read the algorithm, populate the paper's tables,
plot the reported measurements, inspect the accepted/rejected query timing
samples, and inspect the exact GT/calibration arrays used at each stage.
It deliberately does not include the multi-gigabyte vector/attribute inputs,
large graph checkpoints, live-ID trace arrays, or compiled native extension.
Their identities and original locations remain recorded in the provenance.
This is a paper-writing evidence package, **not** a full-data execution bundle.

No new experiment, altered predicate, synthetic result, extra maintenance pass,
or query retiming was performed to create this handoff. No ZIP is required.

## Non-negotiable writing constraints

- Distinguish implementation facts, measured results, and hypotheses. Do not
  turn the approximate incoming/witness searches into an exhaustive algorithm.
- Keep the complete-point update semantics: delete the old point, then insert
  its complete replacement with a fresh label. This is not attribute-only
  update, in-place `updatePoint`, or deleted-slot reuse.
- All operations use the same ordered 1000 queries and their corresponding
  predicates. The live data and therefore exact GT/selectivity change by stage.
- Do not fill the three missing deletion QPS90 values with observed QPS or
  extrapolate below the minimum legal `ef=k=10`.
- Five sequential million-record batches are not five independent experiment
  repetitions. Seven clean query-timing rounds are not seven independently
  constructed graphs.
- There are no other-system benchmark results in this package. Do not invent
  comparison-system numbers, significance claims, or IP-DiskANN equivalence.
- The current data cover one approximately 10%-selective mixed-DNF workload.
  They do not establish performance across all selectivities or attribute types.
- Do not repeat broad promotional claims such as "attribute filtering is
  essentially free" as conclusions of this experiment.
- If producing paper figures, use R or LaTeX-native plotting, not Python
  plotting. The CSV/TSV tables are provided for this purpose.
- Do not imply the V11 changes have been committed or publicly released.
  Public source availability needs a separate repository publication step.

The small-graph performance launch gate had previously failed. The full-scale
run was subsequently explicitly requested for direct measurement; that did not
retroactively change the old result. This procedural history is preserved in
the provenance, not presented as a new scientific acceptance theorem.
