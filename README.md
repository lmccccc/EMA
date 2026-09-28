# EMA (HashANN)

**EMA** (Edge Marker) — a hybrid vector-and-attribute approximate nearest
neighbour index. EMA augments an HNSW graph with **per-edge Edge Markers**
(small bloom-style signatures) that route the search away from neighbours
that cannot satisfy the predicate. The Edge Marker bits are bundled into
the same cache lines as the neighbour list, so attribute filtering is
essentially free at query time.

> **Naming note** — the paper calls the per-edge signature **Edge Marker**
> (EM). Throughout the source tree it is abbreviated **`ft`** (a legacy
> name for the same data structure: "fingerprint"). Read `ft_bits`,
> `edge_level_ft`, `update_node_ft`, etc. as
> Edge-Marker–related code. Old "HashANN" identifiers (module name
> `hashannlib`, scripts `hashann_build.py` / `hashann_query.py`) are
> retained for binary compatibility.

## Features

- **Numerical and categorical attributes** — each item can carry one or more
  numeric values or categorical label sets. Numerical predicates use inclusive
  lower/upper bounds. Both kinds share the same Edge-Marker bloom layout.
- **Multi-predicate queries (DNF)** — a query predicate is an OR of ANDs,
  e.g. `(attr0 ∈ [0.1, 0.3] AND attr1 ∈ {9}) OR (attr1 ∈ {12})`. Predicates
  may mix numerical bounds and label lists arbitrarily, on any subset of
  attributes. Pure attribute-AND and single-bound queries are special cases.
  Node and edge DNF Markers share SIMD bitmap checks on supported CPUs,
  with scalar handling for small bitmaps, trailing bytes, and other platforms.
- **Dynamic insertion / deletion / update** — use `add_items` for insertion
  and synchronous `delete_items` followed by insertion at fresh labels for
  full-point replacement. Attribute-update APIs remain available separately.
  `delete_items` performs synchronous IP-DiskANN-style graph repair and
  heuristic reverse RNG cleanup of deleted points' Marker contributions.
- **Hybrid search** — queries return top-K nearest vectors among the
  attribute-filtered subset. The selectivity-aware Edge-Marker routing
  gives large QPS gains over post-filtering across the 1 %–100 %
  selectivity range.

## Paper artifacts and reproduction coverage

Start with the [paper-to-code map](exp_benchmark/README.md#paper-to-code-map).
The repository includes the current native implementation and Python bindings,
static EMA runners, the three-operation dynamic pipeline, regression tests,
and the [R/TikZ sources for Figures 5--7](paper/README.md).
The [appendix source](paper/appendix/appendix.tex) builds
[`EMA_appendix.pdf`](EMA_appendix.pdf); its obsolete attribute-only,
tombstone/patch and old replacement experiments have been removed.

**This is not a self-contained reproduction of every baseline and every
historical experiment.** Raw dataset subsets, original attribute/predicate
files, large saved indexes, and several modified baseline dependencies are not
distributed here. Parameter sweeps using newly generated inputs are not replays
of the published measurements. The coverage map states which commands exist
and which inputs or implementations are still external.

For Figure 7, the
[frozen V11 recall95 evidence](paper_handoff/README.md) includes the native source
snapshot, query code, original maintenance times, and raw calibration/timing
records. It does not include the original native binary or graph checkpoints.
The current native has subsequent fixes/features and must not be substituted
for that frozen binary when claiming an unchanged-checkpoint replay.

## Incremental Codebook registration

`initAttrMapping(initial_attrs)` initializes the Codebook from the supplied
complete attribute records; the sample need not fill `max_elements`.
Only observed categorical labels are assigned initially. Insertion and update
operations preserve numerical bucket boundaries and all assigned label buckets.
Values below or above the initial numerical range use the first or last bucket.

Insertion and attribute-update entry points register unseen categorical labels
before writing attributes or constructing Markers. A new label receives the
bucket with the fewest assigned labels, with the lowest bucket index breaking
ties; a full Codebook shares buckets rather than moving existing assignments.
Batch registration precedes parallel point work, and each assignment is made
once. Queries do not allocate buckets, and unregistered query labels cannot
satisfy a categorical conjunction.

```python
# Usually automatic in add_items/update_attr and their batch/combined variants.
# Optional pre-registration changes the Codebook, not vectors, attributes or edges.
assigned = index.register_attr_values([[[120000], [new_label]]])

# Preserve assignments and still-unregistered labels across process restarts.
index.save_index("updated.bin")
```

This keeps the existing record layout: categorical IDs must remain in
`[0, max_cate_size)`. Out-of-capacity labels are rejected before registration;
there is no implicit bitmap expansion or index rebuild. Attribute columns and
their types are unchanged. Calling `initAttrMapping` on a populated index is
rejected because remapping would invalidate old Markers.
Registration and mutations require stop-the-world use. Native
`save_index`/`load_index` are the persistence interface for this feature, not
pickle. Existing auxiliary-index refresh and attribute-update Marker limitations
are unchanged.

## Synchronous in-place deletion

```python
# Repair the graph and its Markers before returning.
stats = index.delete_items(labels, num_threads=8)

# Marker-only maintenance for previously tombstoned points; no graph repair.
stats = index.cleanup_deleted_marker_bits(labels, num_threads=8)
```

Deletion search always uses the index's `ef_construction`, including the value
restored from a saved index. There is no separate deletion ef setting, and
query `ef` is independent.

`delete_items` adapts Algorithms 5 and 6 of
[IP-DiskANN](https://arxiv.org/abs/2502.13826) to EMA's layers and source-specific
Markers. For each point, a normal construction-width search records the
**expanded search trace**, not just the final nearest candidates. Discovered
incoming edges are replaced with up to three nearby candidates. Each original
live outgoing neighbor also receives edges from up to three nearby candidate
sources. The shared replacement pool contains the first 50 search candidates.
Only overflowing rows are pruned, using distance-ordered RNG and EMA's existing
CHT edge-coverage constraints rather than introducing DiskANN's separate
pruning-alpha parameter.

RNG pruning and CHT augmentation rank candidates by vector distance, with node ID
as their sorting tie-breaker. Unfiltered construction and maintenance searches do not use a query
predicate. Predicate/Marker checks govern filtered-query eligibility and routing,
not a score added to vector distance. CHT remains a separate edge-coverage
constraint; it does not change candidate ranks. The legacy `set_attr_sort_alpha`
setter accepts only zero and rejects attribute-weighted ranking.

Surviving edges retain their source-specific Markers. New edges start with their
target's attributes; additional bits require an actual pruning witness or a
discovered live point satisfying `dist(target,x) < dist(source,x)` and
`dist(source,target) <= dist(source,x)`. Old Markers are never blindly copied to a
different source or target.
Transfer considers each discovered point's complete attribute mask: existing
geometrically valid coverage is reused, otherwise the point is assigned to one
closest eligible replacement. A single witness is not independently copied to
every replacement edge merely because each edge could geometrically support it.

Each deletion shares a temporary request-mask-to-compatible-record table across
its repair rows. Numerical posting lists narrow candidates before full-record
matching; categorical-only records use full containment checks. The table caches
only immutable mask matches. Current owner Markers and source-specific geometric
support are checked separately, in the original witness and edge tie order.
Its own-record masks are prepared before replacement selection and reused by
cleanup, new edges, and pruning witnesses. A lazy per-point distance cache shares
evaluated ordered pairs across selection, cleanup, RNG, and transfer. Its dense
pool portion is capped at 512 points, with a fixed-size overflow cache; these
memory limits do not restrict the search or witness pool. Actual direct RNG
owner/witness proofs can skip redundant coverage searches in that same row.

Marker cleanup first checks the existing edges for removable bits, protecting
each target's own attributes. If the whole row has none, owner-distance
calculations are skipped. Otherwise the original closest-owner rule is retained;
an unaffected nearer owner is not skipped in favor of cleaning a farther edge.
`empty_marker_rows` counts early-skipped cleanup source rows. The diagnostic
`matched_edges` count covers overlapping selected owners in examined rows and
excludes these skipped no-op rows; graph repair and support/clearing semantics
are unchanged.

The deleted point's outgoing lists are cleared at every layer. Before the batch
returns, a synchronous, distance-free adjacency scan removes missed incoming
references and clears stale dirty/repair-candidate records. This scan is not
the legacy query-guided patch operation and does not rebuild the graph. A deleted
entry point is replaced by a live entry; queries also support a remaining
base-layer-only graph or an index with no live points.

With `num_threads=1`, point transitions retain input order. With multiple threads,
the same dynamic `ParallelFor` helper used by insertion schedules complete point
deletions, not speculative preparation windows or serial commits. Each worker
retires its own point before discovery; at most the active workers' points are
in flight. The retiring point remains eligible as a placeholder in its own
search so neither the construction-width result nor its first-50-position
replacement budget shifts by one. Retired outgoing rows remain frozen and
navigable until all workers finish. The entry is also stable until batch end.

Search expansion locks each source row. Repair rebases edge additions/removals
against the current row and holds its lock through geometry, Marker transfer,
and publication; retiring sources are never rewritten. Removing another active
deletion's incoming edge hands its current complete Marker request to that
point's pending repair queue. A producer reserves that task during registry
lookup; closure requires both an empty queue and no outstanding reservations.
The producer publishes its complete request before releasing the reservation.
If discovery precedes closure but registry lookup arrives too late, a deferred
mailbox preserves the discovered source/layer and complete request. Point workers
drain these requests before returning, outside all source and queue locks.
Point-start and recipient-completion epochs distinguish these in-flight removals
from genuinely old completed dangling edges, which retain scrub-only semantics.
The point watermark is captured before any of its snapshots or planning and
shared by all its rows. Each deferred repair captures its own pre-search
watermark; already-promised descriptors are serviced regardless of their age.
Explicit new/pending requests are preserved independently of those epochs.
Final target-liveness checking and row publication share a short batch gate
with retirement. C++14 builds let independent source rows publish under shared
ownership, while retirement takes exclusive ownership before registration and
MARK publication. The gate's C++11 fallback retains exclusive row publication. Registry
bookkeeping stays separately mutex-protected; neither publisher handoffs nor
searches hold the publication gate. Retirement locks its row, then the exclusive
gate, then the registry; publishers never acquire another row or the registry
under their shared gate. A row is either published before its target retires or
its complete request is handed off; a post-retirement publication cannot disguise a
new promise as an old stored edge. Explicit proposed endpoint IDs carry this
provenance even for empty masks or requests adding no new bits. Search,
geometric preparation, and queue publication remain outside that gate.
No reader/writer fairness is assumed. Within this finite, synchronous call,
retired state is fixed between announcements: existing retired links and
prepared requests are consumed, and fresh repairs cannot recreate those links.
Publication work therefore exhausts between the finitely many retirements;
this is not a wall-time starvation guarantee for externally sustained readers.
A late recipient uses a fresh construction-width search and the same first-50
positional replacement budget; its context is discarded after the local repair.
No completed deletion contexts are retained until batch end.
If the original source also retired, its outgoing-source repair is composed
with the recipient's incoming repair, each using its own bounded pool and
up-to-three endpoint replacements. The incoming stage prepares a local
intermediate row using its own witnesses, direct original-source geometry, and
ordinary RNG/CHT/Marker rules. Only new or changed intermediate edges are then
repaired through live sources; their resulting Markers, not the original raw
request, are revalidated against the source-stage pool and final-edge geometry.
Pending work is not discarded and dead rows are never rewritten.
A post-join drain also services any final obligations and
their local cascades before the adjacency scrub.
Retiring new proposals retain their original pending requests even if pruning
or liveness checks prevented those requests from reaching a result Marker.
Receivers still validate live complete records and current source geometry.
Queue closure and draining are synchronized, with no queue lock held while
taking a source lock. This avoids capacity-reserving
dead-edge placeholders even at small `M`. Candidate/owner liveness is refreshed.
Each locked base-layer search expansion captures an ephemeral source adjacency
version. Cleanup may reuse the completed-search exclusion only when that exact
version still matches under the source's mutation lock. Topology writes at any
layer advance the source version; Marker-only writes do not. Changed or
unexpanded rows keep the conservative geometry checks, without retries or
global search/write phases.

Parallel graph interleavings are intentionally nondeterministic, as with
insertion, but `delete_items` still waits for every repair and the final scrub.
`parallel_deletions` and `peak_parallel_deletions` count full-point tasks and peak
active points (`parallel_points` / `max_active_points` are aliases).
`incoming_handoffs`, `distance_cache_hits`, `distance_cache_misses`, and
`rng_witness_reuses` report local reuse/overlap work. Legacy speculative
preparation counters remain zero. No format version changes are required.
`cleanup_bound_reused_rows` counts cleanup rows with a validated expansion
version; `cleanup_bound_stale_rows` counts rows requiring the conservative path
because their adjacency changed or no expansion version was recorded.
`deferred_incoming_handoffs` and `deferred_incoming_repairs` count deferred
requests queued and serviced. `deferred_retired_source_repairs` counts requests
that also require live replacement sources.
`deferred_intermediate_rows` counts local intermediate row preparations; these
are not publications to retired sources.
`publication_retired_targets` counts prepared targets rejected by the final
retirement/publication handshake.
`ignored_completed_ghosts` counts plain dangling edges whose recipient completed
before the producer's causal boundary. Deferred request attribution is split
between `deferred_overlapping_edges`, `deferred_pending_requests`, and
`deferred_retired_source_requests`; these sum to `deferred_incoming_handoffs`.
Query threading is separate; single-thread QPS measurements must still call the
query API with `num_threads=1`.
Retired slots keep their deletion flags and
vector storage, so `get_current_count()` remains the occupied-slot count rather
than the live-point count. Compaction remains a separate `rebuild_graph` operation.

The candidate-pool budget deliberately trades support coverage for runtime;
increasing `ef_construction` expands the discovery budget without requiring
exhaustive recovery. The OR-compressed representation does not retain historical
contributor identities. No global support scan is added. The
standalone `cleanup_deleted_marker_bits` API only clears unsupported
contributions on existing live-target edges; it does not perform in-place
deletion. Neither operation introduces a global pairwise distance table.

These are **stop-the-world maintenance operations**: no concurrent queries,
insertions, updates, or other maintenance. All labels are validated
before deletion; duplicate, missing, or already-deleted input labels raise.
Worker failures propagate rather than being reported as successful deletions;
resource failures after mutation begins can leave a partially maintained batch.
Once in-place deletion or Marker cleanup has been used, deleted labels in that
index cannot be revived or their slots reused. This conservative restriction
persists across save/load and resets
after a full rebuild. Resume maintenance with `load_index(..., dynamic=True)`.
Insert replacements with fresh labels,
`replace_deleted=False`, and sufficient capacity.
Saved dynamic indexes support subsequent insertion and replacement without an
in-process rebuild workaround.

Numeric Markers use bounded floor buckets over the stored quantile boundaries,
and query masks include both endpoint buckets. Pruning witnesses are aligned
with their actual owner IDs after neighbor heaps are reordered, including
forward insertion, reverse pruning, and point updates. Node Markers initialize
after adjacency storage is cleared and retain other rows' witness contributions
during incremental updates.

Indexes with complete categorical mappings use formats **9 (node Markers) and
10 (edge Markers)**. An index with unassigned categorical entries is saved as
**11 (node) or 12 (edge)**, with the same record layout, so older V11 readers reject
it instead of treating an unassigned bucket as a bit offset. The current native
loads all four formats; existing 9/10 files need no migration for registration.
The frozen V11 benchmark profile remains format10 and is not silently upgraded.
To use earlier formats with the **current native**,
older attribute-bearing indexes must be **rebuilt**: previous numeric lookup could overflow a bitmap, and earlier heap
reordering attached witnesses to the wrong owners. Formats 7/8 corrected ownership
but did not record whether construction used attribute-weighted ranking, so they
cannot be reused as distance-ordered graphs. The loader rejects old formats rather
than retagging them. Initialized pickle metadata requires serialization version 4,
numeric Marker version 2, owner version 2, and distance-order version 1.
Benchmark cache names use `_mo2_do1`, with `_nb2_mo2_do1` for numeric layouts,
to preserve old files.

This does not require rebuilding indexes for a read-only historical replay.
The [existing-index replay guide](exp_benchmark/README.md#existing-index-figure-56-replay-no-construction-or-clustering)
uses an isolated original-source native with legacy-layout and inclusive
numerical query-mask fixes.
It reuses the saved graph, levels, Markers and query inputs without clustering,
rebuilding, retagging or writing index files. Its results are explicitly
separate from V11/current-format experiments.

For compatibility, `mark_deleted(label)` and `batch_mark_deleted(labels)`
remain tombstone-only; use `delete_items` for synchronous graph and Marker repair.
Tombstone-only deletion remains reversible on indexes that have not used
reverse cleanup. `batched_patch_deletes`,
`set_maintenance_thresholds`, and `maintain_deletes` retain their existing
graph-patching/compaction semantics, distinct from reverse Marker cleanup.
`delete_items` does not invoke those legacy policies or automatically rebuild.
Without `edge_level_ft`, `delete_items` still repairs adjacency, while standalone
`cleanup_deleted_marker_bits` is unsupported.

Query-side deletion accounting uses a maintained one-bit-per-slot cache rather
than scattered reads of neighboring node records. Dirty-node and repair-candidate
tracking remain enabled. This cache adds no on-disk fields and is reconstructed
on load. Read-only loads (`dynamic=False`)
also honor stored tombstones; `dynamic=True` enables label lookup for updates.

In-place statistics include `search_candidates`, `search_expanded`,
`incoming_edges_repaired`, `outgoing_edges_added`, `rewired_nodes`,
`pruned_edges`, and `scrubbed_edges`. Marker statistics include
`candidate_sources` (visits, not unique sources), `matched_edges`, `cleared_bits`,
and `support_checks`. `cleared_bits` counts heuristic bit removal, not bits
discarded with retired edges. `scrubbed_edges` counts residual live-to-deleted
adjacency entries removed across all layers, not all scanned edges or the
outgoing rows cleared for retired sources.

The returned dictionary also includes `scrub_wall_s`, a floating-point,
monotonic wall-clock duration in seconds for the batch's final
`scrubDeletedEdges()` call. This includes the full scan, retired outgoing-row
clearing, adjacency/Marker compaction and scrub bookkeeping. It excludes local
and deferred repairs and the subsequent entry-point repair. This is elapsed
time for the whole parallel phase, not summed worker CPU time or a work counter.
An empty batch reports `0.0` because it does not run the scrub. Older native
binaries omit the field; their scrub duration is unknown, not zero.

Attribute-only updates still use the existing conservative
OR-based Marker refresh; this cleanup applies to deleted points, not
replacement attribute records.

## Repository layout

| path                | purpose                                                         |
|---------------------|-----------------------------------------------------------------|
| `hnswlib/`          | C++ implementation of EMA (HNSW + per-edge Edge Markers) and Python bindings |
| `tests/`            | Python entry points: `hashann_build.py`, `hashann_query.py`, GT/predicate generators (legacy filenames; same code is EMA) |
| `exp_benchmark/`    | EMA measurement/replay tools; see the paper-to-code coverage map  |
| `paper/`            | Active appendix sources and current main-figure R/TikZ sources   |
| `paper_handoff/`    | Frozen V11 source snapshots and recall90/recall95 evidence        |
| `docs/`             | Project map, design notes                                       |
| `tools/`            | Repo-guard and other developer scripts                          |

Historical `exp*/` folders were removed from this checkout. Remaining root-level
baseline shell scripts are legacy, machine-specific adapters, not a complete
baseline reproduction suite. Use the documented `exp_benchmark/` entrypoints.

## Quick start

1. Use Python 3.10 or newer, a C++ compiler with OpenMP, and an isolated
   environment. Run from the repository root:

    ```bash
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ./hnswlib
    python -m pip install faiss-cpu
    ```

   The shell scripts keep this environment. They activate conda only if
   `EMA_CONDA_ENV` is explicitly set. FAISS Python is used for representative
   level selection and dynamic exact GT. The pinned paper launchers also need
   Linux `numactl`.

2. Supply the dataset files described in the
   [input requirements](exp_benchmark/README.md#prerequisites).
   For **newly generated** workloads, `GT_BACKEND=numpy` selects the bundled
   brute-force generator. The alternative C++ `generate_groundtruth_arbi`
   target needs an external modified FAISS tree; it is not an upstream FAISS
   build target. Neither generator replaces missing frozen paper inputs.

3. To rerun existing Figure5/6 indexes, follow the
   [query-only replay commands](exp_benchmark/README.md#reproduction-commands).
   They load each index once and collect the original workloads without
   regenerating inputs or rebuilding graphs.

   For separately requested new-build experiments and ablations:

    ```bash
    cd exp_benchmark

    # EMA only, new inputs: four datasets and six DNF selectivity settings.
    # This does not run the competing systems or reproduce all paper tables.
    GT_BACKEND=numpy ./main_experiment.sh
    python aggregate.py logs/main

    # Parameter sweeps on generated workloads, not component ablations.
    GT_BACKEND=numpy ./ablation/M_sweep.sh
    GT_BACKEND=numpy ./ablation/min_deg_sweep.sh
    GT_BACKEND=numpy ./ablation/ft_bits_sweep.sh

    # Dynamic paper pipeline: insert, delete, and full-point update only.
    # See the reproduction guide for input paths and native-binary identity.
    ./dynamic/reproduce.sh --help
    ```

    See `exp_benchmark/README.md` for the full knob list and per-dataset
    paths. The [dynamic reproduction guide](exp_benchmark/dynamic/README.md)
    specifies the mixed-DNF traces, single-thread QPS at 95% recall,
    checkpoint/resume protocol, and three-operation paper output.

## Datasets

| name          | n          | dim  | metric | attribute layout                |
|---------------|-----------:|-----:|--------|---------------------------------|
| `sift10m`     | 10 000 000 | 128  | L2     | one numerical + one categorical |
| `youtube_rgb` |  1 000 000 | 1024 | IP     | one numerical + one categorical |
| `wiki_15_4M`  | 15 435 516 | 1024 | IP     | one numerical + one categorical |
| `Redcaps_4M`  |  4 000 000 | 512  | IP     | one numerical + one categorical |

Each dataset can be configured via the `DATA_ROOT` env var pointing at the
directory that holds the per-dataset sub-folders (defaults to
`/mnt/data/mocheng/dataset/`).

## Developer notes (legacy)

`tools/repo_guard.sh` and `docs/PROJECT_MAP.md` describe legacy workflows.
`repo_guard.sh` performs
basic syntax / sourcing / mutation checks over `*.sh` files; run it
when working with those legacy scripts. It is not a paper reproduction test:

```bash
./tools/repo_guard.sh           # full repo
```
