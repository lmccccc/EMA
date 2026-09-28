# Current V11 deletion strategy: text and pseudocode

## 1. Authority, scope, and terminology

This describes retained V11. Native citations use paths relative to the frozen
`point-parallel-v11-native/` source root. Their copies in this folder are under
`source_reference/hnswlib/`: for example, `hnswlib/hnswalg.h` resolves to
`source_reference/hnswlib/hnswlib/hnswalg.h`, and `python_bindings/bindings.cpp`
resolves to `source_reference/hnswlib/python_bindings/bindings.cpp`.
Pipeline citations are repository-relative and resolve under `source_reference/`.
Repository HEAD `28e07e3` alone is insufficient: the retained implementation
also contains uncommitted/untracked changes. Use the following identities.

| Artifact | SHA256 |
| --- | --- |
| Native extension | `608e9ae692fa2fd9d5ec160a370a63b90419356c6b64d67285f10d6bd1e4f84e` |
| `hnswlib/hnswalg.h` | `7de1d2edcaee93c098712db21f9ccd797cf4620e8386798fa2bef3842e9cf9b2` |
| `python_bindings/bindings.cpp` | `4b208544cab570e4e79eb53dac0a569d1f1df938d637825402195b03b4d14a20` |
| `hnswlib/parallel_for.h` | `f2fe12aa1d1072933473444cdec8706fce82f75cafee4fb710aec5d0c5f6125f` |
| `hnswlib/deletion_cache.h` | `53591c83fb62a0eb42a7348ca946a5df0325e4e7e357f8ec6d0dca00005ad6bb` |
| `hnswlib/deletion_publication_gate.h` | `3dd4b5eaf205163f624d49abcd6bc244dcd321d73269272fbad9e676c7c1a325` |
| `hnswlib/space_l2.h` | `c599d024657896412250b4fde1a95e39f8bc55cccd263f36c8948fa35779edf1` |

The aim is local graph-route repair and updating compressed attribute-routing information.
Native deletion is neither tombstone-only nor delete/reinsert emulation. It is IP-DiskANN-style local repair,
not a verified identical implementation of an upstream IP-DiskANN algorithm.
No novelty, priority, or recall theorem is asserted. The completed experimental
protocol and measurements are documented in `EXPERIMENTS.md` and `RESULTS.md`.

`delete_items(labels, num_threads)` is a synchronous public operation:
a successful nonempty call completes point repairs, deferred obligations,
the final adjacency scrub, and entry repair before returning statistics.
Empty input returns immediately. Unknown, duplicate, or already-deleted labels
are rejected during initial label resolution, before point processing.
[python_bindings/bindings.cpp:1496-1551; hnswlib/hnswalg.h:7262-7292,7314-7354]

The described use requires externally serialized public operations, excluding
concurrent external queries, insertions, attribute changes, or maintenance.
This is a usage assumption, not a claim that a global API lock enforces it.
Vectors and attributes are immutable within that call; graph state is not.
Internal concurrency does not establish linearizability against arbitrary user API calls.
Exceptions after work starts have no rollback promise.
[hnswlib/hnswalg.h:7036-7188,7314-7354; hnswlib/parallel_for.h:25-54]

## 2. Notation and stored routing information

| Symbol | Meaning |
| --- | --- |
| `a` | Point currently being deleted; internal graph ID, not external label. |
| `u -> v` | Directed edge from source row `u` to owner/target ID `v`. |
| `d(x,y)` | Native vector distance returned by the configured distance function. |
| `FT(x)` | Complete encoded attribute mask of point `x`, over all columns. |
| `M(u,v)` | Source-specific base-layer edge Marker stored beside `u -> v`. |
| `R_l(a)` | Ordered retained result of the deletion search at layer `l`. |
| `X_l(a)` | IDs whose rows that search expanded; distinct from retained results. |
| `C(a)` | Shared base-layer witness context formed from `R_0(a)`. |
| `Q_j` | Complete request mask associated with a proposed/retained owner edge. |
| `live(x)` | Current deletion bitmap says that `x` is not retired. |

For the SIFT L2 configuration, `d(x,y) = sum_i (x_i-y_i)^2`, without a square
root. Use this native quantity consistently in every inequality below.
The cache stores ordered pairs even though this particular distance is symmetric.
[hnswlib/space_l2.h:6-19; python_bindings/bindings.cpp:97-105;
hnswlib/hnswalg.h:5787-5793; hnswlib/deletion_cache.h:10-79]

Each attribute column occupies its own FT segment. Numerical attributes set a
bucket bit; categorical records set their mapped category bits. `FT(x)` means
the entire concatenated encoded record, not one independently selected bit.
Bucket encoding is compressed information, not an exact raw-attribute tuple.
[hnswlib/hnswalg.h:637-645,3540-3563]

An edge Marker includes target-own attributes and may represent geometrically covered records.
It is not merely `FT(v)`, nor an unrestricted union transferable unchanged to a different source/target.
Target-own bits are protected by `~FT(v)` during old-edge cleanup.
[hnswlib/hnswalg.h:5365-5375,5920-5940,6325-6338]

Owner-ID alignment is essential: witnesses and Markers follow target IDs, not priority-queue extraction order.
Construction explicitly realigns witnesses by owner ID; deletion requests the
pruner's selected order and retrieves retained old Markers by target ID.
[hnswlib/hnswalg.h:5302-5317,5337-5358,6307-6334,6464-6481]

## 3. Search, incoming discovery, and structural replacement budgets

The sole native deletion-search width is `ef_construction_`, initialized to `max(ef_construction, M)`.
It is 300 for the retained experiment. Query `ef` is not a separate deletion-width control.
A result can contain fewer points, and expanded rows can outnumber that width.
[hnswlib/hnswalg.h:624-628,1098-1186,1326-1487,6021-6068]

For each layer containing `a`, search uses `a`'s vector as the query, descends
from the stable entry through higher layers, and collects retained and expanded
IDs. Source rows are locked during concurrent traversal. Retired rows remain
navigable; retired IDs are excluded from retained results except that `a`
remains eligible as its own live placeholder if encountered.
This preserves its positional contribution without forcing it into results or guaranteeing its discovery.
[hnswlib/hnswalg.h:1109-1176,1371-1409,1458-1482,6027-6068]

The primary Marker context copies the full base result, excluding `a` and IDs
already retired when it is built. It stores masks, ID positions, and per-bit
supporter postings; later uses recheck liveness. It is not truncated to 50.
[hnswlib/hnswalg.h:5724-5784,7069-7080]

Incoming discovery is approximate, without an exhaustive reverse-edge index.
Expanded rows are examined for `u -> a`; base-layer witness-context rows
are also scheduled for cleanup and can discover incoming edges on rebase.
Outgoing proposals and mandatory handoffs add sources; there is no initial global incoming-neighbor search.
[hnswlib/hnswalg.h:6681-6704,7082-7120]

Structural ranking examines only the original first `min(50, |R_l(a)|)`
positions. It then excludes `a`, the anchor itself, and currently retired IDs,
and sorts eligible IDs by `(d(anchor,id), id)`. It does not scan onward to obtain 50 live/filtered matches.
[hnswlib/hnswalg.h:6555-6570]

For discovered `u -> a`, propose up to three currently live replacement targets,
ranked around `u`, with the old `M(u,a)` as a transfer request.
For each live outgoing `a -> v`, propose up to three live replacement sources,
ranked around `v`, carrying `M(a,v)` as their requests.
Existing edges can receive requests without becoming new edges. Retiring selected sources
are refilled from the same first-50 pool. "Three" is a proposal/attempt budget,
not three guaranteed new, distinct, finally retained edges.
[hnswlib/hnswalg.h:6694-6748,7109-7154]

## 4. Overflow pruning, CHT admission, and Marker initialization

A source row is pruned only when the proposed neighbor count exceeds its
layer capacity. `maxM_ = M`; `maxM0_ = M * M0_mul_`, with default multiplier 2.
For the retained M=40 configuration, the base-layer cap is 80 and upper-layer
cap is 40. These are maximum degrees, not enforced exact degrees.
[hnswlib/hnswalg.h:75-76,616-626,6299-6313]

The relative-neighborhood-style (RNG) pruner orders by vector distance and ID.
Candidate `w` is dominated when an already selected live owner `v` satisfies
`d(v,w) < d(u,w)`. Its earlier selection also gives `d(u,v) <= d(u,w)`.
The recorded direct witness is attached to that owner ID.
[hnswlib/hnswalg.h:3679-3719]

With base-layer edge recording enabled, RNG-admissible candidates additionally
pass the existing Counting Hash Table (CHT) coverage admission rule.
The rule accepts an empty/low-degree selected set (`size < maxM_/3`), or an
undercovered inclusive numerical source-candidate bucket interval, or an undercovered
bucket for a category label present in both source and candidate; thresholds use `minM0_`.
Accepted candidates update CHT with their own attributes; admission is separate from distance ordering.
Attribute-sort alpha is fixed at 0; nonzero values are rejected.
[hnswlib/hnswalg.h:502-520,3598-3653,3721-3740;
python_bindings/bindings.cpp:278-286]

CHT rejection and an unprocessed capacity tail are not RNG certificates.
Deletion edge recording/CHT admission is base-layer edge-Marker behavior, not upper-layer reconstruction.
[hnswlib/hnswalg.h:3701-3740,6284-6313]

After selection, retain each surviving old edge's source-specific Marker,
with the local cleanup patch applied. A new edge begins with `FT(target)`.
Recorded direct RNG witnesses contribute their complete own masks, not their
historical edge Markers. Their owner-aligned proofs avoid redundant migration tests in this row.
These direct witnesses come from actual row candidates and may lie outside `C(a)`,
unlike general request reconstruction. Target-own initialization is another distinct record source.
[hnswlib/hnswalg.h:6285-6297,6325-6340,6371-6397]

## 5. Exact old-edge cleanup rule

Cleanup runs inside the source-row transaction, before structural repair and
Marker reconstruction, not a third global pass. Primary cleanup uses live base-layer context sources.
[hnswlib/hnswalg.h:6648-6670,7136-7140]

The whole-row precheck uses cheap candidate, liveness, and valid search-bound exclusions.
If no possibly relevant `M(u,v) & FT(a) & ~FT(v)` bit exists, skip owner-distance/support work.
Do not filter owners by "has removable bits" before closest-owner selection:
an unaffected nearer legal owner must not be bypassed.
[hnswlib/hnswalg.h:5838-5874]

Choose the current live owner minimizing `d(u,v)` subject to
`d(v,a) < d(u,a)` and `d(u,v) <= d(u,a)`; ties keep the first encountered row owner.
Only that owner's Marker is considered for clearing.
[hnswlib/hnswalg.h:5876-5925]

For each removable bit `b`, retain it if there is a currently live
`w in C(a)`, distinct from `u` and `v`, with `b in FT(w)` and both
`d(u,v) <= d(u,w)` and `d(v,w) < d(u,w)`. Otherwise clear `b`.
Cleanup is per bit: it does NOT require `FT(w) subset_of FT(a)` or a complete-record match to `a`.
There is no extra upper bound on `d(u,w)` and no invented witness interval.
[hnswlib/hnswalg.h:5924-5944]

```text
CLEAN_OLD_MARKER(a, u, C):                 # caller owns u's row lock
    if whole-row cheap precheck finds no relevant removable bit: return none
    v = first closest current live owner satisfying
            d(v,a) < d(u,a) and d(u,v) <= d(u,a)
        using a completed-search exclusion only when its row proof is valid
    if no such v: return none
    patched = copy M(u,v); initialize per-witness cleanup geometry memo to UNKNOWN
    for bit b in M(u,v) AND FT(a) AND NOT FT(v):
        supported = false
        for w in C's posting for b:
            if w == u or w == v or not live(w): continue
            if memo[w] is UNKNOWN: memo[w] = (d(u,v) <= d(u,w) and d(v,w) < d(u,w))
            if memo[w]:
                supported = true
                break
        if not supported: clear b in patched
    return a single-owner patch if any bit changed, otherwise none
```

The parallel completed-search exclusion is source-specific: base expansion captures
an adjacency version under the row lock. Cleanup may exclude an out-of-pool owner
only when this retained source was expanded and its captured version still matches.
Stale/unavailable proof does not justify that shortcut; fresh deferred contexts disable it.
Adjacency-sequence changes bump versions; Marker-only changes do not. Serial rows have no competing writer.
[hnswlib/hnswalg.h:1399-1409,5830-5837,5845-5851,5885-5891,
6842-6847,7071-7079]

## 6. Complete-record transfer and reconstruction

If overflow pruning omits old neighbors, combine their locally cleaned old
Markers into a lost-request mask. Each selected edge's wanted mask is its
explicit request OR that lost mask. Only edges missing some wanted bits are
requesting owners for the following recovery procedure.
[hnswlib/hnswalg.h:6315-6353]

A record is compatible with owner request `Q_j` only if complete `FT(w)` is contained in `Q_j`.
Independent matching bits from unrelated records are not a substitute.
Numerical postings narrow the scan, followed by full-mask containment across all columns.
[hnswlib/hnswalg.h:6176-6196,6230-6269]

Exclude the source and selected targets from requested witnesses; target-own records need no such migration.
Also exclude live direct RNG witnesses already assigned to live selected owners.
For every remaining live witness, first look for existing selected-edge coverage:
`FT(w) subset_of M(u,v)` AND `d(u,v) <= d(u,w)` AND `d(v,w) < d(u,w)`.
If such coverage exists, do not copy that record onto another replacement.
[hnswlib/hnswalg.h:6355-6397,6401-6431]

Otherwise, among requesting owners whose wanted mask contains the COMPLETE
record, choose the closest geometrically eligible live owner and OR all of `FT(w)` into its Marker.
Owner order is `(d(u,v), selected_slot)`: ties use row position, not universally target ID.
Do not copy the historical request mask directly to that edge.
[hnswlib/hnswalg.h:6399-6440]

```text
RECOVER_RECORDS(u, selected, wanted, C, direct_RNG_proofs):
    witnesses = union of complete-record matches for requesting selected edges
    remove u, selected targets, and currently valid direct-RNG-covered witnesses
    owners = selected edges ordered by (d(u,target), selected_slot)
    for each remaining currently live witness w:
        if some owner already covers all FT(w) and satisfies LIVE_SUPPORT(u,v,w):
            continue
        for requested owner v in owners:
            if FT(w) subset_of wanted[v] and LIVE_SUPPORT(u,v,w):
                M(u,v) = M(u,v) OR FT(w)
                break
    # LIVE_SUPPORT checks live(v), live(w), then d(u,v) <= d(u,w) AND d(v,w) < d(u,w)
```

This bounded reconstruction does not recover all historical contributors.
Retained old Markers are not globally re-proved record by record.
The cleanup bit-wise test and new-record complete-mask admission are distinct.
[hnswlib/hnswalg.h:5924-5944,6329-6334,6230-6269,6408-6440]

## 7. Coordinator and point/layer pseudocode

The parallel path follows. Section 9 details queue closure, late obligations, and retired sources.
[hnswlib/hnswalg.h:7036-7188,7314-7354]

```text
BATCH_DELETE(labels, P):
    ids = validate and resolve all labels to currently live internal IDs
    if ids is empty: return zero-work statistics
    P = clip resolved thread count to number of ids
    repair entry if already retired; enable the in-place maintenance mode
    if P == 1:
        for a in input order: serial prepare, mark, apply rows, retire outgoing
    else:
        create batch registry, publication gate, queues, epochs, row versions
        ParallelFor(ids, P): DELETE_POINT(a, batch)
        join all workers and aggregate statistics
        drain any remaining deferred obligations to a fixed point
    scrub every row: remove dead targets and clear retired sources' outgoing
    clear dirty/repair-candidate bookkeeping and replace a retired entry
    return statistics

DELETE_POINT(a, batch):
    E = completion watermark                         # before ANY search/snapshot
    lock row(a), exclusive publication gate, registry
    register active recipient; publish actual DELETE_MARK and bitmap for a
    release those locks; keep a's outgoing navigation intact
    for layer l containing a: compute R[l], X[l], and applicable row versions
    C = build immutable masks/postings/match cache from full R[0]
    attach E and source-specific expansion-version evidence to C
    for layer l containing a:
        seed source deltas with X[l] and, at layer0, C's source IDs
        for each live outgoing a -> v in a's frozen row:
            rank sources from ORIGINAL first50 positions of R[l] around v
            propose up to3 live sources u, carrying M(a,v) when applicable
        for each collected source delta:
            apply ROW_TRANSACTION(delta, C, R[l])
            note failed retired sources for outgoing-proposal refill
        refill failed outgoing sources from the SAME saved ranking, up to3
    service incoming descriptors; close only with empty queue and no producers
    repeatedly drain deferred descriptors before this worker returns
```

The separate serial path preserves input order and searches/plans before marking each point,
with a local selected-top-three `closest(anchor)` cache.
The parallel path recomputes rankings per call; outgoing endpoints keep individual refill lists, not an anchor cache.
Parallel interleavings need not reproduce the serial graph byte for byte.
[hnswlib/hnswalg.h:6071-6173,6555-6570,7112-7154,7295-7338]

## 8. Fused source-row transaction pseudocode

Row locks below release on every return; Marker operations apply only to base-layer edge-FT rows.

```text
ROW_TRANSACTION(delta, C, R_layer):
    u = delta.source
    lock row(u)
    if not live(u):
        if delta is an accepted incoming obligation: defer its original request
        return SOURCE_RETIRED                     # outgoing caller can refill
    cleanup = CLEAN_OLD_MARKER(C.deleted, u, C) if scheduled, else none
    read current adjacency and Markers under this same lock
    if no pending/additions/dead-or-self edges:
        apply a cleanup-only clearing patch, if any; return SOURCE_LIVE
    plan = current live nonself neighbors; accumulate any current u -> a request
    if u -> a was found, or an incoming obligation is pending:
        propose up to3 live targets ranked from ORIGINAL first50 of R_layer
    merge explicit outgoing additions and their complete requests
    for each attempted target already retired:
        hand off its original request as a mandatory obligation immediately
    if no structural/proposal change: apply cleanup-only patch; return SOURCE_LIVE
    selected, proofs = overflow-only RNG+CHT selection, or unchanged plan
    initialize selected Markers by target identity using the cleanup patch
    RECOVER_RECORDS(u, selected, wanted masks, C, proofs)
    harvest omitted old edges and original requests, including pruner omissions
    acquire shared publication gate              # still own row(u)
    recheck target liveness; compact neighbors and aligned Markers together
    preserve every retiring omission's stored/materialized/original request
    remember whether an omission was an explicit proposal, even with empty mask
    bump source version if neighbor IDs/order changed; publish the actual prepared row
    release publication gate
    enqueue qualifying retired-target obligations, never treating live prunes as deletes
    unlock row(u); return SOURCE_LIVE
```

The transaction rebases on the current locked row, not an old whole-row plan.
Already-retired proposals receive mandatory handoffs before plan admission.
Every admitted explicit proposal reaches the fence even if its edge exists,
its mask is empty, or it adds no new bits. Pure cleanup clearing can use the
early source-lock-only path: it introduces no edge or new support promise.
[hnswlib/hnswalg.h:6648-6757,6760-6855]

## 9. Engineering lifecycle safeguards

These safeguards preserve local obligations, not a new scoring rule or experimental maintenance policy.

`ParallelFor` claims whole point tasks via `std::thread` and an atomic next index.
There is no whole-batch premark or serial point-commit window; at most the worker
count of tasks is in progress, although completed points remain retired.
[hnswlib/parallel_for.h:11-54; hnswlib/hnswalg.h:7036-7051,7340-7349]

C++14 publishers share a publication gate while holding their own source row.
Actual registration plus retirement MARK takes that gate exclusively, then the
registry mutex. Publication therefore precedes target retirement, or observes
it and preserves a handoff; final liveness testing alone would be insufficient.
Publishers release the gate before registry/queue work. Registry-only reservation
and closure never acquire the gate or a row; closure may acquire its queue lock.
No search, geometry, Marker matching, or queue wait is inside the shared gate.
Pre-C++14 gate aliases retain exclusive publication. Progress assumes finite synchronous
work and normally progressing workers, not standard RW-lock fairness or writer preference.
[hnswlib/deletion_publication_gate.h:3-17;
hnswlib/hnswalg.h:6572-6598,6808-6854,7039-7046]

An active-recipient lookup reserves a producer under the registry lock;
the request is copied into its queue before releasing that reservation.
Closure requires an empty queue and zero producers, and records a completion
epoch under the registry lock. Queue waits release their mutex before row work.
[hnswlib/hnswalg.h:6494-6522,6572-6611,7157-7179]

A producer can arrive after recipient closure without ever having reserved.
For an actually retired target, explicit pending/new/consequential requests
remain mandatory. Incidental old edges are deferred only if recipient completion
postdates the producer's pre-search watermark; older completed ghosts retain
scrub-only treatment. Live neighbors omitted by ordinary pruning are not retired
recipients. Empty and no-new-bit promises cannot be inferred away from mask size.
[hnswlib/hnswalg.h:6627-6645,6715-6732,6780-6798,6826-6854]

Deferred descriptors retain only deleted target, source, layer, and request mask;
queue/source/gate/registry locks are not retained during their searches.
Each gets a new watermark before its first fresh construction-width search;
the same watermark is carried through any recipient and source contexts.
Accepted descriptors are not reclassified away as old incidental edges.
[hnswlib/hnswalg.h:6485-6492,6613-6625,6946-7007]

If source `S` retires during an `S -> D` obligation, inability to rewrite `S` does not cancel it.
First prepare a virtual incoming-repair row for frozen `S` in `D`'s context:
use `D`'s bounded witnesses and direct `S/V` geometry to materialize Markers.
Do not publish that row to retired `S`. Only its new/changed surviving edges
and materialized Markers feed the second stage, which searches around `S` and
revalidates final `U -> V` transfers in `S`'s context with direct `U/V` geometry.
Raw historical masks are not treated as validated intermediate Markers.
Retiring intermediate omissions instead preserve original requests as obligations.
[hnswlib/hnswalg.h:6659-6666,6858-6943,6967-7032]

Workers drain cascades; a final post-join drain precedes scrub and entry replacement.
Parallel-mode retired outgoing navigation remains available until quiescence.
Completion and scrub do not globally re-prove surviving Marker contributors.
[hnswlib/hnswalg.h:7110-7113,7181-7188,7205-7260,7345-7353]

## 10. Reuse actually present in V11

Own masks/postings are constructed once per context before structural ranking,
then reused by cleanup, new-edge initialization, and transfer.
Request-mask matching caches only immutable complete-record compatibility.
Current owner Markers, liveness, and geometric ownership are checked separately.
[hnswlib/hnswalg.h:5749-5815,6198-6271,6408-6440]

The per-point ordered-pair cache allocates dense storage for at most 512 IDs,
with a validity bitmap and 4096-entry direct-mapped overflow; distances are evaluated lazily.
These storage limits neither truncate witnesses nor imply eager all-pairs evaluation.
Direct RNG owner/witness evidence is reused within its row.
Cleanup and reconstruction have separate local geometry-status vectors:
there is no cross-phase Boolean geometry memo or parallel complete-ranking cache.
The retained scored vector and `plan.color_requests` have no new capacity reserves.
[hnswlib/deletion_cache.h:10-79; hnswlib/hnswalg.h:5924-5944,
6381-6397,6407-6422,6555-6570,6689-6732]

## 11. Limits, diagnostics, and paper-facing boundaries

The last whole-graph pass removes residual dead-target references at all layers,
clears retired sources' outgoing rows, and compacts aligned edge Markers.
It clears dirty/repair-candidate bookkeeping. A dead entry is replaced by a live
highest-level entry, or an invalid-entry/level-minus-one state if none remains.
It does not compact away stored vectors/labels or rebuild all surviving Markers.
[hnswlib/hnswalg.h:7190-7260,7262-7292,8461-8477]

Approximate incoming discovery and bounded witnesses are intentional:
"no support found" means none in this context, not none globally.
Shared, bucket-colliding, and historical bits can remain; `FT(a)` is not zeroed everywhere.
The algorithm does not guarantee unchanged recall, global connectivity, complete
historical Marker recovery, or three retained new edges per affected endpoint.
Expanded work, repeated handoffs, and the final scan mean the result width
is not a strict total-work bound or a complexity theorem.
Row locks and publication/registry coordination can contend; shared publication
does not imply unconditional speedup or a wall-time fairness guarantee.
[hnswlib/hnswalg.h:6021-6068,5924-5944,6946-7033,7223-7260]

Returned counters describe work, not maintenance-trigger decisions:
`candidate_sources` counts cleanup visits; `support_checks` counts first cleanup/reconstruction
support-geometry evaluations, not all distances or RNG tests. `rewired_nodes` counts row work, not
necessarily distinct sources or net new edges. Deferred searches contribute work.
Legacy speculative-preparation fields remain zero; point-parallel aliases exist.
Distance-cache hits/misses exclude uncached search-distance calls.
[hnswlib/hnswalg.h:5658-5692,5787-5793,5928,6419,6448,6993-7031;
python_bindings/bindings.cpp:1513-1550]

`delete_items` invokes neither query-triggered patching nor threshold-based `maintainDeletes`.
Explicit patch/rebuild and attribute-only APIs still exist as different operations.
Experiment-owner terminology: paper `point_update` means externally orchestrated
delete plus fresh-label add here, not native `update_point`/`updatePoint`.
The experiment orchestration and exact complete-record trace are documented in
`EXPERIMENTS.md` and the copied pipeline source.
[hnswlib/hnswalg.h:7314-7354,7702-7724;
python_bindings/bindings.cpp:1616-1663,1670-1734,2760-2773;
exp_benchmark/dynamic/runner.py:351-388; exp_benchmark/dynamic/protocol.py:143-220]

In this experiment, `get_deleted_ratio`, `get_dirty_count`, and
`repair_candidates_size` are observational state checks. They verify the
expected retired count and cleared maintenance bookkeeping; they do not
trigger a ratio/dirty-threshold repair policy. The timed operation is the
synchronous `delete_items` call itself.
[exp_benchmark/dynamic/runtime.py:526-550; exp_benchmark/dynamic/runner.py:351-388]
