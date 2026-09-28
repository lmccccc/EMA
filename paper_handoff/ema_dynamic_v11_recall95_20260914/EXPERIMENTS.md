# Recall95 experimental protocol

## What changed

Only the recall target changed from90% to95%. The same native, actual archived
graphs, original1000 query vectors, fixed per-query mixed DNF, exact live-set GT,
trace and routing settings were retained. Every source90 calibration prefix
must reproduce the original ef, recall, returned-label and distance hashes before
a new95% result is admitted.

This is a read-only query remeasurement of saved graph states. The maintenance
times come from the original execution that produced those exact checkpoints.
They are not newly timed, inferred from QPS, or rescaled for the recall change.
Rows explicitly carry `queries_retimed=true`, `maintenance_reused=true`, and
`mutations_executed=false`.

## Data and predicate

SIFT10M has10M original128-dimensional vectors and original numerical/category-set
attributes. Distances are squared L2; k=10. Use the same first1000 original queries,
in the same order, with

```text
P_i(x) = ((lo_i <= numeric(x) <= hi_i) AND 9 in labels(x)) OR 12 in labels(x)
```

Bounds are inclusive and never regenerated after mutation. Mean selectivity is
9.79325947% on the full10M source; each live set has its own measured selectivity.
The predicate file is byte-identical to the existing Figure4(c) T10 file.

## Stages and lineage

Each operation has stage0 plus five1M-record stages. Insertion grows the original
first5M records to10M. Deletion starts from that exact insertion-final checkpoint
and removes the fixed `default_rng(12345).permutation(10M)` prefixes, ending with5M
live points. Complete-point update starts from the unchanged original5M prefix.
Each update stage deletes1M original points before adding1M complete replacement
source records at fresh labels; both vectors and all attributes are replaced.
There is no retired-slot reuse, attribute-only approximation, query-triggered
patch, or extra maintenance pass. Update has5M live points at completed stages.

## Fixed configuration

| Parameter | Value |
|---|---|
| Native | Retained V11, SHA `608e9ae692fa2fd9d5ec160a370a63b90419356c6b64d67285f10d6bd1e4f84e` |
| M / maximum bottom-layer adjacency | 40 / 80 |
| Construction and deletion-search ef | 300 |
| Edge Marker width | 128 bits per attribute |
| Marker/ranking metadata | format10, numeric2, owner2, distance-order1 |
| Candidate ordering | vector-distance-only-v1; CHT still constrains admission |
| Query ef_top | 1 |
| Query routing minimum degree | 8, unchanged |
| FT and routing / tail backfill | enabled / disabled |
| Query threads | one physical core |
| Recall target | 95% external-ID recall@10 |
| Mutation / original exact-GT threads | 32, on one NUMA node |
| Query frequency admission | 3.267--3.333GHz |

Static Figure4 uses routing minimum degree16. Restoring the shared95% recall
target does not make the routing settings identical. Compare absolute QPS only
with the same graph population, routing parameters, recall target and workload.

The source codebook and centroid-representative level plan remain the frozen
whole-source plan. No stage checkpoint is rebuilt for this revision.
New insertion-final and deletion-initial95% calibration results must agree for
their shared10M graph.

## GT, calibration and timing

Each stage reuses its original exact filtered top10 over that stage's live set.
Full10M stages retain the canonical GT IDs, including the original tie convention.
Returned IDs must be unique, live and predicate-valid, and their distances must
match direct float64 squared-L2 distances for the original source-row labels.
Point-update logical-slot and fresh-external-ID recall must agree.

Calibrate integer ef upward from k=10 to the first complete recall>=95%, using
recall only. Freeze the adjacent bracket before timing; calibration is not a
timed QPS curve. Warm up, then time current and initial graph endpoints in
rotating paired rounds on the same physical core. Retain the first seven complete
CPU-clean rounds on one core, taking median endpoint QPS and linear interpolation
against recall at95%. Do not choose fastest samples or pool successful rounds
from different cores. The original limits remain21 rounds per core and three
candidate cores; contaminated attempts are retained and cannot supply paper QPS.

If minimum legal ef10 is already above95%, strict `qps95` remains null and an
observed-recall value is reported separately; never extrapolate or relabel QPS90.
All figure values must come from this new revision, not the historical source90
timing records. The exact attained recalls and metric kinds are in the CSV.

## Index modification time

Time is minutes per1M logical record changes, not cumulative time. It includes
the original synchronous native calls and timed conversion/chunk dispatch, but
excludes GT, calibration, query timing, checkpoint IO, initial codebook/level
preparation and initial graph construction. Update adds its delete and insert
phase times for one1M-replacement batch, not two million independent operations.
