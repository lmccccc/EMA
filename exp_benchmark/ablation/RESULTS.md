# YouTube1M component ablation: 2026-11-01

**This run does not support a claim that every component improves performance.**
C2 provides substantial gains at intermediate selectivities but costs time at
100%. C3 restores complete top-10 returns in sparse cells, while adding a
measurable cost at 20-40%. C1 has no consistently positive effect in this
graph pair; its construction-to-construction variability remains unmeasured.

These are new current-implementation measurements, not replacements for the
historical appendix figures. The manuscript and published figures are unchanged.

## Protocol and complete result matrix

The study uses 1M YouTube-RGB vectors, 1024 dimensions, raw inner product,
and the first 1,000 original queries in each original range-AND-label cell.
Parameters are M40, ef-construction300, 128-bit edge Markers per attribute,
ef-top1, routing minimum degree16, and top-10 results. Two new graphs share
one explicit level plan and native seed, with CHT construction admission
enabled or disabled. Queries run on one physical core.

Each cell has seven complete CPU-clean paired timing rounds. Ratios below
are medians of within-round QPS ratios at **95% mean recall@10**. The target
is either observed exactly or interpolated inside a measured recall bracket.
It is not a requirement that every individual query have 95% recall.

| Nominal selectivity (%) | Actual mean (%) | C1 alone | Adding C2 | Adding C3 | C1 in full method | Full / baseline |
|---|---|---|---|---|---|---|
| 1 | 0.995 | -0.1% | N/A | N/A | -4.0% | 1.554x |
| 2 | 1.993 | -2.0% | N/A | N/A | -3.3% | 1.474x |
| 3 | 2.988 | -4.9% | N/A | N/A | -3.8% | 1.498x |
| 5 | 5.314 | -3.6% | N/A | N/A | -7.9% | 1.436x |
| 7 | 6.994 | -4.6% | N/A | N/A | -9.0% | 1.373x |
| 10 | 9.995 | +3.7% | N/A | N/A | -1.0% | 1.535x |
| 20 | 20.007 | -2.1% | +61.7% | -5.2% | -1.2% | 1.499x |
| 40 | 40.042 | +0.9% | +33.4% | -2.4% | -2.1% | 1.314x |
| 60 | 60.019 | +1.9% | +14.6% | approximately 0% | -2.5% | 1.169x |
| 80 | 81.036 | +2.5% | +6.4% | -0.4% | +2.6% | 1.086x |
| 100 | 100.000 | +1.9% | -3.3% | approximately 0% | +1.9% | 0.985x |

"C1 alone" is `c1/baseline - 1`; "Adding C2" is `c1_c2/c1 - 1`;
"Adding C3" is `full/c1_c2 - 1`; "C1 in full method" is
`full/without_c1 - 1`. These conditional effects are not additive.
The baseline is the same implementation/layout without C1/C2/C3, not an
upstream HNSW binary.

**N/A is not zero recall or zero QPS.** At 1-10%, the C1+C2 variant raises
the native fewer-than-k exception at every sampled ef, including ef4096.
At least one original query cannot return ten results, preventing a valid
complete-batch recall/QPS observation. This does not establish that its
mean recall, under some different partial-result API, would be below 95%.
There is therefore no finite C2-only or C3-increment QPS95 ratio in these cells.

The actual selectivity of the nominal 5% and 80% cells is 5.314% and 81.036%.
Both numeric ranges and categorical labels follow the original workloads;
the cells are not a fixed-label numeric-range-only sweep.

## Why recovery matters in the sparse cells

An independent diagnostic-only replay finds the first failing original query
at ef4096, then changes one query switch on the same graph, query, predicate,
and ef. No wall times from this replay enter the performance results.

| Nominal (%) | First failing query ID (zero-based) | C1+C2 expanded rows | Rows with no passing Marker | C1+C2 passing edge positions |
|---|---|---|---|---|
| 1 | 0 | 2 | 1 | 1 |
| 2 | 22 | 4 | 1 | 4 |
| 3 | 16 | 1 | 1 | 0 |
| 5 | 102 | 2 | 0 | 2 |
| 7 | 76 | 2 | 0 | 2 |
| 10 | 741 | 3 | 0 | 4 |

For all six witnesses, ef10 and ef4096 fail after the same small number of
expansions. Disabling C2, or enabling C3 at the same ef4096, restores ten
valid results. The full method also returns ten results at its original
selected ef32-48 for each witness. Completeness is restored; this is not a
claim of 100% recall for each query.

The failure is not simply an insufficient ef budget. The Marker-restricted
traversal exhausts a tiny reachable component. Importantly, the 5%, 7%, 10%
witnesses contain **no all-negative Marker row**: their passing edges lead
among already visited nodes. Thus, handling only completely empty Marker
rows would not address all the observed failures.

C3 supplies routing-only edges out of these restricted components.
This demonstrates recovery for the measured queries, not a proof of universal
connectivity. A negative Marker is also not a certificate that every two-hop
or distant point is invalid. These deliberately selected first failures do
not estimate failure frequency or mean recall.

## Recovery can also cost throughput

At 20%, C3 reduces the selected ef from 67 to 54, and the observed operating
points expand fewer rows: 180.955 to 172.221 per query. Nevertheless, C3
selects 446.327 backfill edge positions and newly visits 348.660 routing-only
targets per query. Newly scored base-layer neighbors increase from 1934.682
to 2071.099, while paired QPS95 falls 5.2%.

At 40%, C3 costs 2.4%; at 60-100%, its measured effect is approximately neutral.
The correct description is a selectivity-dependent reachability/throughput
tradeoff, not an unconditional acceleration from recovery or from lowering ef.
Work counts describe changed processing; they do not identify a hardware
bottleneck or prove that distance arithmetic is expensive.

## Marker gains disappear when there is nothing to filter

At 20%, 40%, 60%, 80%, adding C2 without recovery improves paired QPS95 by
61.7%, 33.4%, 14.6%, 6.4%, respectively.

The 100% cell is a particularly clean control. C1, C1+C2, and the full method
use the same selected ef63 and return exactly the same labels and distances.
Their expanded-row and newly scored-neighbor counts are also identical:
65.556 and 2332.223 per query. Every scanned edge passes the Marker, and no
backfill edge is selected. Enabling C2 nonetheless costs approximately 3.3%
QPS. Enabling C3 on top of C2 has effectively no further effect.

The full method consequently ends 1.5% below the baseline at 100%, despite
being 1.37-1.55x faster across the measured 1-20% cells. This run does not support
a blanket claim of a speedup over the entire 1-100% range.

## C1 and the limits of this construction comparison

C1 alone ranges from a 4.9% loss to a 3.7% gain. Within the full method it is
negative in nine of eleven cells, including 7.9% and 9.0% losses at 5% and 7%.
The CHT-off/on graphs have mean base-layer degrees 20.049 and 19.590.

These observations do not justify describing C1 as a consistently beneficial
step on this dataset. However, there is only **one graph per construction
policy**. Parallel insertion interleavings differ, and seven query timing
rounds do not provide seven independent construction samples. Small C1
effects must not be presented as construction-robust causal estimates.
A repeated construction study would be needed before generalizing their sign.

## Connection to the sparse-routing question

At 1%, the full method expands 2064.040 rows per query, versus 1706.437 for C1
without Marker or recovery: about 21% **more**, not fewer. It scans 31.7% more
edge positions, but performs 81.5% fewer actual predicate evaluations.
Its newly scored neighbors fall from 29610.589 to 16087.146; 13480.715 of those
are routing-only targets, approximately 83.8%.

Thus the sparse full method is heavily dependent on backfill, even while it
outperforms the unfiltered-routing control. The result cannot be explained
as "Marker simply reduces expanded rows." Nor does the reduction in distance
counts establish the runtime bottleneck. A future geometric skipping design
would need to preserve paths out of small passing-edge components, not just
react to rows where all Markers are negative.

## Reproduction and artifact identity

See [README.md](README.md) for the executable native build, main runner,
first-failure diagnostic, and R/PDF/TikZ plotting commands.

The complete local artifact root is:

```text
/mnt/data/mocheng/dataset/youtube1m/hashann/benchmarks/component_ablation_20261101_FDIj854s/full
```

It contains `request.json`, source snapshots, the two graph build records,
all raw calibration arrays, accepted/rejected timing samples, separate
work diagnostics, `failure_diagnostics/`, the three result CSVs, and `plots/`.
The large vectors and graphs are not bundled into the repository.
`add_wall_s` records native insertion time, not end-to-end construction cost.

| Artifact | SHA256 |
|---|---|
| Native patch | `7a23594124b7ea3ad27369ff2e94ccd0e1948b976054f9c17e1ed43f029720be` |
| Timing native | `9e923038e269fe6d53f8549ad0ff1a21bf55cba63111167f691dbfd19e6ee021` |
| Diagnostic native | `7452386759a2c2d845b73f5a3f44d31c5a9dbb8a8e142c897e8196692926c235` |
| `summary.csv` | `6cf3f2c1bf18aeefbed47fe5e5e41b45d9b9efa0ed49bd21725c575bd6eed23b` |
| `curves.csv` | `49814e8fdc3e5f685f458cc306b98c30819889112ff2ca253f71ad1099698750` |
| `contrasts.csv` | `065efc6f00da8b3ffc3d1bc0527a97670def84b81b0c6da244127b628db14313` |

The native base commit is `d95fb49973f4fe53f5a2c1ac444dd5fdb879f551`.
The CHT-off graph SHA256 is
`42f4f1e3d3c963dc1326e177fb50b265ffe8d421982008c9638e58b917a56568`;
the CHT-on graph SHA256 is
`cbad22f76020a59dfe6a26e4e0c43881a7e9765b8447865ed367699c58f18feb`.
Their shared level payload SHA256 is
`35fc1fc4431b24b089a77128c9b9681ef66631f73a7716280508e9ca709a81f2`.
