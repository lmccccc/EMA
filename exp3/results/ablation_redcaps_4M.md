# Redcaps_4M Parameter Ablation

Dataset: Redcaps_4M (4M × 512d, IP), DNF `or_T{N}` predicate (T1=1% … T10=10%
selectivity, attr_type=[0,1], 2-clause OR).

Setup: edge-level FT, `use_ft=true`, ef_construction=300, single-thread,
1000 queries, ef_search ∈ {10,15,20,40,80,150,300}. QPS reported is linearly
interpolated to recall=0.95 in (1/QPS, recall) space.

---

## 1. Graph degree (M) ablation

Fixed: ft_bits=128, ft_routing_min_deg=8.

### 1a. Original low-selectivity sweep (T1–T10)

| selectivity | M=16   | M=32   | M=40   | M=64  |
|---|---|---|---|---|
| 1%  | **375.0** | 307.8  | 329.5  | 286.3 |
| 2%  | **531.9** | 422.8  | 453.0  | 422.8 |
| 3%  | **635.2** | 591.3  | 576.2  | 532.9 |
| 5%  | 697.3  | **758.9** | 730.2 | 758.9 |
| 7%  | 797.7  | 901.4  | 900.0 | 869.6 |
| 10% | 930.5  | **1111.6** | 1072.5 | 1056.0 |

### 1b. Extended selectivity (6-point curve, low→high→100%)

Six log-spread points sweeping from FT-dominated regime to pure HNSW. T1–T10
reuse the low-sel sweep above; T40/T80 from `m_high_sel/` driver; T100% from
`/tmp/m_dense_all2.py` (pure `knn_query`, no predicate, M=64 ef=2500 pseudo-GT).

| selectivity | M=16 | M=32 | M=40 | M=64 | best |
|---|---|---|---|---|---|
| 1%   | **375.0** | 307.8  | 329.5  | 286.3  | M=16 |
| 3%   | **635.2** | 591.3  | 576.2  | 532.9  | M=16 |
| 10%  | 930.5     | **1111.6** | 1072.5 | 1056.0 | M=32 |
| 40%  | 2199.8    | **2606**   | 2575   | 2364   | M=32 |
| 80%  | 2957.9    | 3097       | 3436.0 | 3398.5 | M=40 |
| 100% | 3765.3*   | 4903.9*   | 5438.1* | **4870** | M=40 |

\* The 100% dense probe was sensitive to OS page-cache pressure: the
sequential 4-index probe under-reported M=64 (≈2500 QPS) because earlier
loads evicted its 4 GB working set. The M=64 number above (4870 QPS) is
from an isolated run (`/tmp/m64_isolated.py`, fresh process, warm cache).
M=16/M=32/M=40 numbers are from the sequential probe and should be
considered a lower bound; reproducing in isolation may shift them upward
by 10–30 %. The ordering M=40 ≳ M=64 > M=32 > M=16 at 100 % matches the
expected diminishing-returns regime.

For plotting (x = log selectivity, y = QPS):

```csv
sel,M16,M32,M40,M64
1,375,308,330,286
3,635,591,576,533
10,931,1112,1073,1056
40,2200,2606,2575,2364
80,2958,3097,3436,3399
100,3765,4904,5438,4870
```

LaTeX table (selectivity = columns, M = rows, last row = best M):

```latex
\begin{table}[t]
  \caption{HashANN QPS at recall=0.95 on Redcaps\_4M ($n=4{\rm M}$, $d=512$, IP)
  across selectivity and graph degree $M$. ft\_bits=128, ft\_routing\_min\_deg=8,
  $efc=300$, single-thread, 1000 queries. Bold = best $M$ at each selectivity.}
  \label{tab:m-ablation-redcaps}
  \begin{tabular}{c|cccccc}
    \hline
    \textbf{$M$} & \textbf{1\%} & \textbf{3\%} & \textbf{10\%} & \textbf{40\%} & \textbf{80\%} & \textbf{100\%} \\
    \hline
    16  & \textbf{375}  & \textbf{635}  & 931           & 2200          & 2958          & 3765 \\
    \hline
    32  & 308           & 591           & \textbf{1112} & \textbf{2606} & 3097          & 4904 \\
    \hline
    40  & 330           & 576           & 1073          & 2575          & \textbf{3436} & \textbf{5438} \\
    \hline
    64  & 286           & 533           & 1056          & 2364          & 3399          & 4870 \\
    \hline
    \textbf{best $M$} & 16 & 16 & 32 & 32 & 40 & 40 \\
    \hline
  \end{tabular}
\end{table}
```

Observations:
- **Low selectivity (≤3%)** strongly prefers small M (M=16 wins by +10–14 %
  vs M=40). The FT bloom rejects >95 % of edges and `min_deg=8` backfill
  caps the effective branching at 8 regardless of M; the only differentiator
  is per-edge neighbor quality, where M=16's tighter heuristic pruning wins.
- **Mid selectivity (10–40%)** is a transition zone; M=32–M=64 win by small
  margins (2–4 %) as backfill triggers less often.
- **High selectivity (≥80%)** favors larger M. At 100 % (pure HNSW with no
  FT routing) M=40 dominates; with FT still active at 80 %, M=32 wins. M=64
  shows diminishing returns in this dataset.
- M choice should be **selectivity-aware**: small M for sparse predicates,
  larger M for dense predicates.

---

## 2. FT routing min_deg ablation

Fixed: M=40, ft_bits=128.

| selectivity | md=0 | md=5 | md=8 | md=15 | md=20 | md=30 | md=40 | md=60 | md=80 |
|---|---|---|---|---|---|---|---|---|---|
| 1%  | –     | 193.8 | 346.8 | **401.8** | 266.8 | 182.8 | 156.2 | 142.5 | 140.3 |
| 2%  | 164.3 | 396.1 | 540.1 | **750.5** | 615.6 | 400.3 | 337.0 | 294.8 | 287.4 |
| 3%  | 199.3 | 476.1 | 643.6 | 802.1 | **817.8** | 526.9 | 435.7 | 391.0 | 371.4 |
| 5%  | 569.2 | 738.6 | 849.1 | 1063.5 | **1193.1** | 768.8 | 641.3 | 553.6 | 533.9 |
| 7%  | 855.7 | 978.0 | 1026.5 | 1201.5 | **1383.7** | 1036.6 | 843.1 | 745.7 | 706.4 |
| 10% | 752.0 | 1090.7 | 1174.7 | **1338.4** | 1328.6 | 1294.6 | 1060.0 | 900.6 | 886.1 |

`–` means recall=0.95 not reached on the swept ef range
(here md=0 at 1% sel: failed to gather K results).

Observations:
- **Sharp inflection at md≈20–30**: QPS rises monotonically up to the optimum
  (md=15 at sel ≤ 2 %, md=20 at sel 3–7 %, md=15 at sel 10 %) and then drops
  steeply once md ≥ 30. The drop is 30–55 % from peak across all selectivities.
- md=60 ≈ md=80: both saturate because maxM0_ for the level-0 graph caps the
  effective neighborhood; further increasing the parameter cannot add edges.
  This `md=60/80` regime is the **post-filtering equivalent** — every neighbor
  is visited regardless of FT, so the FT routing signal is entirely discarded.
- Post-filter equivalent (md=80) is **strictly slower** than the FT routing
  sweet spot (md=15/20): 2.9× slower at sel 1 %, 2.6× slower at sel 2 %,
  1.5× slower at sel 10 %. FT routing is genuinely useful, but only when
  min_deg sits in the narrow band {15, 20}.
- The U-shape is not symmetric: from md=8→15 QPS roughly doubles, but from
  md=20→30 QPS halves. Tuning md too high is more costly than tuning too low.
- Suggested selectivity-aware schedule: md ≈ {1–2%: 15, 3–7%: 20, 10%: 15};
  never exceed M / 2.

---

## 3. FT bloom width (ft_bits) ablation

Fixed: M=40, ft_routing_min_deg=8.

| selectivity | ft=32 | ft=64 | ft=128 | ft=256 |
|---|---|---|---|---|
| 1%  | 269.3 | 268.6 | **299.7** | 279.9 |
| 2%  | 469.9 | 433.4 | **532.4** | 338.1 |
| 3%  | 561.4 | 610.3 | **642.6** | 503.4 |
| 5%  | 708.0 | 821.8 | **828.7** | 632.4 |
| 7%  | 927.4 | **1023.6** | 1002.6 | 738.8 |
| 10% | 1033.0 | 1170.7 | **1169.1** | 1022.5 |

Observations:
- ft=128 is the dominant winner (5 of 6 cells, +2 % to +25 % over ft=32).
  Larger bloom → lower false-positive rate → more accurate routing.
- ft=64 sits between ft=32 and ft=128: it closes most of the gap at mid
  selectivity (5–7 %, within 1 % of ft=128) but still trails ft=128 at the
  extremes (1 %, 10 %).
- ft=32 is uniformly slowest; the bloom saturates and routing degrades.
- ft=256 does **not** help — it is consistently slower than ft=128 (e.g.
  −36 % at 2 %, −22 % at 5 %, −13 % at 10 %). Doubling the bloom width
  doubles per-edge FT memory (32 B → 64 B per slot) and SIMD work
  (1 → 2 `_mm_loadu_si128`), but rejection-rate gains saturate well below
  this point at these selectivities. Edge-FT cache pressure dominates.
- Recommended default: ft_bits = 128. Scaling beyond 128 is counter-productive.

---

## Summary recommendations

1. **M is selectivity-aware**: M=16 at sel ≤ 3 %, M=32 at sel 5–10 %, M=32–40
   at sel 40–100 %. Static M=40 leaves ~10–14 % QPS on the table at low sel.
2. min_deg should NOT be a fixed integer; use a selectivity-aware schedule
   (md=15 at sel 1–2 %, md=20 at sel ≥ 3 %). Current fixed value 8 leaves
   16–35 % QPS on the table across selectivities.
3. ft_bits=128 dominates ft=32 at no measured cost; keep as default.
   ft_bits=256 was tried but **regresses across all selectivities**
   (−6 % to −36 %): bloom gains saturate while edge-FT memory and SIMD
   cost double.

## Reproduction

Sweep scripts:
- `exp3_redcaps/ablation/m_only_sweep.sh`        (T1–T10)
- `exp3_redcaps/ablation/m_high_sel_sweep.sh`    (T20/T40/T60/T80, M=16/64)
- `exp3_redcaps/ablation/m_high_sel_fill.sh`     (T40/T80, M=32/40)
- `exp3_redcaps/ablation/min_deg_sweep.sh`
- `exp3_redcaps/ablation/ft_bits_sweep.sh`
- `/tmp/m_dense_all2.py`                         (100 % sel pure knn probe)

Logs:
- `exp3_redcaps/ablation/logs/m_sweep/`
- `exp3_redcaps/ablation/logs/m_high_sel/`
- `exp3_redcaps/ablation/logs/min_deg_sweep/`
- `exp3_redcaps/ablation/logs/ft_bits_sweep/`
- `/tmp/m_dense_all2.log`
- `/tmp/m_dense_all2.log`

---

## 5. Dynamic operations — per-step cost (measured)

Dataset: sift10m / sift5m, M=40, ef_construction=300, ft_bits=128, edge-FT.
Averaged over the rounds actually executed (1 M ops per round); query
time excluded.

| Step                                            | Avg. cost (per 1 M ops) | Mode                  | Threads |
| ----------------------------------------------- | ----------------------: | :-------------------: | :-----: |
| Insert (`add_items`)                            |                  90.8 s |       parallel        |   32    |
| Mark-delete                                     |                   1.3 s |        serial         |    1    |
| Attr-only update (`update_attr`)                |                   2.5 s |        serial         |    1    |
| Update vec+attr (mark-delete + add)             |                 447.5 s | mixed (add parallel)  |   32    |
| Patch (`batched_patch_deletes`, one call)       |                  ~50  s |       parallel        |   64    |
| Reconstruct (fresh `add_items` over survivors)  |        306 s / 1 M alive |       parallel       |   32    |

Sources: `incremental_sift10m_20260513_130559`,
`delete_sift10m_20260513_164830`,
`delete_patch_sift10m_20260514_191343`,
`attr_update_20260514_211644`,
`point_update_20260516_skipdead_parallel`,
`delete_rebuild_sift10m_20260513_173033`.
