# Completed V11 experimental results

All numbers below are derived from the byte-identical published `data/paper.json`.
Use `data/stage_metrics.csv` for full-precision portable numeric data.
QPS is single-thread queries/second; mutation phases used 32 workers.

## 1. Maintenance totals

Each operation contains five sequential 1M-record batches. Stage0 is excluded.
Times include the timed mutation call boundaries, not GT, query measurement,
initial construction, or checkpoint I/O. Point update is one logical
replacement comprising a delete followed by a fresh complete-record add.

| Operation | Records | Total seconds | Total minutes | Mean minutes / 1M |
| --- | --- | --- | --- | --- |
| insert | 5000000 | 603.829180 | 10.0638 | 2.0128 |
| delete | 5000000 | 1759.797508 | 29.3300 | 5.8660 |
| point_update | 5000000 | 2310.988279 | 38.5165 | 7.7033 |

The first 10M->9M deletion batch took **321.055919 seconds (5.3509 minutes)**.
These are actual full-scale observations, not extrapolations from a small graph.

### Every 1M mutation batch, seconds

| Batch | Insert | Delete | Complete point update |
| --- | --- | --- | --- |
| 1 | 119.160934 | 321.055919 | 458.699308 |
| 2 | 120.772895 | 345.904941 | 459.984158 |
| 3 | 123.410962 | 365.121941 | 462.605361 |
| 4 | 119.632481 | 359.985734 | 464.007417 |
| 5 | 120.851908 | 367.728972 | 465.692036 |

## 2. QPS at 90% ID recall

Rows use cumulative operation count, not the same live population across columns.
Insertion grows5M->10M; deletion shrinks10M->5M; point update keeps5M live points.
Stage0 is the corresponding initial graph. QPS90 is obtained using the
median paired-round endpoint QPS and the legal adjacent recall bracket.

| Stage | Cumulative records | Insert QPS90 | Delete QPS90 | Point-update QPS90 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 2506.09 | 2232.85 | 2515.80 |
| 1 | 1000000 | 2426.21 | 2227.79 | 2537.02 |
| 2 | 2000000 | 2316.24 | 2334.94 | 2549.42 |
| 3 | 3000000 | 2240.12 | NA | 2608.03 |
| 4 | 4000000 | 2242.17 | NA | 2587.26 |
| 5 | 5000000 | 2230.73 | NA | 2551.61 |

### Minimum-ef observations: not QPS90 values

For the following rows, ef10 already exceeds90% recall. The QPS90 column
is intentionally null. The observed points must not be substituted into
a strict QPS90 curve or extrapolated below the legal minimum ef=k.

| Operation | Stage | Live points | ef | Observed recall (%) | Observed QPS |
| --- | --- | --- | --- | --- | --- |
| delete | 3 | 7000000 | 10 | 90.23 | 2496.53 |
| delete | 4 | 6000000 | 10 | 91.03 | 2377.10 |
| delete | 5 | 5000000 | 10 | 92.18 | 2289.90 |

## 3. Full stage alignment

### insert

| Stage | Occupied | Deleted | Live | QPS90 | Selected ef | Recall at ef (%) | Observed QPS | Mean live selectivity (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5000000 | 0 | 5000000 | 2506.09 | 26 | 90.29 | 2449.98 | 9.788069 |
| 1 | 6000000 | 0 | 6000000 | 2426.21 | 27 | 90.38 | 2352.48 | 9.788678 |
| 2 | 7000000 | 0 | 7000000 | 2316.24 | 28 | 90.15 | 2287.02 | 9.786415 |
| 3 | 8000000 | 0 | 8000000 | 2240.12 | 29 | 90.28 | 2199.76 | 9.793659 |
| 4 | 9000000 | 0 | 9000000 | 2242.17 | 29 | 90.05 | 2232.31 | 9.792458 |
| 5 | 10000000 | 0 | 10000000 | 2230.73 | 29 | 90.09 | 2214.37 | 9.793259 |

### delete

| Stage | Occupied | Deleted | Live | QPS90 | Selected ef | Recall at ef (%) | Observed QPS | Mean live selectivity (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10000000 | 0 | 10000000 | 2232.85 | 29 | 90.09 | 2214.95 | 9.793259 |
| 1 | 10000000 | 1000000 | 9000000 | 2227.79 | 16 | 90.66 | 2104.75 | 9.789866 |
| 2 | 10000000 | 2000000 | 8000000 | 2334.94 | 12 | 90.40 | 2266.99 | 9.789949 |
| 3 | 10000000 | 3000000 | 7000000 | NA | 10 | 90.23 | 2496.53 | 9.781280 |
| 4 | 10000000 | 4000000 | 6000000 | NA | 10 | 91.03 | 2377.10 | 9.784914 |
| 5 | 10000000 | 5000000 | 5000000 | NA | 10 | 92.18 | 2289.90 | 9.792059 |

### point_update

| Stage | Occupied | Deleted | Live | QPS90 | Selected ef | Recall at ef (%) | Observed QPS | Mean live selectivity (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5000000 | 0 | 5000000 | 2515.80 | 26 | 90.29 | 2472.57 | 9.788069 |
| 1 | 6000000 | 1000000 | 5000000 | 2537.02 | 16 | 90.65 | 2428.69 | 9.783441 |
| 2 | 7000000 | 2000000 | 5000000 | 2549.42 | 15 | 90.75 | 2394.00 | 9.784663 |
| 3 | 8000000 | 3000000 | 5000000 | 2608.03 | 14 | 90.39 | 2538.42 | 9.797758 |
| 4 | 9000000 | 4000000 | 5000000 | 2587.26 | 14 | 90.24 | 2541.99 | 9.796031 |
| 5 | 10000000 | 5000000 | 5000000 | 2551.61 | 14 | 90.07 | 2537.48 | 9.798450 |

## 4. Paired relative query change

This is the published current/paired-initial QPS90 minus one, not a ratio
to an independently timed baseline from another stage. NA is retained
when no legal current QPS90 bracket exists.

| Stage | Insert change (%) | Delete change (%) | Point-update change (%) |
| --- | --- | --- | --- |
| 0 | -1.3371 | -0.5634 | -0.6607 |
| 1 | -4.5953 | -0.7229 | -0.1563 |
| 2 | -8.8338 | 3.8662 | 0.2925 |
| 3 | -11.8503 | NA | 1.9174 |
| 4 | -12.7223 | NA | 1.9210 |
| 5 | -13.0519 | NA | -0.3216 |

## 5. Sampling and interpretation

- Recorded timed query batches: 556.
- Timed batches belonging to accepted paired rounds: 483.
- Individually rejected/noisy timed batches: 22.
- Other timed batches not selected for the paper: 51.
- The selection is complete paired rounds, not independent filtering of each case.
- Published endpoint QPS uses medians of seven accepted rounds on one core.
- Five mutation batches are evolving stages, not independent repetitions.
- The data do not separately time structural deletion, Marker reconstruction,
  and Marker cleanup inside a native delete call. Do not infer that split
  from work counters or label work counts as unique affected nodes/edges.
- This package does not contain cross-system comparisons or confidence
  intervals over independently rebuilt indexes.

The original raw stage JSON contains every timing attempt, sample, core,
admission flag, and rejection reason. The CSVs are derived conveniences;
the published originals and committed stage results are authoritative.
