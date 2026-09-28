# Completed V11 recall95 results

Query QPS was remeasured from the archived graphs. Mutation timings are unchanged original observations.

## QPS at95% ID recall

| Stage | Cumulative records | Insert QPS95 | Delete QPS95 | Update QPS95 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1596.89 | 1310.11 | 1593.98 |
| 1 | 1000000 | 1540.28 | 1371.96 | 1601.29 |
| 2 | 2000000 | 1497.07 | 1460.28 | 1617.73 |
| 3 | 3000000 | 1402.36 | 1533.70 | 1587.88 |
| 4 | 4000000 | 1304.09 | 1552.92 | 1634.73 |
| 5 | 5000000 | 1316.67 | 1682.09 | 1562.65 |

## Original maintenance totals

| Operation | Records | Total minutes | Minutes per1M |
| --- | --- | --- | --- |
| insert | 5000000 | 10.063820 | 2.012764 |
| delete | 5000000 | 29.329958 | 5.865992 |
| point_update | 5000000 | 38.516471 | 7.703294 |

## Full stage alignment

| Operation | Stage | Live | QPS95 | ef | Attained recall (%) | Observed QPS | Live selectivity (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| insert | 0 | 5000000 | 1596.891 | 41 | 95.0900 | 1583.759 | 9.788069 |
| insert | 1 | 6000000 | 1540.277 | 43 | 95.1500 | 1502.284 | 9.788678 |
| insert | 2 | 7000000 | 1497.072 | 44 | 95.1900 | 1464.489 | 9.786415 |
| insert | 3 | 8000000 | 1402.357 | 47 | 95.1000 | 1373.967 | 9.793659 |
| insert | 4 | 9000000 | 1304.090 | 49 | 95.0500 | 1297.969 | 9.792458 |
| insert | 5 | 10000000 | 1316.674 | 49 | 95.2200 | 1295.549 | 9.793259 |
| delete | 0 | 10000000 | 1310.109 | 49 | 95.2200 | 1284.395 | 9.793259 |
| delete | 1 | 9000000 | 1371.963 | 25 | 95.1100 | 1350.886 | 9.789866 |
| delete | 2 | 8000000 | 1460.276 | 20 | 95.2600 | 1393.561 | 9.789949 |
| delete | 3 | 7000000 | 1533.704 | 17 | 95.1900 | 1491.330 | 9.781280 |
| delete | 4 | 6000000 | 1552.917 | 16 | 95.3500 | 1489.664 | 9.784914 |
| delete | 5 | 5000000 | 1682.088 | 14 | 95.1100 | 1662.321 | 9.792059 |
| point_update | 0 | 5000000 | 1593.978 | 41 | 95.0900 | 1577.749 | 9.788069 |
| point_update | 1 | 5000000 | 1601.294 | 25 | 95.1300 | 1581.490 | 9.783441 |
| point_update | 2 | 5000000 | 1617.735 | 23 | 95.0000 | 1617.735 | 9.784663 |
| point_update | 3 | 5000000 | 1587.877 | 23 | 95.0300 | 1580.511 | 9.797758 |
| point_update | 4 | 5000000 | 1634.728 | 23 | 95.2100 | 1592.405 | 9.796031 |
| point_update | 5 | 5000000 | 1562.655 | 24 | 95.1300 | 1534.586 | 9.798450 |

## Interpretation

Insertion changes graph size; deletion changes both the live population and repaired graph; update preserves5M live points only at stage boundaries. These are different evolving traces, not interchangeable static baselines or independent-build replications.

504 timing samples belong to accepted complete paired rounds out of 696 recorded samples.
