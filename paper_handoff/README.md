# Frozen dynamic source and measurement evidence

The current paper uses
[`ema_dynamic_v11_recall95_20260914`](ema_dynamic_v11_recall95_20260914/README.md).
The earlier [`ema_dynamic_v11_20260913`](ema_dynamic_v11_20260913/README.md)
is retained only as the original recall90 and mutation provenance. Its QPS90
values must not be relabeled as QPS95.

The packages include the frozen V11 native source, original and recall95 query
code, stage metrics, exact-GT results, calibration arrays, timing records,
manifests and checksums. They do **not** include the multi-gigabyte source
vectors, graph checkpoints, or original compiled native binary.

These packages are immutable snapshots. Statements inside them about a
"local working tree" or code not yet released describe their creation time.
The snapshots are now versioned in this repository; their internal files and
hashes have not been rewritten to imply different experimental provenance.
The outer Git commit identifies publication, not a new benchmark execution.

The current `hnswlib/` source includes later changes, including incremental
categorical registration and scrub-time diagnostics. For an original V11
source inspection, use the package's `source_reference/hnswlib/`. Compiling it
produces a new binary artifact; do not assign the archived binary's SHA256 to
that build or use it as an undocumented substitute in a checkpoint replay.
Original machine paths in frozen manifests are provenance, not portable setup
instructions.

Verify either package from its directory:

```bash
sha256sum --check SHA256SUMS
```

For fresh runs versus compatible-checkpoint remeasurement, follow
[`exp_benchmark/dynamic/README.md`](../exp_benchmark/dynamic/README.md).
For Figure 7 rendering from the recorded values, follow
[`paper/README.md`](../paper/README.md).
