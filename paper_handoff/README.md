# Frozen dynamic implementation

`ema_dynamic_v11_recall95_20260914/source_reference/` preserves the V11
native and experiment sources used for the dynamic result. The original
recall90 snapshot was identical for the files it shared with this version;
the recall95 snapshot also includes the query remeasurement code.

These are source references, not a complete standalone build tree: files
unchanged from the original repository are not duplicated here. The current
`hnswlib/` and `exp_benchmark/dynamic/` contain later changes and must not be
presented as the frozen V11 binary or an unchanged-checkpoint replay.
Raw measurements, calibration arrays, run manifests and checkpoints are not
distributed. The [paper CSV](../paper/figures/data/ema_dynamic_v11.csv)
is retained solely as the dynamic figure's plotting input.
