# HashANN Repo Audit (2026-03-13)

## Inventory Snapshot
- Shell scripts: 152
- Python files: 60
- Repeated experiment tracks: `root`, `exp2`, `exp3`, `exp4`

## Architecture Summary
- Configuration hub: `conf.sh` in each track.
- Backend overlays: `*_conf.sh` (`diskann`, `navix`, `irange`, `milvus`, `msvbase`, `acorn`).
- Build/query scripts source config and execute backend binaries or python entrypoints.
- Batch sweep scripts mutate `conf.sh` with `sed -i` for dataset/selectivity loops.

## Findings (from `./tools/repo_guard.sh`)

### P0: Broken Script Syntax
- `exp2/exp_loop.sh` has shell syntax error (unexpected EOF).
- Impact: orchestration can stop mid-run and leave modified `conf.sh` state.

### P1: Missing Sourced Files
- `exp2/milvus_hnsw_index2.sh` sources `./conf2.sh` (missing).
- Impact: this script is not runnable in current state.

### P2: In-Place Global Config Mutation
- Many loop scripts perform `sed -i ... conf.sh` across tracks.
- Impact: hidden state drift between runs and high risk of accidental cross-experiment contamination.

### P2: External Binary Path Fragility
- Multiple scripts reference `../.../build/...` paths not resolved in this workspace.
- Examples include DiskANN/iRange/Navix/ACORN binaries.
- Impact: script behavior depends heavily on local directory layout and external build status.

## What Was Added For Ongoing Management
- `docs/PROJECT_MAP.md`: flow map, dependency model, and operating rules.
- `tools/repo_guard.sh`: one-command guardrail checks.
- README now includes safety workflow and guard command.

## Recommended Next Steps
1. Fix `exp2/exp_loop.sh` syntax first.
2. Decide whether `milvus_hnsw_index2.sh` should use `conf.sh` or add `conf2.sh`.
3. Replace direct `sed -i conf.sh` with temporary copies (or trap-based restore).
4. Centralize external binary roots (single variable per backend config).
5. Add small smoke-run script that validates one dataset + one selectivity end-to-end.
