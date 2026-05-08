# HashANN Codebase Map

## Scope
This repository is an experiment orchestration workspace with shell-first pipelines.
Main workload lives in four parallel tracks:
- `./` (root)
- `./exp2`
- `./exp3`
- `./exp4`

Each track repeats a similar script set and differs by parameter presets and data paths.

## Core Flow
Typical end-to-end flow inside one track:
1. Configure run in `conf.sh` (`dataset`, `attr_type`, `query_sel`, ANN params).
2. Generate metadata:
   - `attr_generator.sh`
   - `predicate_generator.sh`
   - `ground_truth_generator.sh`
3. Build index for one backend:
   - `hashann.sh`, `diskann_index.sh`, `navix_index.sh`, `irange_index.sh`, `milvus_hnsw_index.sh`, `msvbase_hnsw_index.sh`, `acorn_index.sh`
4. Query/evaluate:
   - `query.sh`, `diskann_query.sh`, `navix_query.sh`, `irange_query.sh`, `milvus_hnsw_query.sh`, `msvbase_hnsw_query.sh`, `acorn_query.sh`
5. Batch orchestration:
   - `exps.sh`, `exp_loop.sh`, `exp_loop2.sh`, `exp_cons.sh`, and `exp_query.sh` in experiment folders.

## Dependency Pattern
Most scripts source shared config first:
- `source ./conf.sh`
- plus backend config such as `diskann_conf.sh`, `navix_conf.sh`, `irange_conf.sh`, `milvus_conf.sh`, `msvbase_conf.sh`, `acorn_conf.sh`

This means changing one variable in `conf.sh` impacts many scripts in the same folder.

## High-Risk Couplings
- In-place config rewrites in batch scripts (`sed -i ... conf.sh`) can leave persistent state for later runs.
- Many scripts rely on relative paths to external binaries under `../../code/.../build/...`.
- Same script names exist in multiple folders, so running from wrong `cwd` can execute the wrong experiment variant.

## Management Rules
1. Run from target experiment folder (`pwd` must match intended track).
2. Before edits: run `./tools/repo_guard.sh`.
3. Keep backend-specific paths only in `*_conf.sh`, not duplicated in index/query scripts.
4. Prefer adding new variables in `conf.sh` once, then consume from downstream scripts.
5. After parameter sweeps, reset `conf.sh` to a known baseline and commit that baseline.

## Recommended Working Loop
1. Edit one `conf.sh` or one backend script.
2. Run `./tools/repo_guard.sh`.
3. Run one small smoke command (single dataset/single selectivity).
4. Review logs in local `logs/` before full loops.
