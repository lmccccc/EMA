# HashANN

Shell-first experiment workspace for ANN backends (HashANN, DiskANN, iRange, Navix, Milvus, MSVBase, ACORN).

## Quick Orientation
- Main script families exist in `./`, `./exp2`, `./exp3`, `./exp4`.
- Shared run parameters are defined in each folder's `conf.sh`.
- Index/query scripts source `conf.sh` plus backend-specific `*_conf.sh` files.

Read full map: `docs/PROJECT_MAP.md`

## Safety Workflow (Recommended)
Before and after any code/config change:

```bash
./tools/repo_guard.sh
```

Optional scope check:

```bash
./tools/repo_guard.sh exp2
```

`repo_guard.sh` checks:
- shell syntax for all `*.sh`
- missing sourced config files
- scripts that mutate `conf.sh` in place (`sed -i` hotspots)
- unresolved external `../.../build/...` binary paths
- baseline keys in each `conf.sh`
