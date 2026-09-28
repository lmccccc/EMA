#!/usr/bin/env bash
# Build legacy layout/query compatibility fixes; never open an index or dataset.
set -euo pipefail
PYTHON="${1:?Usage: build_legacy_query_native.sh /path/to/python /new/output/directory}"
OUTPUT="${2:?Usage: build_legacy_query_native.sh /path/to/python /new/output/directory}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE=28e07e33cb2dd4e7c173ca1030e6c5bbd7f4f7bb
EXPECTED_HEADER=461dbe5a3bcfff4eb4a078a00e0e24f5743b51196d92599e4e261903c7c94058
if [ -e "$OUTPUT" ]; then
    echo "Refusing to overwrite native build output: $OUTPUT" >&2
    exit 1
fi
mkdir -p "$OUTPUT/source"
OUTPUT="$(cd "$OUTPUT" && pwd)"
git -C "$REPO" archive "$BASE" hnswlib | tar -x -C "$OUTPUT/source"
HEADER="$OUTPUT/source/hnswlib/hnswlib/hnswalg.h"
if [ "$(sha256sum "$HEADER" | cut -d ' ' -f1)" != "$EXPECTED_HEADER" ]; then
    echo "Original legacy query source identity mismatch" >&2
    exit 1
fi
git -C "$OUTPUT/source" apply --check "$REPO/exp_benchmark/legacy_query_compat.patch"
git -C "$OUTPUT/source" apply "$REPO/exp_benchmark/legacy_query_compat.patch"
git -C "$OUTPUT/source" apply --check "$REPO/exp_benchmark/legacy_query_mask_compat.patch"
git -C "$OUTPUT/source" apply "$REPO/exp_benchmark/legacy_query_mask_compat.patch"
(
    cd "$OUTPUT/source/hnswlib"
    "$PYTHON" setup.py build_ext --build-lib "$OUTPUT/native" --build-temp "$OUTPUT/objects"
)
"$PYTHON" - "$OUTPUT" "$BASE" "$REPO/exp_benchmark/legacy_query_compat.patch" \
    "$REPO/exp_benchmark/legacy_query_mask_compat.patch" <<'PY'
import hashlib
import json
import pathlib
import sys

output, base, patch = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
query_patch = pathlib.Path(sys.argv[4])
modules = list((output / "native").glob("hashannlib*.so"))
if len(modules) != 1:
    raise RuntimeError("Expected exactly one compiled native module")
def info(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
with (output / "provenance.json").open("x") as stream:
    json.dump({"purpose": "saved-index query replay; layout and inclusive-range compatibility fixes, not V11",
               "source_commit": base, "patch": info(patch), "native": info(modules[0]),
               "query_patch": info(query_patch),
               "patched_header": info(output / "source/hnswlib/hnswlib/hnswalg.h"),
               "python": sys.version, "index_rebuilt": False, "index_files_modified": False},
              stream, indent=2)
    stream.write("\n")
PY
