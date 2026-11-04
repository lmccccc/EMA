#!/usr/bin/env bash
set -euo pipefail
PYTHON="${1:?Usage: build_component_native.sh /path/to/python /new/output/directory}"
OUTPUT="${2:?Usage: build_component_native.sh /path/to/python /new/output/directory}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH="$REPO/exp_benchmark/ablation/component_native.patch"
if [ -e "$OUTPUT" ]; then
    echo "Refusing to overwrite ablation native output: $OUTPUT" >&2
    exit 1
fi
git -C "$REPO" diff --quiet HEAD -- hnswlib || {
    echo "Commit or preserve pending native edits before selecting the source commit." >&2
    exit 1
}
BASE="$(git -C "$REPO" rev-parse HEAD)"
mkdir -p "$OUTPUT/source"
OUTPUT="$(cd "$OUTPUT" && pwd)"
git -C "$REPO" archive "$BASE" hnswlib | tar -x -C "$OUTPUT/source"
cp "$PATCH" "$OUTPUT/component_native.patch"
git -C "$OUTPUT/source" apply --check "$OUTPUT/component_native.patch"
git -C "$OUTPUT/source" apply "$OUTPUT/component_native.patch"
for flavor in timing diagnostic; do
    diagnostics=0
    [ "$flavor" = diagnostic ] && diagnostics=1
    (
        cd "$OUTPUT/source/hnswlib"
        EMA_ABLATION_DIAGNOSTICS="$diagnostics" "$PYTHON" setup.py build_ext \
            --build-lib "$OUTPUT/$flavor" --build-temp "$OUTPUT/objects-$flavor"
    ) > "$OUTPUT/build-$flavor.log" 2>&1 || {
        tail -n 60 "$OUTPUT/build-$flavor.log"
        exit 1
    }
done
"$PYTHON" - "$OUTPUT" "$BASE" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
def info(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
modules = {}
for flavor in ("timing", "diagnostic"):
    paths = list((root / flavor).glob("hashannlib*.so"))
    if len(paths) != 1:
        raise RuntimeError("Expected one extension per ablation flavor")
    modules[flavor] = info(paths[0])
sources = {
    str(path.relative_to(root / "source")): info(path)
    for path in sorted((root / "source/hnswlib").rglob("*"))
    if path.is_file() and not path.is_symlink() and path.suffix in (".h", ".cpp", ".py", ".toml")
}
provenance = {
    "schema": "ema-component-native-v1", "source_commit": sys.argv[2],
    "patch": info(root / "component_native.patch"), "modules": modules, "sources": sources,
    "python": sys.version,
    "build_environment": {key: os.environ.get(key) for key in
                          ("CC", "CXX", "CFLAGS", "CXXFLAGS", "HNSWLIB_NO_NATIVE")},
    "diagnostics_macro": {"timing": False, "diagnostic": True},
    "construction_policy": "CHT admission only; RNG and Marker witnesses unchanged",
    "checkpoint_policy": "Ablation graphs are read-only after construction; policy is in the run manifest",
}
with (root / "provenance.json").open("x") as stream:
    json.dump(provenance, stream, indent=2)
    stream.write("\n")
print(root / "provenance.json")
PY
