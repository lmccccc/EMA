"""Build query-only strict-filter extensions from a frozen component-native source."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ..dynamic.runtime import file_info, immutable_json
from ..static_paper import read_json


REPO = Path(__file__).resolve().parents[2]


def build(provenance, output):
    provenance = Path(provenance).resolve(strict=True)
    base = read_json(provenance)
    if base["schema"] != "ema-component-native-v1":
        raise ValueError("Select the original isolated component-native provenance")
    if not re.fullmatch(r"[0-9a-f]{40}", base["source_commit"]):
        raise ValueError("Expected a complete source commit identifier")
    component_patch = Path(base["patch"]["path"])
    if file_info(component_patch)["sha256"] != base["patch"]["sha256"]:
        raise RuntimeError("The frozen component-native patch changed")
    output = Path(output).resolve()
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(provenance, output / "base-native-provenance.json")
    shutil.copyfile(component_patch, output / "component_native.patch")
    shutil.copyfile(REPO / "exp_benchmark/ablation/strict_native.patch",
                    output / "strict_native.patch")
    archive = output / "native-source.tar"
    with archive.open("xb") as stream:
        subprocess.run(["git", "-C", str(REPO), "archive", base["source_commit"], "hnswlib"],
                       stdout=stream, check=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
    archive.unlink()
    for patch_name in ("component_native.patch", "strict_native.patch"):
        patch_path = str(output / patch_name)
        subprocess.run(["git", "-C", str(source), "apply", "--check", patch_path], check=True)
        subprocess.run(["git", "-C", str(source), "apply", patch_path], check=True)
        if patch_name == "component_native.patch":
            for name, info in base["sources"].items():
                if file_info(source / name)["sha256"] != info["sha256"]:
                    raise RuntimeError(f"Original compiled source differs: {name}")
    environment = dict(os.environ)
    for key, value in base["build_environment"].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    modules = {}
    for flavor in ("timing", "diagnostic"):
        environment["EMA_ABLATION_DIAGNOSTICS"] = "1" if flavor == "diagnostic" else "0"
        log = output / f"build-{flavor}.log"
        with log.open("x") as stream:
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--build-lib", str(output / flavor),
                 "--build-temp", str(output / f"objects-{flavor}")],
                cwd=source / "hnswlib", env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Native {flavor} build failed; see {log}")
        binaries = list((output / flavor).glob("hashannlib*.so"))
        if len(binaries) != 1:
            raise RuntimeError("Expected one native extension per flavor")
        modules[flavor] = file_info(binaries[0])
    immutable_json(output / "provenance.json", {
        "schema": "ema-strict-filter-native-v1", "source_commit": base["source_commit"],
        "base_provenance": file_info(provenance),
        "component_patch": file_info(output / "component_native.patch"),
        "strict_patch": file_info(output / "strict_native.patch"),
        "builder": file_info(Path(__file__).resolve()), "modules": modules,
        "sources": {
            str(path.relative_to(source)): file_info(path)
            for path in sorted(source.rglob("*"))
            if not path.is_symlink() and path.is_file() and path.suffix in (".h", ".cpp", ".py", ".toml")
        },
        "python": sys.version, "build_environment": base["build_environment"],
        "diagnostics_macro": {"timing": False, "diagnostic": True},
        "strict_policy": "Exact predicate before neighbor distance/queue insertion; no duplicate result predicate check",
        "entry_policy": "Common unfiltered upper-layer navigation and initial seed; only newly reached base neighbors are gated",
        "checkpoint_policy": "Reuse the unchanged original CHT-on graph read-only; no rebuild or format retagging",
    })
    print(output / "provenance.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-native-provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.base_native_provenance, args.output)


if __name__ == "__main__":
    main()
