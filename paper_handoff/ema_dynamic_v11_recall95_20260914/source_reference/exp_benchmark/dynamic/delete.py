"""Canonical mixed-DNF synchronous structural deletion; no mark/cleanup-only mode."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp_benchmark.dynamic.pipeline import main

if __name__ == "__main__":
    sys.exit(main(["run", "--operations", "delete", *sys.argv[1:]]))
