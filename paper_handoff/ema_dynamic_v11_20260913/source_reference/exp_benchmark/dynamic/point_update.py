"""Complete source-record replacement: synchronous delete + fresh-label add, no slot reuse."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp_benchmark.dynamic.pipeline import main

if __name__ == "__main__":
    sys.exit(main(["run", "--operations", "point_update", *sys.argv[1:]]))
