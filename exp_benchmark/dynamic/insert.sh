#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python3}" -m exp_benchmark.dynamic.pipeline run --operations insert "$@"
