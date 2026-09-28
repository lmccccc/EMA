#!/usr/bin/env bash
# Creates the shared first-prefix cache and validates stage0; never works around native load/add.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python3}" -m exp_benchmark.dynamic.pipeline run --operations insert --through-stage 0 "$@"
