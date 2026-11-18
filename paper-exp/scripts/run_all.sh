#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run() { "${PYTHON:-python}" -u bench.py "$@"; }
run validate "$@"
run smoke "$@"
run balanced "$@"
run online_ir "$@"
run ablation "$@"
run ablation --online "$@"
run efficiency "$@"
run report
