#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
dataset="${1:?Specify cifar10-c, cifar100-c, or imagenet-c}"
shift
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}"
upstream="${PEA_UPSTREAM:-$PWD/../paper-exp/.upstream}"
out="${BALANCED_512_OUTPUT:-$PWD/results}"
mkdir -p "$out/logs"
exec 9>"$out/logs/$dataset.lock"
flock -n 9 || { echo "$dataset is already running" >&2; exit 1; }
ready=()
fail=0
for method in plead pea; do
  if python -u run.py --dataset "$dataset" --method "$method" --upstream "$upstream" --output "$out" --smoke "$@" > "$out/logs/$dataset-$method-smoke.log" 2>&1; then
    ready+=("$method")
  else
    cat "$out/logs/$dataset-$method-smoke.log" >&2
    if [[ "$method" == pea && "$dataset" == imagenet-c && $# -eq 0 ]]; then
      python -u run.py --dataset "$dataset" --method "$method" --upstream "$upstream" --output "$out" > "$out/logs/$dataset-$method.log" 2>&1 || fail=1
    else
      fail=1
    fi
  fi
done
for method in "${ready[@]}"; do
  python -u run.py --dataset "$dataset" --method "$method" --upstream "$upstream" --output "$out" "$@" > "$out/logs/$dataset-$method.log" 2>&1 || fail=1
  (flock -x 8; python report.py --output "$out") 8>"$out/logs/report.lock"
done
exit "$fail"
