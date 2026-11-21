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
case "$dataset" in
  cifar10-c|cifar100-c) methods=(plead pea) ;;
  imagenet-c) methods=(plead) ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
for method in "${methods[@]}"; do
  if python -u run.py --dataset "$dataset" --method "$method" --upstream "$upstream" --output "$out" --smoke "$@" 2>&1 | tee "$out/logs/$dataset-$method-smoke.log"; then
    ready+=("$method")
  else
    fail=1
  fi
done
for method in "${ready[@]}"; do
  python -u run.py --dataset "$dataset" --method "$method" --upstream "$upstream" --output "$out" "$@" 2>&1 | tee "$out/logs/$dataset-$method.log" || fail=1
  (flock -x 8; python report.py --output "$out") 8>"$out/logs/report.lock"
done
exit "$fail"
