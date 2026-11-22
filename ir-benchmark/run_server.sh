#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="${1:?Use all, cifar10-c, cifar100-c, or imagenet-c}"
shift
case "$dataset" in
  all) datasets=(cifar10-c cifar100-c imagenet-c) ;;
  cifar10-c|cifar100-c|imagenet-c) datasets=("$dataset") ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
policy="${IR_CAPACITY_POLICY:-strict}"
case "$policy" in strict|cap) ;; *) echo "Unknown capacity policy: $policy" >&2; exit 2 ;; esac
export HF_HOME="${VIT_CHECKPOINT_ROOT:-/data/user3/chenc/checkpoints/Vit}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" HF_HUB_OFFLINE=1
python_bin="${IR_PYTHON:-$PWD/balanced-512/.runtime/server-$(id -u)/env/bin/python}"
output="${IR_OUTPUT:-$PWD/results/ir-benchmark}"
mkdir -p "$output/$policy/logs"
shell_result=0
exec 9>"$output/$policy/run.lock"
flock -n 9 || { echo "An IR run already owns $output/$policy" >&2; exit 1; }
if [[ ! -x "$python_bin" ]]; then
  echo "Set IR_PYTHON to the Python interpreter with the balanced-512 dependencies." >&2
  exit 1
fi
bash balanced-512/setup.sh
"$python_bin" -u -m unittest discover -s ir-benchmark -p 'test_*.py'
for data in "${datasets[@]}"; do
  for phase in smoke full; do
    options=()
    if [[ "$phase" == smoke ]]; then options+=(--smoke); fi
    "$python_bin" -u ir-benchmark/evaluate.py --dataset "$data" --output "$output" \
      --capacity-policy "$policy" "${options[@]}" "$@" 2>&1 | tee "$output/$policy/logs/$data-$phase.log" || shell_result=1
  done
  "$python_bin" ir-benchmark/report.py "$output/$policy"
done
exit "$shell_result"
