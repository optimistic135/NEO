#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
server_root="${BALANCED_512_STORAGE:-/data/user3/chenc/balanced-512}"
checkpoint_root="${VIT_CHECKPOINT_ROOT:-/data/user3/chenc/checkpoints/Vit}"
data_root="${CLEAN_CIFAR_ROOT:-/data/user3/chenc/datasets}"
python_base="${PYTHON_BOOTSTRAP:-python3.10}"
mkdir -p "$server_root"/{tmp,cache,upstream} "$checkpoint_root" || {
  echo "Need write access to $server_root and $checkpoint_root" >&2
  exit 1
}
exec 9>"$server_root/server.lock"
flock -n 9 || { echo "balanced-512 is already running in $server_root" >&2; exit 1; }
export TMPDIR="$server_root/tmp"
export XDG_CACHE_HOME="$server_root/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$checkpoint_root"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$XDG_CACHE_HOME/hf-xet"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PEA_UPSTREAM="$server_root/upstream"
export BALANCED_512_OUTPUT="${BALANCED_512_OUTPUT:-$server_root/results}"

find_clean_root() {
  local name="$1" train_file="$2" parent
  local matches=()
  while IFS= read -r -d '' parent; do
    if [[ -s "$parent/$train_file" ]]; then
      matches+=("$(dirname "$parent")")
    fi
  done < <(find -L "$data_root" -maxdepth 4 -type d -name "$name" -print0)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "Expected one extracted $name containing $train_file under $data_root; found ${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

cifar10_source="$(find_clean_root cifar-10-batches-py data_batch_1)"
cifar100_source="$(find_clean_root cifar-100-python train)"
"$python_base" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
if [[ ! -x "$server_root/env/bin/python" ]]; then
  "$python_base" -m venv "$server_root/env"
fi
export PATH="$server_root/env/bin:$PATH"
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2+cu118 torchvision==0.16.2+cu118
python -m pip install --no-cache-dir -r requirements-server.txt
python -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name())'
bash setup.sh

pea_pin=9c358ba682052862df0739fbbadb71f1b586614d
if [[ ! -d "$PEA_UPSTREAM/pea/.git" ]]; then
  git -C "$PEA_UPSTREAM" clone https://github.com/TheMaXiao/PEA_TTA.git pea
fi
if [[ "$(git -C "$PEA_UPSTREAM/pea" rev-parse HEAD)" != "$pea_pin" ]]; then
  git -C "$PEA_UPSTREAM/pea" checkout --detach "$pea_pin"
fi
python -u prepare_models.py
python -m unittest discover -s . -p 'test_protocol.py'

fail=0
bash run_dataset.sh cifar10-c --source-root "$cifar10_source" || fail=1
bash run_dataset.sh cifar100-c --source-root "$cifar100_source" || fail=1
bash run_dataset.sh imagenet-c || fail=1
python report.py --output "$BALANCED_512_OUTPUT"
exit "$fail"
