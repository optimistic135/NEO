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
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "Multiple copies of $name under $data_root. Set CLEAN_CIFAR_ROOT to the intended location." >&2
    return 1
  fi
  if [[ ${#matches[@]} -eq 1 ]]; then
    printf '%s\n' "${matches[0]}"
  fi
}

prepare_clean_root() {
  local dataset_class="$1" name="$2" train_file="$3" root
  root="$(find_clean_root "$name" "$train_file")" || return 1
  if [[ -z "$root" ]]; then
    if [[ ! -w "$data_root" ]]; then
      echo "Need write access to $data_root to download clean $dataset_class" >&2
      return 1
    fi
    root="$data_root"
  fi
  # Torchvision verifies the official files and only downloads if they are missing or invalid.
  python -u -c 'import sys; from torchvision import datasets; getattr(datasets, sys.argv[1])(sys.argv[2], train=True, download=True)'     "$dataset_class" "$root" >&2 || return 1
  printf '%s\n' "$root"
}

"$python_base" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
if [[ ! -x "$server_root/env/bin/python" ]]; then
  "$python_base" -m venv "$server_root/env"
fi
export PATH="$server_root/env/bin:$PATH"
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2+cu118 torchvision==0.16.2+cu118
python -m pip install -r requirements-server.txt
python -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name())'
cifar10_source="$(prepare_clean_root CIFAR10 cifar-10-batches-py data_batch_1)"
cifar100_source="$(prepare_clean_root CIFAR100 cifar-100-python train)"
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
