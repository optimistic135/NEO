#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
repo_root="$(cd .. && pwd)"
server_root="${BALANCED_512_STORAGE:-$PWD/.runtime/server-$(id -u)}"
checkpoint_root="${VIT_CHECKPOINT_ROOT:-/data/user3/chenc/checkpoints/Vit}"
data_root="${CLEAN_CIFAR_ROOT:-/data/user3/chenc/datasets}"
python_base="${PYTHON_BOOTSTRAP:-python3.10}"

if (( EUID == 0 )); then
  echo "Run this script as your login user. Only the dedicated asset directories may require sudo." >&2
  exit 1
fi

ensure_asset_dir() {
  local path="$1"
  if [[ ! -d "$path" ]] && ! mkdir -p "$path" 2>/dev/null; then
    if [[ "${BALANCED_512_CREATE_ASSET_DIRS:-0}" == 1 ]]; then
      sudo install -d -o "$(id -u)" -g "$(id -g)" -- "$path" || return 1
    fi
  fi
  if [[ ! -d "$path" || ! -w "$path" ]]; then
    echo "Need write access to $path. Ask the server administrator to provision this directory." >&2
    return 1
  fi
}

mkdir -p "$server_root"/{tmp,cache,upstream}
exec 9>"$server_root/server.lock"
flock -n 9 || { echo "balanced-512 is already running in $server_root" >&2; exit 1; }
ensure_asset_dir "$checkpoint_root"
export TMPDIR="$server_root/tmp"
export XDG_CACHE_HOME="$server_root/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$checkpoint_root"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$XDG_CACHE_HOME/hf-xet"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PEA_UPSTREAM="${PEA_UPSTREAM:-$server_root/upstream}"
export BALANCED_512_OUTPUT="${BALANCED_512_OUTPUT:-$repo_root/results/balanced-512}"

find_clean_root() {
  local dataset_class="$1" name="$2" train_file="$3" parent root
  local matches=()
  while IFS= read -r -d '' parent; do
    root="$(dirname "$parent")"
    if [[ -s "$parent/$train_file" ]] &&
       python -u -c 'import sys; from torchvision import datasets; getattr(datasets, sys.argv[1])(sys.argv[2], train=True, download=False)' "$dataset_class" "$root" >&2; then
      matches+=("$root")
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
  local dataset_class="$1" name="$2" train_file="$3" root download_root
  root="$(find_clean_root "$dataset_class" "$name" "$train_file")" || return 1
  if [[ -n "$root" ]]; then
    printf '%s\n' "$root"
    return 0
  fi
  download_root="$data_root/CIFAR-clean"
  ensure_asset_dir "$download_root" || return 1
  python -u -c 'import sys; from torchvision import datasets; getattr(datasets, sys.argv[1])(sys.argv[2], train=True, download=True)' \
    "$dataset_class" "$download_root" >&2 || return 1
  printf '%s\n' "$download_root"
}

"$python_base" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
if [[ ! -x "$server_root/env/bin/python" ]]; then
  "$python_base" -m venv "$server_root/env"
fi
export PATH="$server_root/env/bin:$PATH"
python -m pip install --upgrade 'pip>=24.2'
# Resolve regular dependencies on PIP_INDEX_URL. Only the two CUDA wheels
# are fetched from the official PyTorch package server.
python -m pip install --only-binary=:all: -r requirements-server.txt \
  'torch @ https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl' \
  'torchvision @ https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-linux_x86_64.whl'
python -m pip check
python -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name())'
cifar10_source="$(prepare_clean_root CIFAR10 cifar-10-batches-py data_batch_1)"
cifar100_source="$(prepare_clean_root CIFAR100 cifar-100-python train)"
bash setup.sh

pea_pin=9c358ba682052862df0739fbbadb71f1b586614d
existing_pea="$repo_root/paper-exp/.upstream"
if [[ "$PEA_UPSTREAM" == "$server_root/upstream" && -d "$existing_pea/pea/.git" ]] &&
   [[ "$(git -C "$existing_pea/pea" rev-parse HEAD)" == "$pea_pin" ]]; then
  export PEA_UPSTREAM="$existing_pea"
fi
mkdir -p "$PEA_UPSTREAM"
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
