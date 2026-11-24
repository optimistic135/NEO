#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME="${VIT_CHECKPOINT_ROOT:-/data/user3/chenc/checkpoints/Vit}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX
python_bin="${PAPER_EXP_PYTHON:-$PWD/balanced-512/.runtime/server-$(id -u)/env/bin/python}"
[[ -x "$python_bin" ]] || { echo "Set PAPER_EXP_PYTHON to the existing balanced-512 environment." >&2; exit 1; }
"$python_bin" -m pip install --only-binary=:all: -r balanced-512/requirements-server.txt \
  -r paper-exp/requirements/methods/deyo.txt torch==2.1.2 torchvision==0.16.2
"$python_bin" -u paper-exp/scripts/run_online_baselines.py "$@"
