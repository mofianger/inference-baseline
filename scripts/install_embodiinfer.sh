#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="${UV_BIN:-/root/.local/bin/uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/uv-cache}"
export UV_LOCK_TIMEOUT=1200
if [[ ! -d third_party/EmbodiInfer/.git ]]; then
  git clone https://github.com/BUAA-CI-LAB/EmbodiInfer.git third_party/EmbodiInfer
  git -C third_party/EmbodiInfer checkout 2cd9a37a0f6f555c2a65991f1cf4ccb578caf23b
fi
if [[ ! -x .venv-embodiinfer/bin/python ]]; then
  "$uv_bin" venv .venv-embodiinfer --python 3.12
fi
"$uv_bin" pip install --python .venv-embodiinfer/bin/python \
  torch==2.10.0 torchvision==0.25.0 transformers==5.3.0 diffusers==0.35.1 \
  accelerate safetensors pillow einops numpy
# The config imports this exact checkout via python_paths, without changing its lock.
