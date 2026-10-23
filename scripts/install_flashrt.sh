#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD/third_party/FlashRT"
uv_bin="${UV_BIN:-/root/.local/bin/uv}"
if [[ ! -d "$root/.git" ]]; then
  git clone https://github.com/flashrt-project/FlashRT.git "$root"
fi
git -C "$root" checkout --detach 1fbe995089af4148bacd10ead61764d0aeb80f7a
if [[ ! -d "$root/third_party/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git "$root/third_party/cutlass"
fi
if [[ ! -x "$root/.venv/bin/python" ]]; then
  "$uv_bin" venv --python python3.12 "$root/.venv"
fi
"$uv_bin" pip install --python "$root/.venv/bin/python" 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
"$uv_bin" pip install --python "$root/.venv/bin/python" pybind11 cmake ninja 'transformers==4.55.4' safetensors sentencepiece pillow numpy pyyaml einops
"$uv_bin" pip install --python "$root/.venv/bin/python" --no-deps -e "$root"
export PATH="$root/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
cmake -B "$root/build" -S "$root" -DGPU_ARCH=120 -DFLASHRT_ENABLE_MOTUS=OFF
cmake --build "$root/build" -j "${BUILD_JOBS:-8}"
"$uv_bin" pip freeze --python "$root/.venv/bin/python" > "$root/environment-freeze.txt"
python -c 'import flash_rt; from flash_rt import flash_rt_kernels, flash_rt_fa2, flash_rt_fp4; print(flash_rt.__file__); print(flash_rt_kernels.__file__)'
