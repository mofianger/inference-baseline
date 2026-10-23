#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
flashrt_root="$PWD/third_party/FlashRT"
export PATH="$flashrt_root/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
cmake -B "$flashrt_root/build" -S "$flashrt_root" -DGPU_ARCH=120 -DFLASHRT_ENABLE_MOTUS=OFF -DFLASHRT_ENABLE_PI05_SKINNY=ON -DFLASHRT_ENABLE_PI05_NVFP4=ON
cmake --build "$flashrt_root/build" -j "${BUILD_JOBS:-8}"
