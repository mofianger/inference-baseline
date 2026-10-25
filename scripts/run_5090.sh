#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
engine="${1:?Usage: run_5090.sh ENGINE NEW_OUTPUT_DIR [DATASET]}"
output="${2:?Provide a new output directory}"
dataset="${3:-datasets/groot-n17-v3}"
case "$engine" in
  reference) python_bin=/data/5090-machine/Isaac-GR00T/.venv/bin/python ;;
  flashrt|flashrt_nvfp4|flashrt-fp16-head) python_bin=/data/inference-baseline/third_party/FlashRT/.venv/bin/python ;;
  embodiinfer) python_bin=/data/inference-baseline/.venv-embodiinfer/bin/python ;;
  phyai|phyai-llm-*) python_bin=/data/inference-baseline/third_party/phyai/.venv/bin/python ;;
  groot-trt-*) python_bin=/data/inference-baseline/.venv-groot-trt/bin/python ;;
  foldquant-*) python_bin=/data/inference-baseline/.venv-foldquant/bin/python ;;
  *) echo "Unknown engine: $engine" >&2; exit 2 ;;
esac
export NO_ALBUMENTATIONS_UPDATE=1
export PATH="$(dirname "$python_bin"):/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Cooperative lock prevents two runs of this script from sharing the GPU.
mkdir -p artifacts
exec 9>artifacts/gpu-0.lock
flock -n 9 || { echo 'Another benchmark holds GPU 0 lock' >&2; exit 3; }
"$python_bin" -m inference_baseline.cli run --config "configs/5090/$engine.json" \
  --dataset "$dataset" --output "$output"
