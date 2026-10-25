#!/usr/bin/env bash
# Run from repository root on 5090-machine. Use a NEW output directory for each run.
set -euo pipefail
RUN=profile/flashrt-nvfp4-5090-20261025
PY="$PWD/third_party/FlashRT/.venv/bin/python"
export PATH="$PWD/third_party/FlashRT/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
OUT="$RUN/reports/repro-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
"$PY" "$RUN/harness/collect.py" --mode torch --out "$OUT/timeline"
"$PY" "$RUN/harness/collect.py" --mode mapping --out "$OUT/mapping"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o "$OUT/nsys" "$PY" "$RUN/harness/collect.py" --mode nsys --out "$OUT/nsys-meta"
# Obtain official NVBit 1.8.1 in harness/nvbit_release_x86_64 before this step:
# https://github.com/NVlabs/NVBit/releases/download/v1.8.1/nvbit-Linux-x86_64-1.8.1.tar.bz2
cmake -S "$RUN/harness" -B "$RUN/harness/build" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DNVBIT_TOOL_CUDA_ARCHITECTURES=120
cmake --build "$RUN/harness/build" -j4
LD_PRELOAD="$PWD/$RUN/harness/build/cta_trace.so" "$PY" "$RUN/harness/collect.py" --mode cta --out "$OUT/cta"
