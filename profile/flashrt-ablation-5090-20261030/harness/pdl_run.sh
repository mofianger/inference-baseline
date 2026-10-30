#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
RUN=profile/flashrt-ablation-5090-20261030
mkdir -p "$RUN/reports/pdl"
nvidia-smi -q > "$RUN/reports/pdl/gpu-before.txt"
trap 'nvidia-smi -rgc; nvidia-smi -rmc' EXIT
nvidia-smi -lgc 2400,2400
nvidia-smi -lmc 14001,14001 || true
idx=0
for setting in 0 1 1 0 0 1; do
 FLASHRT_GROOT_N17_DIT_PDL=$setting third_party/FlashRT/.venv/bin/python "$RUN/harness/pdl.py" --out "$RUN/reports/pdl/run-$idx.json"
 idx=$((idx+1))
done
for setting in 0 1; do
 FLASHRT_GROOT_N17_DIT_PDL=$setting nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$RUN/reports/pdl/trace-$setting" third_party/FlashRT/.venv/bin/python "$RUN/harness/pdl.py" --trace --out "$RUN/reports/pdl/trace-$setting.json"
done
