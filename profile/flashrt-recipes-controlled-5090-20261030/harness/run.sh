#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
R=profile/flashrt-recipes-controlled-5090-20261030
P=third_party/FlashRT/.venv/bin/python
mkdir -p "$R/reports"
nvidia-smi -q > "$R/reports/gpu-before.txt"
nvidia-smi -lgc 2400,2400
nvidia-smi -lmc 14001,14001
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate --format=csv --loop-ms=100 > "$R/reports/telemetry.csv" &
MONITOR=$!
trap 'kill "$MONITOR" || true; nvidia-smi -rgc; nvidia-smi -rmc' EXIT
date +%z > "$R/reports/timezone.txt"
for round in 0 1 2; do
 recipes='nvfp4-fused fp8-fused fp8-unfused nvfp4-unfused'
 if [ "$round" = 1 ]; then recipes='nvfp4-unfused fp8-unfused fp8-fused nvfp4-fused'; fi
 for recipe in $recipes; do
  RECIPE=$recipe "$P" "$R/harness/collect.py" --out "$R/reports/$recipe/wall-$round" > "$R/reports/$recipe-$round.log" 2>&1
 done
done
for recipe in nvfp4-fused fp8-fused fp8-unfused nvfp4-unfused; do
 RECIPE=$recipe nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$R/reports/$recipe/trace" "$P" "$R/harness/collect.py" --mode trace --out "$R/reports/$recipe/trace-data" > "$R/reports/$recipe-trace.log" 2>&1
done
