#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
R=profile/flashrt-gain-breakdown-5090-20261031
P=third_party/FlashRT/.venv/bin/python
C=profile/flashrt-recipes-controlled-5090-20261030/harness/collect.py
mkdir -p "$R/reports"
nvidia-smi -q > "$R/reports/gpu-before.txt"
nvidia-smi -lgc 2400,2400
nvidia-smi -lmc 14001,14001
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate --format=csv --loop-ms=100 > "$R/reports/telemetry.csv" &
MONITOR=$!
trap 'kill "$MONITOR" || true; nvidia-smi -rgc; nvidia-smi -rmc' EXIT
date +%z > "$R/reports/timezone.txt"
for round in 0 1 2; do
 labels='fp8 nvfp4-off nvfp4-on'
 if [ "$round" = 1 ]; then labels='nvfp4-on nvfp4-off fp8'; fi
 for label in $labels; do
  recipe=nvfp4-fused; pdl=0
  if [ "$label" = fp8 ]; then recipe=fp8-fused; fi
  if [ "$label" = nvfp4-on ]; then pdl=1; fi
  RECIPE=$recipe BENCH_PDL=$pdl "$P" "$C" --out "$R/reports/$label/wall-$round" > "$R/reports/$label-$round.log" 2>&1
 done
done
for label in fp8 nvfp4-off nvfp4-on; do
 recipe=nvfp4-fused; pdl=0
 if [ "$label" = fp8 ]; then recipe=fp8-fused; fi
 if [ "$label" = nvfp4-on ]; then pdl=1; fi
 RECIPE=$recipe BENCH_PDL=$pdl nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$R/reports/$label/trace" "$P" "$C" --mode trace --out "$R/reports/$label/trace-data" > "$R/reports/$label-trace.log" 2>&1
done
