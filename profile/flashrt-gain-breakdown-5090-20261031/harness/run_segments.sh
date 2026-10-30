#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
R=profile/flashrt-gain-breakdown-5090-20261031
P=third_party/FlashRT/.venv/bin/python
nvidia-smi -lgc 2400,2400
nvidia-smi -lmc 14001,14001
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate --format=csv --loop-ms=100 > "$R/reports/telemetry-segments.csv" &
MONITOR=$!
trap 'kill "$MONITOR" || true; nvidia-smi -rgc; nvidia-smi -rmc' EXIT
"$P" "$R/harness/segments.py" --out "$R/reports/segments" > "$R/reports/segments.log" 2>&1
nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$R/reports/hybrid-trace" "$P" "$R/harness/segments.py" --trace --out "$R/reports/segments-traced" > "$R/reports/segments-trace.log" 2>&1
