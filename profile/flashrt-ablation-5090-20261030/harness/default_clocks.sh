#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
RUN=profile/flashrt-ablation-5090-20261030/reports/default-clocks-retest
mkdir -p "$RUN"
nvidia-smi -rgc
nvidia-smi -rmc
nvidia-smi -q > "$RUN/gpu-before.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate,utilization.gpu --format=csv --loop-ms=50 > "$RUN/telemetry.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" || true' EXIT
date +%z > "$RUN/timezone.txt"
for round in 1 2 3; do
 third_party/FlashRT/.venv/bin/python profile/flashrt-nvfp4-nsys-5090-20261030/harness/collect.py --mode wall --out "$RUN/round-$round" > "$RUN/round-$round.log" 2>&1
done
nvidia-smi -q > "$RUN/gpu-after.txt"
