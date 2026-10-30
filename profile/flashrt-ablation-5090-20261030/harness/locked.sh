#!/usr/bin/env bash
set -euo pipefail
RUN=/data/inference-baseline/profile/flashrt-ablation-5090-20261030
TELEMETRY="$RUN/reports/telemetry-$(date +%s).csv"
MONITOR_PID=''
trap 'if [ -n "$MONITOR_PID" ]; then kill "$MONITOR_PID" || true; fi; nvidia-smi -rgc; nvidia-smi -rmc' EXIT
nvidia-smi -lgc 2400,2400
nvidia-smi -lmc 14001,14001
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate --format=csv --loop-ms=200 > "$TELEMETRY" &
MONITOR_PID=$!
echo "Telemetry: $TELEMETRY"
"$@"
