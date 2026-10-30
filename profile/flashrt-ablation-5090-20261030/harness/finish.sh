#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
RUN=profile/flashrt-ablation-5090-20261030
PYTHON=third_party/FlashRT/.venv/bin/python
LOCK="$RUN/harness/locked.sh"
bash "$LOCK" "$PYTHON" "$RUN/harness/fusion.py" --out "$RUN/reports/fusion-measured" > "$RUN/reports/fusion-measured.log" 2>&1
bash "$LOCK" "$PYTHON" "$RUN/harness/precision.py" --fixtures "$RUN/reports/fusion-measured/fixtures.pt" --out "$RUN/reports/precision-measured" > "$RUN/reports/precision-measured.log" 2>&1
bash "$LOCK" nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$RUN/reports/local-trace" "$PYTHON" "$RUN/harness/local_trace.py" --fixtures "$RUN/reports/fusion-measured/fixtures.pt" > "$RUN/reports/local-trace.log" 2>&1
bash "$LOCK" nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$RUN/reports/fusion-trace" "$PYTHON" "$RUN/harness/fusion.py" --trace --out "$RUN/reports/fusion-traced" > "$RUN/reports/fusion-trace.log" 2>&1
bash "$LOCK" nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$RUN/reports/precision-trace" "$PYTHON" "$RUN/harness/precision.py" --trace --fixtures "$RUN/reports/fusion-measured/fixtures.pt" --out "$RUN/reports/precision-traced" > "$RUN/reports/precision-trace.log" 2>&1
