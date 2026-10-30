#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
RUN=profile/flashrt-nvfp4-nsys-5090-20261030
PYTHON=third_party/FlashRT/.venv/bin/python
# Run once in an empty reports directory; existing Nsys outputs are not overwritten.
"$PYTHON" "$RUN/harness/collect.py" --mode wall --out "$RUN/reports"
nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --export=sqlite --output="$RUN/reports/frontend-hw" \
  "$PYTHON" "$RUN/harness/collect.py" --mode nsys --out "$RUN/reports"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --export=sqlite --output="$RUN/reports/frontend-sw" \
  "$PYTHON" "$RUN/harness/collect.py" --mode nsys --out "$RUN/reports/sw"
"$PYTHON" "$RUN/analysis/export_symbols.py"
"$PYTHON" "$RUN/analysis/prepare_template.py"
"$PYTHON" "$RUN/analysis/build.py"
