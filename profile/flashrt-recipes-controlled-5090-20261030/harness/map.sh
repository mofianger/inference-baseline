#!/usr/bin/env bash
set -euo pipefail
cd /data/inference-baseline
R=profile/flashrt-recipes-controlled-5090-20261030
P=third_party/FlashRT/.venv/bin/python
for recipe in nvfp4-fused fp8-fused fp8-unfused nvfp4-unfused; do
 RECIPE=$recipe nsys profile --trace=cuda-hw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite -o "$R/reports/$recipe/mapping-v2" "$P" "$R/harness/collect.py" --mode map --out "$R/reports/$recipe/mapping-v2-data" > "$R/reports/$recipe-mapping-v2.log" 2>&1
done
