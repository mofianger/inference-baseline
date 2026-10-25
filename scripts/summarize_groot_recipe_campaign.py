"""Resolve this campaign against the frozen candidate catalog (not a new benchmark)."""
import json
from pathlib import Path

root = Path("reports/2026-10-25-5090-groot-new-recipes")
index = json.loads((root / "run-index.json").read_text())
catalog = json.loads(Path("recipes/groot-n17-5090-candidates.json").read_text(encoding="utf-8"))
labels = {
    "F03": "flashrt-fp8-fp16-head",
    "P02": "phyai-llm-fp8-a8", "P03": "phyai-llm-fp8-block-a8",
    "P04": "phyai-llm-fp8-a16", "P05": "phyai-llm-nvfp4-a16",
    "P06": "phyai-llm-int8-a8", "P07": "phyai-llm-int4-a16", "P08": "phyai-llm-int4-fp8-a8",
    "N02": "groot-trt-compile", "N03": "groot-trt-baseline", "N04": "groot-trt-optimized",
    "N05": "groot-trt-fp8", "N06": "groot-trt-nvfp4",
    "Q01": "foldquant-w8a8", "Q02": "foldquant-awq", "Q03": "foldquant-smoothquant",
}
rows = []
for recipe in catalog["recipes"]:
    rid = recipe["id"]
    row = {"id": rid, "name": recipe["name"]}
    if rid in labels:
        label = labels[rid]
        runs = [r for r in index if r.get("recipe") == label]
        # Export/build failures carry the recipe in the pipeline provenance.
        for r in index:
            if r["status"] != "stage_failed":
                continue
            p = root / "runs" / r["run"] / "provenance.json"
            if p.exists() and json.loads(p.read_text())["config"]["label"] == label:
                runs.append(r)
        runs.sort(key=lambda r: r["run"])
        config_name = "flashrt-fp16-head" if rid == "F03" else label
        row.update(config=f"configs/5090/{config_name}.json", attempts=runs,
                   status=runs[-1]["status"] if runs else "pending")
    elif rid in {"F01", "F02", "E01", "P01", "N01"}:
        row.update(status="skipped_existing_measurement", reason="Existing successful Run retained; user requested no repeat.")
    elif rid in {"E02", "E03", "E04", "E05", "P10"}:
        row.update(status="excluded_requires_engine_implementation", reason="Existing generic quantization is not wired into this GR00T path.")
    elif rid.startswith(("V", "C")) or rid == "F09":
        row.update(status="excluded_boundary", reason="Existing API does not match fresh prepared GPU tensor to GPU action protocol without engine changes.")
    elif rid in {"F08", "Q04", "Q05"}:
        row.update(status="deferred_experimental", reason="Separate experimental/hardware-preflight recipes; not part of this primary campaign.")
    else:
        row.update(status="deferred_ablation", reason="Execution ablation; new primary recipes prioritized in this campaign.")
    rows.append(row)
(root / "candidate-coverage.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
measured = [r for r in index if r["status"] == "measured"]
table = ["# 本轮有效测量", "", "全部采用相同三视角评估 fixture；数值门槛未设置。误差为所有校验样本的最大绝对误差。", "",
         "| 配置 / Recipe | mean (ms) | p95 (ms) | max abs error | 原始记录 |",
         "|---|---:|---:|---:|---|"]
for r in sorted(measured, key=lambda r: r["mean_ms"]):
    table.append(f"| `{r['recipe']}` | {r['mean_ms']:.3f} | {r['p95_ms']:.3f} | {r['max_abs']:.6f} | [summary]({r['source']}) |")
table += ["", "候选 ID 的覆盖状态与每次尝试见 [candidate-coverage.json](candidate-coverage.json)；包括重试的全部记录见 [run-index.json](run-index.json)。", ""]
(root / "measured.md").write_text("\n".join(table), encoding="utf-8")
print(f"{len(measured)} measured runs; {len(rows)} candidates accounted for")
