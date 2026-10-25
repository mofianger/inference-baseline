"""Export this campaign's measured runs and failures without large assets."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_baseline.io import sha256, write_json

root = Path("artifacts")
out = Path("reports/2026-10-25-5090-groot-new-recipes")
out.mkdir(parents=True, exist_ok=True)
rows = []
prefixes = ("20261025-phyai-", "20261025-flashrt-fp16-", "20261025-groot-trt-", "20261025-foldquant-")
campaign = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefixes)]
for directory in sorted(campaign):
    for path in sorted(directory.rglob("*.json")):
        if path.name not in ("summary.json", "failure.json", "correctness.json", "stages.json", "batch.json", "provenance.json"):
            continue
        relative = path.relative_to(root)
        target = out / "runs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        value = json.loads(path.read_text())
        if path.name == "summary.json":
            latency = path.parent / "latency.jsonl"
            shutil.copy2(latency, target.parent / latency.name)
            rows.append({"run": relative.parent.as_posix(), "recipe": value["config"].get("label", value["config"]["engine"]),
                         "status": "measured", "mean_ms": value["latency"]["mean_ms"],
                         "p50_ms": value["latency"]["p50_ms"], "p95_ms": value["latency"]["p95_ms"],
                         "max_abs": max(x["max_abs"] for x in value["correctness"]),
                         "gate": value["accuracy_gate"], "source": (Path("runs") / relative).as_posix()})
        elif path.name == "failure.json":
            rows.append({"run": relative.parent.as_posix(), "recipe": value["config"].get("label", value["config"]["engine"]),
                         "status": "failed", "error": value["error"], "source": (Path("runs") / relative).as_posix()})
        elif path.name == "stages.json":
            for stage in value:
                if stage["exit_code"]:
                    rows.append({"run": relative.parent.as_posix(), "status": "stage_failed", "stage": stage["stage"],
                                 "exit_code": stage["exit_code"], "source": (Path("runs") / relative).as_posix()})
logs = [p for p in root.glob("*.log") if p.name.startswith(prefixes + ("groot-", "foldquant-"))]
logs += [p for d in campaign for p in d.rglob("*.log")]
log_manifest = []
for path in sorted(logs):
    relative = path.relative_to(root)
    target = out / "log-tails" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # The full server log remains the authoritative artifact. Tails are for
    # reviewing diagnostics without committing large TensorRT compiler dumps.
    lines = path.read_text(errors="replace").splitlines()
    target.write_text("\n".join(line[:1600] for line in lines[-100:]) + "\n")
    log_manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path), excerpt=str(target.relative_to(out))))
for directory in root.iterdir():
    if not directory.is_dir() or not directory.name.startswith(("groot-trt-", "foldquant-")):
        continue
    for filename in ("fixture_binding.json", "retired-intermediates.json"):
        path = directory / filename
        if path.is_file():
            target = out / "assets" / directory.name / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    engines = directory / "engines"
    if engines.exists():
        items = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(engines.iterdir()) if p.is_file()]
        write_json(out / "assets" / directory.name / "engine-files.json", items)
calibration = Path("datasets/groot-n17-calibration-v2/manifest.json")
if calibration.exists():
    shutil.copy2(calibration, out / "calibration-manifest.json")
shutil.copy2("datasets/groot-n17-v3/manifest.json", out / "evaluation-manifest.json")
write_json(out / "run-index.json", rows)
write_json(out / "server-log-manifest.json", log_manifest)
print(json.dumps(rows, indent=2))
