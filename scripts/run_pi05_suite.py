"""Resume the official pi05 conversion/fixture/run sequence on the GPU server.

Every benchmark executes in its own environment and fresh process. Failures
are recorded independently; no failed recipe is reported as measured.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait-for-download", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--recipes", nargs="+", help="Run only these recipe names (for isolated retries)")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Suite output must be new")
    output.mkdir(parents=True, exist_ok=True)
    official = Path("/data/huggingface/openpi/pi05_base")
    canonical = Path("/data/huggingface/openpi/pi05_base_pytorch_fp32")
    dataset = root / "datasets/pi05-base-v3-v1"
    reference = root / "third_party/openpi/.venv/bin/python"
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing", "stages": []}
    def save():
        (output / "suite.json").write_text(json.dumps(status, indent=2) + "\n")
    def execute(name, command):
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "JAX_PLATFORMS": "cpu"})
        status["stages"].append({"name": name, "returncode": result.returncode})
        save()
        return result.returncode == 0
    save()
    while not (official / "verified-files.json").is_file():
        if not args.wait_for_download:
            raise FileNotFoundError("Official download is not fully verified")
        status["status"] = "waiting_for_verified_official_weights"
        save()
        time.sleep(30)
    status["status"] = "preparing"
    save()
    if not (canonical / "conversion-provenance.json").is_file():
        if not execute("conversion", [reference, "scripts/convert_pi05_official.py", "--source", official, "--output", canonical]):
            status["status"] = "conversion_failed"
            save()
            return
    if not (dataset / "manifest.json").is_file():
        if not execute("prepare", [reference, "-m", "inference_baseline.cli", "prepare", "--config",
                                   "configs/5090/pi05-reference.json", "--output", dataset]):
            status["status"] = "fixture_failed"
            save()
            return
    runs = [("reference", reference),
            ("phyai", root / "third_party/phyai/.venv/bin/python"),
            ("embodiinfer", root / "third_party/EmbodiInfer/.venv-pi05/bin/python"),
            ("flashrt-bf16", root / "third_party/FlashRT/.venv/bin/python"),
            ("flashrt-fp8", root / "third_party/FlashRT/.venv/bin/python"),
            ("flashrt-nvfp4", root / "third_party/FlashRT/.venv/bin/python")]
    if args.recipes:
        unknown = set(args.recipes) - {name for name, _ in runs}
        if unknown:
            raise ValueError(f"Unknown recipes: {sorted(unknown)}")
        runs = [(name, python) for name, python in runs if name in args.recipes]
    status["status"] = "running"
    save()
    for name, python in runs:
        execute(name, [python, "-m", "inference_baseline.cli", "run", "--config", f"configs/5090/pi05-{name}.json",
                       "--dataset", dataset, "--output", output / name])
    status["status"] = "finished_with_failures" if any(s["returncode"] for s in status["stages"]) else "finished"
    save()


if __name__ == "__main__":
    main()
