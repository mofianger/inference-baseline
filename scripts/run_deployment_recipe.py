"""Run an existing deployment pipeline, retaining each stage's log/exit code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--dataset", default="datasets/groot-n17-v3")
p.add_argument("--assets", required=True)
p.add_argument("--output", required=True)
p.add_argument("--start", choices=["quantize", "export", "build", "run"], default="export")
p.add_argument("--retire-intermediates", action="store_true")
args = p.parse_args()
config = json.loads(Path(args.config).read_text())
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
fold = config["label"].startswith("foldquant")
stages = (["quantize"] if fold else []) + ["export", "build", "run"]
if args.start not in stages:
    raise ValueError("Invalid start stage")
records = []
for stage in stages[stages.index(args.start):]:
    if stage == "run":
        command = [sys.executable, "-m", "inference_baseline.cli", "run", "--config", args.config,
                   "--dataset", args.dataset, "--output", str(out / "run")]
    else:
        command = [sys.executable, "scripts/export_foldquant_fixture.py" if fold else "scripts/export_groot_trt_fixture.py",
                   "--config", args.config, "--dataset", args.dataset, "--output", args.assets, "--stage", stage]
        if not fold:
            command += ["--profile", config["options"]["mode"], "--quantization", config["options"]["quantization"]]
    started = datetime.now(timezone.utc).isoformat()
    with (out / f"{stage}.log").open("w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(stage=stage, command=command, exit_code=code, started_utc=started,
                        finished_utc=datetime.now(timezone.utc).isoformat()))
    (out / "stages.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records[-1]), flush=True)
    if code:
        if args.retire_intermediates and any(s["stage"] != "run" for s in records):
            subprocess.run([sys.executable, "scripts/retire_deployment_intermediates.py",
                            "--assets", args.assets, "--failed-stages", str(out / "stages.json")], check=True)
        sys.exit(code)
if args.retire_intermediates:
    subprocess.run([sys.executable, "scripts/retire_deployment_intermediates.py",
                    "--assets", args.assets, "--summary", str(out / "run" / "summary.json")], check=True)
