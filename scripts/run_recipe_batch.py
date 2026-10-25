"""Sequential, failure-tolerant runs of explicitly selected existing recipes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--python", required=True)
p.add_argument("--configs", nargs="+", required=True)
p.add_argument("--dataset", default="datasets/groot-n17-v3")
p.add_argument("--output", required=True)
args = p.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
rows = []
for config in args.configs:
    name = Path(config).stem
    command = [args.python, "-m", "inference_baseline.cli", "run", "--config", config,
               "--dataset", args.dataset, "--output", str(root / name)]
    started = datetime.now(timezone.utc).isoformat()
    with (root / f"{name}.log").open("w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    rows.append(dict(config=config, command=command, exit_code=code, started_utc=started,
                     finished_utc=datetime.now(timezone.utc).isoformat()))
    (root / "batch.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows[-1]), flush=True)
