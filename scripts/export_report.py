"""Copy selected small run summaries into Git-tracked reports, with a comparison."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Run directories containing summary.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run in args.runs:
        source = Path(run) / "summary.json"
        data = json.loads(source.read_text())
        if data["status"] != "measured":
            raise ValueError(f"Not a completed measurement: {run}")
        dest = target / (data["config"]["engine"] + ".json")
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite report {dest}")
        shutil.copyfile(source, dest)
        shutil.copyfile(Path(run) / "latency.jsonl", target / (data["config"]["engine"] + "-latency.jsonl"))
        summaries.append(str(dest))
    subprocess.run([sys.executable, "-m", "inference_baseline.cli", "report", *summaries,
                    "--output", str(target / "comparison.md")], check=True)


if __name__ == "__main__":
    main()
