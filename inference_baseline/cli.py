import argparse
import os
from pathlib import Path
import sys
from .io import read_json, write_json


def configure(config):
    for k, v in config.get("environment", {}).items():
        os.environ[k] = str(v)
    # Invoking a venv's Python directly does not activate its executables.
    # JIT backends must find that environment's ninja and other build tools.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    for p in reversed(config.get("python_paths", [])):
        sys.path.insert(0, str(Path(p).resolve()))


def main():
    parser = argparse.ArgumentParser(description="VLA prepared tensor → action tensor benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if name == "prepare":
            p.add_argument("--samples", type=int, default=5)
        else:
            p.add_argument("--dataset", required=True)
    p = sub.add_parser("report")
    p.add_argument("summaries", nargs="+")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "report":
        data = [read_json(p) for p in args.summaries]
        if len({d["dataset_manifest_sha256"] for d in data}) != 1:
            raise ValueError("Cannot compare different datasets/protocols")
        lines = ["# Model-only benchmark", "", "Accuracy gates are not implied by latency measurements.", "",
                 "| Engine | Mean ms | p50 ms | p95 ms | p99 ms | Max abs error | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
        for d in data:
            v = d["latency"]
            label = d['config'].get('label', d['config']['engine'])
            lines.append(f"| {label} | {v['mean_ms']:.3f} | {v['p50_ms']:.3f} | {v['p95_ms']:.3f} | {v['p99_ms']:.3f} | {max(x['max_abs'] for x in d['correctness']):.6f} | {d['accuracy_gate']} |")
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    config = read_json(args.config)
    configure(config)
    if args.command == "prepare":
        from .dataset import prepare
        prepare(config, args.output, args.samples)
    else:
        from .runner import run
        run(config, args.dataset, args.output)


if __name__ == "__main__":
    main()
