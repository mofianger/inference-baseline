"""Hash and retire reproducible intermediates after a successful measured run.

Engine binaries, fixture bindings, original weights and datasets are retained.
Only onnx/ and quantized/ under an explicitly named recipe asset dir qualify.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument("--assets", required=True)
proof = p.add_mutually_exclusive_group(required=True)
proof.add_argument("--summary")
proof.add_argument("--failed-stages")
args = p.parse_args()
root = Path(args.assets).resolve()
allowed = (Path(__file__).resolve().parents[1] / "artifacts").resolve()
if root.parent != allowed or not root.name.startswith(("groot-trt-", "foldquant-")):
    raise ValueError("Asset path is outside the allowed generated deployment directories")
if args.summary:
    summary = json.loads(Path(args.summary).read_text())
    if summary.get("status") != "measured" or Path(summary["config"]["options"]["engine_dir"]).resolve() != root / "engines":
        raise ValueError("Need a measured run of these exact engines")
else:
    stages = json.loads(Path(args.failed_stages).read_text())
    if not stages or not stages[-1]["exit_code"]:
        raise ValueError("Need a recorded failed stage")
    asset_commands = [s["command"] for s in stages if s["stage"] != "run"]
    if not asset_commands or any(Path(c[c.index("--output") + 1]).resolve() != root for c in asset_commands):
        raise ValueError("Failed stages do not belong to these assets")
targets = [root / n for n in ("onnx", "quantized") if (root / n).exists()]
records = []
for target in targets:
    if target.is_symlink() or target.resolve().parent != root:
        raise ValueError("Unexpected intermediate symlink")
    for path in sorted(target.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            records.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=digest.hexdigest()))
record_path = root / "retired-intermediates.json"
previous = json.loads(record_path.read_text()) if record_path.exists() else []
record_path.write_text(json.dumps(previous + records, indent=2) + "\n")
for target in targets:
    shutil.rmtree(target)
print(json.dumps({"files": len(records), "freed_bytes": sum(r["bytes"] for r in records)}))
