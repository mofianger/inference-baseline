import hashlib
import json
import subprocess
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_manifest(path):
    root = Path(path)
    files = sorted(set(root.glob("*.safetensors")) | set(root.glob("*.json")))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"No safetensors checkpoint at {root}")
    return {p.name: sha256(p) for p in files}


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def revision(path):
    return {"commit": command(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "dirty": command(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"])}
