"""Convert verified official Orbax weights with upstream mapping and audited loading."""
import argparse
import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Conversion destination must be new")
    files = json.loads((source / "verified-files.json").read_text())
    listing = json.loads((source / "official-source.json").read_text())
    expected = {x["name"].removeprefix("checkpoints/pi05_base/") for x in listing["objects"]}
    if {x["path"] for x in files} != expected:
        raise ValueError("Incomplete verified source manifest")
    for entry in files:
        path = (source / entry["path"]).resolve()
        if not path.is_relative_to(source) or digest(path) != entry["sha256"]:
            raise ValueError(f"Source verification failed: {entry['path']}")
    checkout = Path(__file__).resolve().parents[1] / "third_party/openpi"
    sys.path.insert(0, str(checkout / "src"))
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    script = checkout / "examples/convert_jax_model_to_pytorch.py"
    tree = ast.parse(script.read_text())
    # Training config imports are used only by the upstream CLI; do not require
    # datasets or simulator dependencies for the conversion functions.
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module == "openpi.training" or
        isinstance(n, ast.Import) and any(a.name.startswith("openpi.training") for a in n.names))]
    namespace = {"__name__": "benchmark_conversion"}
    exec(compile(tree, str(script), "exec"), namespace)
    audit = {}
    original = PI0Pytorch.load_state_dict
    def audited_load(self, state_dict, strict=True, **kwargs):
        result = original(self, state_dict, strict=False, **kwargs)
        # Both heads are unused by continuous action sampling. PaliGemma's
        # head is tied to its loaded input embedding; the expert has no tokens.
        allowed = {"paligemma_with_expert.paligemma.lm_head.weight",
                   "paligemma_with_expert.gemma_expert.lm_head.weight"}
        if set(result.missing_keys) - allowed or result.unexpected_keys:
            raise RuntimeError(f"Unaccounted checkpoint keys: {result}")
        audit.update(missing_unused_heads=result.missing_keys, unexpected_keys=result.unexpected_keys,
                     loaded_tensor_count=len(state_dict))
        return result
    PI0Pytorch.load_state_dict = audited_load
    # Construct in FP32 so upstream load_state_dict cannot round weights before
    # writing the canonical export. Recipe-specific casts happen at engine load.
    config = Pi0Config(pi05=True, dtype="float32", pytorch_compile_mode=None)
    try:
        namespace["convert_pi0_checkpoint"](str(source), "float32", str(output), config)
    finally:
        PI0Pytorch.load_state_dict = original
    provenance = {"source": listing["source"], "source_files": files,
                  "openpi_commit": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
                  "conversion_script_sha256": digest(script), "model_config": dataclasses.asdict(config),
                  "load_audit": audit, "export_sha256": digest(output / "model.safetensors")}
    (output / "conversion-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["load_audit"]), flush=True)


if __name__ == "__main__":
    main()
