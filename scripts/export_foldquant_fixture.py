"""Bind frozen calibration tensors to FoldQuant's existing GR00T pipeline."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_baseline.cli import configure
from inference_baseline.io import checkpoint_manifest, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="datasets/groot-n17-v3")
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=["quantize", "export", "build"], required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    configure(config)
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(20261025)
    out = Path(args.output)
    if args.stage == "build":
        from foldquant_integration.build_engines import BuildConfig, build
        build(BuildConfig(onnx_dir=str(out / "onnx"), engine_dir=str(out / "engines"),
                          max_batch=1, workspace_mb=4096))
        return
    from inference_baseline.dataset import load_dataset, move_sample
    from inference_baseline.adapters.reference import ReferenceAdapter
    manifest, protocol, samples = load_dataset(args.dataset)
    if checkpoint_manifest(config["checkpoint"]) != manifest["checkpoint"]:
        raise ValueError("Checkpoint mismatch")
    calibration_path = config["options"].get("calibration_dataset", args.dataset)
    cm, cp, calibration_samples = load_dataset(calibration_path)
    if cm["checkpoint"] != manifest["checkpoint"] or cp != protocol:
        raise ValueError("Calibration checkpoint/protocol mismatch")
    samples = [move_sample(s, config["device"]) for s in calibration_samples if s["role"] == "calibration"]
    if not samples:
        raise ValueError("No calibration partition")
    from foldquant_integration import calibration, quantized
    from gr00t.data.embodiment_tags import EmbodimentTag

    def policy_factory(*a, **kw):
        adapter = ReferenceAdapter(config, protocol, samples)
        return SimpleNamespace(model=adapter.model, get_action=adapter.infer,
                               embodiment_tag=EmbodimentTag.resolve(manifest["embodiment_tag"].removesuffix("_3view")))

    if args.stage == "quantize":
        from foldquant_integration.quantize import QuantizeConfig, main as quantize
        calibration.load_policy = policy_factory
        calibration.load_dataset = lambda *a, **kw: samples
        calibration.sample_observations = lambda *a, **kw: (
            [calibration.SampleId(i, 0) for i in range(len(samples))], samples)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "fixture_binding.json", {
            "dataset_manifest_sha256": sha256(Path(args.dataset) / "manifest.json"),
            "calibration_manifest_sha256": sha256(Path(calibration_path) / "manifest.json"),
            "calibration_sample_ids": [s["id"] for s in samples],
            "checkpoint": manifest["checkpoint"], "seed": 20261025})
        quantize(QuantizeConfig(model_path=config["checkpoint"], dataset_path=str(calibration_path),
            output_dir=str(out / "quantized"), num_calib=len(samples), seed=20261025,
            embodiment_tag=manifest["embodiment_tag"].removesuffix("_3view"),
            llm_scheme=config["options"]["llm_scheme"], dit_scheme=config["options"]["dit_scheme"],
            base_model_id="nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495"))
    else:
        from foldquant.quantized import export_onnx
        # Public replay API checks the quantization state against the frozen
        # base checkpoint. Quantized weights come from that recorded state;
        # the unquantized components are loaded from the same frozen base.
        state = quantized.load_state_for(out / "quantized", config["checkpoint"])
        policy = policy_factory()
        with torch.inference_mode():
            export_onnx(state, quantized.module_paths(policy), out / "onnx",
                        source=out / "quantized", adapter=quantized, policy=policy)
            quantized.complete_onnx(policy, out / "onnx", state)


if __name__ == "__main__":
    main()
