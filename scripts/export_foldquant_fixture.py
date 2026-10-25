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
        # ModelOpt 0.39 imports this public type from the old package location
        # under torch 2.9. Bind the same upstream type; no symbolic/math changes.
        import torch.onnx._internal.torchscript_exporter as legacy_exporter
        if not hasattr(legacy_exporter, "JitScalarType"):
            from torch.onnx._internal.torchscript_exporter._type_utils import JitScalarType
            legacy_exporter.JitScalarType = JitScalarType
        # Native FoldQuant packs are replayed from QuantState. ModelOpt also
        # modifies floating-point weights during calibration; those live in
        # the quantized checkpoint, not in its quantizer state.
        state = quantized.load_state_for(out / "quantized", config["checkpoint"])
        policy = policy_factory()
        if any("modelopt_state" in ms.config for ms in state.modules.values()):
            from safetensors import safe_open
            parameters = dict(policy.model.named_parameters())
            buffers = dict(policy.model.named_buffers())
            loaded = set()
            with torch.no_grad():
                for shard in sorted((out / "quantized").glob("*.safetensors")):
                    with safe_open(str(shard), framework="pt", device="cpu") as f:
                        for name in f.keys():
                            target = parameters.get(name, buffers.get(name))
                            if target is not None:
                                value = f.get_tensor(name)
                                if value.shape != target.shape:
                                    raise ValueError(f"Quantized checkpoint shape mismatch: {name}")
                                target.copy_(value)
                                loaded.add(name)
                            elif not name.startswith("foldquant."):
                                raise ValueError(f"Unexpected quantized checkpoint tensor: {name}")
            missing = set(parameters) - loaded
            if missing:
                raise ValueError(f"Missing ModelOpt checkpoint parameters: {sorted(missing)[:10]}")
            write_json(out / "checkpoint_replay.json", {
                "source": str(out / "quantized"), "loaded_model_tensors": len(loaded),
                "floating_weights": "upstream quantized checkpoint (including smoothing)"})
        with torch.inference_mode():
            export_onnx(state, quantized.module_paths(policy), out / "onnx",
                        source=out / "quantized", adapter=quantized, policy=policy)
            quantized.complete_onnx(policy, out / "onnx", state)


if __name__ == "__main__":
    main()
