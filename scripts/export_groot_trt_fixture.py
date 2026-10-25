"""Feed the frozen calibration partition to NVIDIA's unmodified exporter.

Only the dataset/policy I/O boundary is replaced: no preprocessing is repeated,
and no evaluation inputs are used for PTQ. Upstream capture, calibration,
quantization, ONNX export and TensorRT build functions remain unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_baseline.cli import configure
from inference_baseline.io import checkpoint_manifest, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="datasets/groot-n17-v3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantization", choices=["none", "fp8", "mixed_nvfp4"], default="none")
    parser.add_argument("--profile", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--stage", choices=["export", "build"], default="export")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    configure(config)
    import torch
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261025)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        from scripts.deployment.trt.build_tensorrt_engine import build_full_pipeline
        build_full_pipeline(str(output / "onnx"), str(output / "engines"),
                            precision="bf16", workspace_mb=4096)
        from scripts.deployment.build_trt_pipeline import _copy_export_metadata
        _copy_export_metadata(str(output / "onnx"), str(output / "engines"))
        return
    from inference_baseline.dataset import load_dataset, move_sample
    from inference_baseline.adapters.reference import ReferenceAdapter
    manifest, protocol, samples = load_dataset(args.dataset)
    if checkpoint_manifest(config["checkpoint"]) != manifest["checkpoint"]:
        raise ValueError("Checkpoint mismatch")
    calibration = [move_sample(s, config["device"]) for s in samples if s["role"] == "calibration"]
    if not calibration:
        raise ValueError("No calibration-only fixtures")
    from scripts.deployment.trt import export_onnx_n1d7 as exporter
    from scripts.deployment.trt import n1d7_calibration as calibrator

    class PreparedPolicy:
        def __init__(self, **kwargs):
            self.adapter = ReferenceAdapter(config, protocol, calibration)
            self.model = self.adapter.model
            self.modality_configs = {}

        def get_modality_config(self):
            return {}

        def get_action(self, sample):
            return self.adapter.infer(sample)

    class PreparedLoader:
        def __init__(self, **kwargs):
            pass

        def __len__(self):
            return len(calibration)

        def __getitem__(self, index):
            return [calibration[index]]

    exporter.Gr00tPolicy = PreparedPolicy
    exporter.LeRobotEpisodeLoader = PreparedLoader
    exporter.prepare_observation = lambda policy, dataset, traj_idx=0: calibration[traj_idx]
    calibrator.LeRobotEpisodeLoader = PreparedLoader
    calibrator._prepare_step_observation = lambda loader, ep_idx, step_idx, policy: calibration[ep_idx]
    calibrator._collate_inputs = lambda policy, sample: dict(sample["inputs"])
    export_config = exporter.ExportConfig(
        model_path=config["checkpoint"], dataset_path=args.dataset,
        embodiment_tag=manifest["embodiment_tag"], output_dir=str(output / "onnx"),
        export_mode="full_pipeline", execution_profile=args.profile,
        quantization=args.quantization, calib_size=len(calibration), batch_size=1)
    write_json(output / "fixture_binding.json", {
        "dataset_manifest_sha256": sha256(Path(args.dataset) / "manifest.json"),
        "calibration_sample_ids": [s["id"] for s in calibration],
        "quantization": args.quantization, "profile": args.profile,
        "seed": 20261025, "checkpoint": manifest["checkpoint"]})
    with torch.inference_mode():
        exporter.main(export_config)


if __name__ == "__main__":
    main()
