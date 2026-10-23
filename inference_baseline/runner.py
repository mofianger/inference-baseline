from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

from .adapters import create
from .dataset import load_dataset, move_sample
from .hardware import TorchDevice
from .io import checkpoint_manifest, revision, sha256, write_json
from .metrics import errors, latency_summary


def run(config, dataset, output):
    import torch
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Run output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    try:
        return _run(config, dataset, out, torch)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "config": config})
        raise


def _run(config, dataset, out, torch):
    if config.get("warmup", 10) < 1 or config.get("iterations", 100) < 1:
        raise ValueError("warmup and iterations must be positive")
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, protocol, cpu_samples = load_dataset(dataset)
    if checkpoint_manifest(config["checkpoint"]) != manifest["checkpoint"]:
        raise ValueError("Checkpoint differs from frozen reference; refusing comparison")
    device = TorchDevice(config.get("device", "cuda:0"))
    before = device.metadata()
    samples = [move_sample(s, device.device) for s in cpu_samples]
    eval_samples = [s for s in samples if s["role"] == "evaluation"]
    if len(eval_samples) < 2:
        raise ValueError("Need at least two distinct evaluation observations")
    start = time.perf_counter()
    with torch.inference_mode():
        adapter = create(config["engine"], config, protocol, samples)
        # Some engines change process-global Torch settings during construction.
        # Reapply the declared benchmark controls before correctness/timing.
        torch.set_num_threads(config.get("cpu_threads", 8))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device.sync()
        setup_seconds = time.perf_counter() - start
        for i in range(config.get("warmup", 10)):
            adapter.infer(eval_samples[i % len(eval_samples)])
        device.sync()
        runtime_before = adapter.runtime_stats() if hasattr(adapter, "runtime_stats") else None
        correctness = []
        # A/B/A detects stale cached outputs and output-buffer aliasing.
        sequence = eval_samples + [eval_samples[0]]
        frozen = []
        for sample in sequence:
            actual = adapter.infer(sample)
            device.sync()
            if actual.device != device.device or list(actual.shape) != [protocol.batch, protocol.action_horizon, protocol.action_dim]:
                raise ValueError("Adapter returned wrong action shape/device")
            array = actual.detach().float().cpu().numpy().copy()
            frozen.append(array)
            correctness.append({"sample_id": sample["id"], **errors(array, sample["reference_actions"].float().cpu().numpy(), config.get("thresholds"))})
        import numpy as np
        repeat_error = errors(frozen[-1], frozen[0])
        if not np.allclose(frozen[0], frozen[-1], atol=1e-5, rtol=1e-5):
            raise ValueError("Repeated input produced unstable output (A/B/A check)")
        if all(np.array_equal(frozen[0], x) for x in frozen[1:-1]):
            raise ValueError("Different observations produced identical outputs; possible stale cache")
        if any(x["gate"] == "failed" for x in correctness):
            write_json(out / "correctness.json", correctness)
            raise ValueError("Configured accuracy gate failed")
        # Change only one dynamic input at a time to detect graph buffers that
        # accidentally retain old pixels, state or initial noise.
        input_sensitivity = {}
        for field in ("pixel_values", "state", "noise"):
            probe = dict(eval_samples[0])
            probe["inputs"] = dict(probe["inputs"])
            if field == "noise":
                probe["noise"] = eval_samples[1]["noise"]
            else:
                probe["inputs"][field] = eval_samples[1]["inputs"][field]
            probe_array = adapter.infer(probe).detach().float().cpu().numpy().copy()
            delta = errors(probe_array, frozen[0])
            if delta["max_abs"] == 0:
                raise ValueError(f"Output ignored changed {field}; possible stale graph buffer")
            input_sensitivity[field] = {k: delta[k] for k in ("max_abs", "mae", "cosine")}
        restored = adapter.infer(eval_samples[0]).detach().float().cpu().numpy()
        if not np.allclose(restored, frozen[0], atol=1e-5, rtol=1e-5):
            raise ValueError("Input sensitivity probes changed subsequent baseline output")
        if device.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device.device)
        rows = []
        with (out / "latency.jsonl").open("w", encoding="utf-8") as f:
            for i in range(config.get("iterations", 100)):
                sample = eval_samples[i % len(eval_samples)]
                actual, ms = device.time(lambda: adapter.infer(sample))
                row = {"iteration": i, "sample_id": sample["id"], "wall_ms": ms}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
        # Ensure adapters did not mutate their prepared input tensors or initial noise.
        for original, current in zip(cpu_samples, samples):
            for key, value in original["inputs"].items():
                if not torch.equal(value, current["inputs"][key].cpu()):
                    raise ValueError(f"Adapter mutated input {key}")
            if not torch.equal(original["noise"], current["noise"].cpu()):
                raise ValueError("Adapter mutated initial noise")
        runtime_after = adapter.runtime_stats() if hasattr(adapter, "runtime_stats") else None
        if runtime_before != runtime_after:
            raise ValueError(f"Runtime graphs/compilation changed after warmup: {runtime_before} -> {runtime_after}")
        if config.get("options", {}).get("denoise_graph") and (
            not runtime_before or not runtime_before["denoise_graphs"] or
            (config["options"].get("prefix_cuda_graph") and not runtime_before["prefix"]["layouts"]) or
            (config["options"].get("compile_backend") == "inductor" and not runtime_before["compiled_graphs"])
        ):
            raise ValueError("Requested optimization did not produce captured/compiled graphs")
        packages = {}
        for name in ("torch", "transformers", "diffusers", "triton", "flashinfer-python"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        summary = {"status": "measured", "accuracy_gate": "passed" if config.get("thresholds") else "not_configured",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "protocol": protocol.to_dict(), "dataset_manifest_sha256": sha256(Path(dataset) / "manifest.json"),
            "dataset_kind": manifest["data_kind"], "checkpoint": manifest["checkpoint"],
            "runner": revision(Path(__file__).resolve().parents[1]), "engine_source": revision(config["engine_source"]),
            "engine": adapter.metadata, "packages": packages, "hardware_before": before, "hardware_after": device.metadata(),
            "setup_seconds": setup_seconds, "latency": latency_summary([r["wall_ms"] for r in rows]),
            "runtime_before": runtime_before, "runtime_after": runtime_after,
            "correctness": correctness, "repeat_error": repeat_error,
            "input_sensitivity": input_sensitivity,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device.device) if device.device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device.device) if device.device.type == "cuda" else None,
            "limitations": ["synthetic fixture, no task accuracy", "small fixture set; latency iterations reuse observations", "no acceptance claim without configured thresholds"]}
        write_json(out / "summary.json", summary)
        print(json.dumps({"engine": config["engine"], "latency": summary["latency"], "accuracy_gate": summary["accuracy_gate"]}, indent=2), flush=True)
        return summary
