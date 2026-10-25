"""Boundary adapter for NVIDIA's existing GR00T TensorRT deployment recipes.

No kernels, quantization algorithms, or model forward implementation live here.
The upstream exporter/runtime own those; this adapter only binds frozen tensors
and supplies the shared initial noise through the upstream init_actions hook.
"""
from pathlib import Path
from types import SimpleNamespace

from .reference import ReferenceAdapter


class GrootTRTAdapter(ReferenceAdapter):
    def __init__(self, config, protocol, samples):
        import torch
        super().__init__(config, protocol, samples)
        options = config["options"]
        if options.get("mode") == "compile":
            self.model.action_head.model.forward = torch.compile(
                self.model.action_head.model.forward, mode="max-autotune")
            self.metadata.update(implementation="official torch.compile DiT max-autotune")
            self.trt = False
            return
        import json
        from scripts.deployment.trt.trt_model_forward import setup_tensorrt_engines, _engine_path
        directory = Path(options["engine_dir"])
        metadata = json.loads((directory / "export_metadata.json").read_text())
        if metadata["num_inference_timesteps"] != protocol.denoising_steps:
            raise ValueError("TensorRT denoising steps differ from protocol")
        if metadata["vit_grid_thw"] != samples[0]["inputs"]["image_grid_thw"].tolist():
            raise ValueError("TensorRT image grid differs from fixture")
        for component in ("vit", "llm", "dit"):
            if not Path(_engine_path(str(directory), component)).is_file():
                raise FileNotFoundError(f"Missing {component} engine; refusing fallback")
        self.policy = SimpleNamespace(model=self.model)
        setup_tensorrt_engines(self.policy, str(directory), mode="n17_full_pipeline")
        self.trt = True
        self.metadata = {"implementation": "NVIDIA upstream TensorRT full_pipeline",
                         "export_metadata": metadata,
                         "fixture_binding": json.loads((directory.parent / "fixture_binding.json").read_text()),
                         "noise_injection": "upstream init_actions + captured initial_actions copy",
                         "boundary": "prepared GPU inputs before embeddings to full normalized GPU action"}

    def infer(self, sample):
        if not self.trt:
            return super().infer(sample)
        head = self.model.action_head
        # Cast/copy happens inside the timed call. Do not mutate fixture noise.
        runner = getattr(head, "_trt_optimization_runner", None)
        dtype = (runner.state_encoder_engine.dtype_of("state") if runner is not None
                 else head.state_encoder_engine.in_meta[0][2])
        head.init_actions = sample["noise"].to(dtype=dtype).clone()
        state = getattr(runner, "_graph_state", None)
        if state is not None:
            state.initial_actions.copy_(head.init_actions)
        return self.model.get_action(dict(sample["inputs"]))["action_pred"]

    def runtime_stats(self):
        runner = getattr(self.model.action_head, "_trt_optimization_runner", None)
        if self.trt and self.metadata["export_metadata"]["execution_profile"] == "optimized":
            if getattr(runner, "_graph_state", None) is None or getattr(runner, "_capture_failed", False):
                raise RuntimeError("Requested optimized TensorRT graph is not active")
        return {"optimized_graph": getattr(runner, "_graph_state", None) is not None,
                "capture_failed": getattr(runner, "_capture_failed", False)}
