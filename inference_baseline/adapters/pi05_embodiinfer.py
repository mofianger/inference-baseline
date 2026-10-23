"""EmbodiInfer native pi05 runtime, strict canonical model-only weights."""
from pathlib import Path


class Pi05EmbodiInferAdapter:
    def __init__(self, config, protocol, samples):
        import torch
        from safetensors.torch import load_model
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from embodiinfer.policies.config import VLAPolicyConfig
        from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
        from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch
        from embodiinfer.policies.pi05.inference import Pi05OptimizationConfig
        self.batch_type = Pi05Batch
        self.protocol = protocol
        options = config.get("options", {})
        self.request_graph = options.get("request_cuda_graph", True)
        native_config = PI05Config(device="cpu", dtype="float32", chunk_size=50, n_action_steps=50,
            max_state_dim=32, max_action_dim=32, num_inference_steps=10, tokenizer_max_length=200,
            input_features={**{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in
                               ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,))},
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(32,))})
        # Already-normalized fixtures need no robot processor or norm stats.
        # This mirrors upstream's OpenPI strict model loader, bypassing only
        # its raw-observation processor setup.
        loaded = PI05Policy(native_config)
        pwe = loaded.model.paligemma_with_expert
        pwe.paligemma.lm_head = None
        pwe.gemma_expert.lm_head = None
        load_model(loaded.model, str(Path(config["checkpoint"]) / "model.safetensors"), strict=True)
        pwe.to_bfloat16_for_selected_params("bfloat16")
        loaded.to(config["device"]).eval()
        self.policy = Pi05Policy(VLAPolicyConfig(name="pi05", action_dim=32, action_horizon=50,
            default_num_steps=10, dtype="float32"), loaded, native_inference=True, native_embeddings=True,
            prefix_cuda_graph=options.get("prefix_cuda_graph", True), request_cuda_graph=self.request_graph,
            optimizations=Pi05OptimizationConfig() if options.get("fused_ops", False) else None).eval()
        self.metadata = {"implementation": "EmbodiInfer native Pi05Runtime",
                         "precision": "BF16 text/expert with FP32 full vision, projector, norms and action projections (LeRobot policy)",
                         "request_cuda_graph": self.request_graph,
                         "fused_ops": options.get("fused_ops", False),
                         "weight_loading": "strict canonical OpenPI tensors; no raw-observation processor",
                         "boundary": "prepared GPU tensors through prefix and full denoising"}

    def infer(self, sample):
        t = sample["inputs"]
        batch = self.batch_type(images=[t["pixel_values"][:, i] for i in range(3)],
            img_masks=[t["image_mask"][:, i] for i in range(3)], tokens=t["input_ids"], masks=t["attention_mask"])
        runtime = self.policy._runtime
        if self.request_graph:
            return runtime.request(batch, sample["noise"], self.protocol.denoising_steps)
        prefix = self.policy.encode_prefix(batch)
        return runtime.decode(sample["noise"], prefix, self.protocol.denoising_steps, use_graph=True)

    def runtime_stats(self):
        stats = self.policy._runtime.stats()
        if self.request_graph and not stats["request_graphs"]:
            raise RuntimeError("Requested pi05 full-request graph was not captured")
        return stats
