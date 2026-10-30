"""FlashRT RTX FP8, including learned patch/token embeddings inside infer."""
from pathlib import Path
from ..io import read_json, sha256


class FlashRTAdapter:
    def __init__(self, config, protocol, samples):
        import torch
        from safetensors import safe_open
        from flash_rt.frontends.torch.groot_n17_rtx_fp8 import GrootN17TorchFrontendRtxFP8
        from flash_rt.core.quant import calibrator
        opts = config.get("options", {})
        # Upstream action-head calibration may sample noise during setup.
        # Timed inference still receives the shared fixture noise explicitly.
        calibration_seed = opts.get("calibration_seed", 0)
        torch.manual_seed(calibration_seed)
        calibrator.CACHE_DIR = Path(opts["calibration_cache"]).resolve()
        frontend_class = GrootN17TorchFrontendRtxFP8
        if opts.get("fp16_action_head", False):
            if opts.get("use_fp4", False):
                raise ValueError("FP16 action head and NVFP4 are mutually exclusive")
            from flash_rt.frontends.torch.groot_n17_rtx_fp8 import GrootN17TorchFrontendRtxFP8FP16DiT
            frontend_class = GrootN17TorchFrontendRtxFP8FP16DiT
        if opts.get("use_fp4", False):
            from flash_rt.frontends.torch.groot_n17_rtx_fp4 import GrootN17TorchFrontendRtxFP4
            frontend_class = GrootN17TorchFrontendRtxFP4
        self.torch = torch
        self.protocol = protocol
        self.device = config["device"]
        root = Path(config["checkpoint"])
        index = read_json(root / "model.safetensors.index.json")["weight_map"]
        def load(name):
            with safe_open(str(root / index[name]), framework="pt", device="cpu") as f:
                return f.get_tensor(name).to(device=self.device, dtype=torch.bfloat16)
        prefix = "backbone.model.model."
        self.patch_weight = load(prefix + "visual.patch_embed.proj.weight")
        self.patch_bias = load(prefix + "visual.patch_embed.proj.bias") if prefix + "visual.patch_embed.proj.bias" in index else None
        self.token_weight = load(prefix + "language_model.embed_tokens.weight")
        # Frontend maps this spelling to the checkpoint's same category 24.
        self.frontend = frontend_class(config["checkpoint"], num_views=protocol.views,
            embodiment_tag="oxe_droid_relative_eef_relative_joint", device=self.device)
        if self.frontend._embodiment_id != protocol.embodiment_id:
            raise ValueError("FlashRT embodiment does not match fixture")
        if protocol.action_horizon != 40 or protocol.action_dim != 132:
            raise ValueError("This pinned FlashRT frontend requires H40/A132; no silent override")
        self.frontend.set_prompt(aux=self.aux(samples[0]))
        from flash_rt.frontends.torch.groot_n17_rtx_fp8 import _fused_backbone_available
        actual_fused = _fused_backbone_available(self.frontend._fvk)
        if actual_fused != opts.get("fused_backbone", True):
            raise RuntimeError("Requested FlashRT backbone fusion mode does not match selected path")
        self.metadata = {"precision": "FP8 backbone / mixed FP8-BF16 action head", "cuda_graph": True,
            "implementation": "GrootN17TorchFrontendRtxFP8 + timed BF16 patch/token embedding",
            "calibration_sample": samples[0]["id"], "constant_cache": "position/RoPE only; no cached observation features"}
        self.metadata.update(implementation=frontend_class.__name__ + " + timed BF16 patch/token embedding",
            calibration_cache=str(calibrator.CACHE_DIR), calibration_seed=calibration_seed,
            use_fp4=opts.get("use_fp4", False))
        if opts.get("use_fp4", False):
            self.metadata["precision"] = "FP8 backbone / NVFP4 DiT and cross-KV / mixed BF16 remaining ops"
        if opts.get("fp16_action_head", False):
            self.metadata["precision"] = "FP8 backbone / FP16 action head; upstream FP8-loaded then dequantized DiT weights"
        # SM120 GR00T NVFP4 lives in flash_rt_kernels. The separate
        # flash_rt_fp4 add-on is built for SM100/110, not this target.
        from flash_rt import flash_rt_kernels, flash_rt_fa2
        self.metadata["binaries"] = {m.__name__: {"path": m.__file__, "sha256": sha256(m.__file__)}
            for m in (flash_rt_kernels, flash_rt_fa2)}
        self.metadata["fused_backbone"] = actual_fused
        self.metadata["calibration_files"] = {p.name: sha256(p) for p in calibrator.CACHE_DIR.glob("*.json")}

    def runtime_stats(self):
        f = self.frontend
        if self.metadata["implementation"].startswith("GrootN17TorchFrontendRtxFP8FP16DiT"):
            graphs = getattr(f, "_dit_graphs", [])
            if not hasattr(f, "_kbb_graph") or len(graphs) != self.protocol.denoising_steps:
                raise RuntimeError("FP16 action head did not capture all step graphs")
            return {"backbone_graph": True, "action_step_graphs": len(graphs), "dit_quant": "FP16"}
        if not hasattr(f, "_kbb_graph") or not hasattr(f, "_k_dit_graph"):
            raise RuntimeError("FlashRT backbone/action graph was not captured")
        if f._k_nsteps != self.protocol.denoising_steps:
            raise RuntimeError("FlashRT captured a different denoising step count")
        if self.metadata["use_fp4"] and not hasattr(f, "_ck_nvfp4"):
            raise RuntimeError("NVFP4 cross-KV was not enabled")
        return {"backbone_graph": True, "action_graph": True, "steps": f._k_nsteps,
                "dit_quant": f._DIT_QUANT, "dit_use_fp8": f._DIT_USE_FP8,
                "nvfp4_cross_kv": hasattr(f, "_ck_nvfp4")}

    def aux(self, sample):
        t = sample["inputs"]
        constants = sample["flashrt_constants"]
        torch = self.torch
        w = self.patch_weight
        pixels = t["pixel_values"].reshape(-1, w.shape[1], *w.shape[2:])
        patches = torch.nn.functional.conv3d(pixels, w, self.patch_bias, stride=w.shape[2:]).reshape(-1, w.shape[0])
        patches = patches + constants["vision_pos_embed"]
        embeds = torch.nn.functional.embedding(t["input_ids"], self.token_weight)
        return {"pixel_features": patches, "llm_input_embeds": embeds,
                "input_ids": t["input_ids"], "grid_thw": t["image_grid_thw"],
                **{k: constants[k] for k in ("position_ids", "rope_cos", "rope_sin", "visual_pos_masks")}}

    def infer(self, sample):
        # Passing aux is essential: without it the frontend reuses the previous backbone!
        return self.frontend.infer(sample["inputs"]["state"], aux=self.aux(sample),
            initial_noise=sample["noise"], num_inference_timesteps=self.protocol.denoising_steps,
            action_horizon=self.protocol.action_horizon, use_dit_graph=True)
