"""FlashRT RTX FP8, including learned patch/token embeddings inside infer."""
from pathlib import Path
from ..io import read_json


class FlashRTAdapter:
    def __init__(self, config, protocol, samples):
        import torch
        from safetensors import safe_open
        from flash_rt.frontends.torch.groot_n17_rtx_fp8 import GrootN17TorchFrontendRtxFP8
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
        self.frontend = GrootN17TorchFrontendRtxFP8(config["checkpoint"], num_views=protocol.views,
            embodiment_tag="oxe_droid_relative_eef_relative_joint", device=self.device)
        if self.frontend._embodiment_id != protocol.embodiment_id:
            raise ValueError("FlashRT embodiment does not match fixture")
        if protocol.action_horizon != 40 or protocol.action_dim != 132:
            raise ValueError("This pinned FlashRT frontend requires H40/A132; no silent override")
        self.frontend.set_prompt(aux=self.aux(samples[0]))
        self.metadata = {"precision": "FP8 backbone / mixed BF16 action head", "cuda_graph": True,
            "implementation": "GrootN17TorchFrontendRtxFP8 + timed BF16 patch/token embedding",
            "calibration_sample": samples[0]["id"], "constant_cache": "position/RoPE only; no cached observation features"}

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
