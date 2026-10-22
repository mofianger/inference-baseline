class EmbodiInferAdapter:
    def __init__(self, config, protocol, samples):
        from embodiinfer.policies.config import VLAPolicyConfig
        from embodiinfer.policies.gr00t.modeling_gr00t import Gr00tPolicy
        from embodiinfer.policies.gr00t.processor_gr00t import Gr00tBatch
        self.batch_type = Gr00tBatch
        self.steps = protocol.denoising_steps
        opts = config.get("options", {})
        policy_config = VLAPolicyConfig(name="gr00t", action_dim=protocol.action_dim,
            action_horizon=protocol.action_horizon, default_num_steps=self.steps, dtype="bfloat16")
        self.model = Gr00tPolicy(policy_config, checkpoint=config["checkpoint"],
            cosmos_path=config["cosmos"], attention="sdpa", native_inference=opts.get("native_inference", True),
            prefix_cuda_graph=opts.get("prefix_cuda_graph", False)).to(config["device"]).eval()
        self.metadata = {"precision": "BF16", "native_inference": opts.get("native_inference", True),
                         "prefix_cuda_graph": opts.get("prefix_cuda_graph", False), "denoise_graph": False,
                         "implementation": "Gr00tPolicy.sample_actions"}

    def infer(self, sample):
        t = sample["inputs"]
        batch = self.batch_type(backbone_inputs={k: t[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")},
                               state=t["state"], embodiment_id=t["embodiment_id"], request_ids=[sample["id"]])
        return self.model.sample_actions(batch, x0=sample["noise"].clone(), num_steps=self.steps)
