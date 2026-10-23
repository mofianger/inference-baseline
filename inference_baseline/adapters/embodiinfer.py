class EmbodiInferAdapter:
    def __init__(self, config, protocol, samples):
        import torch
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
        self.graphs = None
        if opts.get("denoise_graph", False):
            from embodiinfer.engine.graph import GraphManager
            self.graphs = GraphManager(self.model, torch.device(config["device"]), torch.bfloat16, full_loop=True)
            if opts.get("compile_backend") == "inductor":
                self.model.denoise_step = torch.compile(self.model.denoise_step, backend="inductor",
                    options={"triton.cudagraphs": False})
            self.metadata.update(denoise_graph=True, compile_backend=opts.get("compile_backend"),
                implementation="encode_prefix + FlowActionDecoder.integrate + GraphManager(full_loop=True)")

    def runtime_stats(self):
        from torch._dynamo.utils import counters
        return {"prefix": self.model._prefix_runtime.stats(),
                "denoise_graphs": len(self.graphs._graphs) if self.graphs else 0,
                "compiled_graphs": counters["stats"].get("unique_graphs", 0)}

    def infer(self, sample):
        t = sample["inputs"]
        batch = self.batch_type(backbone_inputs={k: t[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")},
                               state=t["state"], embodiment_id=t["embodiment_id"], request_ids=[sample["id"]])
        if self.graphs is None:
            return self.model.sample_actions(batch, x0=sample["noise"].clone(), num_steps=self.steps)
        prefix = self.model.encode_prefix(batch)
        return self.model.decoder.integrate(sample["noise"].clone(), prefix, self.steps, 1, self.graphs)
