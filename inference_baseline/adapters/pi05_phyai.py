"""PhyAI pi05 inline engine using the shared prepared tensor contract."""


class Pi05PhyAIAdapter:
    def __init__(self, config, protocol, samples):
        import torch
        from phyai import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.models.pi05.configuration_pi05 import PI05Config
        from phyai.models.pi05.main_pi05 import PI05Args
        from phyai.models.pi05.scheduler_pi05 import PI05Request
        self.request = PI05Request
        options = config.get("options", {})
        graph = options.get("cuda_graph", True)
        vision_fp32 = options.get("vision_fp32", False)
        geometry = PI05Config(chunk_size=protocol.action_horizon, max_action_dim=protocol.action_dim,
                              num_inference_steps=protocol.denoising_steps, tokenizer_max_length=protocol.max_token_len)
        # Official safetensors save_model may retain either tied alias. PhyAI
        # expects the lm_head spelling for the language input embedding.
        def remap(key):
            if key == "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight":
                return "paligemma_with_expert.paligemma.lm_head.weight"
            return key
        self.engine = Engine(EngineArgs(plugin="pi05", plugin_args=PI05Args(
            checkpoint_dir=config["checkpoint"], config=geometry, max_batch_size=1,
            inputs_image_shape=[[224, 224, 3]] * 3, weight_remap=remap, weight_strict=True,
            vision_params_dtype=torch.float32 if vision_fp32 else None),
            config=EngineConfig(device=DeviceConfig(target=config["device"], params_dtype=torch.bfloat16),
                                runtime=RuntimeConfig(use_cuda_graph=graph,
                                    flashinfer_workspace_bytes=options.get("flashinfer_workspace_bytes", 268435456)))))
        if self.engine.mode != "inline":
            raise ValueError("Model-only protocol requires an inline single-GPU engine")
        self.metadata = {"implementation": "PhyAI pi05 inline Engine.step",
                         "precision": "BF16 with FP32 vision" if vision_fp32 else "BF16 with FP32 stem",
                         "cuda_graph_requested": graph, "weight_strict": True,
                         "flashinfer_workspace_bytes": options.get("flashinfer_workspace_bytes", 268435456),
                         "boundary": "GPU images/tokens/noise through vision, prefix and complete Euler loop"}

    def infer(self, sample):
        t = sample["inputs"]
        return self.engine.step(self.request(pixel_values=t["pixel_values"], input_ids=t["input_ids"],
            lang_lens=t["attention_mask"].sum(dim=1), noise=sample["noise"]))

    def runtime_stats(self):
        scheduler = self.engine.entry.scheduler
        stats = {"vision_graph": scheduler.vision_runner.graph is not None,
                 "prefix_graph_buckets": sorted(scheduler.llm_runner.graphs),
                 "complete_euler_graph": scheduler.expert_runner.graph is not None}
        if self.metadata["cuda_graph_requested"] and not all(stats.values()):
            raise RuntimeError("PhyAI requested graphs are not all captured")
        return stats
