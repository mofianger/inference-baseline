class PhyAIAdapter:
    def __init__(self, config, protocol, samples):
        import os
        import hashlib
        from pathlib import Path
        import torch
        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
        from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
        from phyai.models.gr00t_n17.scheduler_gr00t_n17 import GR00TN17Request
        from phyai.utils import load_config
        self.request_type = GR00TN17Request
        c = load_config(config["checkpoint"], GR00TN17Config)
        # PhyAI resolves HF symlinks before testing filename extensions. HF blob
        # names have none; hardlinks preserve both bytes and .safetensors names.
        source = Path(config["checkpoint"])
        staging = Path("artifacts/checkpoint-links") / hashlib.sha256(str(source).encode()).hexdigest()[:16]
        staging.mkdir(parents=True, exist_ok=True)
        for item in list(source.glob("*.json")) + list(source.glob("*.safetensors")):
            target = staging / item.name
            if not target.exists():
                os.link(item.resolve(), target)
            if not os.path.samefile(item, target):
                raise ValueError("Staged checkpoint differs from original HF snapshot")
        graph = config.get("options", {}).get("cuda_graph", True)
        self.engine = Engine(EngineArgs(plugin="gr00t_n17", plugin_args=GR00TN17Args(
            checkpoint_dir=staging, config=c, max_batch_size=protocol.batch,
            capture_profiles=tuple(self.request(s) for s in samples)), config=EngineConfig(
                device=DeviceConfig(target=config["device"], params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=graph))))
        self.metadata = {"precision": "BF16", "cuda_graph": graph, "implementation": "GR00TN17 Engine"}

    def request(self, sample):
        return self.request_type(tensors=sample["inputs"], noise=sample["noise"])

    def infer(self, sample):
        return self.engine.step(self.request(sample))
