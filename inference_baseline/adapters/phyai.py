class PhyAIAdapter:
    def __init__(self, config, protocol, samples):
        import torch
        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
        from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
        from phyai.models.gr00t_n17.scheduler_gr00t_n17 import GR00TN17Request
        from phyai.utils import load_config
        self.request_type = GR00TN17Request
        c = load_config(config["checkpoint"], GR00TN17Config)
        graph = config.get("options", {}).get("cuda_graph", True)
        self.engine = Engine(EngineArgs(plugin="gr00t_n17", plugin_args=GR00TN17Args(
            checkpoint_dir=config["checkpoint"], config=c, max_batch_size=protocol.batch,
            capture_profiles=tuple(self.request(s) for s in samples)), config=EngineConfig(
                device=DeviceConfig(target=config["device"], params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=graph))))
        self.metadata = {"precision": "BF16", "cuda_graph": graph, "implementation": "GR00TN17 Engine"}

    def request(self, sample):
        return self.request_type(tensors=sample["inputs"], noise=sample["noise"])

    def infer(self, sample):
        return self.engine.step(self.request(sample))
