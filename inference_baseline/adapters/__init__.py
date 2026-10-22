import importlib

ADAPTERS = {"reference": "reference:ReferenceAdapter", "embodiinfer": "embodiinfer:EmbodiInferAdapter",
            "phyai": "phyai:PhyAIAdapter", "flashrt": "flashrt:FlashRTAdapter"}


def create(name, config, protocol, samples):
    path = ADAPTERS.get(name, name)
    module, cls = path.split(":")
    if name in ADAPTERS:
        module = "inference_baseline.adapters." + module
    return getattr(importlib.import_module(module), cls)(config, protocol, samples)
