import importlib

ADAPTERS = {"reference": "reference:ReferenceAdapter", "embodiinfer": "embodiinfer:EmbodiInferAdapter",
            "phyai": "phyai:PhyAIAdapter", "flashrt": "flashrt:FlashRTAdapter"}


def create(name, config, protocol, samples):
    if protocol.model == "pi0.5" and name in ADAPTERS:
        if name == "phyai":
            from .pi05_phyai import Pi05PhyAIAdapter
            return Pi05PhyAIAdapter(config, protocol, samples)
        if name != "reference":
            raise NotImplementedError(f"pi05 {name} adapter has not been validated yet")
        from .pi05_reference import Pi05ReferenceAdapter
        return Pi05ReferenceAdapter(config, protocol, samples)
    path = ADAPTERS.get(name, name)
    module, cls = path.split(":")
    if name in ADAPTERS:
        module = "inference_baseline.adapters." + module
    return getattr(importlib.import_module(module), cls)(config, protocol, samples)
