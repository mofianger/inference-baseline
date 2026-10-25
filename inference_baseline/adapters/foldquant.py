"""Prepared-tensor boundary for the upstream FoldQuant GR00T TRT runtime."""
import json
from pathlib import Path
from types import SimpleNamespace
from .reference import ReferenceAdapter
from ..io import sha256


class FoldQuantAdapter(ReferenceAdapter):
    def __init__(self, config, protocol, samples):
        super().__init__(config, protocol, samples)
        from foldquant_integration._upstream import PIPELINE_COMPONENTS, ensure_deployment_on_path
        from foldquant.runtime.plugins import load_plugins
        directory = Path(config["options"]["engine_dir"])
        for _, _, filename in PIPELINE_COMPONENTS:
            if not (directory / filename).is_file():
                raise FileNotFoundError(filename)
        manifest = json.loads((directory / "foldquant_export.json").read_text())
        plugins = load_plugins(manifest["plugin_libs"])
        ensure_deployment_on_path()
        from trt_model_forward import setup_tensorrt_engines
        self.policy = SimpleNamespace(model=self.model)
        setup_tensorrt_engines(self.policy, str(directory), mode="n17_full_pipeline")
        self.metadata = {"implementation": "FoldQuant upstream seven-engine GR00T pipeline",
                         "precision": manifest["schemes"], "manifest": manifest,
                         "plugins": {str(p): sha256(p) for p in plugins},
                         "noise_injection": "upstream init_actions, copy inside timing"}

    def infer(self, sample):
        head = self.model.action_head
        head.init_actions = sample["noise"].clone()
        return self.model.get_action(dict(sample["inputs"]))["action_pred"]
