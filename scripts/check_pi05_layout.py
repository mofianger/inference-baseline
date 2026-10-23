"""Compare model parameter layouts without loading or inventing checkpoint weights."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=("openpi", "embodiinfer"), required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
import torch
if args.kind == "openpi":
    sys.path.insert(0, str(root / "third_party/openpi/src"))
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    with torch.device("meta"):
        model = PI0Pytorch(Pi0Config(pi05=True, dtype="float32", pytorch_compile_mode=None))
else:
    sys.path.insert(0, str(root / "third_party/EmbodiInfer"))
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
    with torch.device("meta"):
        model = PI05Pytorch(PI05Config(dtype="float32", chunk_size=50, n_action_steps=50,
                                     max_state_dim=32, max_action_dim=32))
model.paligemma_with_expert.paligemma.lm_head = None
model.paligemma_with_expert.gemma_expert.lm_head = None
layout = {k: list(v.shape) for k, v in model.state_dict().items()}
Path(args.output).write_text(json.dumps(layout, indent=2) + "\n")
print(json.dumps({"kind": args.kind, "tensors": len(layout), "torch": torch.__version__}))
