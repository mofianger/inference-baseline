"""Fail early on missing OpenPI inference dependencies or unsupported GPU wheel."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "third_party/openpi/src"), str(root / "third_party/openpi/packages/openpi-client/src")]
import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

probe = torch.ones(16, 16, device="cuda")
probe = probe @ probe
torch.cuda.synchronize()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "config": str(Pi0Config(pi05=True)),
                  "cuda_probe": probe[0, 0].item()}))
