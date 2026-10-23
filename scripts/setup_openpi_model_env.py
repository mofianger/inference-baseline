"""Configure only the isolated OpenPI environment used by the benchmark."""
import importlib.util
from pathlib import Path
import shutil
import sys


root = Path(__file__).resolve().parents[1]
source = root / "third_party/openpi"
venv = source / ".venv"
if Path(sys.prefix).resolve() != venv.resolve():
    raise RuntimeError("Run with third_party/openpi/.venv/bin/python")
target = Path(importlib.util.find_spec("transformers").origin).parent
if not target.is_relative_to(venv.resolve()):
    raise RuntimeError("Refusing to patch a different environment")
shutil.copytree(source / "src/openpi/models_pytorch/transformers_replace", target, dirs_exist_ok=True)
print(f"Applied upstream transformers replacements to {target}")
