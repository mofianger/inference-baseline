import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None
from inference_baseline.pi05_protocol import Pi05Protocol
from inference_baseline.io import sha256, write_json


@unittest.skipIf(torch is None, "Requires an engine Torch environment")
class Pi05ContractTests(unittest.TestCase):
    def sample(self):
        return {"id": "p0", "role": "evaluation", "noise": torch.zeros(1, 50, 32), "inputs": {
            "pixel_values": torch.zeros(1, 3, 3, 224, 224), "image_mask": torch.ones(1, 3, dtype=torch.bool),
            "state": torch.zeros(1, 32), "input_ids": torch.ones(1, 200, dtype=torch.int64),
            "attention_mask": torch.ones(1, 200, dtype=torch.bool)}}

    def test_roundtrip_and_grouped_state(self):
        from inference_baseline.dataset import load_dataset
        p = Pi05Protocol()
        sample = self.sample()
        p.validate(sample)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(sample, root / "p0.pt")
            write_json(root / "manifest.json", {"protocol": p.to_dict(), "samples": [
                {"id": "p0", "role": "evaluation", "file": "p0.pt", "sha256": sha256(root / "p0.pt")}]})
            restored = load_dataset(root)[1]
            self.assertEqual(restored.sensitivity_groups()["state_tokens"], ("state", "input_ids", "attention_mask"))

    def test_rejects_masked_view_and_bad_noise(self):
        sample = self.sample()
        sample["inputs"]["image_mask"][0, 2] = False
        with self.assertRaisesRegex(ValueError, "active"):
            Pi05Protocol().validate(sample)
        sample = self.sample()
        sample["noise"] = sample["noise"].bfloat16()
        with self.assertRaisesRegex(ValueError, "noise"):
            Pi05Protocol().validate(sample)

    def test_rejects_raw_pixels_and_noncontiguous_mask(self):
        sample = self.sample()
        sample["inputs"]["pixel_values"].fill_(255)
        with self.assertRaisesRegex(ValueError, "normalized"):
            Pi05Protocol().validate(sample)
        sample = self.sample()
        sample["inputs"]["attention_mask"][0, 5] = False
        with self.assertRaisesRegex(ValueError, "right-padded"):
            Pi05Protocol().validate(sample)
