import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

from inference_baseline.io import sha256, write_json
from inference_baseline.protocol import ModelProtocol


@unittest.skipIf(torch is None, "Torch fixture checks run in engine environments")
class FixtureContractTests(unittest.TestCase):
    def setUp(self):
        self.p = ModelProtocol("test", "GR00T N1.7", 1, 3, 4, 40, 132, 132, 1, [256, 256], 24)
        self.sample = {"id": "s0", "role": "evaluation", "inputs": {
            "state": torch.zeros(1, 1, 132, dtype=torch.bfloat16),
            "embodiment_id": torch.tensor([24]), "input_ids": torch.zeros(1, 200, dtype=torch.int64),
            "attention_mask": torch.ones(1, 200, dtype=torch.int64),
            "image_grid_thw": torch.tensor([[1, 16, 16]] * 3),
            "pixel_values": torch.zeros(768, 1536, dtype=torch.bfloat16)},
            "noise": torch.zeros(1, 40, 132, dtype=torch.bfloat16)}

    def test_rejects_two_views_and_wrong_dtype(self):
        self.p.validate(self.sample)
        self.sample["inputs"]["state"] = self.sample["inputs"]["state"].float()
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.p.validate(self.sample)
        self.sample["inputs"]["state"] = self.sample["inputs"]["state"].bfloat16()
        self.sample["inputs"]["image_grid_thw"] = torch.tensor([[1, 16, 16]] * 2)
        with self.assertRaisesRegex(ValueError, "three"):
            self.p.validate(self.sample)

    def test_fixture_hash_detects_changes(self):
        from inference_baseline.dataset import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(self.sample, root / "s0.pt")
            write_json(root / "manifest.json", {"protocol": self.p.to_dict(), "samples": [
                {"id": "s0", "role": "evaluation", "file": "s0.pt", "sha256": sha256(root / "s0.pt")}]})
            self.assertEqual(len(load_dataset(root)[2]), 1)
            with (root / "s0.pt").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_dataset(root)

    def test_noise_injection_is_instance_local(self):
        from inference_baseline.adapters.reference import inject_noise
        class Head:
            @torch.no_grad()
            def get_action_with_features(self):
                return torch.randn((1, 2))
        first, second = Head(), Head()
        inject_noise(first)
        first._benchmark_initial_noise = torch.full((1, 2), 7.)
        self.assertTrue(torch.equal(first.get_action_with_features(), first._benchmark_initial_noise))
        self.assertIsNot(first.get_action_with_features.__func__, second.get_action_with_features.__func__)
        self.assertFalse(torch.equal(second.get_action_with_features(), first._benchmark_initial_noise))
