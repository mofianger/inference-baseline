import unittest
import numpy as np
from inference_baseline.metrics import errors, latency_summary
from inference_baseline.protocol import ModelProtocol


class MetricsTests(unittest.TestCase):
    def test_zero_reference_is_finite_and_not_auto_passed(self):
        r = errors(np.ones((1, 40, 3)), np.zeros((1, 40, 3)))
        self.assertEqual(r["gate"], "not_configured")
        self.assertEqual(r["max_abs"], 1)
        self.assertEqual(r["per_action_dim_mae"], [1, 1, 1])
        self.assertTrue(np.isfinite(r["relative_rmse"]))

    def test_gate_and_shape(self):
        a = np.array([[[1., 2.]]])
        self.assertEqual(errors(a, a, {"max_abs": 0})["gate"], "passed")
        self.assertEqual(errors(a + 1, a, {"max_abs": 0.1})["gate"], "failed")
        with self.assertRaises(ValueError): errors(a, a.reshape(-1))
        with self.assertRaises(ValueError): errors(a * np.nan, a)
        with self.assertRaises(ValueError): errors(a, a, {"typo": 1})

    def test_percentiles_and_invalid_samples(self):
        self.assertEqual(latency_summary([1, 2, 3])["p50_ms"], 2)
        for x in ([], [np.nan], [-1]):
            with self.assertRaises(ValueError): latency_summary(x)

    def test_checkpoint_drives_workload(self):
        p = ModelProtocol.from_checkpoint({"model_type": "Gr00tN1d7", "num_inference_timesteps": 7,
            "action_horizon": 20, "max_action_dim": 9, "max_state_dim": 12, "image_target_size": [256, 256]}, 24)
        self.assertEqual((p.denoising_steps, p.action_horizon, p.views), (7, 20, 3))


if __name__ == "__main__":
    unittest.main()
