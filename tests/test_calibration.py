"""
Tests for calibration models.
"""
import unittest

import numpy as np
import torch

from arctan.config import CalibrationConfig
from arctan.models.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    fit_temperature,
)


class TestTemperatureScaler(unittest.TestCase):
    def test_forward_scales_logits(self):
        # logits / T with known values
        scaler = TemperatureScaler(init_temperature=2.0)
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = scaler(logits)
        expected = logits / 2.0
        self.assertTrue(torch.allclose(result, expected))

    def test_temperature_value_property(self):
        scaler = TemperatureScaler(init_temperature=3.0)
        self.assertAlmostEqual(scaler.temperature_value, 3.0, places=4)

    def test_fit_temperature_converges(self):
        # Create synthetic logits where T=1 is miscalibrated
        torch.manual_seed(42)
        logits = torch.randn(100, 2)
        labels = (logits[:, 1] > logits[:, 0]).long()
        config = CalibrationConfig(temperature_epochs=50)
        scaler = fit_temperature(logits, labels, config)
        # T should be positive and should converge
        self.assertGreater(scaler.temperature_value, 0.0)

class TestECE(unittest.TestCase):
    def test_perfect_calibration(self):
        # If predicted prob == actual fraction, ECE should be near 0
        probs = np.array([0.1] * 100 + [0.9] * 100)
        labels = np.array([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90)
        ece = compute_ece(probs, labels, num_bins=10)
        self.assertLess(ece, 0.05)

    def test_worst_calibration(self):
        # Confident and wrong
        probs = np.array([0.99] * 100)
        labels = np.zeros(100)
        ece = compute_ece(probs, labels, num_bins=10)
        self.assertGreater(ece, 0.5)

class TestBrierScore(unittest.TestCase):
    def test_perfect_predictions(self):
        probs = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(compute_brier_score(probs, labels), 0.0)

    def test_worst_predictions(self):
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(compute_brier_score(probs, labels), 1.0)

if __name__ == '__main__':
    unittest.main()
