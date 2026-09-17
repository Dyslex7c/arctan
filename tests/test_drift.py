"""
Tests for distribution drift detection.
"""
import unittest

import numpy as np

from arctan.config import DriftConfig
from arctan.drift import (
    compute_ks_test,
    compute_psi,
    detect_feature_drift,
    detect_prediction_drift,
)


class TestPSI(unittest.TestCase):
    def test_identical_distributions(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        self.assertAlmostEqual(psi, 0.0, places=5)

    def test_shifted_distribution(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # big shift
        psi = compute_psi(ref, cur)
        self.assertGreater(psi, 0.2)

    def test_non_negative(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(0.5, 1.2, 500)
        psi = compute_psi(ref, cur)
        self.assertGreaterEqual(psi, 0.0)

class TestKSTest(unittest.TestCase):
    def test_same_distribution(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        result = compute_ks_test(data, data)
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertAlmostEqual(result['statistic'], 0.0)

    def test_different_distributions(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(5, 1, 500)
        result = compute_ks_test(ref, cur)
        self.assertGreater(result['statistic'], 0.5)
        self.assertLess(result['p_value'], 0.001)

class TestDetectFeatureDrift(unittest.TestCase):
    def test_no_drift(self):
        np.random.seed(42)
        features = np.random.randn(500, 3)
        config = DriftConfig()
        result = detect_feature_drift(
            features, features, ['f1', 'f2', 'f3'], config
        )
        self.assertEqual(result['num_drifted'], 0)
        self.assertEqual(result['total_features'], 3)

    def test_all_drift(self):
        np.random.seed(42)
        ref = np.random.randn(500, 2)
        cur = np.random.randn(500, 2) + 10
        config = DriftConfig()
        result = detect_feature_drift(ref, cur, ['f1', 'f2'], config)
        self.assertEqual(result['num_drifted'], 2)

class TestDetectPredictionDrift(unittest.TestCase):
    def test_no_drift(self):
        np.random.seed(42)
        probs = np.random.uniform(0, 1, 500)
        config = DriftConfig()
        result = detect_prediction_drift(probs, probs, config)
        self.assertFalse(result['drifted'])

if __name__ == '__main__':
    unittest.main()
