"""
Tests for epistemic uncertainty estimation.
"""
import unittest

import torch
import torch.nn as nn

from arctan.models.uncertainty import (
    enable_mc_dropout,
    mc_dropout_predict,
    predictive_entropy,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(4, 2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.drop(x)
        return self.fc(x)

class TestEnableMCDropout(unittest.TestCase):
    def test_dropout_active_batchnorm_frozen(self):
        model = SimpleModel()
        model.eval()
        with enable_mc_dropout(model) as m:
            # Dropout should be in training mode
            self.assertTrue(m.drop.training)
            # BatchNorm should be in eval mode
            self.assertFalse(m.bn.training)
        # After context, model should be back to eval
        self.assertFalse(model.training)

    def test_restores_original_state(self):
        model = SimpleModel()
        model.train()
        with enable_mc_dropout(model):
            pass
        # Should still be in train mode
        self.assertTrue(model.training)

class TestMCDropoutPredict(unittest.TestCase):
    def test_returns_expected_keys(self):
        model = SimpleModel()
        x = torch.randn(10, 4)
        result = mc_dropout_predict(model, lambda: model(x), n_samples=5)
        self.assertIn('mean', result)
        self.assertIn('variance', result)
        self.assertIn('entropy', result)
        self.assertIn('samples', result)

    def test_shapes(self):
        model = SimpleModel()
        x = torch.randn(10, 4)
        result = mc_dropout_predict(model, lambda: model(x), n_samples=5)
        self.assertEqual(result['mean'].shape, (10, 2))
        self.assertEqual(result['variance'].shape, (10,))
        self.assertEqual(result['entropy'].shape, (10,))
        self.assertEqual(result['samples'].shape, (5, 10, 2))

    def test_variance_positive_with_dropout(self):
        model = SimpleModel()
        x = torch.randn(20, 4)
        result = mc_dropout_predict(model, lambda: model(x), n_samples=50)
        # With 50% dropout, variance should be non-zero
        self.assertGreater(result['variance'].mean().item(), 0.0)

class TestPredictiveEntropy(unittest.TestCase):
    def test_uniform_high_entropy(self):
        probs = torch.tensor([[0.5, 0.5]])
        entropy = predictive_entropy(probs)
        self.assertGreater(entropy.item(), 0.6)  # log(2) ≈ 0.693

    def test_certain_low_entropy(self):
        probs = torch.tensor([[0.99, 0.01]])
        entropy = predictive_entropy(probs)
        self.assertLess(entropy.item(), 0.1)

if __name__ == '__main__':
    unittest.main()
