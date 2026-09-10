"""Unit tests for model and FocalLoss."""

import unittest

import torch

from arctan.config import ModelConfig
from arctan.models.fraud_gnn import FocalLoss, FraudGNN


class TestFraudGNN(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(in_features=8, hidden_dim=32, out_dim=2)
        self.model = FraudGNN(self.config)

    def test_forward_shape(self) -> None:
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = self.model(x, edge_index)
        self.assertIsInstance(out, dict)
        self.assertIn("fraud", out)
        self.assertIn("ring", out)
        self.assertEqual(out["fraud"].shape, (10, 2))
        self.assertEqual(out["ring"].shape, (10, 2))

    def test_predict_proba_sums_to_one(self) -> None:
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        probs = self.model.predict_proba(x, edge_index)
        sums = probs.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_forward_with_edge_features(self) -> None:
        """GATv2Conv should use edge_attr when edge_dim is set."""
        config = ModelConfig(in_features=8, hidden_dim=32, out_dim=2, edge_dim=2)
        model = FraudGNN(config)
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 2)  # 4 edges, 2 features (log_amount, norm_step)
        out = model(x, edge_index, edge_attr=edge_attr)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["fraud"].shape, (10, 2))
        self.assertEqual(out["ring"].shape, (10, 2))

    def test_edge_dim_activates_gat(self) -> None:
        """With edge_dim=2, the GATv2Conv layer should have edge_dim set."""
        config = ModelConfig(in_features=8, hidden_dim=32, out_dim=2, edge_dim=2)
        model = FraudGNN(config)
        self.assertEqual(model.edge_dim, 2)

    def test_ring_head_independent(self) -> None:
        """Ring head should produce distinct representations from fraud head."""
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = self.model(x, edge_index)
        self.assertFalse(torch.allclose(out["fraud"], out["ring"]))


class TestFocalLoss(unittest.TestCase):
    def test_focal_loss_runs(self) -> None:
        logits = torch.randn(20, 2)
        targets = torch.randint(0, 2, (20,))
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        self.assertTrue(loss.item() >= 0)

    def test_gamma_zero_equals_ce(self) -> None:
        """With gamma=0 and no alpha, FocalLoss should equal cross-entropy."""
        torch.manual_seed(0)
        logits = torch.randn(50, 2)
        targets = torch.randint(0, 2, (50,))

        focal = FocalLoss(gamma=0.0)(logits, targets)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        self.assertAlmostEqual(focal.item(), ce.item(), places=4)


if __name__ == "__main__":
    unittest.main()
