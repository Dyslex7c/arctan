"""
Tests for counterfactual explanations.
"""
import unittest

import torch
from torch_geometric.data import Data

from arctan.config import ModelConfig
from arctan.models.counterfactual import CounterfactualExplainer
from arctan.models.fraud_gnn import FraudGNN


class TestCounterfactualExplainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(in_features=4, hidden_dim=16, edge_dim=None, ring_hidden_dim=8)
        self.model = FraudGNN(self.config)
        # Create a small graph: 5 nodes, 6 edges
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0,1,2,3,0,2], [1,2,3,4,3,0]], dtype=torch.long)
        y = torch.tensor([0, 0, 1, 1, 0])
        self.graph = Data(x=x, edge_index=edge_index, y=y)
        self.explainer = CounterfactualExplainer(
            self.model, feature_names=['f1', 'f2', 'f3', 'f4']
        )

    def test_returns_expected_keys(self):
        result = self.explainer.find_counterfactual(
            node_idx=2, graph=self.graph, max_iters=10
        )
        self.assertIn('success', result)
        self.assertIn('original_prob', result)
        self.assertIn('counterfactual_prob', result)
        self.assertIn('perturbation', result)
        self.assertIn('num_features_changed', result)
        self.assertIn('total_perturbation_norm', result)

    def test_perturbation_is_dict(self):
        result = self.explainer.find_counterfactual(
            node_idx=2, graph=self.graph, max_iters=10
        )
        self.assertIsInstance(result['perturbation'], dict)

    def test_original_prob_is_float(self):
        result = self.explainer.find_counterfactual(
            node_idx=2, graph=self.graph, max_iters=10
        )
        self.assertIsInstance(result['original_prob'], float)
        self.assertGreaterEqual(result['original_prob'], 0.0)
        self.assertLessEqual(result['original_prob'], 1.0)

    def test_feature_names_in_perturbation(self):
        result = self.explainer.find_counterfactual(
            node_idx=2, graph=self.graph, max_iters=50
        )
        for key in result['perturbation']:
            self.assertIn(key, ['f1', 'f2', 'f3', 'f4'])

    def test_norm_non_negative(self):
        result = self.explainer.find_counterfactual(
            node_idx=2, graph=self.graph, max_iters=10
        )
        self.assertGreaterEqual(result['total_perturbation_norm'], 0.0)

if __name__ == '__main__':
    unittest.main()
