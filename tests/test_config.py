"""Unit tests for Arctan configuration."""

import unittest

from arctan.config import (
    InferenceConfig,
    ModelConfig,
    PathConfig,
    PipelineConfig,
    ServerSettings,
    TrainingConfig,
    get_default_config,
)


class TestPipelineConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = get_default_config()
        self.assertIsInstance(cfg, PipelineConfig)
        self.assertIsInstance(cfg.paths, PathConfig)
        self.assertIsInstance(cfg.model, ModelConfig)
        self.assertIsInstance(cfg.training, TrainingConfig)
        self.assertIsInstance(cfg.inference, InferenceConfig)

    def test_paths(self) -> None:
        cfg = get_default_config()
        self.assertTrue(str(cfg.paths.data_root).endswith("data"))
        self.assertTrue(str(cfg.paths.graph_path).endswith("fraud_graph.pt"))
        self.assertTrue(str(cfg.paths.best_model_path).endswith("fraud_gnn_best.pt"))

    def test_model_defaults(self) -> None:
        cfg = get_default_config()
        self.assertEqual(cfg.model.out_dim, 2)
        self.assertEqual(cfg.model.gat_heads, 4)
        self.assertEqual(cfg.model.aggr, "mean")
        self.assertEqual(cfg.model.edge_dim, 2)

    def test_server_settings(self) -> None:
        settings = ServerSettings()
        self.assertEqual(settings.port, 8001)
        self.assertEqual(settings.host, "0.0.0.0")


if __name__ == "__main__":
    unittest.main()
