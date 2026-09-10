"""Unit tests for the temporal GNN architecture.

Tests cover the time encoder, entity memory module, temporal GNN model,
and their integration. Each test is self-contained with mock data.
"""

import unittest

import torch

from arctan.config import TemporalModelConfig


class TestTimeEncoder(unittest.TestCase):
    """Tests for the learnable time encoder."""

    def test_output_shape(self):
        """Time encoder should produce [N, out_channels] embeddings."""
        from arctan.models.time_encoder import TimeEncoder

        enc = TimeEncoder(out_channels=64)
        t = torch.tensor([1.0, 2.0, 5.0, 10.0])
        out = enc(t)
        self.assertEqual(out.shape, (4, 64))

    def test_output_range(self):
        """Cosine activation should produce values in [-1, 1]."""
        from arctan.models.time_encoder import TimeEncoder

        enc = TimeEncoder(out_channels=32)
        t = torch.randn(100)
        out = enc(t)
        self.assertTrue(out.min().item() >= -1.0)
        self.assertTrue(out.max().item() <= 1.0)

    def test_different_times_different_embeddings(self):
        """Different Δt values should produce different embeddings."""
        from arctan.models.time_encoder import TimeEncoder

        enc = TimeEncoder(out_channels=16)
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([100.0])
        out1 = enc(t1)
        out2 = enc(t2)
        self.assertFalse(torch.allclose(out1, out2))

    def test_reset_parameters(self):
        """reset_parameters should run without error."""
        from arctan.models.time_encoder import TimeEncoder

        enc = TimeEncoder(out_channels=16)
        enc.reset_parameters()


class TestEntityMemory(unittest.TestCase):
    """Tests for the entity memory module (TGNMemory wrapper)."""

    def test_build_entity_memory(self):
        """build_entity_memory should return a TGNMemory with correct dims."""
        from arctan.models.entity_memory import build_entity_memory

        config = TemporalModelConfig(memory_dim=32, time_dim=16, raw_msg_dim=2)
        mem = build_entity_memory(num_nodes=100, config=config)

        self.assertEqual(mem.memory.shape, (100, 32))
        self.assertEqual(mem.num_nodes, 100)
        self.assertEqual(mem.memory_dim, 32)

    def test_memory_reset(self):
        """After reset, all memory vectors should be zero."""
        from arctan.models.entity_memory import build_entity_memory

        config = TemporalModelConfig(memory_dim=16, time_dim=8, raw_msg_dim=2)
        mem = build_entity_memory(num_nodes=50, config=config)
        mem.reset_state()

        self.assertTrue(torch.all(mem.memory == 0))
        self.assertTrue(torch.all(mem.last_update == 0))

    def test_memory_update_changes_state(self):
        """Updating memory with events should change the memory vectors."""
        from arctan.models.entity_memory import build_entity_memory

        config = TemporalModelConfig(memory_dim=16, time_dim=8, raw_msg_dim=2)
        mem = build_entity_memory(num_nodes=50, config=config)
        mem.reset_state()
        mem.eval()

        initial_mem = mem.memory.clone()

        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 5])
        t = torch.tensor([1, 2, 3])
        msg = torch.randn(3, 2)

        mem.update_state(src, dst, t, msg)

        # Memory should have changed for involved nodes
        involved = torch.tensor([0, 1, 2, 3, 4, 5])
        self.assertFalse(
            torch.allclose(mem.memory[involved], initial_mem[involved])
        )

    def test_save_load_memory_state(self):
        """Memory state should survive save/load round-trip."""
        import tempfile

        from arctan.models.entity_memory import (
            build_entity_memory,
            load_memory_state,
            save_memory_state,
        )

        config = TemporalModelConfig(memory_dim=16, time_dim=8, raw_msg_dim=2)
        mem = build_entity_memory(num_nodes=20, config=config)
        mem.eval()

        # Update some state
        src = torch.tensor([0, 1])
        dst = torch.tensor([2, 3])
        t = torch.tensor([10, 20])
        msg = torch.randn(2, 2)
        mem.update_state(src, dst, t, msg)

        original_memory = mem.memory.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            save_memory_state(mem, f.name)

            # Reset and reload
            mem.reset_state()
            self.assertTrue(torch.all(mem.memory == 0))

            load_memory_state(mem, f.name)
            self.assertTrue(torch.allclose(mem.memory, original_memory))


class TestTemporalGNN(unittest.TestCase):
    """Tests for the TemporalFraudGNN model."""

    def setUp(self):
        """Create a small model for testing."""
        self.config = TemporalModelConfig(
            memory_dim=32,
            time_dim=16,
            embedding_dim=64,
            num_attention_heads=2,
            raw_msg_dim=2,
            dropout=0.0,
            out_dim=2,
        )

    def test_forward_shape(self):
        """Forward pass should produce [N, out_dim] logits for both fraud and ring heads."""
        from arctan.models.temporal_gnn import TemporalFraudGNN

        model = TemporalFraudGNN(self.config)
        model.eval()

        memory = torch.randn(10, 32)  # 10 nodes, memory_dim=32
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        edge_attr = torch.randn(3, 16 + 2)  # time_dim + raw_msg_dim

        out = model(memory, edge_index, edge_attr)
        self.assertIsInstance(out, dict)
        self.assertIn("fraud", out)
        self.assertIn("ring", out)
        self.assertEqual(out["fraud"].shape, (10, 2))
        self.assertEqual(out["ring"].shape, (10, 2))

    def test_predict_proba_sums_to_one(self):
        """Softmax probabilities should sum to 1."""
        from arctan.models.temporal_gnn import TemporalFraudGNN

        model = TemporalFraudGNN(self.config)
        model.eval()

        memory = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        edge_attr = torch.randn(2, 18)

        probs = model.predict_proba(memory, edge_index, edge_attr)
        sums = probs.sum(dim=-1)
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        )

    def test_forward_no_edges(self):
        """Model should handle empty edge index gracefully."""
        from arctan.models.temporal_gnn import TemporalFraudGNN

        model = TemporalFraudGNN(self.config)
        model.eval()

        memory = torch.randn(5, 32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 18))

        out = model(memory, edge_index, edge_attr)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["fraud"].shape, (5, 2))
        self.assertEqual(out["ring"].shape, (5, 2))

    def test_gradient_flow(self):
        """Gradients should flow through both task heads."""
        from arctan.models.temporal_gnn import TemporalFraudGNN

        model = TemporalFraudGNN(self.config)
        model.train()

        memory = torch.randn(8, 32, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        edge_attr = torch.randn(3, 18)

        out = model(memory, edge_index, edge_attr)
        loss = out["fraud"].sum() + out["ring"].sum()
        loss.backward()

        self.assertIsNotNone(memory.grad)
        self.assertTrue(memory.grad.abs().sum() > 0)


class TestTemporalIntegration(unittest.TestCase):
    """Integration tests combining memory + model."""

    def test_memory_to_model_pipeline(self):
        """Entity memory output should feed correctly into the temporal GNN."""
        from arctan.models.entity_memory import build_entity_memory
        from arctan.models.temporal_gnn import TemporalFraudGNN

        config = TemporalModelConfig(
            memory_dim=32, time_dim=16, embedding_dim=64,
            num_attention_heads=2, raw_msg_dim=2, dropout=0.0, out_dim=2,
        )

        # Build memory and update with some events
        mem = build_entity_memory(num_nodes=20, config=config)
        mem.eval()

        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([4, 5, 6, 7])
        t = torch.tensor([10, 20, 30, 40])
        msg = torch.randn(4, 2)
        mem.update_state(src, dst, t, msg)

        # Get memory for a subset of nodes
        query_nodes = torch.tensor([0, 1, 4, 5])
        z, last_update = mem(query_nodes)

        self.assertEqual(z.shape, (4, 32))

        # Feed into model
        model = TemporalFraudGNN(config)
        model.eval()

        edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        edge_attr = torch.randn(2, 18)

        out = model(z, edge_index, edge_attr)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["fraud"].shape, (4, 2))
        self.assertEqual(out["ring"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
