"""Unit tests for temporal motif feature engineering."""

import unittest

import polars as pl

from arctan.config import MotifConfig
from arctan.data.motif_features import (
    MOTIF_FEATURE_COLS,
    _compute_fan_in_bursts,
    _compute_fan_out_bursts,
    _compute_ping_pong_ratio,
    _compute_t2_cycles,
    _compute_t3_cycles,
    _compute_temporal_clustering,
    compute_motif_features,
)


class TestTemporalMotifs(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MotifConfig(
            cycle_window=48,
            burst_window=10,
            burst_min_targets=3,
            enabled=True,
        )

    def test_t2_cycle_detection(self) -> None:
        """A->B at t=10 and B->A at t=20 should form a 2-cycle for entity A."""
        edges = pl.DataFrame({
            "src_id": [0, 1, 2],
            "dst_id": [1, 0, 3],
            "step": [10.0, 20.0, 15.0],
            "amount": [100.0, 100.0, 50.0],
        })
        cycles = _compute_t2_cycles(edges, window=48)
        self.assertIn("node_id", cycles.columns)
        self.assertIn("t2_cycle_count", cycles.columns)

        node_0_count = cycles.filter(pl.col("node_id") == 0)["t2_cycle_count"].to_list()
        self.assertEqual(len(node_0_count), 1)
        self.assertEqual(node_0_count[0], 1)

    def test_t2_cycle_outside_window(self) -> None:
        """A->B at t=10 and B->A at t=100 with window=48 should NOT form a 2-cycle."""
        edges = pl.DataFrame({
            "src_id": [0, 1],
            "dst_id": [1, 0],
            "step": [10.0, 100.0],
            "amount": [100.0, 100.0],
        })
        cycles = _compute_t2_cycles(edges, window=48)
        node_0_count = cycles.filter(pl.col("node_id") == 0)["t2_cycle_count"].to_list()
        self.assertEqual(len(node_0_count), 0)

    def test_t3_cycle_detection(self) -> None:
        """A->B at t=10, B->C at t=20, C->A at t=30 should form a 3-cycle for entity A."""
        edges = pl.DataFrame({
            "src_id": [0, 1, 2],
            "dst_id": [1, 2, 0],
            "step": [10.0, 20.0, 30.0],
            "amount": [100.0, 100.0, 100.0],
        })
        cycles = _compute_t3_cycles(edges, window=48)
        node_0_count = cycles.filter(pl.col("node_id") == 0)["t3_cycle_count"].to_list()
        self.assertEqual(len(node_0_count), 1)
        self.assertEqual(node_0_count[0], 1)

    def test_fan_out_burst(self) -> None:
        """Entity sending to >=K targets in the same time bin triggers fan-out burst."""
        edges = pl.DataFrame({
            "src_id": [0, 0, 0, 1],
            "dst_id": [1, 2, 3, 2],
            "step": [5.0, 6.0, 7.0, 5.0],
            "amount": [10.0, 20.0, 30.0, 10.0],
        })
        bursts = _compute_fan_out_bursts(edges, burst_window=10, min_targets=3)
        node_0 = bursts.filter(pl.col("node_id") == 0)["fan_out_burst_count"].to_list()
        self.assertEqual(len(node_0), 1)
        self.assertEqual(node_0[0], 1)

    def test_fan_in_burst(self) -> None:
        """Entity receiving from >=K sources in the same time bin triggers fan-in burst."""
        edges = pl.DataFrame({
            "src_id": [1, 2, 3],
            "dst_id": [0, 0, 0],
            "step": [5.0, 6.0, 7.0],
            "amount": [10.0, 20.0, 30.0],
        })
        bursts = _compute_fan_in_bursts(edges, burst_window=10, min_sources=3)
        node_0 = bursts.filter(pl.col("node_id") == 0)["fan_in_burst_count"].to_list()
        self.assertEqual(len(node_0), 1)
        self.assertEqual(node_0[0], 1)

    def test_ping_pong_ratio(self) -> None:
        """Entity 0 sends to 1 and 2, but only 1 sends back: ping-pong ratio should be 0.5."""
        edges = pl.DataFrame({
            "src_id": [0, 0, 1],
            "dst_id": [1, 2, 0],
            "step": [10.0, 12.0, 20.0],
            "amount": [100.0, 50.0, 100.0],
        })
        pp = _compute_ping_pong_ratio(edges, window=48)
        ratio = pp.filter(pl.col("node_id") == 0)["ping_pong_ratio"].to_list()
        self.assertEqual(len(ratio), 1)
        self.assertAlmostEqual(ratio[0], 0.5, places=4)

    def test_temporal_clustering(self) -> None:
        """Triangle among nodes 0, 1, 2: clustering coefficient for node 0 should be 1.0."""
        edges = pl.DataFrame({
            "src_id": [0, 0, 1],
            "dst_id": [1, 2, 2],
            "step": [5.0, 10.0, 15.0],
            "amount": [10.0, 20.0, 30.0],
        })
        tc = _compute_temporal_clustering(edges)
        node_0 = tc.filter(pl.col("node_id") == 0)["temporal_clustering_coeff"].to_list()
        self.assertEqual(len(node_0), 1)
        self.assertAlmostEqual(node_0[0], 1.0, places=4)

    def test_empty_edges_df(self) -> None:
        """Empty edges DataFrame should produce empty motif DataFrame with correct columns."""
        empty_edges = pl.DataFrame({
            "src_id": pl.Series([], dtype=pl.Int64),
            "dst_id": pl.Series([], dtype=pl.Int64),
            "step": pl.Series([], dtype=pl.Float64),
            "amount": pl.Series([], dtype=pl.Float64),
        })
        result = compute_motif_features(empty_edges, self.config)
        self.assertIn("node_id", result.columns)
        for col in MOTIF_FEATURE_COLS:
            self.assertIn(col, result.columns)
        self.assertEqual(result.height, 0)

    def test_compute_motif_features_all_cols(self) -> None:
        """compute_motif_features should output all 6 motif feature columns."""
        edges = pl.DataFrame({
            "src_id": [0, 1, 2, 0],
            "dst_id": [1, 2, 0, 3],
            "step": [10.0, 20.0, 30.0, 35.0],
            "amount": [100.0, 100.0, 100.0, 50.0],
        })
        result = compute_motif_features(edges, self.config)
        self.assertIn("node_id", result.columns)
        for col in MOTIF_FEATURE_COLS:
            self.assertIn(col, result.columns)
        self.assertTrue(result.height > 0)


if __name__ == "__main__":
    unittest.main()
