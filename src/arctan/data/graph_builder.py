"""Converts preprocessed DataFrames into a PyTorch Geometric ``Data`` object.

Key improvements over the baseline:
  1. **Temporal masks**: train/val/test are assigned by temporal period
     (not random permutation), preventing future-information leakage.
  2. **Edge attribute normalization**: ``amount`` is log-transformed and
     ``timestamp`` is min-max normalized so GATv2Conv receives meaningful
     scaled edge features.
  3. Persists the graph to disk for fast reload during training.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import Data

from arctan.config import PipelineConfig, get_default_config
from arctan.data.preprocess import TemporalSplitResult, preprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_graph(
    split_result: TemporalSplitResult, config: PipelineConfig
) -> Data:
    """Build a PyTorch Geometric Data object with temporal masks and normalized edges."""
    nodes_df = split_result.nodes_df
    edges_df = split_result.edges_df
    train_entities = split_result.train_entity_ids
    val_entities = split_result.val_entity_ids
    test_entities = split_result.test_entity_ids

    logger.info("Building PyTorch Geometric Data object (temporal split)...")

    if nodes_df.height == 0:
        logger.warning("Empty nodes DataFrame. Returning empty Data object.")
        return Data()

    # Node features (x)
    exclude_cols = ["node_id", "entity_id", "is_fraud", "source"]
    feature_cols = [col for col in nodes_df.columns if col not in exclude_cols]

    if feature_cols:
        x_np = nodes_df.select(feature_cols).to_numpy()
        x = torch.tensor(x_np, dtype=torch.float32)
    else:
        logger.warning("No feature columns found. Creating dummy feature tensor.")
        x = torch.ones((nodes_df.height, 1), dtype=torch.float32)

    config.model.in_features = x.shape[1]

    # Labels (y)
    y_np = nodes_df.select("is_fraud").to_numpy().squeeze()
    y = torch.tensor(y_np, dtype=torch.int64)

    # Edges (edge_index)
    if edges_df.height > 0:
        src = edges_df.select("src_id").to_numpy().squeeze()
        dst = edges_df.select("dst_id").to_numpy().squeeze()

        if src.ndim == 0:
            src = np.array([src])
            dst = np.array([dst])

        edge_index_np = np.stack([src, dst], axis=0)
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)

        # Edge attribute normalization
        # [log(amount + 1), min-max normalized timestamp]
        amounts = edges_df.select("amount").to_numpy().squeeze().astype(np.float32)
        steps = edges_df.select("step").to_numpy().squeeze().astype(np.float32)

        # Log-transform amount (handles heavy-tailed distribution)
        log_amounts = np.log1p(amounts)

        # Min-max normalize timestamp to [0, 1]
        step_min = steps.min()
        step_max = steps.max()
        if step_max > step_min:
            norm_steps = (steps - step_min) / (step_max - step_min)
        else:
            norm_steps = np.zeros_like(steps)

        edge_attr_np = np.stack([log_amounts, norm_steps], axis=1)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

        logger.info(
            "Edge attributes: shape=%s, "
            "log_amount range=[%.2f, %.2f], "
            "norm_step range=[%.2f, %.2f]",
            edge_attr.shape,
            log_amounts.min(), log_amounts.max(),
            norm_steps.min(), norm_steps.max(),
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    # Temporal train / val / test masks
    entity_list = (
        nodes_df.select("entity_id").to_series().to_list()
        if "entity_id" in nodes_df.columns
        else []
    )

    num_nodes = nodes_df.height
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for i, eid in enumerate(entity_list):
        if eid in train_entities:
            train_mask[i] = True
        elif eid in val_entities:
            val_mask[i] = True
        elif eid in test_entities:
            test_mask[i] = True

    logger.info(
        "Temporal masks: train=%d, val=%d, test=%d (total=%d)",
        train_mask.sum().item(),
        val_mask.sum().item(),
        test_mask.sum().item(),
        num_nodes,
    )

    # Log fraud distribution per split
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n_total = mask.sum().item()
        n_fraud = (y[mask] == 1).sum().item() if n_total > 0 else 0
        prevalence = (n_fraud / n_total * 100) if n_total > 0 else 0.0
        logger.info(
            "  %s: %d entities, %d fraud (%.2f%% prevalence)",
            name, n_total, n_fraud, prevalence,
        )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        entity_id=entity_list,
    )

    # Persist
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.paths.graph_path
    logger.info(f"Saving graph to {out_path}")
    torch.save(data, out_path)

    return data


def load_graph(config: PipelineConfig) -> Data:
    """Load a previously saved graph from disk."""
    graph_path = config.paths.graph_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found at {graph_path}")

    logger.info(f"Loading graph from {graph_path}")
    data = torch.load(graph_path, weights_only=False)

    if hasattr(data, "x") and data.x is not None:
        config.model.in_features = data.x.shape[1]

    return data


if __name__ == "__main__":
    cfg = get_default_config()
    cfg.paths.ensure_dirs()

    logger.info("Starting temporal data pipeline...")
    split_result = preprocess(cfg)
    graph_data = build_graph(split_result, cfg)

    logger.info("Pipeline completed successfully.")
    logger.info(
        f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, "
        f"{graph_data.num_node_features} features, "
        f"edge_attr shape={graph_data.edge_attr.shape}"
    )
