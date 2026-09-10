"""Converts preprocessed DataFrames into PyTorch Geometric graph objects.

Supports two graph representations:

**Static graph** (``build_graph``):
  • Single ``Data`` object with all edges flattened into one snapshot.
  • Temporal masks split nodes into train/val/test by first-activity time.
  • Used by the static FraudGNN baseline.

**Temporal event stream** (``build_temporal_data``):
  • ``TemporalData`` object with events sorted chronologically.
  • Each event is a ``(src, dst, t, msg)`` tuple for the TGN pipeline.
  • Preserves raw integer timestamps (time encoder learns its own representation).
  • Used by the TemporalFraudGNN.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import Data, TemporalData

from arctan.config import PipelineConfig, get_default_config
from arctan.data.preprocess import FEATURE_COLS, TemporalSplitResult, preprocess

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

    # Ring membership labels (for multi-task learning)
    if "is_ring_member" in nodes_df.columns:
        ring_y_np = nodes_df.select("is_ring_member").to_numpy().squeeze()
        ring_y = torch.tensor(ring_y_np, dtype=torch.int64)
    else:
        ring_y = torch.zeros(nodes_df.height, dtype=torch.int64)

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

    # Log ring distribution per split
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n_total = mask.sum().item()
        n_ring = (ring_y[mask] == 1).sum().item() if n_total > 0 else 0
        logger.info(
            "  %s: %d ring members (%.2f%%)",
            name, n_ring, (n_ring / max(1, n_total) * 100),
        )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        ring_y=ring_y,
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
    """Load a previously saved static graph from disk."""
    graph_path = config.paths.graph_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found at {graph_path}")

    logger.info(f"Loading graph from {graph_path}")
    data = torch.load(graph_path, weights_only=False)

    if hasattr(data, "x") and data.x is not None:
        config.model.in_features = data.x.shape[1]

    return data


# Transaction type encoding for temporal edge features
_TXN_TYPE_MAP = {
    "PAYMENT": 0.0,
    "TRANSFER": 0.25,
    "CASH_OUT": 0.5,
    "CASH_IN": 0.75,
    "DEBIT": 1.0,
}


def build_temporal_data(
    split_result: TemporalSplitResult, config: PipelineConfig
) -> dict:
    """Build a TemporalData object from the preprocessed event stream.

    Unlike ``build_graph`` which flattens all edges into a single static snapshot,
    this function preserves chronological order. Each event is stored as
    ``(src, dst, t, msg)`` where ``msg = [log(amount+1), txn_type_encoded]``.

    Returns a dict with:
      - ``temporal_data``: The PyG TemporalData object (sorted by time).
      - ``node_labels``: Tensor of node-level fraud labels ``[N]``.
      - ``entity_ids``: List of entity ID strings.
      - ``train_entity_ids``, ``val_entity_ids``, ``test_entity_ids``: Entity split sets.
      - ``num_nodes``: Total node count.
    """
    nodes_df = split_result.nodes_df
    edges_df = split_result.edges_df

    logger.info("Building TemporalData object for TGN pipeline...")

    if edges_df.height == 0:
        logger.warning("Empty edges DataFrame.")
        return {}

    # Sort edges chronologically (should already be sorted, but enforce it)
    edges_sorted = edges_df.sort("step")

    # Extract source, destination, timestamp
    src = torch.tensor(
        edges_sorted.select("src_id").to_numpy().squeeze(), dtype=torch.long
    )
    dst = torch.tensor(
        edges_sorted.select("dst_id").to_numpy().squeeze(), dtype=torch.long
    )
    t = torch.tensor(
        edges_sorted.select("step").to_numpy().squeeze(), dtype=torch.long
    )

    # Build edge message features: [log(amount+1), txn_type_encoded]
    amounts = edges_sorted.select("amount").to_numpy().squeeze().astype(np.float32)
    log_amounts = np.log1p(amounts)

    if "txn_type" in edges_sorted.columns:
        txn_types = edges_sorted.select("txn_type").to_series().to_list()
        txn_encoded = np.array(
            [_TXN_TYPE_MAP.get(t_type, 0.5) for t_type in txn_types],
            dtype=np.float32,
        )
    else:
        txn_encoded = np.zeros(len(log_amounts), dtype=np.float32)

    msg = torch.tensor(
        np.stack([log_amounts, txn_encoded], axis=1), dtype=torch.float32
    )

    temporal_data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    # Node labels
    y_np = nodes_df.select("is_fraud").to_numpy().squeeze()
    node_labels = torch.tensor(y_np, dtype=torch.int64)

    # Ring membership labels (for multi-task learning)
    if "is_ring_member" in nodes_df.columns:
        ring_np = nodes_df.select("is_ring_member").to_numpy().squeeze()
        ring_labels = torch.tensor(ring_np, dtype=torch.int64)
    else:
        ring_labels = torch.zeros(nodes_df.height, dtype=torch.int64)

    # Structural node features (16 features from feature engineering)
    available_cols = [c for c in FEATURE_COLS if c in nodes_df.columns]
    if available_cols:
        feature_np = nodes_df.select(available_cols).to_numpy().astype(np.float32)
        node_features = torch.tensor(feature_np, dtype=torch.float32)
    else:
        node_features = torch.zeros(nodes_df.height, 0, dtype=torch.float32)

    # Entity IDs
    entity_ids = (
        nodes_df.select("entity_id").to_series().to_list()
        if "entity_id" in nodes_df.columns
        else []
    )

    # Persist
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.paths.temporal_graph_path
    result = {
        "temporal_data": temporal_data,
        "node_labels": node_labels,
        "ring_labels": ring_labels,
        "node_features": node_features,
        "entity_ids": entity_ids,
        "train_entity_ids": split_result.train_entity_ids,
        "val_entity_ids": split_result.val_entity_ids,
        "test_entity_ids": split_result.test_entity_ids,
        "num_nodes": nodes_df.height,
    }
    torch.save(result, out_path)
    logger.info(f"Saved TemporalData to {out_path}")

    logger.info(
        "TemporalData: %d events, %d nodes, msg_dim=%d, "
        "step range=[%d, %d]",
        len(temporal_data.src),
        nodes_df.height,
        msg.shape[1],
        int(t.min()),
        int(t.max()),
    )

    return result


def load_temporal_data(config: PipelineConfig) -> dict:
    """Load a previously saved TemporalData dict from disk."""
    path = config.paths.temporal_graph_path
    if not path.exists():
        raise FileNotFoundError(f"Temporal data file not found at {path}")

    logger.info(f"Loading TemporalData from {path}")
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    cfg = get_default_config()
    cfg.paths.ensure_dirs()

    logger.info("Starting data pipeline...")
    split_result = preprocess(cfg)

    # Build static graph (for baseline FraudGNN)
    graph_data = build_graph(split_result, cfg)
    logger.info(
        f"Static graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, "
        f"{graph_data.num_node_features} features, "
        f"edge_attr shape={graph_data.edge_attr.shape}"
    )

    # Build temporal data (for TGN)
    temporal_result = build_temporal_data(split_result, cfg)
    logger.info(
        f"Temporal data: {len(temporal_result['temporal_data'].src)} events, "
        f"{temporal_result['num_nodes']} nodes"
    )

    logger.info("Pipeline completed successfully.")

