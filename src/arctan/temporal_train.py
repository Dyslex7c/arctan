"""Temporal training loop for the TGN-based fraud detection model.

Processes transactions as a chronological event stream rather than a
static graph snapshot. For each mini-batch of events:

  1. Sample temporal neighborhoods via LastNeighborLoader
  2. Retrieve current entity memory states
  3. Compute time-encoded edge features for sampled neighborhood
  4. Forward pass through temporal graph attention with node features
  5. Compute focal loss on labeled nodes
  6. Backpropagate (memory gradients detached for efficiency)
  7. Update entity memories with this batch's events

This prevents future-information leakage that occurs in the static model's
full-batch training, where message passing aggregates freely across all
time periods.
"""

from __future__ import annotations

import structlog
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from arctan.config import PipelineConfig, get_default_config
from arctan.data.graph_builder import load_temporal_data
from arctan.models.entity_memory import build_entity_memory, save_memory_state
from arctan.models.fraud_gnn import FocalLoss
from arctan.models.temporal_gnn import TemporalFraudGNN
from arctan.models.time_encoder import TimeEncoder

logger = structlog.get_logger(__name__)

# Temporal split boundaries (must match preprocess.py)
TRAIN_END = 445
VAL_END = 594


def _compute_edge_features(
    e_id: torch.Tensor,
    edge_index: torch.Tensor,
    last_update: torch.Tensor,
    edge_t_store: torch.Tensor,
    edge_msg_store: torch.Tensor,
    edge_insert_count: int,
    time_encoder: TimeEncoder,
    tconfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute time-encoded edge features for sampled neighborhood edges.

    Uses stored timestamps and messages from previously inserted edges,
    with the time encoder computing learnable Dt embeddings.
    """
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return torch.zeros(0, tconfig.time_dim + tconfig.raw_msg_dim, device=device)

    if e_id.numel() > 0 and edge_insert_count > 0:
        # Clamp e_id to valid range (some may reference edges not yet stored)
        valid_mask = e_id < edge_insert_count
        clamped_eid = e_id.clamp(max=max(0, edge_insert_count - 1))

        # Look up stored edge timestamps and messages
        sampled_t = edge_t_store[clamped_eid]
        sampled_msg = edge_msg_store[clamped_eid]

        # Zero out invalid entries
        sampled_t = sampled_t * valid_mask.float()
        sampled_msg = sampled_msg * valid_mask.unsqueeze(-1).float()

        # Compute relative time: source node's last update - edge timestamp
        src_last = last_update[edge_index[0]]
        delta_t = (src_last - sampled_t).float()
        t_enc = time_encoder(delta_t)

        return torch.cat([t_enc, sampled_msg], dim=-1)

    return torch.zeros(num_edges, tconfig.time_dim + tconfig.raw_msg_dim, device=device)


def train_temporal_model(config: PipelineConfig) -> TemporalFraudGNN:
    """Train the temporal GNN model with chronological event processing.

    Args:
        config: Pipeline configuration.

    Returns:
        The trained TemporalFraudGNN model.
    """
    torch.manual_seed(config.training.seed)
    device = torch.device(config.training.device)
    tconfig = config.temporal_model

    logger.info("Loading temporal data...")
    data_dict = load_temporal_data(config)
    temporal_data = data_dict["temporal_data"]
    node_labels = data_dict["node_labels"].to(device)
    num_nodes = data_dict["num_nodes"]

    # Load node features
    node_features = data_dict.get("node_features")
    if node_features is not None and node_features.size(1) > 0:
        node_features = node_features.to(device)
        tconfig.node_feature_dim = node_features.size(1)
        logger.info("Node features loaded: %d dimensions", tconfig.node_feature_dim)
    else:
        node_features = None
        tconfig.node_feature_dim = 0

    # Split events by temporal boundaries
    train_mask = temporal_data.t <= TRAIN_END
    val_mask = (temporal_data.t > TRAIN_END) & (temporal_data.t <= VAL_END)

    train_data = temporal_data[train_mask]
    val_data = temporal_data[val_mask]

    logger.info(
        "Temporal splits: train=%d events, val=%d events, total=%d events",
        len(train_data.src),
        len(val_data.src),
        len(temporal_data.src),
    )

    # Build model components
    memory = build_entity_memory(num_nodes, tconfig).to(device)
    model = TemporalFraudGNN(tconfig).to(device)
    time_encoder = TimeEncoder(tconfig.time_dim).to(device)
    neighbor_loader = LastNeighborLoader(
        num_nodes, size=tconfig.num_neighbors, device=device
    )

    # Compute class weights (capped to prevent over-prediction)
    train_entity_ids = data_dict["train_entity_ids"]
    entity_ids = data_dict["entity_ids"]
    num_pos = sum(
        1 for i, eid in enumerate(entity_ids)
        if eid in train_entity_ids and node_labels[i].item() == 1
    )
    train_entity_count = sum(1 for eid in entity_ids if eid in train_entity_ids)
    num_neg = train_entity_count - num_pos

    if train_entity_count > 0 and num_pos > 0:
        raw_ratio = num_neg / num_pos
        capped_ratio = min(raw_ratio, tconfig.max_class_weight_ratio)
        class_weights = torch.tensor(
            [1.0, capped_ratio], dtype=torch.float32
        ).to(device)
        logger.info(
            "Class weights: neg=1.0, pos=%.1f (raw ratio=%.1f, capped at %.1f)",
            capped_ratio, raw_ratio, tconfig.max_class_weight_ratio,
        )
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)

    criterion = FocalLoss(
        alpha=class_weights, gamma=config.training.focal_loss_gamma
    )

    # Ring membership labels (for multi-task learning)
    ring_labels = data_dict.get("ring_labels")
    if ring_labels is not None:
        ring_labels = ring_labels.to(device)
    else:
        ring_labels = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Ring membership criterion
    if config.multitask.enabled:
        train_entity_ids_set = data_dict["train_entity_ids"]
        ring_pos = sum(
            1 for i, eid in enumerate(entity_ids)
            if eid in train_entity_ids_set and ring_labels[i].item() == 1
        )
        ring_neg = train_entity_count - ring_pos
        if ring_pos > 0:
            raw_ring_ratio = ring_neg / ring_pos
            capped_ring_ratio = min(raw_ring_ratio, tconfig.max_class_weight_ratio)
            ring_class_weights = torch.tensor(
                [1.0, capped_ring_ratio], dtype=torch.float32
            ).to(device)
        else:
            ring_class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
        ring_criterion = FocalLoss(
            alpha=ring_class_weights, gamma=config.training.focal_loss_gamma
        )
    else:
        ring_criterion = None

    # Optimizer covers model, memory, and time encoder parameters
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(memory.parameters())
        + list(time_encoder.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Pre-allocate edge feature storage (indexed by insertion order)
    total_events = len(temporal_data.src)
    edge_t_store = torch.zeros(total_events, device=device)
    edge_msg_store = torch.zeros(total_events, tconfig.raw_msg_dim, device=device)

    # Training loop
    logger.info("Starting temporal training loop...")
    best_val_auroc = -1.0
    patience_counter = 0
    best_model_state = None
    best_time_encoder_state = None

    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
    train_loader = TemporalDataLoader(train_data, batch_size=200)
    val_loader = TemporalDataLoader(val_data, batch_size=200)

    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        memory.train()
        memory.reset_state()
        neighbor_loader.reset_state()
        edge_insert_count = 0

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

            # Sample temporal neighborhoods for batch nodes
            batch_nodes = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(batch_nodes)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get current memory states
            z, last_update = memory(n_id)

            # Compute edge features using stored timestamps + time encoder
            edge_feat = _compute_edge_features(
                e_id, edge_index, last_update,
                edge_t_store, edge_msg_store, edge_insert_count,
                time_encoder, tconfig, device,
            )

            # Get node features for the subgraph
            nf = node_features[n_id] if node_features is not None else None

            # Forward pass through temporal attention + node features
            outputs = model(z, edge_index, edge_feat, node_features=nf)

            # Compute loss on batch nodes
            batch_local = assoc[batch_nodes]
            batch_labels = node_labels[batch_nodes]
            fraud_logits = outputs["fraud"]
            loss = criterion(fraud_logits[batch_local], batch_labels)

            # Multi-task: add ring classification loss
            if ring_criterion is not None:
                ring_logits_batch = outputs["ring"][batch_local]
                ring_labels_batch = ring_labels[batch_nodes]
                ring_loss = ring_criterion(ring_logits_batch, ring_labels_batch)
                loss = (
                    config.multitask.fraud_task_weight * loss
                    + config.multitask.ring_task_weight * ring_loss
                )

            loss.backward()
            optimizer.step()

            # Update memory with this batch's events (detached from graph)
            memory.update_state(src, dst, t, msg)

            # Store edge features for future neighbor sampling lookups
            bs = src.size(0)
            edge_t_store[edge_insert_count:edge_insert_count + bs] = t.float()
            edge_msg_store[edge_insert_count:edge_insert_count + bs] = msg
            neighbor_loader.insert(src, dst)
            edge_insert_count += bs

            # Detach memory to prevent BPTT across batches
            memory.detach()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(1, num_batches)

        # Validation
        model.eval()
        memory.eval()

        val_preds = []
        val_labels_list = []
        val_logits_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

                batch_nodes = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(batch_nodes)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                z, last_update = memory(n_id)

                edge_feat = _compute_edge_features(
                    e_id, edge_index, last_update,
                    edge_t_store, edge_msg_store, edge_insert_count,
                    time_encoder, tconfig, device,
                )

                nf = node_features[n_id] if node_features is not None else None
                outputs = model(z, edge_index, edge_feat, node_features=nf)

                batch_local = assoc[batch_nodes]
                fraud_logits = outputs["fraud"][batch_local]
                probs = torch.softmax(fraud_logits, dim=-1)[:, 1]

                val_preds.append(probs.cpu())
                val_logits_list.append(fraud_logits.detach().cpu())
                val_labels_list.append(node_labels[batch_nodes].cpu())

                # Update memory during validation (but no grad)
                memory.update_state(src, dst, t, msg)

                bs = src.size(0)
                edge_t_store[edge_insert_count:edge_insert_count + bs] = t.float()
                edge_msg_store[edge_insert_count:edge_insert_count + bs] = msg
                neighbor_loader.insert(src, dst)
                edge_insert_count += bs

        if val_preds:
            all_preds = torch.cat(val_preds)
            all_labels = torch.cat(val_labels_list)

            unique_labels = all_labels.unique()
            has_both = (unique_labels == 0).any() and (unique_labels == 1).any()

            try:
                val_auroc = (
                    roc_auc_score(all_labels.numpy(), all_preds.numpy())
                    if has_both
                    else 0.5
                )
            except Exception:
                val_auroc = 0.5

            preds_binary = (all_preds > 0.5).long()
            tp = ((preds_binary == 1) & (all_labels == 1)).sum().item()
            fp = ((preds_binary == 1) & (all_labels == 0)).sum().item()
            fn = ((preds_binary == 0) & (all_labels == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val_f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            val_auroc = 0.5
            val_f1 = 0.0

        logger.info(
            "Epoch stats",
            epoch=epoch,
            train_loss=f"{avg_train_loss:.4f}",
            val_auroc=f"{val_auroc:.4f}",
            val_f1=f"{val_f1:.4f}",
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            best_time_encoder_state = {
                k: v.cpu().clone() for k, v in time_encoder.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Save best model (includes time encoder state)
    if best_model_state is not None:
        config.paths.models_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model": best_model_state,
            "time_encoder": best_time_encoder_state,
            "node_feature_dim": tconfig.node_feature_dim,
        }

        # Fit temperature scaler on validation predictions
        if config.calibration.enabled and val_logits_list:
            from arctan.models.calibration import fit_temperature
            
            val_logits_t = torch.cat(val_logits_list, dim=0)
            val_labels_t = torch.cat(val_labels_list, dim=0)
            temp_scaler = fit_temperature(
                val_logits_t, val_labels_t, config.calibration
            )
            save_dict["temperature"] = temp_scaler.state_dict()
            logger.info(
                "Temperature scaler fitted",
                temperature=f"{temp_scaler.temperature_value:.4f}",
            )

        torch.save(save_dict, config.paths.temporal_model_path)

        # Save feature reference for drift detection
        if config.drift.enabled:
            import numpy as np
            # Use train-period node features as reference
            ref_path = config.paths.models_dir / "feature_reference.pt"
            # Note: temporal model may not have node features
            # Save what we have from the data dict
            if 'node_features' in data_dict and data_dict['node_features'] is not None:
                train_feats = data_dict['node_features'].numpy()
                torch.save(
                    {
                        "mean": np.mean(train_feats, axis=0),
                        "std": np.std(train_feats, axis=0),
                        "raw": train_feats,
                    },
                    ref_path,
                )
                logger.info(f"Feature reference saved to {ref_path}")

        model.load_state_dict(best_model_state)
        logger.info(f"Best temporal model saved to {config.paths.temporal_model_path}")

    # Build final memory state for inference by replaying train+val events
    logger.info("Building final memory state for inference...")
    memory.reset_state()
    neighbor_loader.reset_state()
    memory.eval()

    all_seen_mask = temporal_data.t <= VAL_END
    all_seen_data = temporal_data[all_seen_mask]
    seen_loader = TemporalDataLoader(all_seen_data, batch_size=200)

    with torch.no_grad():
        for batch in seen_loader:
            batch = batch.to(device)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

    save_memory_state(memory, config.paths.memory_state_path)
    logger.info("Final inference memory state saved.")

    return model


if __name__ == "__main__":
    cfg = get_default_config()
    cfg.model_type = "temporal"
    train_temporal_model(cfg)
