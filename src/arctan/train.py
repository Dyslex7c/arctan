"""Training script for the FraudGNN model.

Implements full-batch training with:
  • Automatic inverse-frequency class weighting
  • Focal Loss to handle severe label imbalance
  • Early stopping on validation F1
  • Per-epoch logging of train loss, val loss, val F1, and val AUROC
"""

import structlog
import torch
from sklearn.metrics import roc_auc_score

from arctan.config import PipelineConfig, get_default_config
from arctan.data.graph_builder import load_graph
from arctan.models.fraud_gnn import FocalLoss, FraudGNN

logger = structlog.get_logger(__name__)


def train_model(config: PipelineConfig) -> FraudGNN:
    """Train the FraudGNN model with the given configuration."""
    torch.manual_seed(config.training.seed)

    logger.info("Loading graph...")
    graph = load_graph(config)

    config.model.in_features = graph.num_node_features

    device = torch.device(config.training.device)
    model = FraudGNN(config.model).to(device)
    graph = graph.to(device)

    # Compute inverse-frequency class weights for Focal Loss
    logger.info("Computing class weights...")
    y_train = graph.y[graph.train_mask]
    num_neg = (y_train == 0).sum().item()
    num_pos = (y_train == 1).sum().item()
    total = num_neg + num_pos

    if total > 0:
        weight_neg = total / (2.0 * max(1, num_neg))
        weight_pos = total / (2.0 * max(1, num_pos))
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=config.training.focal_loss_gamma)

    # Ring membership criterion (for multi-task learning)
    if config.multitask.enabled and hasattr(graph, 'ring_y'):
        ring_y_train = graph.ring_y[graph.train_mask]
        ring_pos = (ring_y_train == 1).sum().item()
        ring_neg = (ring_y_train == 0).sum().item()
        ring_total = ring_pos + ring_neg
        if ring_total > 0 and ring_pos > 0:
            ring_w_neg = ring_total / (2.0 * max(1, ring_neg))
            ring_w_pos = ring_total / (2.0 * max(1, ring_pos))
            ring_weights = torch.tensor([ring_w_neg, ring_w_pos], dtype=torch.float32).to(device)
        else:
            ring_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
        ring_criterion = FocalLoss(alpha=ring_weights, gamma=config.training.focal_loss_gamma)
    else:
        ring_criterion = None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Full-batch training loop
    logger.info("Starting training loop...")
    best_val_f1 = -1.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(graph.x, graph.edge_index, graph.edge_attr)
        train_logits = outputs["fraud"][graph.train_mask]
        train_y = graph.y[graph.train_mask]

        loss = criterion(train_logits, train_y)

        # Multi-task: add ring classification loss
        if ring_criterion is not None:
            ring_logits = outputs["ring"][graph.train_mask]
            ring_labels = graph.ring_y[graph.train_mask]
            ring_loss = ring_criterion(ring_logits, ring_labels)
            loss = (
                config.multitask.fraud_task_weight * loss
                + config.multitask.ring_task_weight * ring_loss
            )
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(graph.x, graph.edge_index, graph.edge_attr)
            val_logits = val_outputs["fraud"][graph.val_mask]
            val_y = graph.y[graph.val_mask]

            val_loss = criterion(val_logits, val_y).item()

            val_preds = val_logits.argmax(dim=-1)
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1]

            tp = ((val_preds == 1) & (val_y == 1)).sum().item()
            fp = ((val_preds == 1) & (val_y == 0)).sum().item()
            fn = ((val_preds == 0) & (val_y == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val_f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            try:
                val_auroc = roc_auc_score(val_y.cpu().numpy(), val_probs.cpu().numpy())
            except Exception:
                val_auroc = 0.5

        logger.info(
            "Epoch stats",
            epoch=epoch,
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_f1:.4f}",
            val_auroc=f"{val_auroc:.4f}",
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        config.paths.models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.paths.best_model_path)
        logger.info(f"Best model saved to {config.paths.best_model_path}")

    return model


if __name__ == "__main__":
    config = get_default_config()
    train_model(config)
