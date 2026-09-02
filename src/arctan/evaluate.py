"""Evaluation script — production-grade fraud detection metrics and diagnostics.

Reports both standard ML metrics and operational fraud-detection KPIs:

Standard:
  • Per-class and macro precision / recall / F1
  • AUROC and PR-AUC

Operational (Threshold & Top-K):
  • Precision@K and Recall@K for K ∈ {50, 100, 200, 500}
  • Lift@K (improvement over random baseline)
  • Optimal F1 threshold (sweeps 0.01-0.99)
  • High-recall threshold (≥95% recall)

Saved artifacts:
  • eval_report.txt — full classification report + top-K table
  • confusion_matrix.png
  • precision_recall_curve.png
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import (
    precision_recall_curve as sk_pr_curve,
)

from arctan.config import PipelineConfig, get_default_config
from arctan.data.graph_builder import load_graph
from arctan.models.fraud_gnn import FraudGNN

logger = structlog.get_logger(__name__)


def _compute_topk_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, k_values: list[int]
) -> dict:
    """Compute Precision@K, Recall@K, and Lift@K.

    Ranks all entities by descending fraud probability, then checks how many
    of the top-K are actually fraudulent.
    """
    n_total = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = n_pos / n_total if n_total > 0 else 0.0

    # Rank entities by descending predicted fraud probability
    ranked_indices = np.argsort(y_probs)[::-1]
    ranked_labels = y_true[ranked_indices]

    results = {}
    for k in k_values:
        k_actual = min(k, n_total)
        top_k_labels = ranked_labels[:k_actual]
        tp_at_k = int(top_k_labels.sum())

        precision_at_k = tp_at_k / k_actual if k_actual > 0 else 0.0
        recall_at_k = tp_at_k / n_pos if n_pos > 0 else 0.0
        lift_at_k = precision_at_k / base_rate if base_rate > 0 else 0.0

        results[f"precision@{k}"] = float(precision_at_k)
        results[f"recall@{k}"] = float(recall_at_k)
        results[f"lift@{k}"] = float(lift_at_k)

    return results


def _compute_threshold_analysis(
    y_true: np.ndarray, y_probs: np.ndarray
) -> dict:
    """Sweep thresholds to find optimal F1 and high-recall operating points."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_f1 = 0.0
    best_f1_thresh = 0.5
    high_recall_thresh = 0.01  # start with lowest threshold

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = float(t)

        # Track the highest threshold that still achieves ≥95% recall
        if recall >= 0.95:
            high_recall_thresh = float(t)

    return {
        "optimal_f1_threshold": best_f1_thresh,
        "optimal_f1_value": float(best_f1),
        "high_recall_95_threshold": high_recall_thresh,
    }


def _save_pr_curve(
    y_true: np.ndarray, y_probs: np.ndarray, save_path: str
) -> None:
    """Save precision-recall curve as PNG."""
    precision_arr, recall_arr, _ = sk_pr_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_arr, precision_arr, linewidth=2, color="#2563eb")
    ax.fill_between(recall_arr, precision_arr, alpha=0.15, color="#2563eb")

    # Baseline (random classifier)
    prevalence = y_true.mean()
    ax.axhline(y=prevalence, color="#dc2626", linestyle="--", linewidth=1,
               label=f"Random baseline ({prevalence:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(config: PipelineConfig) -> dict:
    """Evaluate the best model checkpoint on the held-out test set."""
    device = torch.device("cpu")

    logger.info("Loading graph...")
    graph = load_graph(config).to(device)
    config.model.in_features = graph.num_node_features

    logger.info("Loading model...")
    model = FraudGNN(config.model).to(device)
    if config.paths.best_model_path.exists():
        model.load_state_dict(
            torch.load(config.paths.best_model_path, map_location=device, weights_only=True)
        )
    else:
        logger.warning("No model checkpoint found! Evaluating uninitialised model.")

    model.eval()

    logger.info("Running inference on test set...")
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index, graph.edge_attr)
        test_logits = logits[graph.test_mask]
        test_y = graph.y[graph.test_mask]

    test_preds = test_logits.argmax(dim=-1).cpu().numpy()
    test_probs = torch.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()
    test_y = test_y.cpu().numpy()

    n_test = len(test_y)
    n_test_pos = int(test_y.sum())
    test_prevalence = n_test_pos / n_test if n_test > 0 else 0.0

    logger.info(
        "Test set: %d entities, %d fraud (%.2f%% prevalence)",
        n_test, n_test_pos, test_prevalence * 100,
    )

    # Standard metrics
    acc = accuracy_score(test_y, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, test_preds, labels=[0, 1], average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_y, test_preds, average="macro", zero_division=0
    )

    if len(np.unique(test_y)) >= 2:
        try:
            auroc = float(roc_auc_score(test_y, test_probs))
        except Exception:
            auroc = 0.5
        try:
            pr_auc = float(average_precision_score(test_y, test_probs))
        except Exception:
            pr_auc = 0.0
    else:
        auroc = 1.0 if acc == 1.0 else 0.5
        pr_auc = 1.0 if acc == 1.0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision_class_0": float(precision[0]),
        "precision_class_1": float(precision[1]),
        "recall_class_0": float(recall[0]),
        "recall_class_1": float(recall[1]),
        "f1_class_0": float(f1[0]),
        "f1_class_1": float(f1[1]),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "test_prevalence": test_prevalence,
    }

    # Top-K operational metrics
    k_values = [50, 100, 200, 500]
    topk_metrics = _compute_topk_metrics(test_y, test_probs, k_values)
    metrics.update(topk_metrics)

    # Threshold analysis
    threshold_metrics = _compute_threshold_analysis(test_y, test_probs)
    metrics.update(threshold_metrics)

    # Save classification report + top-K table
    cm = confusion_matrix(test_y, test_preds, labels=[0, 1])
    report = classification_report(test_y, test_preds, zero_division=0)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    report_path = config.paths.processed_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ARCTAN — Temporal Fraud Detection Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("Standard Classification Report\n\n")
        f.write(report)
        f.write("\n\n")

        f.write(f"Test set: {n_test} entities, {n_test_pos} fraud "
                f"({test_prevalence:.4f} prevalence)\n")
        f.write(f"AUROC:  {auroc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")

        f.write("Operational Top-K Metrics\n\n")
        f.write(f"{'K':>6}  {'Precision@K':>12}  {'Recall@K':>10}  {'Lift@K':>8}\n")
        f.write("-" * 42 + "\n")
        for k in k_values:
            p = topk_metrics[f"precision@{k}"]
            r = topk_metrics[f"recall@{k}"]
            lift_ratio = topk_metrics[f"lift@{k}"]
            f.write(f"{k:>6}  {p:>12.4f}  {r:>10.4f}  {lift_ratio:>8.2f}x\n")

        f.write("\nThreshold Analysis\n\n")
        f.write(f"Optimal F1 threshold:      {threshold_metrics['optimal_f1_threshold']:.2f} "
                f"(F1 = {threshold_metrics['optimal_f1_value']:.4f})\n")
        f.write(
            "High-recall (≥95%) threshold: "
            f"{threshold_metrics['high_recall_95_threshold']:.2f}\n"
        )

    logger.info(f"Evaluation report saved to {report_path}")

    # Save confusion matrix plot
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center")

    plot_path = config.paths.processed_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")

    # Save PR curve
    pr_path = config.paths.processed_dir / "precision_recall_curve.png"
    _save_pr_curve(test_y, test_probs, str(pr_path))
    logger.info(f"Precision-recall curve saved to {pr_path}")

    return metrics


if __name__ == "__main__":
    config = get_default_config()
    metrics = evaluate_model(config)
    logger.info("Evaluation complete", metrics=metrics)
