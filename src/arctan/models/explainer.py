"""GNNExplainer-based attribution for per-node risk score interpretability.

For every scored entity the explainer surfaces:
  - which *features* (e.g. transaction velocity, counterparty diversity)
    most influenced the prediction, and
  - which *neighbouring entities* in the graph contributed the most.

This makes the risk score auditable — a reviewer can inspect the
explanation and verify that the model relied on sensible signals.
"""

from typing import Any

import torch

from arctan.models.fraud_gnn import FraudGNN

try:
    from torch_geometric.explain import Explainer, GNNExplainer
except ImportError:
    Explainer, GNNExplainer = None, None  # type: ignore[assignment,misc]


class _FraudOnlyWrapper(torch.nn.Module):
    """Thin wrapper that extracts fraud logits from the multi-task FraudGNN.
    
    GNNExplainer expects a model that returns a single tensor, but
    FraudGNN now returns a dict.  This wrapper adapts the interface.
    """
    def __init__(self, model: FraudGNN) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        outputs = self.model(x, edge_index, edge_attr)
        return outputs["fraud"]


class FraudExplainer:
    """Interpretability wrapper around a trained :class:`FraudGNN` model."""

    def __init__(self, model: FraudGNN, num_hops: int = 2) -> None:
        """Initialise the explainer.

        Args:
            model: A trained FraudGNN model to explain.
            num_hops: Number of message-passing hops in the computational graph.
        """
        self.model = model
        self.model.eval()
        self.num_hops = num_hops

        if Explainer is None or GNNExplainer is None:
            raise ImportError(
                "torch_geometric.explain is required for FraudExplainer. "
                "Install with: pip install torch-geometric"
            )

        self.explainer = Explainer(
            model=_FraudOnlyWrapper(self.model),
            algorithm=GNNExplainer(epochs=20),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Explain the model prediction for a single node.

        Returns a dict containing ``feature_importance``, ``edge_importance``,
        ``node_importance``, and a human-readable ``summary`` string.
        """
        explanation = self.explainer(
            x=x,
            edge_index=edge_index,
            index=node_idx,
            edge_attr=edge_attr,
        )

        # Feature importance
        if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
            feature_importance_tensor = explanation.node_mask.mean(dim=0)
        else:
            feature_importance_tensor = torch.zeros(x.size(1))

        feature_importance = feature_importance_tensor.detach().cpu().numpy()

        # Edge importance
        edge_importance: dict[tuple[int, int], float] = {}
        if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None:
            edge_mask = explanation.edge_mask.detach().cpu().numpy()
            for i, score in enumerate(edge_mask):
                src = int(edge_index[0, i].item())
                dst = int(edge_index[1, i].item())
                edge_importance[(src, dst)] = float(score)

        # Aggregate to neighbour-level importance
        node_importance: dict[int, float] = {}
        for (src, dst), score in edge_importance.items():
            if src == node_idx:
                node_importance[dst] = node_importance.get(dst, 0.0) + score
            elif dst == node_idx:
                node_importance[src] = node_importance.get(src, 0.0) + score

        # Build human-readable summary
        top_features = sorted(
            range(len(feature_importance)),
            key=lambda i: feature_importance[i],
            reverse=True,
        )[:5]
        top_neighbors = sorted(
            node_importance.items(), key=lambda item: item[1], reverse=True
        )[:3]

        feature_str = ", ".join(
            f"feature_{i} ({feature_importance[i]:.4f})" for i in top_features
        )
        neighbor_str = ", ".join(
            f"node_{n} ({score:.4f})" for n, score in top_neighbors
        )

        summary = (
            f"Prediction for node {node_idx} was most influenced by: "
            f"Features [{feature_str}], "
            f"Neighbor Connections [{neighbor_str if neighbor_str else 'None'}]"
        )

        return {
            "node_importance": node_importance,
            "feature_importance": feature_importance,
            "edge_importance": edge_importance,
            "summary": summary,
        }
