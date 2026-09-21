"""
This module implements gradient-based counterfactual explanations for the Arctan fraud detection
system. It provides tools to find the minimal feature perturbation that changes a fraud prediction
to legitimate.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from arctan.models.fraud_gnn import FraudGNN

logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    """
    Finds minimal feature perturbations to change a model's prediction for a specific node.
    """

    def __init__(self, model: FraudGNN, feature_names: list[str] | None = None):
        """
        Initialize the explainer.
        
        Args:
            model: The trained FraudGNN model.
            feature_names: Optional list of feature names for human-readable output.
        """
        self.model = model
        self.feature_names = feature_names

    def find_counterfactual(
        self,
        node_idx: int,
        graph: Data,
        target_class: int = 0,
        max_iters: int = 200,
        lr: float = 0.01,
        lambda_l1: float = 0.1,
    ) -> dict:
        """
        Finds a counterfactual perturbation for the given node.

        Args:
            node_idx: The index of the node to explain.
            graph: The PyG Data object containing the graph.
            target_class: The desired target class (default: 0, legitimate).
            max_iters: Maximum number of optimization iterations.
            lr: Learning rate for the perturbation optimizer.
            lambda_l1: L1 regularization weight for the perturbation.

        Returns:
            A dictionary containing the counterfactual explanation details.
        """
        self.model.eval()

        with torch.no_grad():
            initial_logits = self.model(graph.x, graph.edge_index, graph.edge_attr)["fraud"]
            initial_probs = F.softmax(initial_logits, dim=-1)
            original_prob = initial_probs[node_idx, 1].item()

        delta = torch.zeros(graph.x.size(1), requires_grad=True, device=graph.x.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        target_tensor = torch.tensor([target_class], dtype=torch.long, device=graph.x.device)

        success = False
        final_prob = original_prob

        with torch.enable_grad():
            for _ in range(max_iters):
                optimizer.zero_grad()

                # Build perturbed features without in-place ops
                node_feats = (graph.x[node_idx] + delta).clamp(min=0)
                perturbed_x = torch.cat([
                    graph.x[:node_idx],
                    node_feats.unsqueeze(0),
                    graph.x[node_idx + 1:],
                ], dim=0)

                outputs = self.model(
                    perturbed_x, graph.edge_index, graph.edge_attr
                )
                logits = outputs["fraud"]

                ce_loss = F.cross_entropy(
                    logits[node_idx].unsqueeze(0), target_tensor
                )
                l1_loss = lambda_l1 * delta.abs().sum()
                loss = ce_loss + l1_loss

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    check_feats = (
                        graph.x[node_idx] + delta
                    ).clamp(min=0)
                    check_x = torch.cat([
                        graph.x[:node_idx],
                        check_feats.unsqueeze(0),
                        graph.x[node_idx + 1:],
                    ], dim=0)
                    check_logits = self.model(
                        check_x, graph.edge_index, graph.edge_attr
                    )["fraud"]
                    pred_class = (
                        check_logits[node_idx].argmax().item()
                    )

                    if pred_class == target_class:
                        success = True
                        check_probs = F.softmax(
                            check_logits, dim=-1
                        )
                        final_prob = (
                            check_probs[node_idx, 1].item()
                        )
                        break

        if not success:
            with torch.no_grad():
                final_feats = (
                    graph.x[node_idx] + delta
                ).clamp(min=0)
                final_x = torch.cat([
                    graph.x[:node_idx],
                    final_feats.unsqueeze(0),
                    graph.x[node_idx + 1:],
                ], dim=0)
                final_logits = self.model(
                    final_x, graph.edge_index, graph.edge_attr
                )["fraud"]
                final_probs = F.softmax(final_logits, dim=-1)
                final_prob = final_probs[node_idx, 1].item()

        final_delta = (graph.x[node_idx] + delta).clamp(min=0) - graph.x[node_idx]
        
        perturbation_dict = {}
        changed_features = 0
        total_norm = final_delta.norm(p=2).item()
        
        final_delta_np = final_delta.detach().cpu().numpy()
        for j, change in enumerate(final_delta_np):
            if abs(change) > 0.01:
                changed_features += 1
                feat_name = (
                    self.feature_names[j]
                    if self.feature_names and j < len(self.feature_names)
                    else f"feature_{j}"
                )
                perturbation_dict[feat_name] = float(change)

        return {
            "success": success,
            "original_prob": float(original_prob),
            "counterfactual_prob": float(final_prob),
            "perturbation": perturbation_dict,
            "num_features_changed": changed_features,
            "total_perturbation_norm": float(total_norm),
        }
