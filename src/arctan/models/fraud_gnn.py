"""Hybrid GraphSAGE + GATv2 architecture for node-level anomaly detection.

The network stacks a SAGEConv neighbourhood-aggregation layer, a multi-head
GATv2Conv attention layer, and a second SAGEConv layer before a linear
classifier.  This design lets the model learn *both* mean-pooled structural
patterns (SAGE) and fine-grained pairwise attention over counterparty edges
(GATv2).

FocalLoss is included here because it is tightly coupled to the model's
training objective — standard cross-entropy fails when fraud labels are a
tiny minority of nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, SAGEConv

from arctan.config import ModelConfig


class FraudGNN(nn.Module):
    """Hybrid GraphSAGE + GATv2 multi-task model for fraud and ring classification."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialise the FraudGNN model.

        Args:
            config: Model configuration containing architecture hyper-parameters.
        """
        super().__init__()

        # Layer 1: SAGEConv — neighbourhood mean/max/add aggregation
        self.conv1 = SAGEConv(
            config.in_features,
            config.hidden_dim,
            aggr=config.aggr,
        )
        self.bn1 = BatchNorm(config.hidden_dim)

        # Layer 2: GATv2Conv — multi-head dynamic attention
        self.edge_dim = getattr(config, "edge_dim", None)
        gat_kwargs: dict = dict(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            heads=config.gat_heads,
            concat=False,
        )
        if self.edge_dim is not None:
            gat_kwargs["edge_dim"] = self.edge_dim
        self.conv2 = GATv2Conv(**gat_kwargs)
        self.bn2 = BatchNorm(config.hidden_dim)

        # Layer 3: SAGEConv — dimensionality reduction before classifier
        self.conv3 = SAGEConv(
            config.hidden_dim,
            config.hidden_dim // 2,
            aggr=config.aggr,
        )
        self.bn3 = BatchNorm(config.hidden_dim // 2)

        # Output head
        self.lin = nn.Linear(config.hidden_dim // 2, config.out_dim)

        # Ring membership classification head
        self.ring_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.ring_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ring_hidden_dim, config.ring_out_dim),
        )

        self.dropout_prob = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node feature matrix of shape ``[N, in_features]``.
            edge_index: COO edge connectivity of shape ``[2, E]``.
            edge_attr: Optional edge feature matrix of shape ``[E, D_e]``.

        Returns:
            Dictionary containing fraud logits and ring logits.
        """
        # SAGEConv Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # GATv2Conv Layer — attend over counterparty edges
        if self.edge_dim is not None and edge_attr is not None:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # SAGEConv Layer 2
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Linear classifier
        shared = x  # shared representation
        fraud_logits = self.lin(shared)
        ring_logits = self.ring_head(shared)
        return {"fraud": fraud_logits, "ring": ring_logits}

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class probabilities (softmax over logits)."""
        outputs = self.forward(x, edge_index, edge_attr)
        return F.softmax(outputs["fraud"], dim=-1)


class FocalLoss(nn.Module):
    """Focal Loss for severely imbalanced node classification.

    ``FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)``

    When γ = 0 this reduces to weighted cross-entropy.  Higher γ
    down-weights easy negatives so the model focuses on hard-to-classify
    fraud nodes.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        """Initialise FocalLoss.

        Args:
            gamma: Focusing parameter — modulates ``(1 − p_t)`` factor.
            alpha: Per-class weight tensor of shape ``[C]``.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``[N, C]``.
            targets: Ground-truth labels of shape ``[N]``.

        Returns:
            Scalar loss tensor.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
