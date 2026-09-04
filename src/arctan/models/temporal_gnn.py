"""Temporal Graph Network for fraud detection with online entity memory.

Architecture (Rossi et al., 2020 adapted for node classification):
  1. **Entity Memory** — GRU-updated per-node state vectors that capture each
     entity's behavioral trajectory as transactions arrive chronologically.
  2. **Temporal Graph Attention** — Multi-head attention over each node's K most
     recent neighbors, using time-encoded Δt as part of the attention mechanism.
  3. **MLP Classifier** — Maps the concatenation of memory and attention
     embeddings to a fraud probability.

Unlike the static FraudGNN which processes the entire graph in one pass,
the TemporalFraudGNN processes events chronologically and can detect evolving
fraud patterns (e.g. an account that transitions from normal to burst activity).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from arctan.config import TemporalModelConfig


class TemporalFraudGNN(nn.Module):
    """TGN-based fraud detection with online entity memory.

    The model receives pre-computed memory vectors from the TGNMemory module
    and combines them with temporal graph attention over causal neighborhoods
    to produce per-node fraud predictions.

    This module handles the **embedding** and **classification** stages of the
    TGN pipeline. Memory management (GRU updates, message computation) is
    handled externally by the TGNMemory module during training.
    """

    def __init__(self, config: TemporalModelConfig) -> None:
        """Initialise the temporal fraud detection model.

        Args:
            config: Temporal model configuration with memory_dim, time_dim,
                    embedding_dim, and attention head settings.
        """
        super().__init__()
        self.config = config

        # Temporal graph attention over causal neighborhoods
        # Input: memory vectors (memory_dim), edge features include time encoding
        self.attention = TransformerConv(
            in_channels=config.memory_dim,
            out_channels=config.embedding_dim,
            heads=config.num_attention_heads,
            concat=False,
            edge_dim=config.time_dim + config.raw_msg_dim,
            dropout=config.dropout,
        )

        # MLP classifier: memory || attention_embedding [|| node_features] → fraud probability
        classifier_input_dim = config.memory_dim + config.embedding_dim
        if config.node_feature_dim > 0:
            classifier_input_dim += config.node_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim // 2, config.out_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass producing per-node logits.

        Args:
            memory: Node memory vectors of shape ``[N_batch, memory_dim]`` where
                    N_batch is the number of nodes in the sampled subgraph.
            edge_index: COO edge connectivity ``[2, E_batch]`` for the sampled
                        temporal neighborhood (relabeled to local indices).
            edge_attr: Edge features ``[E_batch, time_dim + raw_msg_dim]``
                       containing concatenated time encoding and raw message.
            node_features: Optional structural node features ``[N_batch, node_feature_dim]``.

        Returns:
            Raw logits of shape ``[N_batch, out_dim]``.
        """
        # Temporal graph attention over causal neighborhood
        attn_out = self.attention(memory, edge_index, edge_attr)

        # Concatenate memory + attention + optional node features
        parts = [memory, attn_out]
        if self.config.node_feature_dim > 0:
            if node_features is not None:
                parts.append(node_features)
            else:
                parts.append(torch.zeros(
                    memory.size(0), self.config.node_feature_dim,
                    device=memory.device,
                ))
        combined = torch.cat(parts, dim=-1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def predict_proba(
        self,
        memory: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class probabilities (softmax over logits)."""
        logits = self.forward(memory, edge_index, edge_attr, node_features)
        return F.softmax(logits, dim=-1)
