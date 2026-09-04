"""Learnable time encoder for temporal graph networks.

Maps scalar elapsed-time values Δt into d-dimensional embeddings via
``cos(W · Δt + b)`` with learnable frequency W and phase b. This allows
the TGN to distinguish temporal patterns — e.g. a burst of 30 transactions
in one hour versus 30 transactions spread across 30 days.

Reference: Xu et al., "Inductive Representation Learning on Temporal Graphs" (2020).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Maps scalar Δt values to d-dimensional time embeddings.

    Uses a single linear projection followed by a cosine activation:
    ``time_embedding(Δt) = cos(W · Δt + b)`` where W ∈ R^{1×d} and b ∈ R^d
    are learnable parameters.
    """

    def __init__(self, out_channels: int) -> None:
        """Initialise the time encoder.

        Args:
            out_channels: Dimensionality of the time embedding (d).
        """
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)

    def reset_parameters(self) -> None:
        """Re-initialise learnable weights."""
        self.lin.reset_parameters()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode elapsed-time scalars into d-dimensional embeddings.

        Args:
            t: Tensor of elapsed-time values, any shape. Each element is
               a scalar Δt (e.g. ``t_current - t_last_update``).

        Returns:
            Time embeddings of shape ``[*t.shape, out_channels]``.
        """
        return self.lin(t.view(-1, 1)).cos()
