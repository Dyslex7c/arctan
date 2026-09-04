"""Online entity memory module for the Temporal Graph Network.

Each entity maintains a d-dimensional memory vector that evolves over time
via GRU updates as new transactions arrive. The memory captures an entity's
behavioral trajectory — a fraud account that starts with normal activity and
then bursts into fan-out transfers will develop a different memory signature
than a purely legitimate account.

Builds on PyG's TGNMemory (Rossi et al., 2020) with:
  • IdentityMessage: concatenates [memory_src, memory_dst, edge_features, time_enc]
  • LastAggregator: keeps only the most recent message per node
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    TGNMemory,
)

from arctan.config import TemporalModelConfig


def build_entity_memory(
    num_nodes: int,
    config: TemporalModelConfig,
) -> TGNMemory:
    """Construct a TGNMemory module with Arctan's default message and aggregator.

    The memory module stores a ``memory_dim``-dimensional state vector per node,
    updated via GRU whenever a transaction event involves that node. The update
    message is formed by concatenating the source memory, destination memory,
    raw edge features, and a learnable time encoding of the elapsed time since
    the node's last update.

    Args:
        num_nodes: Total number of entities in the graph.
        config: Temporal model configuration.

    Returns:
        A configured ``TGNMemory`` instance ready for training.
    """
    msg_module = IdentityMessage(
        raw_msg_dim=config.raw_msg_dim,
        memory_dim=config.memory_dim,
        time_dim=config.time_dim,
    )
    aggr_module = LastAggregator()

    return TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=config.raw_msg_dim,
        memory_dim=config.memory_dim,
        time_dim=config.time_dim,
        message_module=msg_module,
        aggregator_module=aggr_module,
    )


def save_memory_state(
    memory: TGNMemory,
    path: str | Path,
) -> None:
    """Persist the current memory state (vectors + last-update timestamps).

    Saved state can be loaded at inference time so new transactions can
    incrementally update entity memories without reprocessing the full history.

    Args:
        memory: The TGNMemory module whose state to save.
        path: Filesystem path to write the state dict.
    """
    state = {
        "memory": memory.memory.detach().cpu(),
        "last_update": memory.last_update.detach().cpu(),
    }
    torch.save(state, path)


def load_memory_state(
    memory: TGNMemory,
    path: str | Path,
) -> None:
    """Restore a previously saved memory state.

    Args:
        memory: The TGNMemory module to restore into.
        path: Filesystem path of the saved state dict.
    """
    state = torch.load(path, weights_only=True)
    memory.memory.copy_(state["memory"].to(memory.memory.device))
    memory.last_update.copy_(state["last_update"].to(memory.last_update.device))
