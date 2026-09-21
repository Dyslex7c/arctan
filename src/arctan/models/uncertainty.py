"""
This module provides epistemic uncertainty estimation via Monte Carlo Dropout,
allowing the model to quantify its own confidence.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@contextmanager
def enable_mc_dropout(model: nn.Module):
    """
    Context manager to enable Monte Carlo Dropout during inference.
    Sets the model to train mode to activate dropout layers, but freezes
    BatchNorm layers in eval mode to prevent updating running statistics.
    """
    is_training = model.training
    model.train()
    
    # Keep BatchNorm in eval mode (covers nn.BatchNorm* and PyG BatchNorm)
    for m in model.modules():
        if hasattr(m, "running_mean"):
            m.eval()
            
    try:
        yield model
    finally:
        if not is_training:
            model.eval()


def predictive_entropy(mean_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy from mean probabilities.
    
    Args:
        mean_probs: Mean probabilities [N, C].
        
    Returns:
        Entropy [N].
    """
    eps = 1e-10
    return -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)


def mc_dropout_predict(
    model: nn.Module, forward_fn: Callable[[], torch.Tensor], n_samples: int = 30
) -> dict[str, torch.Tensor]:
    """
    Perform Monte Carlo Dropout predictions.
    
    Args:
        model: PyTorch model.
        forward_fn: Callable that performs one forward pass and returns logits [N, C].
        n_samples: Number of MC dropout samples.
        
    Returns:
        Dictionary containing mean probabilities, variance, predictive entropy, and all samples.
    """
    with enable_mc_dropout(model):
        samples_list = []
        for _ in range(n_samples):
            logits = forward_fn()
            probs = F.softmax(logits, dim=-1)
            samples_list.append(probs)
            
    samples = torch.stack(samples_list, dim=0)  # [n_samples, N, C]
    mean_probs = torch.mean(samples, dim=0)     # [N, C]
    
    # Variance of the positive class probabilities (index 1 if available, else 0)
    pos_class_idx = 1 if mean_probs.shape[-1] > 1 else 0
    variance = torch.var(samples[..., pos_class_idx], dim=0)
    
    entropy = predictive_entropy(mean_probs)
    
    return {
        "mean": mean_probs,
        "variance": variance,
        "entropy": entropy,
        "samples": samples,
    }
