"""
This module implements Platt-style temperature scaling for post-hoc calibration
and related calibration metrics for the Arctan fraud detection system.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arctan.config import CalibrationConfig

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """
    A single learnable temperature parameter for temperature scaling calibration.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies temperature scaling to the logits."""
        return logits / self.temperature

    @property
    def temperature_value(self) -> float:
        """Returns the current temperature as a float."""
        return self.temperature.item()


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: CalibrationConfig
) -> TemperatureScaler:
    """
    Fits the temperature scaler using LBFGS optimizer on the provided validation logits and labels.
    """
    scaler = TemperatureScaler(
        init_temperature=config.temperature_init
    )
    optimizer = torch.optim.LBFGS(
        [scaler.temperature],
        lr=config.temperature_lr,
        max_iter=config.temperature_epochs,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    logger.info(f"Fitted temperature: {scaler.temperature_value:.4f}")
    return scaler


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    # Use right=False so that 0 is included in the first bin if probs is [0, 1]
    bin_indices = np.digitize(probs, bins, right=False)

    ece = 0.0
    for b in range(1, num_bins + 1):
        # np.digitize returns indices 1 to len(bins)-1 for values within range
        mask = bin_indices == b
        # include probs == 1.0 in the last bin
        if b == num_bins:
            mask = mask | (bin_indices == b + 1)
            
        if np.any(mask):
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            bin_weight = np.sum(mask) / len(probs)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the Brier score.
    """
    return float(np.mean((probs - labels) ** 2))


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
    save_path: str
) -> None:
    """
    Plots a reliability diagram and saves it to the specified path.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(probs, bins, right=False)

    mean_probs = []
    frac_pos = []

    for b in range(1, num_bins + 1):
        mask = bin_indices == b
        if b == num_bins:
            mask = mask | (bin_indices == b + 1)
            
        if np.any(mask):
            mean_probs.append(probs[mask].mean())
            frac_pos.append(labels[mask].mean())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.bar(
        mean_probs,
        frac_pos,
        width=1/num_bins,
        edgecolor='black',
        alpha=0.7,
        label='Outputs'
    )
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
