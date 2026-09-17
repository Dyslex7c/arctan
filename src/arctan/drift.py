"""
This module implements distribution drift detection for the Arctan fraud detection system.
It detects distribution drift between training and inference-time data using Population
Stability Index (PSI) and the Kolmogorov-Smirnov (KS) test.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ks_2samp

from arctan.config import DriftConfig

logger = logging.getLogger(__name__)


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute the Population Stability Index (PSI) between two 1-D arrays.
    """
    min_val = min(np.min(reference), np.min(current))
    max_val = max(np.max(reference), np.max(current))
    
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)
    
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)
    
    epsilon = 1e-6
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)
    
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Perform a two-sample Kolmogorov-Smirnov test.
    """
    ks_stat, p_value = ks_2samp(reference, current)
    return {"statistic": float(ks_stat), "p_value": float(p_value)}


def detect_feature_drift(
    ref_features: np.ndarray,
    cur_features: np.ndarray,
    feature_names: list[str],
    config: DriftConfig
) -> dict:
    """
    Detect drift per feature using PSI and KS-test.
    """
    features_drift = {}
    num_drifted = 0
    total_features = len(feature_names)
    
    for i, feature_name in enumerate(feature_names):
        ref_col = ref_features[:, i]
        cur_col = cur_features[:, i]
        
        psi_val = compute_psi(ref_col, cur_col, num_bins=config.num_bins)
        ks_res = compute_ks_test(ref_col, cur_col)
        
        psi_drifted = psi_val > config.psi_threshold
        ks_drifted = ks_res["p_value"] < config.ks_alpha
        is_drifted = bool(psi_drifted or ks_drifted)
        
        features_drift[feature_name] = {
            "psi": psi_val,
            "ks_statistic": ks_res["statistic"],
            "ks_p_value": ks_res["p_value"],
            "drifted": is_drifted,
        }
        
        if is_drifted:
            num_drifted += 1
            
    summary = f"{num_drifted}/{total_features} features show significant drift"
    
    return {
        "features": features_drift,
        "num_drifted": num_drifted,
        "total_features": total_features,
        "summary": summary,
    }


def detect_prediction_drift(
    ref_probs: np.ndarray,
    cur_probs: np.ndarray,
    config: DriftConfig
) -> dict:
    """
    Detect drift on the fraud probability distribution.
    """
    psi_val = compute_psi(ref_probs, cur_probs, num_bins=config.num_bins)
    ks_res = compute_ks_test(ref_probs, cur_probs)
    
    psi_drifted = psi_val > config.psi_threshold
    ks_drifted = ks_res["p_value"] < config.ks_alpha
    is_drifted = bool(psi_drifted or ks_drifted)
    
    return {
        "psi": psi_val,
        "ks_statistic": ks_res["statistic"],
        "ks_p_value": ks_res["p_value"],
        "drifted": is_drifted,
    }


def generate_drift_report(
    feature_drift: dict,
    prediction_drift: dict,
    save_path: str
) -> None:
    """
    Write a text file summary of the drift detection results.
    """
    with open(save_path, "w") as f:
        f.write("Drift Detection Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Feature Drift Analysis\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Feature Name':<30} | {'PSI':<10} | {'KS Stat':<10} | "
            f"{'p-value':<10} | {'Drifted'}\n"
        )
        f.write("-" * 80 + "\n")
        
        for feature_name, metrics in feature_drift["features"].items():
            psi = f"{metrics['psi']:.4f}"
            ks_stat = f"{metrics['ks_statistic']:.4f}"
            p_val = f"{metrics['ks_p_value']:.4e}"
            drifted = str(metrics['drifted'])
            
            f.write(
                f"{feature_name:<30} | {psi:<10} | {ks_stat:<10} | "
                f"{p_val:<10} | {drifted}\n"
            )
            
        f.write("-" * 80 + "\n")
        f.write(f"Summary: {feature_drift['summary']}\n\n")
        
        f.write("Prediction Drift Analysis\n")
        f.write("-" * 80 + "\n")
        f.write(f"PSI:        {prediction_drift['psi']:.4f}\n")
        f.write(f"KS Stat:    {prediction_drift['ks_statistic']:.4f}\n")
        f.write(f"KS p-value: {prediction_drift['ks_p_value']:.4e}\n")
        f.write(f"Drifted:    {prediction_drift['drifted']}\n")
