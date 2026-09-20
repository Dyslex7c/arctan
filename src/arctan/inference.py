"""Production inference module — scores entities by fraud probability.

Supports two scoring backends:

**Static** (:class:`FraudScorer`):
  Loads the static FraudGNN and pre-built graph. Runs a full forward pass
  over all nodes to score a single entity. Best for batch scoring.

**Temporal** (:class:`TemporalFraudScorer`):
  Loads the TGN model and entity memory state. Scores entities using their
  current memory vector and temporal neighborhood. Supports incremental
  updates via ``ingest_transaction()`` without reprocessing the full graph.

Design decisions:
  • Singleton pattern so the model stays warm across API requests.
  • Graceful fallback when the model/graph isn't trained yet.
  • Entity lookup by string ID for API ergonomics.
  • Auto-selects temporal scorer if temporal artifacts exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
import torch

from arctan.config import PipelineConfig, get_default_config
from arctan.data.graph_builder import load_graph
from arctan.models.fraud_gnn import FraudGNN

try:
    from arctan.models.explainer import FraudExplainer

    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False

logger = structlog.get_logger(__name__)


class FraudScorer:
    """Inference scorer for predicting entity-level fraud probability."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or get_default_config()
        self.device = torch.device("cpu")
        self.model: FraudGNN | None = None
        self.graph: Any = None
        self.node_to_idx: dict[str, int] = {}
        self.explainer: Any = None
        self.temperature_scaler: Any = None
        self._is_loaded = False

    def _load(self) -> None:
        """Lazy-load model and graph on first inference call."""
        if self._is_loaded:
            return

        if not self.config.paths.graph_path.exists():
            logger.warning(
                "Graph file not found. Run data pipeline and training first.",
                path=str(self.config.paths.graph_path),
            )
            self._is_loaded = True
            return

        try:
            logger.info("Loading graph for inference...")
            self.graph = load_graph(self.config).to(self.device)
            self.config.model.in_features = self.graph.num_node_features

            if hasattr(self.graph, "entity_id") and self.graph.entity_id:
                for i, eid in enumerate(self.graph.entity_id):
                    self.node_to_idx[eid] = i
            else:
                logger.warning("Graph has no entity_id attribute. ID lookup unavailable.")

            logger.info("Loading model for inference...")
            self.model = FraudGNN(self.config.model).to(self.device)
            if self.config.paths.best_model_path.exists():
                checkpoint = torch.load(
                    self.config.paths.best_model_path,
                    map_location=self.device,
                    weights_only=True,
                )
                # Support both nested dict and flat state_dict
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model.load_state_dict(
                        checkpoint["model"], strict=False
                    )
                    # Load temperature scaler if present
                    if "temperature" in checkpoint:
                        from arctan.models.calibration import (
                            TemperatureScaler,
                        )
                        self.temperature_scaler = TemperatureScaler()
                        self.temperature_scaler.load_state_dict(
                            checkpoint["temperature"]
                        )
                        logger.info(
                            "Temperature scaler loaded (T=%.4f)",
                            self.temperature_scaler.temperature_value,
                        )
                else:
                    self.model.load_state_dict(
                        checkpoint, strict=False
                    )
                logger.info("Trained model checkpoint loaded.")
            else:
                logger.warning("No checkpoint found. Using uninitialised weights.")

            self.model.eval()

            if HAS_EXPLAINER:
                self.explainer = FraudExplainer(self.model)
        except Exception as exc:
            logger.error("Failed to initialise inference engine", error=str(exc))

        self._is_loaded = True

    # Risk-level mapping
    @staticmethod
    def _get_risk_level(risk_score: int) -> str:
        """Map a 0–1000 risk score to a categorical risk level."""
        if risk_score >= 750:
            return "critical"
        if risk_score >= 500:
            return "high"
        if risk_score >= 250:
            return "medium"
        return "low"

    # Scoring
    def score_entity(self, entity_id: str) -> dict:
        """Score a single entity by its identifier."""
        self._load()

        # Fallback when model isn't ready
        if self.graph is None or not self.node_to_idx:
            return {
                "entity_id": entity_id,
                "risk_score": 500,
                "risk_level": "high",
                "confidence": 0.0,
                "fraud_probability": 0.5,
                "explanation": "Model not trained yet. Run the training pipeline first.",
            }

        node_idx = self.node_to_idx.get(entity_id)

        if node_idx is None:
            return {
                "entity_id": entity_id,
                "risk_score": 500,
                "risk_level": "high",
                "confidence": 0.1,
                "fraud_probability": 0.5,
                "explanation": "Entity not found in graph. Default score assigned.",
            }

        with torch.no_grad():
            outputs = self.model(
                self.graph.x, self.graph.edge_index, self.graph.edge_attr
            )
            if isinstance(outputs, dict):
                fraud_logits = outputs["fraud"]
                # Apply temperature scaling if available
                if self.temperature_scaler is not None:
                    scaled = self.temperature_scaler(fraud_logits)
                    p_fraud_cal = torch.softmax(
                        scaled, dim=-1
                    )[node_idx, 1].item()
                else:
                    p_fraud_cal = None
                p_fraud = torch.softmax(
                    fraud_logits, dim=-1
                )[node_idx, 1].item()
                p_ring = (
                    torch.softmax(
                        outputs["ring"], dim=-1
                    )[node_idx, 1].item()
                    if "ring" in outputs
                    else None
                )
            else:
                p_fraud = torch.softmax(
                    outputs, dim=-1
                )[node_idx, 1].item()
                p_fraud_cal = None
                p_ring = None

        # MC Dropout uncertainty estimation
        uncertainty_val = None
        if self.config.uncertainty.enabled and self.model is not None:
            try:
                from arctan.models.uncertainty import mc_dropout_predict

                def _forward_fn() -> torch.Tensor:
                    out = self.model(
                        self.graph.x,
                        self.graph.edge_index,
                        self.graph.edge_attr,
                    )
                    return out["fraud"] if isinstance(out, dict) else out

                unc = mc_dropout_predict(
                    self.model,
                    _forward_fn,
                    n_samples=self.config.uncertainty.mc_samples,
                )
                uncertainty_val = float(
                    unc["entropy"][node_idx].item()
                )
            except Exception as e:
                logger.warning(f"Uncertainty estimation failed: {e}")

        confidence = (
            max(0.0, min(1.0, 1.0 - uncertainty_val))
            if uncertainty_val is not None
            else 0.9
        )

        risk_score = int(p_fraud * 1000)
        risk_level = self._get_risk_level(risk_score)

        explanation = "Score derived from GNN graph topology and entity features."
        if HAS_EXPLAINER and self.explainer is not None:
            try:
                exp = self.explainer.explain_node(
                    node_idx, self.graph.x, self.graph.edge_index
                )
                if isinstance(exp, dict) and "summary" in exp:
                    explanation = exp["summary"].replace("\n", " ")
                else:
                    explanation = str(exp)
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")

        res = {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "fraud_probability": float(p_fraud),
            "explanation": explanation,
        }
        if p_fraud_cal is not None:
            res["calibrated_fraud_probability"] = float(p_fraud_cal)
        if uncertainty_val is not None:
            res["uncertainty"] = float(uncertainty_val)
        if p_ring is not None:
            res["ring_probability"] = float(p_ring)
            res["is_ring_member"] = bool(p_ring >= 0.5)
        return res

    def score_batch(self, entity_ids: list[str]) -> list[dict]:
        """Score multiple entities with a single batched forward pass.

        Much faster than calling ``score_entity`` in a loop because the
        GNN forward pass (the expensive part) runs only once.  MC Dropout
        uncertainty is skipped in batch mode for performance.
        """
        self._load()

        # Fallback when model isn't ready
        if self.graph is None or not self.node_to_idx:
            return [
                {
                    "entity_id": eid,
                    "risk_score": 500,
                    "risk_level": "high",
                    "confidence": 0.0,
                    "fraud_probability": 0.5,
                    "explanation": (
                        "Model not trained yet. "
                        "Run the training pipeline first."
                    ),
                }
                for eid in entity_ids
            ]

        # Resolve entity IDs to node indices
        found: list[tuple[int, int, str]] = []  # (position, idx, eid)
        not_found: list[tuple[int, str]] = []   # (position, eid)
        for pos, eid in enumerate(entity_ids):
            idx = self.node_to_idx.get(eid)
            if idx is not None:
                found.append((pos, idx, eid))
            else:
                not_found.append((pos, eid))

        results: list[dict | None] = [None] * len(entity_ids)

        # Fallback for unknown entities
        for pos, eid in not_found:
            results[pos] = {
                "entity_id": eid,
                "risk_score": 500,
                "risk_level": "high",
                "confidence": 0.1,
                "fraud_probability": 0.5,
                "explanation": (
                    "Entity not found in graph. "
                    "Default score assigned."
                ),
            }

        if not found:
            return results  # type: ignore[return-value]

        # Single forward pass
        node_indices = torch.tensor(
            [idx for _, idx, _ in found],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                self.graph.x,
                self.graph.edge_index,
                self.graph.edge_attr,
            )
            if isinstance(outputs, dict):
                fraud_logits = outputs["fraud"]
                fraud_probs = torch.softmax(fraud_logits, dim=-1)
                batch_p_fraud = fraud_probs[node_indices, 1]

                if self.temperature_scaler is not None:
                    scaled = self.temperature_scaler(fraud_logits)
                    cal_probs = torch.softmax(scaled, dim=-1)
                    batch_p_cal = cal_probs[node_indices, 1]
                else:
                    batch_p_cal = None

                if "ring" in outputs:
                    ring_probs = torch.softmax(
                        outputs["ring"], dim=-1
                    )
                    batch_p_ring = ring_probs[node_indices, 1]
                else:
                    batch_p_ring = None
            else:
                fraud_probs = torch.softmax(outputs, dim=-1)
                batch_p_fraud = fraud_probs[node_indices, 1]
                batch_p_cal = None
                batch_p_ring = None

        # Build result dicts
        explanation = (
            "Score derived from GNN graph topology "
            "and entity features."
        )
        for i, (pos, _idx, eid) in enumerate(found):
            p_fraud = float(batch_p_fraud[i].item())
            risk_score = int(p_fraud * 1000)
            res = {
                "entity_id": eid,
                "risk_score": risk_score,
                "risk_level": self._get_risk_level(risk_score),
                "confidence": 0.9,
                "fraud_probability": p_fraud,
                "explanation": explanation,
            }
            if batch_p_cal is not None:
                res["calibrated_fraud_probability"] = float(
                    batch_p_cal[i].item()
                )
            if batch_p_ring is not None:
                p_ring = float(batch_p_ring[i].item())
                res["ring_probability"] = p_ring
                res["is_ring_member"] = bool(p_ring >= 0.5)
            results[pos] = res

        return results  # type: ignore[return-value]


class TemporalFraudScorer:
    """Inference scorer using the Temporal Graph Network with entity memory.

    Unlike ``FraudScorer`` which requires a full forward pass over the entire
    static graph, this scorer uses pre-computed entity memory vectors that can
    be incrementally updated as new transactions arrive.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or get_default_config()
        self.device = torch.device("cpu")
        self.model: Any = None
        self.memory: Any = None
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}
        self.num_nodes: int = 0
        self._memory_vectors: torch.Tensor | None = None
        self.temperature_scaler: Any = None
        self._is_loaded = False

    def _load(self) -> None:
        """Lazy-load temporal model and memory state."""
        if self._is_loaded:
            return

        from arctan.models.temporal_gnn import TemporalFraudGNN

        tconfig = self.config.temporal_model

        if not self.config.paths.temporal_model_path.exists():
            logger.warning(
                "Temporal model not found. Run temporal training first.",
                path=str(self.config.paths.temporal_model_path),
            )
            self._is_loaded = True
            return

        try:
            # Load temporal data for entity ID mapping
            from arctan.data.graph_builder import load_temporal_data

            data_dict = load_temporal_data(self.config)
            entity_ids = data_dict.get("entity_ids", [])
            self.num_nodes = data_dict.get("num_nodes", 0)

            for i, eid in enumerate(entity_ids):
                self.node_to_idx[eid] = i
                self.idx_to_node[i] = eid

            # Load model checkpoint (supports both nested dict and flat state_dict)
            checkpoint = torch.load(
                self.config.paths.temporal_model_path,
                map_location=self.device,
                weights_only=True,
            )
            if "model" in checkpoint:
                tconfig.node_feature_dim = checkpoint.get(
                    "node_feature_dim", 0
                )
                self.model = TemporalFraudGNN(tconfig).to(self.device)
                self.model.load_state_dict(
                    checkpoint["model"], strict=False
                )
                # Load temperature scaler if present
                if "temperature" in checkpoint:
                    from arctan.models.calibration import (
                        TemperatureScaler,
                    )
                    self.temperature_scaler = TemperatureScaler()
                    self.temperature_scaler.load_state_dict(
                        checkpoint["temperature"]
                    )
                    logger.info(
                        "Temperature scaler loaded (T=%.4f)",
                        self.temperature_scaler.temperature_value,
                    )
            else:
                self.model = TemporalFraudGNN(tconfig).to(self.device)
                self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            logger.info("Temporal model loaded.")

            # Load memory state
            if self.config.paths.memory_state_path.exists():
                state = torch.load(
                    self.config.paths.memory_state_path,
                    weights_only=True,
                )
                self._memory_vectors = state["memory"].to(self.device)
                logger.info(
                    "Memory state loaded: %s", self._memory_vectors.shape
                )
            else:
                logger.warning("No memory state found. Using zero memory.")
                self._memory_vectors = torch.zeros(
                    self.num_nodes, tconfig.memory_dim, device=self.device
                )

        except Exception as exc:
            logger.error(
                "Failed to initialise temporal inference", error=str(exc)
            )

        self._is_loaded = True

    @staticmethod
    def _get_risk_level(risk_score: int) -> str:
        """Map a 0–1000 risk score to a categorical risk level."""
        if risk_score >= 750:
            return "critical"
        if risk_score >= 500:
            return "high"
        if risk_score >= 250:
            return "medium"
        return "low"

    def score_entity(self, entity_id: str) -> dict:
        """Score a single entity using its current memory state."""
        self._load()

        if self.model is None or self._memory_vectors is None:
            return {
                "entity_id": entity_id,
                "risk_score": 500,
                "risk_level": "high",
                "confidence": 0.0,
                "fraud_probability": 0.5,
                "model_type": "temporal",
                "explanation": "Temporal model not ready. Run training first.",
            }

        node_idx = self.node_to_idx.get(entity_id)
        if node_idx is None:
            return {
                "entity_id": entity_id,
                "risk_score": 500,
                "risk_level": "high",
                "confidence": 0.1,
                "fraud_probability": 0.5,
                "model_type": "temporal",
                "explanation": "Entity not found. Default score assigned.",
            }

        with torch.no_grad():
            mem = self._memory_vectors[node_idx].unsqueeze(0)
            tconfig = self.config.temporal_model

            edge_index = torch.tensor(
                [[0], [0]], dtype=torch.long, device=self.device
            )
            edge_feat = torch.zeros(
                1,
                tconfig.time_dim + tconfig.raw_msg_dim,
                device=self.device,
            )

            outputs = self.model(mem, edge_index, edge_feat)
            if isinstance(outputs, dict):
                fraud_logits = outputs["fraud"]
                ring_logits = outputs.get("ring")
                # Apply temperature scaling if available
                if self.temperature_scaler is not None:
                    scaled = self.temperature_scaler(fraud_logits)
                    p_fraud_cal = torch.softmax(
                        scaled, dim=-1
                    )[0, 1].item()
                else:
                    p_fraud_cal = None
            else:
                fraud_logits = outputs
                ring_logits = None
                p_fraud_cal = None

            probs = torch.softmax(fraud_logits, dim=-1)
            p_fraud = probs[0, 1].item()
            p_ring = (
                torch.softmax(ring_logits, dim=-1)[0, 1].item()
                if ring_logits is not None
                else None
            )

        # MC Dropout uncertainty estimation
        uncertainty_val = None
        if self.config.uncertainty.enabled and self.model is not None:
            try:
                from arctan.models.uncertainty import mc_dropout_predict

                def _forward_fn() -> torch.Tensor:
                    out = self.model(mem, edge_index, edge_feat)
                    return out["fraud"] if isinstance(out, dict) else out

                unc = mc_dropout_predict(
                    self.model,
                    _forward_fn,
                    n_samples=self.config.uncertainty.mc_samples,
                )
                uncertainty_val = float(unc["entropy"][0].item())
            except Exception as e:
                logger.warning(
                    f"Uncertainty estimation failed: {e}"
                )

        confidence = (
            max(0.0, min(1.0, 1.0 - uncertainty_val))
            if uncertainty_val is not None
            else 0.9
        )

        risk_score = int(p_fraud * 1000)
        res = {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "confidence": confidence,
            "fraud_probability": float(p_fraud),
            "model_type": "temporal",
            "explanation": (
                "Score from TGN entity memory and temporal attention."
            ),
        }
        if p_fraud_cal is not None:
            res["calibrated_fraud_probability"] = float(p_fraud_cal)
        if uncertainty_val is not None:
            res["uncertainty"] = float(uncertainty_val)
        if p_ring is not None:
            res["ring_probability"] = float(p_ring)
            res["is_ring_member"] = bool(p_ring >= 0.5)
        return res

    def ingest_transaction(
        self,
        src_id: str,
        dst_id: str,
        amount: float,
        txn_type: str = "TRANSFER",
    ) -> None:
        """Incrementally update entity memories with a new transaction.

        This is the key advantage of the temporal model over the static one:
        new transactions update the involved entities' memories without
        reprocessing the entire graph.

        Args:
            src_id: Sender entity ID.
            dst_id: Receiver entity ID.
            amount: Transaction amount.
            txn_type: Transaction type string.
        """
        self._load()

        if self._memory_vectors is None:
            logger.warning("Cannot ingest: memory not loaded.")
            return

        src_idx = self.node_to_idx.get(src_id)
        dst_idx = self.node_to_idx.get(dst_id)

        if src_idx is None or dst_idx is None:
            logger.warning(
                "Unknown entity in transaction",
                src=src_id,
                dst=dst_id,
                src_found=src_idx is not None,
                dst_found=dst_idx is not None,
            )
            return

        # Update memory vectors using a simple exponential moving average
        # (Full GRU update would require the TGNMemory module to be loaded)
        log_amount = float(np.log1p(amount))
        update_signal = torch.tensor(
            [log_amount], dtype=torch.float32, device=self.device
        )
        # Blend new information into memory (decay factor 0.1)
        decay = 0.1
        self._memory_vectors[src_idx] = (
            (1 - decay) * self._memory_vectors[src_idx]
            + decay * update_signal.expand_as(self._memory_vectors[src_idx])
        )
        self._memory_vectors[dst_idx] = (
            (1 - decay) * self._memory_vectors[dst_idx]
            + decay * update_signal.expand_as(self._memory_vectors[dst_idx])
        )

        logger.debug(
            "Memory updated",
            src=src_id,
            dst=dst_id,
            amount=amount,
        )

    def score_batch(self, entity_ids: list[str]) -> list[dict]:
        """Score multiple entities with a single batched forward pass.

        Stacks memory vectors for all requested entities and runs
        them through the model at once. MC Dropout uncertainty is
        skipped in batch mode for performance.
        """
        self._load()

        if self.model is None or self._memory_vectors is None:
            return [
                {
                    "entity_id": eid,
                    "risk_score": 500,
                    "risk_level": "high",
                    "confidence": 0.0,
                    "fraud_probability": 0.5,
                    "model_type": "temporal",
                    "explanation": (
                        "Temporal model not ready. "
                        "Run training first."
                    ),
                }
                for eid in entity_ids
            ]

        # Resolve entity IDs
        found: list[tuple[int, int, str]] = []
        not_found: list[tuple[int, str]] = []
        for pos, eid in enumerate(entity_ids):
            idx = self.node_to_idx.get(eid)
            if idx is not None:
                found.append((pos, idx, eid))
            else:
                not_found.append((pos, eid))

        results: list[dict | None] = [None] * len(entity_ids)

        for pos, eid in not_found:
            results[pos] = {
                "entity_id": eid,
                "risk_score": 500,
                "risk_level": "high",
                "confidence": 0.1,
                "fraud_probability": 0.5,
                "model_type": "temporal",
                "explanation": (
                    "Entity not found. Default score assigned."
                ),
            }

        if not found:
            return results  # type: ignore[return-value]

        n = len(found)
        tconfig = self.config.temporal_model

        # Stack memory vectors for all found entities
        mem_indices = [idx for _, idx, _ in found]
        batch_mem = self._memory_vectors[mem_indices]  # [n, dim]

        # Self-loop edges for each node in the batch
        node_ids_t = torch.arange(
            n, dtype=torch.long, device=self.device
        )
        edge_index = torch.stack(
            [node_ids_t, node_ids_t], dim=0
        )  # [2, n]
        edge_feat = torch.zeros(
            n,
            tconfig.time_dim + tconfig.raw_msg_dim,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(batch_mem, edge_index, edge_feat)
            if isinstance(outputs, dict):
                fraud_logits = outputs["fraud"]
                ring_logits = outputs.get("ring")
                if self.temperature_scaler is not None:
                    scaled = self.temperature_scaler(fraud_logits)
                    cal_probs = torch.softmax(
                        scaled, dim=-1
                    )[:, 1]
                else:
                    cal_probs = None
            else:
                fraud_logits = outputs
                ring_logits = None
                cal_probs = None

            fraud_probs = torch.softmax(
                fraud_logits, dim=-1
            )[:, 1]
            ring_probs = (
                torch.softmax(ring_logits, dim=-1)[:, 1]
                if ring_logits is not None
                else None
            )

        explanation = (
            "Score from TGN entity memory "
            "and temporal attention."
        )
        for i, (pos, _idx, eid) in enumerate(found):
            p_fraud = float(fraud_probs[i].item())
            risk_score = int(p_fraud * 1000)
            res = {
                "entity_id": eid,
                "risk_score": risk_score,
                "risk_level": self._get_risk_level(risk_score),
                "confidence": 0.9,
                "fraud_probability": p_fraud,
                "model_type": "temporal",
                "explanation": explanation,
            }
            if cal_probs is not None:
                res["calibrated_fraud_probability"] = float(
                    cal_probs[i].item()
                )
            if ring_probs is not None:
                p_ring = float(ring_probs[i].item())
                res["ring_probability"] = p_ring
                res["is_ring_member"] = bool(p_ring >= 0.5)
            results[pos] = res

        return results  # type: ignore[return-value]


# Module-level singletons
_scorer: FraudScorer | None = None
_temporal_scorer: TemporalFraudScorer | None = None


def get_scorer(
    config: PipelineConfig | None = None,
) -> FraudScorer | TemporalFraudScorer:
    """Return the appropriate scorer singleton.

    Auto-selects the temporal scorer if temporal model artifacts exist,
    otherwise falls back to the static scorer.
    """
    global _scorer, _temporal_scorer
    cfg = config or get_default_config()

    # Prefer temporal model if artifacts exist
    if cfg.model_type == "temporal" and cfg.paths.temporal_model_path.exists():
        if _temporal_scorer is None:
            _temporal_scorer = TemporalFraudScorer(cfg)
        return _temporal_scorer

    if _scorer is None:
        _scorer = FraudScorer(cfg)
    return _scorer

