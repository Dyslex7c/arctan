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
                self.model.load_state_dict(
                    torch.load(
                        self.config.paths.best_model_path,
                        map_location=self.device,
                        weights_only=True,
                    )
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
            probs = self.model.predict_proba(
                self.graph.x, self.graph.edge_index, self.graph.edge_attr
            )
            p_fraud = probs[node_idx, 1].item()

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

        return {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": 0.9,
            "fraud_probability": float(p_fraud),
            "explanation": explanation,
        }

    def score_batch(self, entity_ids: list[str]) -> list[dict]:
        """Score multiple entities."""
        return [self.score_entity(eid) for eid in entity_ids]


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

            # Load model
            self.model = TemporalFraudGNN(tconfig).to(self.device)
            self.model.load_state_dict(
                torch.load(
                    self.config.paths.temporal_model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )
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
            # Use the entity's memory vector directly through the classifier
            # For single-entity scoring, we skip the attention layer and use
            # the memory-only path (no neighborhood to attend over)
            mem = self._memory_vectors[node_idx].unsqueeze(0)
            tconfig = self.config.temporal_model

            # Create a dummy self-loop edge for the attention layer
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
            edge_feat = torch.zeros(
                1,
                tconfig.time_dim + tconfig.raw_msg_dim,
                device=self.device,
            )

            logits = self.model(mem, edge_index, edge_feat)
            probs = torch.softmax(logits, dim=-1)
            p_fraud = probs[0, 1].item()

        risk_score = int(p_fraud * 1000)
        return {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "confidence": 0.9,
            "fraud_probability": float(p_fraud),
            "model_type": "temporal",
            "explanation": "Score from TGN entity memory and temporal attention.",
        }

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
        """Score multiple entities."""
        return [self.score_entity(eid) for eid in entity_ids]


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

