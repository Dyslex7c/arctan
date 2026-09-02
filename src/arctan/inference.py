"""Production inference module — scores entities by fraud probability.

The :class:`FraudScorer` lazy-loads the trained model and graph on first
call, converts the GNN's per-node fraud probability into a 0-1000 risk
score, and optionally generates a GNNExplainer attribution summary.

Design decisions:
  • Singleton pattern so the model stays warm across API requests.
  • Graceful fallback when the model/graph isn't trained yet.
  • Entity lookup by string ID for API ergonomics.
"""

from typing import Any

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


# Module-level singleton
_scorer: FraudScorer | None = None


def get_scorer() -> FraudScorer:
    """Return the singleton FraudScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer()
    return _scorer
