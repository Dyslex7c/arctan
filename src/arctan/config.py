"""Hyperparameter configuration and path settings for the Arctan GNN pipeline.

All tuneable values live here so that data ingestion, training, evaluation,
and inference share a single source of truth.  Paths default to ``data/``
relative to the project root directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:

    class BaseSettings(BaseModel):  # type: ignore[no-redef]
        pass

    def SettingsConfigDict(**kwargs):  # type: ignore[misc]
        return kwargs


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent


# Path configuration
class PathConfig(BaseModel):
    """Filesystem paths for raw data, processed artefacts, and model checkpoints."""

    data_root: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data")
    raw_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    transactions_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "raw" / "transactions"
    )
    processed_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "processed")
    models_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "models")

    # Static model artefacts
    graph_file: str = "fraud_graph.pt"
    best_model_file: str = "fraud_gnn_best.pt"

    # Temporal model artefacts
    temporal_graph_file: str = "temporal_data.pt"
    temporal_model_file: str = "temporal_gnn_best.pt"
    memory_state_file: str = "entity_memory_state.pt"

    def ensure_dirs(self) -> None:
        """Create every configured directory if it doesn't exist."""
        for d in (
            self.data_root,
            self.raw_dir,
            self.transactions_dir,
            self.processed_dir,
            self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def graph_path(self) -> Path:
        return self.processed_dir / self.graph_file

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / self.best_model_file

    @property
    def temporal_graph_path(self) -> Path:
        return self.processed_dir / self.temporal_graph_file

    @property
    def temporal_model_path(self) -> Path:
        return self.models_dir / self.temporal_model_file

    @property
    def memory_state_path(self) -> Path:
        return self.models_dir / self.memory_state_file


# Model architecture
class ModelConfig(BaseModel):
    """Architecture hyper-parameters for the hybrid GraphSAGE + GATv2 model."""

    in_features: int = 16          # set dynamically from graph.num_node_features
    hidden_dim: int = 128
    out_dim: int = 2               # binary: legitimate / fraudulent
    sage_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.3
    aggr: Literal["mean", "max", "add"] = "mean"
    edge_dim: int | None = 2       # edge features: [log_amount, norm_timestamp]


# Temporal GNN architecture (TGN)
class TemporalModelConfig(BaseModel):
    """Architecture parameters for the Temporal Graph Network (TGN).

    The TGN maintains per-entity memory vectors updated via GRU as transactions
    arrive chronologically, enabling time-aware fraud pattern recognition.
    """

    memory_dim: int = 100          # GRU hidden state per entity
    time_dim: int = 100            # learnable time encoding dimension
    embedding_dim: int = 128       # final node embedding before classifier
    num_attention_heads: int = 2   # heads in temporal graph attention
    num_neighbors: int = 20        # K most recent neighbors per node
    raw_msg_dim: int = 2           # edge event features: [log_amount, txn_type]
    node_feature_dim: int = 0      # structural node features (set to 16 when available)
    dropout: float = 0.1
    out_dim: int = 2               # binary: legitimate / fraudulent
    max_class_weight_ratio: float = 10.0  # cap positive class weight


# Training
class TrainingConfig(BaseModel):
    """Training hyper-parameters."""

    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 1024
    num_epochs: int = 200
    patience: int = 20              # early-stopping patience (epochs)
    num_neighbors: list[int] = Field(default_factory=lambda: [15, 10])
    focal_loss_gamma: float = 2.0   # 0 → standard CE
    device: str = "cpu"
    seed: int = 42


# Inference & Server Settings
class InferenceConfig(BaseModel):
    """Settings for inference scoring."""

    device: str = "cpu"
    score_scale: int = 1000         # maps P(fraud) to 0–1000 risk score


class ServerSettings(BaseSettings):
    """Server runtime configuration loaded from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8001
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8000"]
    )


# Aggregate config
class PipelineConfig(BaseModel):
    """Top-level configuration container."""

    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    temporal_model: TemporalModelConfig = Field(default_factory=TemporalModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    model_type: Literal["static", "temporal"] = "temporal"


def get_default_config() -> PipelineConfig:
    """Return a ``PipelineConfig`` with sensible defaults."""
    return PipelineConfig()
