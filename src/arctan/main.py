"""FastAPI microservice for GNN-based entity risk scoring.

Endpoints:
  • ``GET  /healthz``                          liveness probe
  • ``GET  /readyz``                           readiness probe
  • ``GET  /api/v1/scores/{entity_id}``        score a single entity
  • ``POST /api/v1/scores/batch``              batch scoring (≤100)
  • ``POST /api/v1/transactions/ingest``       ingest a transaction
"""

from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from arctan.config import ServerSettings
from arctan.inference import get_scorer


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Envelope[T](BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    data: T
    message: str | None = None
    timestamp: str


def ok[T](data: T, message: str | None = None) -> Envelope[T]:
    return Envelope(data=data, message=message, timestamp=_now_iso())


class EntityScore(CamelModel):
    """Single entity risk score response."""

    entity_id: str
    risk_score: int = Field(..., ge=0, le=1000)
    risk_level: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    ring_probability: float | None = None
    is_ring_member: bool | None = None
    calibrated_fraud_probability: float | None = None
    uncertainty: float | None = None


class BatchScoreRequest(BaseModel):
    """Request body for batch scoring."""

    entity_ids: list[str] = Field(..., min_length=1, max_length=100)


class TransactionRequest(CamelModel):
    """Request body for transaction ingestion."""

    src_id: str
    dst_id: str
    amount: float = Field(..., gt=0)
    txn_type: str = "TRANSFER"


def create_app() -> FastAPI:
    """Application factory."""
    server_settings = ServerSettings()

    app = FastAPI(
        title="Arctan ML Scoring Service",
        version="0.1.0",
        description=(
            "Decoupled microservice providing GNN-based entity risk scores "
            "and fraud probability estimates."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["infra"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "arctan-ml"}

    @app.get("/readyz", tags=["infra"])
    async def readyz() -> dict[str, str]:
        scorer = get_scorer()
        is_ready = scorer.config.paths.graph_path.exists()
        return {
            "status": "ok" if is_ready else "not_trained",
            "graph_exists": str(is_ready),
        }

    @app.get(
        "/api/v1/scores/{entity_id}",
        response_model=Envelope[EntityScore],
        tags=["scores"],
        summary="Score a single entity",
    )
    def score_entity(entity_id: str) -> Envelope[EntityScore]:
        scorer = get_scorer()
        result = scorer.score_entity(entity_id)
        return ok(EntityScore(**result))

    @app.post(
        "/api/v1/scores/batch",
        response_model=Envelope[list[EntityScore]],
        tags=["scores"],
        summary="Score multiple entities in batch",
    )
    def score_batch(body: BatchScoreRequest) -> Envelope[list[EntityScore]]:
        scorer = get_scorer()
        results = scorer.score_batch(body.entity_ids)
        return ok([EntityScore(**r) for r in results])

    @app.post(
        "/api/v1/transactions/ingest",
        response_model=Envelope[dict],
        tags=["transactions"],
        summary="Ingest a transaction to update entity memory",
    )
    def ingest_transaction(
        body: TransactionRequest,
    ) -> Envelope[dict]:
        scorer = get_scorer()
        if hasattr(scorer, "ingest_transaction"):
            scorer.ingest_transaction(
                body.src_id,
                body.dst_id,
                body.amount,
                body.txn_type,
            )
            return ok(
                {"status": "ingested"},
                message="Entity memories updated.",
            )
        return ok(
            {"status": "skipped"},
            message=(
                "Static model does not support "
                "live transaction ingestion."
            ),
        )

    return app


app = create_app()

if __name__ == "__main__":
    settings = ServerSettings()
    uvicorn.run(
        "arctan.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
