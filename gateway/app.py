"""Async API gateway that proxies scoring requests to the Arctan ML service.

Provides degradation with fallback default scores when the ML service is unavailable,
so upstream consumers never receive an error.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from gateway.config import get_settings

logger = structlog.get_logger(__name__)


# schemas
def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Envelope[T](BaseModel):
    success: bool = True
    data: T
    message: str | None = None
    timestamp: str


def ok[T](data: T, message: str | None = None) -> Envelope[T]:
    return Envelope(data=data, message=message, timestamp=_now_iso())


class EntityScore(CamelModel):
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
    entity_ids: list[str] = Field(..., min_length=1, max_length=100)


def _fallback_score(entity_id: str, reason: str) -> EntityScore:
    return EntityScore(
        entity_id=entity_id,
        risk_score=500,
        risk_level="high",
        confidence=0.0,
        fraud_probability=0.5,
        explanation=reason,
    )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Arctan API Gateway",
        version="0.1.0",
        description="API gateway with async httpx proxy to the Arctan ML scoring service.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", tags=["infra"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "arctan-gateway"}

    @app.get(
        "/api/v1/scores/{entity_id}",
        response_model=Envelope[EntityScore],
        tags=["scores"],
        summary="Score a single entity",
        description="Proxies to the ML microservice with fallback.",
    )
    async def score_entity(entity_id: str) -> Envelope[EntityScore]:
        url = f"{settings.ml_service_url.rstrip('/')}/api/v1/scores/{entity_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    body = response.json()
                    data = body.get("data", body)
                    return ok(EntityScore.model_validate(data))
                else:
                    logger.warning(
                        "ml_service_non_200",
                        status_code=response.status_code,
                        text=response.text,
                    )
        except Exception as exc:
            logger.warning(
                "ml_service_unreachable",
                entity_id=entity_id,
                url=url,
                error=repr(exc),
            )

        return ok(_fallback_score(entity_id, "ML service unavailable. Fallback score assigned."))

    @app.post(
        "/api/v1/scores/batch",
        response_model=Envelope[list[EntityScore]],
        tags=["scores"],
        summary="Score multiple entities in batch",
        description="Proxies batch requests to the ML microservice.",
    )
    async def score_batch(body: BatchScoreRequest) -> Envelope[list[EntityScore]]:
        url = f"{settings.ml_service_url.rstrip('/')}/api/v1/scores/batch"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url, json={"entity_ids": body.entity_ids}
                )
                if response.status_code == 200:
                    res_json = response.json()
                    data_list = res_json.get("data", res_json)
                    return ok([EntityScore.model_validate(r) for r in data_list])
                else:
                    logger.warning(
                        "ml_service_batch_non_200",
                        status_code=response.status_code,
                        text=response.text,
                    )
        except Exception as exc:
            logger.warning(
                "ml_service_batch_unreachable",
                url=url,
                count=len(body.entity_ids),
                error=repr(exc),
            )

        fallbacks = [
            _fallback_score(eid, "ML service unavailable. Fallback score assigned.")
            for eid in body.entity_ids
        ]
        return ok(fallbacks)

    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "gateway.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
