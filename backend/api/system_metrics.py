"""
backend/api/system_metrics.py
GET /api/metrics — Live server metrics.
"""
from __future__ import annotations
import logging
from fastapi import APIRouter
from backend.schemas.common import MetricsResponse
from backend.core.middleware import metrics_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System Metrics",
    description="Returns live server metrics: uptime, request counts, error rate, latency, model status.",
)
async def system_metrics() -> MetricsResponse:
    snap = metrics_store.snapshot()

    # Model loaded?
    model_loaded = False
    try:
        from backend.services.llm_service import _loader
        model_loaded = _loader is not None and _loader._model is not None
    except Exception:
        pass

    # RAG loaded?
    rag_loaded = False
    rag_chunks = 0
    try:
        from backend.services.rag_service import _vector_store
        if _vector_store is not None:
            rag_loaded = True
            rag_chunks = _vector_store.num_chunks
    except Exception:
        pass

    return MetricsResponse(
        uptime_seconds=snap["uptime_seconds"],
        total_requests=snap["total_requests"],
        total_errors=snap["total_errors"],
        error_rate=snap["error_rate"],
        mean_latency_ms=snap["mean_latency_ms"],
        requests_by_path=snap["requests_by_path"],
        errors_by_path=snap["errors_by_path"],
        model_loaded=model_loaded,
        rag_loaded=rag_loaded,
        rag_chunks=rag_chunks,
    )
