"""
backend/api/health.py
GET /health  —  liveness probe for CyberAdapt-LLM.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.config import get_settings

router = APIRouter(tags=["health"])

# Record the time the process started so uptime can be reported
_START_TIME: float = time.time()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    phase: int
    uptime_seconds: float
    environment: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service liveness status. Use this to verify the API is running.",
)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe — no heavy I/O, always fast."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        phase=settings.app_phase,
        uptime_seconds=round(time.time() - _START_TIME, 2),
        environment=settings.app_env,
    )
