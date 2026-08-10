"""
backend/main.py
CyberAdapt-LLM  ·  Phase 9 — Production-Style FastAPI Application.

Responsibilities:
  - Application factory with lifespan context
  - Production middleware (Request ID, access logging, metrics)
  - CORS configuration
  - Structured global exception handlers
  - Router registration (all Phase 1–9 endpoints)
  - Configurable API prefix (/api)
"""

from __future__ import annotations

import datetime
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import health as health_router
from backend.api import chat as chat_router
from backend.api import rag as rag_router
from backend.api import threat as threat_router
from backend.api import vuln as vuln_router
from backend.api import document as document_router
from backend.api import report as report_router
from backend.api import cyber_chat as cyber_chat_router
from backend.api import model_info as model_info_router
from backend.api import evaluation as evaluation_router
from backend.api import system_metrics as metrics_router
from backend.core.config import get_settings
from backend.core.logging_config import configure_logging
from backend.core.middleware import RequestIDMiddleware, RequestLoggingMiddleware

# ── Logging setup ─────────────────────────────────────────────────────────────
settings = get_settings()
configure_logging(log_level=settings.app_log_level, env=settings.app_env)
logger = logging.getLogger(__name__)

# Phase 9 uses /api (without version suffix)
_API_PREFIX = "/api"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown hooks."""
    logger.info(
        "Starting %s v%s  [Phase 9]  env=%s",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    logger.info("API listening on %s:%s", settings.app_host, settings.app_port)
    logger.info("API prefix: %s", _API_PREFIX)
    logger.info("Docs: http://%s:%s/docs", settings.app_host, settings.app_port)

    yield  # application runs here

    logger.info("Shutting down %s", settings.app_name)


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Resource-Efficient Cybersecurity Domain Adaptation for LLMs. "
            "Phase 9 — Production-style backend with threat analysis, vulnerability analysis, "
            "RAG-grounded Q&A, security report generation, and cybersecurity chat."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost runs first) ─────────────────────
    # 1. Request ID (must be first so all downstream has the ID)
    app.add_middleware(RequestIDMiddleware)
    # 2. Access logging + metrics
    app.add_middleware(RequestLoggingMiddleware)
    # 3. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # ── Structured error handlers ─────────────────────────────────────────────

    def _error_body(request: Request, error: str, detail: str, status: int) -> dict:
        return {
            "error":      error,
            "detail":     detail,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        # Build a human-readable message
        msgs = [
            f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
            for e in errors
        ]
        detail = "; ".join(msgs)
        logger.warning("Validation error [%s]: %s", request.url.path, detail)
        return JSONResponse(
            status_code=422,
            content=_error_body(request, "ValidationError", detail, 422),
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_body(
                request, "NotFound",
                f"Endpoint not found: {request.method} {request.url.path}", 404
            ),
        )

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=405,
            content=_error_body(
                request, "MethodNotAllowed",
                f"Method {request.method} not allowed for {request.url.path}", 405
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception [%s]: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content=_error_body(
                request, "InternalServerError",
                "An unexpected error occurred. Check server logs.", 500
            ),
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Health (no prefix — stays at /health)
    app.include_router(health_router.router)

    # Phase 2 — Base LLM chat
    app.include_router(chat_router.router,        prefix=_API_PREFIX)   # POST /api/chat

    # Phase 7 — RAG
    app.include_router(rag_router.router,         prefix=_API_PREFIX)   # POST /api/rag/query
                                                                         # GET  /api/rag/status
    # Phase 8 — Analysis features
    app.include_router(threat_router.router,      prefix=_API_PREFIX)   # POST /api/threat/analyze
    app.include_router(vuln_router.router,        prefix=_API_PREFIX)   # POST /api/vuln/analyze
    app.include_router(document_router.router,    prefix=_API_PREFIX)   # POST /api/document/analyze
                                                                         # POST /api/document/upload
    app.include_router(report_router.router,      prefix=_API_PREFIX)   # POST /api/report/generate
    app.include_router(cyber_chat_router.router,  prefix=_API_PREFIX)   # POST /api/cyber/chat

    # Phase 9 — System endpoints
    app.include_router(model_info_router.router,  prefix=_API_PREFIX)   # GET  /api/model/info
    app.include_router(evaluation_router.router,  prefix=_API_PREFIX)   # GET  /api/evaluation/results
    app.include_router(metrics_router.router,     prefix=_API_PREFIX)   # GET  /api/metrics

    # Compatibility alias: /api/vulnerability/analyze → same router as vuln
    # (Phase 9 spec uses 'vulnerability' instead of 'vuln')
    from fastapi import APIRouter
    _alias = APIRouter(tags=["vulnerability-analysis"])

    from backend.schemas.vuln import VulnRequest, VulnResponse
    from backend.core.safety import SafetyError

    @_alias.post(
        "/vulnerability/analyze",
        response_model=VulnResponse,
        summary="Vulnerability Analysis (canonical path)",
        description="Alias for /api/vuln/analyze — Phase 9 canonical endpoint.",
    )
    async def vuln_alias(request: VulnRequest) -> VulnResponse:
        _logger = logging.getLogger(__name__)
        try:
            from backend.services.vuln_service import analyze_vulnerability
            result = analyze_vulnerability(request.description, top_k=request.top_k)
        except SafetyError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
        except Exception as exc:
            _logger.exception("Vuln analysis error: %s", exc)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Internal server error.")
        return VulnResponse(**result)

    app.include_router(_alias, prefix=_API_PREFIX)                       # POST /api/vulnerability/analyze

    return app


app = create_app()
