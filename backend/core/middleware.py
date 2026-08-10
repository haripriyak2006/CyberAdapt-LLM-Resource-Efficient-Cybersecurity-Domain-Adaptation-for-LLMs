"""
backend/core/middleware.py
Production middleware for CyberAdapt-LLM — Phase 9.

Provides:
  - RequestIDMiddleware  : attaches X-Request-ID to every request/response
  - RequestLoggingMiddleware : structured per-request access logs
  - In-memory metrics counter (request counts, error counts, latency histogram)
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


# ─────────────────────────────────────────────────────────────────────────────
# In-memory metrics store (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

class _MetricsStore:
    """Thread-safe-enough metrics counters for a single-process FastAPI server."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.total_errors: int = 0       # HTTP 4xx + 5xx
        self.requests_by_path: dict[str, int] = defaultdict(int)
        self.errors_by_path: dict[str, int] = defaultdict(int)
        self.latency_sum_ms: float = 0.0
        self.latency_count: int = 0

    def record(self, path: str, status_code: int, latency_ms: float) -> None:
        self.total_requests += 1
        self.requests_by_path[path] += 1
        self.latency_sum_ms += latency_ms
        self.latency_count += 1
        if status_code >= 400:
            self.total_errors += 1
            self.errors_by_path[path] += 1

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.start_time, 1)

    @property
    def mean_latency_ms(self) -> float:
        if self.latency_count == 0:
            return 0.0
        return round(self.latency_sum_ms / self.latency_count, 2)

    def snapshot(self) -> dict:
        return {
            "uptime_seconds":    self.uptime_seconds,
            "total_requests":    self.total_requests,
            "total_errors":      self.total_errors,
            "error_rate":        round(self.total_errors / max(self.total_requests, 1), 4),
            "mean_latency_ms":   self.mean_latency_ms,
            "requests_by_path":  dict(self.requests_by_path),
            "errors_by_path":    dict(self.errors_by_path),
        }


# Single global instance — imported by the metrics endpoint
metrics_store = _MetricsStore()


# ─────────────────────────────────────────────────────────────────────────────
# Request ID middleware
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique request ID to every request and response.

    - Reads X-Request-ID from incoming header if present (caller-supplied ID).
    - Otherwise generates a new UUID4.
    - Echoes the ID in the X-Request-ID response header.
    - Stores on request.state.request_id for use in route handlers / logs.
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[self.header_name] = request_id
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Request logging + metrics middleware
# ─────────────────────────────────────────────────────────────────────────────

import logging
_logger = logging.getLogger("cyberadapt.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured per-request access logging and metrics collection.

    Log format (one line per request):
      METHOD /path  status=200  latency=123ms  req_id=<uuid>
    """

    # Paths to skip (health polling noise)
    _SKIP_PATHS = frozenset({"/health", "/favicon.ico"})

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        path = request.url.path
        request_id = getattr(request.state, "request_id", "-")

        # Record metrics for all paths
        metrics_store.record(path, response.status_code, latency_ms)

        # Structured log (skip noisy health checks)
        if path not in self._SKIP_PATHS:
            _logger.info(
                "%s %s  status=%s  latency=%sms  req_id=%s  ip=%s",
                request.method,
                path,
                response.status_code,
                latency_ms,
                request_id,
                request.client.host if request.client else "-",
            )

        return response
