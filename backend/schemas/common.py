"""
backend/schemas/common.py
Shared Pydantic schemas used across Phase 9 endpoints.

Provides:
  - ErrorResponse     : structured error envelope (all 4xx/5xx)
  - HealthResponse    : GET /health
  - ModelInfoResponse : GET /api/model/info
  - MetricsResponse   : GET /api/metrics
"""

from __future__ import annotations

import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Structured error envelope
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Consistent error format for all 4xx/5xx responses."""

    error:      str = Field(..., description="Short error code / type.")
    detail:     str = Field(..., description="Human-readable error message.")
    request_id: Optional[str] = Field(None, description="Request ID for log correlation.")
    timestamp:  str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of the error.",
    )

    model_config = {"json_schema_extra": {"example": {
        "error":      "SafetyPolicyViolation",
        "detail":     "Generating offensive malware or exploit code is not permitted.",
        "request_id": "a1b2c3d4-...",
        "timestamp":  "2026-08-10T14:00:00+00:00",
    }}}


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:    str = Field(..., description="'ok' when healthy.")
    app:       str = Field(..., description="Application name.")
    version:   str = Field(..., description="Application version.")
    phase:     int = Field(..., description="Current development phase.")
    env:       str = Field(..., description="Deployment environment.")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    model_id:          str   = Field(..., description="HuggingFace model identifier.")
    model_loaded:      bool  = Field(..., description="Whether the model is currently in memory.")
    parameter_count_m: Optional[float] = Field(None, description="Parameter count in millions.")
    device:            Optional[str]   = Field(None, description="Device the model is running on.")
    half_precision:    Optional[bool]  = Field(None, description="Whether fp16 is enabled.")
    load_time_s:       Optional[float] = Field(None, description="Model load time in seconds.")
    embedding_model_id:Optional[str]   = Field(None, description="RAG embedding model.")
    rag_chunks:        Optional[int]   = Field(None, description="Number of RAG index chunks loaded.")
    adapted_model_path:Optional[str]   = Field(None, description="Path to adapted model if loaded.")

    model_config = {"json_schema_extra": {"example": {
        "model_id":           "distilgpt2",
        "model_loaded":       True,
        "parameter_count_m":  81.9,
        "device":             "cpu",
        "half_precision":     False,
        "load_time_s":        3.68,
        "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "rag_chunks":         69,
    }}}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    uptime_seconds:    float              = Field(..., description="Seconds since server start.")
    total_requests:    int                = Field(..., description="Total HTTP requests served.")
    total_errors:      int                = Field(..., description="Total HTTP 4xx/5xx responses.")
    error_rate:        float              = Field(..., description="Error rate [0, 1].")
    mean_latency_ms:   float              = Field(..., description="Mean request latency in ms.")
    requests_by_path:  dict[str, int]     = Field(default_factory=dict)
    errors_by_path:    dict[str, int]     = Field(default_factory=dict)
    model_loaded:      bool               = Field(..., description="Whether the LLM is in memory.")
    rag_loaded:        bool               = Field(..., description="Whether the RAG vector store is loaded.")
    rag_chunks:        int                = Field(default=0)
    timestamp:         str                = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation results
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationResultsResponse(BaseModel):
    available:           bool             = Field(..., description="Whether evaluation results exist.")
    generated_at:        Optional[str]    = Field(None)
    caveat:              Optional[str]    = Field(None)
    base_model:          Optional[str]    = Field(None)
    adapted_model:       Optional[str]    = Field(None)
    base_mcq_accuracy:   Optional[float]  = Field(None)
    adapted_mcq_accuracy:Optional[float]  = Field(None)
    base_gen_recall:     Optional[float]  = Field(None)
    adapted_gen_recall:  Optional[float]  = Field(None)
    base_ppl:            Optional[float]  = Field(None)
    adapted_ppl:         Optional[float]  = Field(None)
    ppl_delta:           Optional[float]  = Field(None, description="Negative = improvement.")
    message:             Optional[str]    = Field(None, description="Status note if results unavailable.")
