"""
backend/api/demo.py
Phase 11 — Live Model Comparison API endpoint.

POST /api/demo/compare
GET  /api/demo/questions
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["demo-comparison"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    question: str = Field(
        ..., min_length=10, max_length=1000,
        description="Cybersecurity question to compare between models.",
    )
    top_k: int = Field(default=3, ge=1, le=10, description="RAG retrieval top-K.")

    model_config = {"json_schema_extra": {"example": {
        "question": "What is CVE-2021-44228 (Log4Shell) and what attack vector does it exploit?",
        "top_k": 3,
    }}}


class ModelResult(BaseModel):
    answer: str
    model: str
    latency_ms: float
    error: str | None = None
    adapted_model_available: bool = True


class EvidenceInfo(BaseModel):
    sources: list[dict]
    evidence_sufficient: bool
    num_chunks: int


class CompareResponse(BaseModel):
    question: str
    base: ModelResult
    adapted: ModelResult
    evidence: EvidenceInfo
    difference_summary: str
    total_latency_ms: float
    disclaimer: str


class DemoQuestion(BaseModel):
    id: str
    category: str
    question: str
    rationale: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/demo/questions",
    response_model=list[DemoQuestion],
    summary="Get pre-designed demonstration questions",
    description=(
        "Returns the 5 carefully designed cybersecurity demonstration questions "
        "for the live model comparison demo."
    ),
)
async def get_demo_questions() -> list[DemoQuestion]:
    from backend.services.demo_service import DEMO_QUESTIONS
    return [DemoQuestion(**q) for q in DEMO_QUESTIONS]


@router.post(
    "/demo/compare",
    response_model=CompareResponse,
    summary="Live Model Comparison",
    description=(
        "Run the same cybersecurity question through the base LLM and CyberAdapt-LLM "
        "simultaneously with shared RAG evidence. Returns side-by-side answers, "
        "latency, evidence, and an AI-generated explanation of differences."
    ),
)
async def compare_models(request: CompareRequest) -> CompareResponse:
    logger.info("Demo comparison | question=%.80s", request.question)
    try:
        from backend.services.demo_service import run_comparison
        result = run_comparison(question=request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Demo comparison failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {type(exc).__name__} — check server logs.",
        )

    return CompareResponse(
        question=result["question"],
        base=ModelResult(**result["base"]),
        adapted=ModelResult(
            answer=result["adapted"]["answer"],
            model=result["adapted"]["model"],
            latency_ms=result["adapted"]["latency_ms"],
            error=result["adapted"]["error"],
            adapted_model_available=result["adapted"]["adapted_model_available"],
        ),
        evidence=EvidenceInfo(**result["evidence"]),
        difference_summary=result["difference_summary"],
        total_latency_ms=result["total_latency_ms"],
        disclaimer=result["disclaimer"],
    )
