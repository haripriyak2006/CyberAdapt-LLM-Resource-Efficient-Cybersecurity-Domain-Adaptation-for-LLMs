"""
backend/api/evaluation.py
GET /api/evaluation/results — Phase 6 benchmark comparison results.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from fastapi import APIRouter
from backend.schemas.common import EvaluationResultsResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "results"


@router.get(
    "/evaluation/results",
    response_model=EvaluationResultsResponse,
    summary="Phase 6 Evaluation Results",
    description=(
        "Returns the Phase 6 baseline-vs-adapted benchmark comparison. "
        "All scores are from our own experiments — not copied from any research paper."
    ),
)
async def evaluation_results() -> EvaluationResultsResponse:
    comparison_path = _RESULTS_DIR / "comparison.json"

    if not comparison_path.exists():
        return EvaluationResultsResponse(
            available=False,
            message="Evaluation results not found. Run: python evaluation/benchmark_runner.py",
        )

    try:
        data = json.loads(comparison_path.read_text("utf-8"))
    except Exception as exc:
        logger.exception("Failed to read comparison.json: %s", exc)
        return EvaluationResultsResponse(
            available=False,
            message="Error reading evaluation results. Check server logs.",
        )

    base    = data.get("base",    {})
    adapted = data.get("adapted", {})
    delta   = data.get("delta",   {})

    return EvaluationResultsResponse(
        available=True,
        generated_at=data.get("generated_at"),
        caveat=data.get("caveat"),
        base_model=base.get("model"),
        adapted_model=adapted.get("model"),
        base_mcq_accuracy=base.get("mcq_accuracy"),
        adapted_mcq_accuracy=adapted.get("mcq_accuracy"),
        base_gen_recall=base.get("gen_keyword_recall"),
        adapted_gen_recall=adapted.get("gen_keyword_recall"),
        base_ppl=base.get("mean_reference_ppl"),
        adapted_ppl=adapted.get("mean_reference_ppl"),
        ppl_delta=delta.get("ppl_delta"),
    )
