"""backend/api/report.py — POST /api/v1/report/generate"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from backend.schemas.report import ReportRequest, SecurityReport
from backend.core.safety import SafetyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["report-generation"])


@router.post("/report/generate", response_model=SecurityReport,
             summary="Security Report Generator",
             description=(
                 "Generate a structured security report from an incident description. "
                 "Includes Executive Summary, Threat Description, IoCs, MITRE mapping, "
                 "Risk Assessment, and Recommendations. For DEFENSIVE use only."
             ))
async def generate_report(request: ReportRequest) -> SecurityReport:
    logger.info("Report generation | len=%s | org=%s", len(request.incident_description), request.organization)
    try:
        from backend.services.report_service import generate_report as _generate
        result = _generate(
            incident_description=request.incident_description,
            affected_assets=request.affected_assets,
            analyst_name=request.analyst_name,
            organization=request.organization,
            top_k=request.top_k,
        )
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
    except Exception as exc:
        logger.exception("Report generation error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
    return SecurityReport(**result)
