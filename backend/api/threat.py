"""backend/api/threat.py — POST /api/v1/threat/analyze"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from backend.schemas.threat import ThreatRequest, ThreatResponse
from backend.core.safety import SafetyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["threat-analysis"])


@router.post("/threat/analyze", response_model=ThreatResponse,
             summary="Cybersecurity Threat Analysis",
             description=(
                 "Analyze a security incident description and return structured threat intelligence. "
                 "Returns threat type, IoCs, attack technique, impact, and defensive actions. "
                 "For DEFENSIVE use only."
             ))
async def analyze_threat(request: ThreatRequest) -> ThreatResponse:
    logger.info("Threat analysis | len=%s", len(request.description))
    try:
        from backend.services.threat_service import analyze_threat as _analyze
        result = _analyze(request.description, top_k=request.top_k)
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
    except Exception as exc:
        logger.exception("Threat analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
    return ThreatResponse(**result)
