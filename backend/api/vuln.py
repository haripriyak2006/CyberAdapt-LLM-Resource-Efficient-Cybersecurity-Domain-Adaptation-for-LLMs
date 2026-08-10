"""backend/api/vuln.py — POST /api/v1/vuln/analyze"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from backend.schemas.vuln import VulnRequest, VulnResponse
from backend.core.safety import SafetyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["vulnerability-analysis"])


@router.post("/vuln/analyze", response_model=VulnResponse,
             summary="Vulnerability Analysis",
             description=(
                 "Analyze a CVE ID or vulnerability description. "
                 "Returns severity, attack vector, impact, and mitigation guidance. "
                 "For DEFENSIVE use only."
             ))
async def analyze_vuln(request: VulnRequest) -> VulnResponse:
    logger.info("Vuln analysis | len=%s", len(request.description))
    try:
        from backend.services.vuln_service import analyze_vulnerability
        result = analyze_vulnerability(request.description, top_k=request.top_k)
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
    except Exception as exc:
        logger.exception("Vuln analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
    return VulnResponse(**result)
