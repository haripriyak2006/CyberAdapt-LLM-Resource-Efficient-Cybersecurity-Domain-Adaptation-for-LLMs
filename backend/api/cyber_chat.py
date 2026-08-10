"""backend/api/cyber_chat.py — POST /api/v1/cyber/chat"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from backend.schemas.cyber_chat import CyberChatRequest, CyberChatResponse
from backend.core.safety import SafetyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cybersecurity-chat"])


@router.post("/cyber/chat", response_model=CyberChatResponse,
             summary="Cybersecurity Chat",
             description=(
                 "Ask a natural-language cybersecurity question. "
                 "Answers are grounded in retrieved cybersecurity references (OWASP, NIST, MITRE, NVD). "
                 "Focused on DEFENSIVE cybersecurity education."
             ))
async def cyber_chat_endpoint(request: CyberChatRequest) -> CyberChatResponse:
    logger.info("Cyber chat | len=%s", len(request.message))
    try:
        from backend.services.cyber_chat_service import cyber_chat
        result = cyber_chat(
            message=request.message,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
        )
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
    except Exception as exc:
        logger.exception("Cyber chat error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
    return CyberChatResponse(**result)
