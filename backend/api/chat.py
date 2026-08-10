"""
backend/api/chat.py
POST /api/v1/chat  —  base model inference endpoint.

Phase 2: raw base-model output (no domain adaptation, no RAG).
The model is identified clearly as the BASE MODEL in all responses.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.llm_service import LLMServiceError, generate_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the base model",
    description=(
        "Send a message to the base language model and receive a generated response. "
        "**Phase 2 — Base Model only.** No domain adaptation or RAG applied yet."
    ),
    responses={
        200: {"description": "Successful generation"},
        422: {"description": "Request validation error (empty message, out-of-range params)"},
        503: {"description": "Model not available or loading failed"},
        500: {"description": "Unexpected server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Generate a completion for the given message using the base LLM.

    - **message**: the user prompt (1–4096 chars)
    - **max_tokens**: tokens to generate (1–1024, default 256)
    - **temperature**: 0.0 = deterministic, higher = creative (0.0–2.0)
    """
    logger.info(
        "Chat request | len=%s | max_tokens=%s | temp=%s",
        len(request.message),
        request.max_tokens,
        request.temperature,
    )

    try:
        result = generate_response(
            prompt=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except LLMServiceError as exc:
        logger.warning("LLMServiceError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Catch-all: log internally, never expose details to the caller
        logger.exception("Unexpected error in /chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc

    return ChatResponse(**result)
