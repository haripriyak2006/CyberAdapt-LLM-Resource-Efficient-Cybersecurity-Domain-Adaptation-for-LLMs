"""
backend/api/rag.py
POST /api/v1/rag/query  — CyberAdapt-LLM RAG endpoint.

Security notes:
  - Retrieved documents are sanitised in rag_service before prompt insertion.
  - Internal stack traces are never exposed to the API caller.
  - Request body is validated by Pydantic before reaching service layer.
  - Question length is capped at 2048 characters.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.schemas.rag import RAGRequest, RAGResponse, SourceReference
from backend.services.rag_service import RAGServiceError, rag_query, get_rag_status

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


@router.post(
    "/rag/query",
    response_model=RAGResponse,
    summary="Cybersecurity RAG Query",
    description=(
        "Answer a cybersecurity question using Retrieval-Augmented Generation. "
        "The system retrieves relevant passages from trusted cybersecurity sources "
        "(OWASP, NIST, MITRE, NVD), grounds the answer in retrieved evidence, "
        "and clearly flags when evidence is insufficient."
    ),
    responses={
        200: {"description": "Successful RAG response with sources."},
        503: {"description": "RAG system not ready (vector store not built)."},
        500: {"description": "Internal error — no details exposed."},
    },
)
async def rag_query_endpoint(request: RAGRequest) -> RAGResponse:
    logger.info(
        "RAG query | len=%s | top_k=%s | min_score=%s",
        len(request.question),
        request.top_k,
        request.min_score,
    )

    try:
        result = rag_query(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            max_new_tokens=request.max_new_tokens,
        )
    except RAGServiceError as exc:
        msg = str(exc)
        if "not found" in msg.lower() or "unavailable" in msg.lower():
            raise HTTPException(status_code=503, detail=msg)
        logger.exception("RAGServiceError: %s", exc)
        raise HTTPException(status_code=503, detail="RAG system unavailable.")
    except Exception as exc:
        logger.exception("Unexpected error in /rag/query: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    return RAGResponse(
        answer=result["answer"],
        evidence_sufficient=result["evidence_sufficient"],
        sources=[
            SourceReference(
                source=s["source"],
                topic=s["topic"],
                document_type=s["document_type"],
                license=s["license"],
                score=s["score"],
                text_preview=s["text_preview"],
                was_sanitised=s["was_sanitised"],
            )
            for s in result["sources"]
        ],
        num_chunks_retrieved=result["num_chunks_retrieved"],
        latency_ms=result["latency_ms"],
        model=result["model"],
    )


@router.get(
    "/rag/status",
    summary="RAG System Status",
    description="Check whether the RAG vector store is loaded and how many chunks it contains.",
)
async def rag_status_endpoint() -> dict:
    return get_rag_status()
