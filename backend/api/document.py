"""
backend/api/document.py
Document analysis endpoints — Phase 9.

Two routes:
  POST /api/document/analyze  — text body (JSON)
  POST /api/document/upload   — multipart file upload

File upload security:
  - File size validated against MAX_FILE_SIZE_BYTES
  - MIME type validated against ALLOWED_MIME_TYPES
  - Extension validated against ALLOWED_EXTENSIONS
  - Content decoded as UTF-8 (rejects invalid byte sequences)
  - PDF extraction attempted if PyPDF2/pypdf available; falls back to error message
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from backend.core.safety import SafetyError
from backend.core.limits import (
    MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB,
    ALLOWED_MIME_TYPES, ALLOWED_EXTENSIONS,
)
from backend.schemas.document_analysis import DocumentAnalysisRequest, DocumentAnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["document-analysis"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_mime(content_type: str | None, filename: str) -> None:
    """Raise 415 if content_type or extension is not allowed."""
    ext = Path(filename).suffix.lower()
    # Check extension
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File extension '{ext}' is not allowed. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    # Check content-type (some clients send 'text/plain; charset=utf-8')
    mime = (content_type or "").split(";")[0].strip()
    if mime and mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"MIME type '{mime}' is not allowed. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )


def _validate_size(size: int) -> None:
    """Raise 413 if file exceeds size limit."""
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {size / (1024*1024):.2f} MB exceeds limit of "
                f"{MAX_FILE_SIZE_MB:.1f} MB."
            ),
        )


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes. Returns empty string if no PDF library available."""
    try:
        import pypdf  # pypdf >= 3.x
        reader = pypdf.PdfReader(io.BytesIO(content))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)
    except ImportError:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        return ""


def _read_file_content(file_bytes: bytes, content_type: str, filename: str) -> str:
    """Extract text from uploaded file based on type."""
    mime = (content_type or "").split(";")[0].strip()

    if mime == "application/pdf" or filename.lower().endswith(".pdf"):
        text = _extract_text_from_pdf(file_bytes)
        if not text:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. Install 'pypdf' or 'PyPDF2', or upload a TXT file.",
            )
        return text

    # Text-based formats
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail="File content could not be decoded as UTF-8. Please upload a plain-text file.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Route 1 — JSON text body
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/document/analyze",
    response_model=DocumentAnalysisResponse,
    summary="Security Document Analyzer (text body)",
    description=(
        "Analyze plain-text security document content. "
        "Extracts threats, vulnerabilities, IoCs, and defensive recommendations."
    ),
)
async def analyze_document_text(request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
    logger.info("Document analysis (text) | len=%s | file=%s", len(request.content), request.filename)
    try:
        from backend.services.document_service import analyze_document as _analyze
        result = _analyze(request.content, filename=request.filename, top_k=request.top_k)
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
    except Exception as exc:
        logger.exception("Document analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
    return DocumentAnalysisResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
# Route 2 — Multipart file upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/document/upload",
    response_model=DocumentAnalysisResponse,
    summary="Security Document Analyzer (file upload)",
    description=(
        "Upload a security document (TXT, MD, JSON, JSONL, PDF — max 2 MB). "
        "File is validated for MIME type and size before analysis."
    ),
)
async def analyze_document_upload(
    file: UploadFile = File(..., description="Security document to analyze (TXT/PDF/JSON, max 2MB)"),
    top_k: int = Form(default=3, ge=1, le=10, description="RAG retrieval top-K"),
) -> DocumentAnalysisResponse:
    filename = file.filename or "upload.txt"
    logger.info(
        "Document upload | file=%s | content_type=%s",
        filename, file.content_type,
    )

    # ── Validation ─────────────────────────────────────────────────────────────
    _validate_mime(file.content_type, filename)

    file_bytes = await file.read()
    _validate_size(len(file_bytes))

    # ── Extract text ───────────────────────────────────────────────────────────
    content = _read_file_content(file_bytes, file.content_type or "", filename)

    if len(content.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Extracted content is too short to analyze (< 50 characters).",
        )

    # ── Analyse ────────────────────────────────────────────────────────────────
    try:
        from backend.services.document_service import analyze_document as _analyze
        result = _analyze(content, filename=filename, top_k=top_k)
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=f"Safety policy violation: {exc.reason}")
    except Exception as exc:
        logger.exception("Document upload analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    return DocumentAnalysisResponse(**result)
