"""
backend/services/rag_service.py
RAG pipeline service for CyberAdapt-LLM — Phase 7.

Orchestrates:
  1. VectorStore loading (singleton, lazy)
  2. EmbeddingModel loading (singleton, lazy)
  3. Retriever (top-K with injection defense)
  4. Prompt building (evidence-grounded, with hard delimiters)
  5. LLM generation (via existing llm_service)

Security guarantees:
  - Retrieved documents are sanitised before prompt insertion.
  - Retrieved content cannot appear before system instructions.
  - Low-relevance results trigger "Insufficient evidence" response.
  - No internal errors or stack traces exposed via the API.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_vector_store = None
_embedding_model = None
_retriever = None
_store_loaded_from: Optional[str] = None


class RAGServiceError(Exception):
    """Raised for expected RAG pipeline failures."""


def _get_settings():
    from backend.core.config import get_settings
    return get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_loaded(
    vector_store_path: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> None:
    """Lazy-load the vector store and embedding model (once per process)."""
    global _vector_store, _embedding_model, _retriever, _store_loaded_from

    cfg = _get_settings()
    store_path = Path(vector_store_path or cfg.vector_store_path)
    embed_name = embedding_model_name or cfg.embedding_model_id

    # Re-use if already loaded from the same path
    if _vector_store is not None and str(store_path) == _store_loaded_from:
        return

    from rag.vector_store import VectorStore
    from rag.embeddings import EmbeddingModel
    from rag.retriever import Retriever

    if not (store_path / "index.faiss").exists():
        raise RAGServiceError(
            f"Vector store not found at {store_path}. "
            "Run: python scripts/ingest_cybersec.py"
        )

    logger.info("Loading vector store from: %s", store_path)
    _vector_store = VectorStore.load(store_path)

    logger.info("Loading embedding model: %s", embed_name)
    _embedding_model = EmbeddingModel(
        model_name=embed_name,
        cache_dir=cfg.model_cache_dir,
    )

    _retriever = Retriever(
        vector_store=_vector_store,
        embedding_model=_embedding_model,
        top_k=top_k,
        min_score_threshold=min_score,
    )
    _store_loaded_from = str(store_path)
    logger.info(
        "RAG service ready | chunks=%s | embed=%s",
        _vector_store.num_chunks, embed_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rag_query(
    question: str,
    top_k: int = 5,
    min_score: float = 0.3,
    max_new_tokens: int = 256,
) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate → return.

    Returns
    -------
    dict with keys:
      answer, evidence_sufficient, sources, num_chunks_retrieved, latency_ms, model
    """
    t_start = time.perf_counter()

    try:
        _ensure_loaded(top_k=top_k, min_score=min_score)
    except RAGServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to load RAG components: %s", exc)
        raise RAGServiceError("RAG system unavailable. Check logs.") from exc

    # Update retriever config per-request
    _retriever.top_k = top_k
    _retriever.min_score_threshold = min_score

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        chunks, evidence_sufficient = _retriever.retrieve(question)
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        raise RAGServiceError("Retrieval failed.") from exc

    # ── Build prompt ──────────────────────────────────────────────────────────
    from rag.retriever import build_rag_prompt
    prompt = build_rag_prompt(
        question=question,
        chunks=chunks,
        evidence_sufficient=evidence_sufficient,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        from backend.services.llm_service import generate_response
        gen_result = generate_response(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.3,   # lower temperature for factual grounded answers
        )
        answer = gen_result["response"]
        model  = gen_result["model"]
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        raise RAGServiceError("LLM generation failed.") from exc

    # Prepend explicit evidence flag if insufficient
    if not evidence_sufficient:
        answer = (
            "[Insufficient Evidence] The retrieved documents did not contain "
            "closely relevant information for this query. "
            "The following is the model's best effort without reliable evidence:\n\n"
            + answer
        )

    # ── Build source references ───────────────────────────────────────────────
    sources = [
        {
            "source":        c.source,
            "topic":         c.topic,
            "document_type": c.document_type,
            "license":       c.license,
            "score":         c.score,
            "text_preview":  c.text[:200].strip(),
            "was_sanitised": c.was_sanitised,
        }
        for c in chunks
    ]

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        "RAG query complete | evidence_sufficient=%s | chunks=%s | latency=%.0fms",
        evidence_sufficient, len(chunks), latency_ms,
    )

    return {
        "answer":               answer,
        "evidence_sufficient":  evidence_sufficient,
        "sources":              sources,
        "num_chunks_retrieved": len(chunks),
        "latency_ms":           latency_ms,
        "model":                model,
    }


def get_rag_status() -> dict:
    """Return the current RAG system status (for health checks)."""
    if _vector_store is None:
        return {"loaded": False, "num_chunks": 0}
    return {
        "loaded":     True,
        "num_chunks": _vector_store.num_chunks,
        "stats":      _vector_store.stats(),
    }
