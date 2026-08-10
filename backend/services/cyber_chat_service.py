"""
backend/services/cyber_chat_service.py
Cybersecurity Chat service — Phase 8.

Natural-language cybersecurity Q&A grounded in the RAG vector store.
Focused entirely on defensive cybersecurity education and analysis.
"""
from __future__ import annotations
import logging
import time
from backend.core.safety import enforce_safety
from backend.services.analysis_engine import (
    _fetch_rag_context, _call_llm, compute_confidence
)

logger = logging.getLogger(__name__)

_TEMPLATE = """\
[CYBERSECURITY Q&A — DEFENSIVE EDUCATION ONLY]
Retrieved Cybersecurity References:
{context}

Using ONLY the above references, answer this cybersecurity question.
If the references do not contain enough information, say so clearly.
Focus on defensive, protective, and educational information only.

Question: {question}

Answer:"""


def cyber_chat(message: str, top_k: int = 3, max_new_tokens: int = 200) -> dict:
    """
    Answer a natural-language cybersecurity question using RAG + LLM.
    """
    enforce_safety(message)

    t_start = time.perf_counter()

    # RAG retrieval
    sources, context, sufficient = _fetch_rag_context(message, top_k=top_k, min_score=0.25)

    context_block = context or "No relevant cybersecurity references retrieved."

    prompt = _TEMPLATE.format(
        context=context_block,
        question=message.strip(),
    )

    raw_answer, model, _llm_ms = _call_llm(prompt, max_tokens=max_new_tokens, temperature=0.4)

    # Clean up response
    answer = raw_answer.strip()
    if not answer or len(answer) < 10:
        answer = (
            "I was unable to generate a detailed answer from the retrieved references. "
            "Please consult official cybersecurity sources such as OWASP, NIST, or MITRE ATT&CK."
        )

    if not sufficient:
        answer = (
            "[Note: Retrieved evidence was limited for this query. "
            "The following answer may be incomplete.]\n\n" + answer
        )

    confidence = compute_confidence(sources)
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "answer":             answer,
        "sources":            sources,
        "evidence_sufficient": sufficient,
        "confidence":         confidence,
        "latency_ms":         latency_ms,
        "model":              model,
    }
