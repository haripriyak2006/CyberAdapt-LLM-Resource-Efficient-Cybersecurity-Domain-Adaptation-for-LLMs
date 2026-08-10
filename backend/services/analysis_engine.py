"""
backend/services/analysis_engine.py
Shared analysis engine for CyberAdapt-LLM Phase 8 features.

All five analysis services (threat, vuln, document, report, cyber_chat)
use this engine. It provides:
  1. RAG-enhanced context retrieval (best-effort — degrades gracefully)
  2. Structured completion prompts for causal LMs
  3. Field extraction from free-form LLM output
  4. Confidence scoring based on retrieval relevance
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    raw_output: str
    fields: dict
    sources: list[dict] = field(default_factory=list)
    evidence_sufficient: bool = False
    confidence: str = "low"
    latency_ms: float = 0.0
    model: str = "distilgpt2"


# ─────────────────────────────────────────────────────────────────────────────
# Field parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_labeled_fields(text: str, labels: list[str]) -> dict[str, str]:
    """
    Extract labeled fields from a structured LLM completion.

    Expects text in the form:
      Label: value ...
      Next Label: value ...

    Fields may span multiple lines until the next label.
    """
    result: dict[str, str] = {}
    # Build a regex that matches any known label
    label_pat = re.compile(
        r'^(' + '|'.join(re.escape(l) for l in labels) + r')\s*:\s*',
        re.MULTILINE | re.IGNORECASE,
    )

    # Find all label positions
    matches = list(label_pat.finditer(text))
    for i, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        # Collapse internal newlines into spaces for single-value fields
        value = re.sub(r'\n+', ' ', value).strip()
        result[label.lower().replace(' ', '_')] = value

    return result


def extract_list_field(text: str, label: str) -> list[str]:
    """
    Extract a bullet-list or numbered-list field value as a Python list.
    """
    # Find the section starting with the label
    pat = re.compile(rf'{re.escape(label)}\s*:\s*\n(.*?)(?=\n[A-Z]|\Z)', re.DOTALL | re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return []
    block = m.group(1)
    items = []
    for line in block.splitlines():
        line = re.sub(r'^[\s\-\*\d\.]+', '', line).strip()
        if line:
            items.append(line)
    return items


def compute_confidence(sources: list[dict]) -> str:
    """Compute a confidence rating from retrieval scores."""
    if not sources:
        return "low"
    best_score = max(s.get("score", 0) for s in sources)
    if best_score >= 0.55:
        return "high"
    elif best_score >= 0.4:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# RAG context fetcher (graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_rag_context(query: str, top_k: int = 3, min_score: float = 0.3) -> tuple[list[dict], str, bool]:
    """
    Fetch RAG context for a query.
    Returns (sources, context_text, evidence_sufficient).
    Never raises — degrades gracefully if RAG not available.
    """
    try:
        from backend.services.rag_service import _ensure_loaded, _retriever
        _ensure_loaded(top_k=top_k, min_score=min_score)
        if _retriever is None:
            return [], "", False
        chunks, sufficient = _retriever.retrieve(query)
        sources = [
            {
                "source":        c.source,
                "topic":         c.topic,
                "document_type": c.document_type,
                "license":       c.license,
                "score":         c.score,
                "text_preview":  c.text[:150].strip(),
            }
            for c in chunks
        ]
        context_parts = []
        for i, c in enumerate(chunks[:3], 1):
            context_parts.append(
                f"[Ref {i}: {c.source} | {c.topic} | score={c.score:.2f}]\n{c.text[:300]}"
            )
        context = "\n\n".join(context_parts)
        return sources, context, sufficient
    except Exception as exc:
        logger.debug("RAG context fetch failed (degrading gracefully): %s", exc)
        return [], "", False


# ─────────────────────────────────────────────────────────────────────────────
# LLM caller
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> tuple[str, str, float]:
    """Call the LLM service. Returns (response_text, model_name, latency_ms)."""
    from backend.services.llm_service import generate_response
    result = generate_response(prompt, max_tokens=max_tokens, temperature=temperature)
    return result["response"], result["model"], result["latency_ms"]


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    input_text: str,
    prompt_template: str,
    field_labels: list[str],
    rag_query: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.3,
    rag_top_k: int = 3,
) -> AnalysisResult:
    """
    Run a full RAG-enhanced analysis pass.

    Parameters
    ----------
    input_text      : The user's raw input (already safety-checked by caller).
    prompt_template : Prompt string with {input} and {context} placeholders.
    field_labels    : Field names to extract from LLM output.
    rag_query       : Optional separate query for RAG (defaults to input_text).
    max_tokens      : Max new tokens to generate.
    temperature     : Sampling temperature.
    rag_top_k       : Number of chunks to retrieve.
    """
    t_start = time.perf_counter()

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    rag_q = rag_query or input_text
    sources, context, evidence_sufficient = _fetch_rag_context(rag_q, top_k=rag_top_k)

    context_block = (
        context if context
        else "No cybersecurity reference documents retrieved for this query."
    )

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = prompt_template.format(
        input=input_text[:2000],   # cap input length
        context=context_block,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    raw_output, model, _llm_ms = _call_llm(prompt, max_tokens=max_tokens, temperature=temperature)

    # ── Parse fields ──────────────────────────────────────────────────────────
    # Parse from the full prompt+output so labels are easier to find
    parse_target = prompt + " " + raw_output
    fields = parse_labeled_fields(parse_target, field_labels)

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = compute_confidence(sources)

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return AnalysisResult(
        raw_output=raw_output,
        fields=fields,
        sources=sources,
        evidence_sufficient=evidence_sufficient,
        confidence=confidence,
        latency_ms=latency_ms,
        model=model,
    )
