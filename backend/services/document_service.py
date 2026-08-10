"""
backend/services/document_service.py
Security Document Analyzer service — Phase 8.

Input  : plain-text content of a security document (TXT or pre-extracted PDF)
Output : summary, threats, vulnerabilities, suspicious_indicators, recommendations
"""
from __future__ import annotations
import logging
import re
from backend.core.safety import enforce_safety
from backend.services.analysis_engine import run_analysis, _fetch_rag_context
from backend.services.threat_service import _extract_iocs_from_text

logger = logging.getLogger(__name__)

_TEMPLATE = """\
[SECURITY DOCUMENT ANALYSIS — DEFENSIVE USE ONLY]
Retrieved Cybersecurity Context:
{context}

Document Content (excerpt):
{input}

Security findings from this document:

Summary: """

_FIELD_LABELS = ["Summary", "Threats", "Vulnerabilities", "Indicators", "Recommendations"]

_THREAT_KEYWORDS = [
    "malware", "ransomware", "phishing", "APT", "intrusion", "backdoor",
    "trojan", "exploit", "attack", "breach", "lateral movement", "exfiltration",
    "C2", "command and control", "botnet", "keylogger",
]

_VULN_KEYWORDS = [
    "CVE-", "SQL injection", "XSS", "buffer overflow", "CSRF", "RCE",
    "remote code execution", "privilege escalation", "path traversal",
    "arbitrary code", "unpatched", "vulnerability", "zero-day",
]


def _keyword_scan(text: str, keywords: list[str]) -> list[str]:
    """Return matched keywords found in text."""
    found = []
    for kw in keywords:
        if re.search(re.escape(kw), text, re.IGNORECASE):
            found.append(kw)
    return list(dict.fromkeys(found))


def analyze_document(content: str, filename: str = "document.txt", top_k: int = 3) -> dict:
    """
    Analyze a security document for threats, vulnerabilities, and indicators.
    """
    enforce_safety(content)

    # Use first 2000 chars as summary context, full content for keyword scan
    excerpt = content[:2000]

    result = run_analysis(
        input_text=excerpt,
        prompt_template=_TEMPLATE,
        field_labels=_FIELD_LABELS,
        rag_query="cybersecurity threats vulnerabilities indicators",
        max_tokens=300,
        temperature=0.3,
        rag_top_k=top_k,
    )

    raw = result.raw_output
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    summary = (
        result.fields.get("summary")
        or (lines[0] if lines else f"Security document analysis of {filename}.")
    )
    summary = summary[:500]

    # Keyword-based extraction from full document content
    threats         = _keyword_scan(content, _THREAT_KEYWORDS)
    vulnerabilities = _keyword_scan(content, _VULN_KEYWORDS)
    indicators      = _extract_iocs_from_text(content)

    # Recommendations — from LLM output or defaults
    recs = _extract_recommendations(raw)
    if not recs:
        recs = [
            "Review and remediate all identified vulnerabilities.",
            "Implement security monitoring and alerting.",
            "Conduct regular security assessments.",
            "Apply defence-in-depth controls.",
        ]

    return {
        "summary":               summary,
        "threats":               threats,
        "vulnerabilities":       vulnerabilities,
        "suspicious_indicators": indicators,
        "recommendations":       recs[:8],
        "evidence":              result.sources,
        "evidence_sufficient":   result.evidence_sufficient,
        "confidence":            result.confidence,
        "char_count":            len(content),
        "latency_ms":            result.latency_ms,
        "model":                 result.model,
    }


def _extract_recommendations(text: str) -> list[str]:
    """Extract recommendation lines from LLM output."""
    recs = []
    in_section = False
    for line in text.splitlines():
        if re.search(r'\b(recommend|action|mitigat|remediat)\b', line, re.IGNORECASE):
            in_section = True
        if in_section:
            stripped = re.sub(r'^[\s\-\*\d\.]+', '', line).strip()
            if stripped and len(stripped) > 10:
                recs.append(stripped)
    return recs[:6]
