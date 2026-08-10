"""
backend/services/vuln_service.py
Vulnerability Analysis service — Phase 8.

Input  : CVE ID or vulnerability description
Output : summary, affected_component, severity, attack_vector,
         potential_impact, mitigation, evidence
"""
from __future__ import annotations
import logging
import re
from backend.core.safety import enforce_safety
from backend.services.analysis_engine import run_analysis

logger = logging.getLogger(__name__)

_FIELD_LABELS = [
    "Vulnerability Summary", "Affected Component", "Severity",
    "Attack Vector", "Potential Impact", "Mitigation",
]

_TEMPLATE = """\
[VULNERABILITY ANALYSIS — DEFENSIVE USE ONLY]
Retrieved Cybersecurity References:
{context}

Vulnerability / CVE Input:
{input}

Defensive vulnerability analysis:

Vulnerability Summary: """

_SEVERITY_WORDS = {
    "critical": "Critical",
    "high":     "High",
    "medium":   "Medium",
    "moderate": "Medium",
    "low":      "Low",
    "info":     "Informational",
}

_CVSS_SEVERITY = {
    (9.0, 10.0): "Critical",
    (7.0, 8.9):  "High",
    (4.0, 6.9):  "Medium",
    (0.1, 3.9):  "Low",
}


def _parse_severity(text: str) -> str:
    """Extract severity from text using CVSS score or keywords."""
    # Try CVSS score
    m = re.search(r'CVSS[:\s]+(\d+\.\d+)', text, re.IGNORECASE)
    if m:
        score = float(m.group(1))
        for (lo, hi), label in _CVSS_SEVERITY.items():
            if lo <= score <= hi:
                return label
    # Try keywords
    for kw, label in _SEVERITY_WORDS.items():
        if re.search(rf'\b{kw}\b', text, re.IGNORECASE):
            return label
    return "Unknown"


def _parse_attack_vector(text: str) -> str:
    """Extract CVSS attack vector from text."""
    for av in ("Network", "Adjacent", "Local", "Physical"):
        if re.search(rf'\b{av}\b', text, re.IGNORECASE):
            return av
    return "Network (assumed)"


def analyze_vulnerability(description: str, top_k: int = 3) -> dict:
    """
    Run vulnerability analysis on a CVE ID or description.
    Returns a structured dict ready for the API response.
    """
    enforce_safety(description)

    # Use a targeted RAG query
    cve_match = re.search(r'CVE-\d{4}-\d{4,7}', description, re.IGNORECASE)
    rag_query = cve_match.group(0).upper() + " vulnerability" if cve_match else description

    result = run_analysis(
        input_text=description,
        prompt_template=_TEMPLATE,
        field_labels=_FIELD_LABELS,
        rag_query=rag_query,
        max_tokens=300,
        temperature=0.3,
        rag_top_k=top_k,
    )

    raw   = result.raw_output
    fields = result.fields

    # Vulnerability summary
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    summary = (
        fields.get("vulnerability_summary")
        or (lines[0] if lines else "Vulnerability analysis pending.")
    )
    summary = summary[:400]

    # Affected component
    affected = fields.get("affected_component", "").strip()
    if not affected:
        # Try to extract from description (library name, version)
        m = re.search(r'(?:in|on|affects?)\s+([A-Za-z0-9\-\.]+(?:\s+v?[\d\.]+)?)', description, re.IGNORECASE)
        affected = m.group(1) if m else "See vulnerability description"

    severity     = _parse_severity(description + " " + raw)
    attack_vector = _parse_attack_vector(description + " " + raw)

    impact = fields.get("potential_impact", "").strip() or "Assess impact per affected environment."
    impact = impact[:300]

    mitigation = fields.get("mitigation", "").strip() or (
        "Apply vendor patches immediately. "
        "Implement network segmentation and access controls. "
        "Monitor for exploit attempts."
    )
    mitigation = mitigation[:400]

    return {
        "vulnerability_summary": summary,
        "affected_component":    affected,
        "severity":              severity,
        "attack_vector":         attack_vector,
        "potential_impact":      impact,
        "mitigation":            mitigation,
        "evidence":              result.sources,
        "evidence_sufficient":   result.evidence_sufficient,
        "confidence":            result.confidence,
        "latency_ms":            result.latency_ms,
        "model":                 result.model,
    }
