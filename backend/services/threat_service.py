"""
backend/services/threat_service.py
Threat Analysis service — Phase 8.

Input  : security incident description
Output : threat_type, indicators, potential_impact, attack_technique,
         defensive_actions, confidence, evidence
"""
from __future__ import annotations
import logging
from backend.core.safety import enforce_safety
from backend.services.analysis_engine import run_analysis, extract_list_field

logger = logging.getLogger(__name__)

_FIELD_LABELS = [
    "Threat Type", "Attack Technique", "Potential Impact",
    "Confidence", "Summary",
]

_TEMPLATE = """\
[CYBERSECURITY THREAT ANALYSIS — DEFENSIVE USE ONLY]
Retrieved Cybersecurity Context:
{context}

Incident Description:
{input}

Based on the context and description above, provide a defensive threat analysis:

Threat Type: """


def _extract_bullets(text: str, section_hint: str) -> list[str]:
    """Extract bullet items after a keyword hint."""
    lines = text.splitlines()
    collecting = False
    items: list[str] = []
    for line in lines:
        if section_hint.lower() in line.lower():
            collecting = True
            continue
        if collecting:
            stripped = line.strip().lstrip("-*•123456789. ")
            if stripped:
                items.append(stripped)
            elif items:
                break
    return items[:8]


def analyze_threat(description: str, top_k: int = 3) -> dict:
    """
    Run threat analysis on a security incident description.
    Returns a structured dict ready for the API response.
    """
    enforce_safety(description)

    result = run_analysis(
        input_text=description,
        prompt_template=_TEMPLATE,
        field_labels=_FIELD_LABELS,
        max_tokens=300,
        temperature=0.3,
        rag_top_k=top_k,
    )

    raw = result.raw_output
    fields = result.fields

    # Extract threat type (first meaningful line of output)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    threat_type = (
        fields.get("threat_type")
        or (lines[0] if lines else "Unknown")
    )
    # Clean up
    threat_type = threat_type.split("\n")[0][:120]

    # Attack technique — look for MITRE patterns
    import re
    mitre_match = re.search(r'T\d{4}(?:\.\d{3})?', raw)
    attack_technique = (
        fields.get("attack_technique")
        or (mitre_match.group(0) if mitre_match else "See MITRE ATT&CK for relevant techniques")
    )

    potential_impact = fields.get("potential_impact", "").strip() or "Impact assessment requires more context."
    potential_impact = potential_impact[:300]

    # Indicators — keyword-based heuristic extraction from description
    indicators = _extract_iocs_from_text(description)

    # Defensive actions — from output or defaults
    defensive_actions = _extract_bullets(raw, "action") or _extract_bullets(raw, "recommend") or [
        "Isolate affected systems immediately",
        "Collect and preserve forensic evidence",
        "Review firewall and proxy logs",
        "Change credentials for affected accounts",
        "Notify incident response team",
    ]

    return {
        "threat_type":        threat_type,
        "indicators":         indicators,
        "potential_impact":   potential_impact,
        "attack_technique":   attack_technique,
        "defensive_actions":  defensive_actions[:8],
        "confidence":         result.confidence,
        "evidence":           result.sources,
        "evidence_sufficient": result.evidence_sufficient,
        "latency_ms":         result.latency_ms,
        "model":              result.model,
    }


def _extract_iocs_from_text(text: str) -> list[str]:
    """Heuristically extract indicators (IPs, domains, hashes, CVEs) from text."""
    import re
    indicators: list[str] = []
    # IP addresses
    for m in re.finditer(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text):
        indicators.append(f"IP: {m.group(0)}")
    # Domains (simple heuristic)
    for m in re.finditer(r'\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|ru|cn|xyz|tk|top)\b', text):
        indicators.append(f"Domain: {m.group(0)}")
    # File hashes (MD5/SHA1/SHA256)
    for m in re.finditer(r'\b[0-9a-fA-F]{32}\b|\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b', text):
        indicators.append(f"Hash: {m.group(0)}")
    # CVE IDs
    for m in re.finditer(r'CVE-\d{4}-\d{4,7}', text, re.IGNORECASE):
        indicators.append(f"CVE: {m.group(0).upper()}")
    # Ports
    for m in re.finditer(r'\bport[s]?\s+(\d{2,5})\b', text, re.IGNORECASE):
        indicators.append(f"Port: {m.group(1)}")
    return list(dict.fromkeys(indicators))[:10]  # dedup, cap at 10
