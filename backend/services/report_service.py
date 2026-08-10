"""
backend/services/report_service.py
Security Report Generator service — Phase 8.

Generates a structured security report with all required sections:
  Executive Summary, Threat Description, Affected Assets, Indicators,
  Risk Assessment, MITRE Mapping, Recommendations, Evidence, Limitations
"""
from __future__ import annotations
import datetime
import logging
import re
import time
import uuid
from backend.core.safety import enforce_safety
from backend.services.analysis_engine import (
    run_analysis, _fetch_rag_context, compute_confidence
)
from backend.services.threat_service import _extract_iocs_from_text

logger = logging.getLogger(__name__)

_EXEC_TEMPLATE = """\
[SECURITY REPORT — EXECUTIVE SUMMARY]
Retrieved Context:
{context}

Incident:
{input}

Executive Summary (non-technical, 2-3 sentences): """

_THREAT_TEMPLATE = """\
[THREAT DESCRIPTION]
{context}

Incident:
{input}

Detailed threat analysis: """

_MITRE_KEYWORDS = {
    "phishing":           "T1566 — Phishing",
    "spear":              "T1566.001 — Spearphishing Attachment",
    "credential":         "T1078 — Valid Accounts",
    "lateral":            "T1021 — Remote Services",
    "exfil":              "T1041 — Exfiltration Over C2 Channel",
    "ransomware":         "T1486 — Data Encrypted for Impact",
    "sql":                "T1190 — Exploit Public-Facing Application",
    "privilege":          "T1068 — Exploitation for Privilege Escalation",
    "backdoor":           "T1505 — Server Software Component",
    "dns":                "T1071.004 — DNS",
    "powershell":         "T1059.001 — PowerShell",
    "remote access":      "T1219 — Remote Access Software",
    "brute force":        "T1110 — Brute Force",
    "command and control": "T1571 — Non-Standard Port",
    "persistence":        "T1547 — Boot or Logon Autostart Execution",
}

_RISK_LEVELS = {
    "ransomware": "Critical — Business disruption, data loss, regulatory exposure",
    "APT":        "Critical — Nation-state level threat, long-term compromise",
    "breach":     "High — Data exfiltration, regulatory and reputational risk",
    "malware":    "High — System compromise, potential lateral movement",
    "phishing":   "Medium — Credential theft, account compromise",
    "default":    "Medium — Requires further investigation to determine full scope",
}


def _map_mitre(text: str) -> list[str]:
    """Map incident description to MITRE ATT&CK techniques."""
    techniques = []
    lower = text.lower()
    for keyword, technique in _MITRE_KEYWORDS.items():
        if keyword in lower:
            techniques.append(technique)
    # Also look for explicit T-codes
    for m in re.finditer(r'T\d{4}(?:\.\d{3})?', text):
        entry = f"{m.group(0)} — (see MITRE ATT&CK)"
        if entry not in techniques:
            techniques.append(entry)
    return list(dict.fromkeys(techniques))[:8]


def _assess_risk(text: str) -> str:
    """Determine risk level from incident description."""
    lower = text.lower()
    for keyword, risk in _RISK_LEVELS.items():
        if keyword in lower:
            return risk
    return _RISK_LEVELS["default"]


def _default_recommendations() -> list[str]:
    return [
        "1. [IMMEDIATE] Isolate affected systems from the network.",
        "2. [IMMEDIATE] Preserve forensic evidence (logs, memory dumps, disk images).",
        "3. [SHORT-TERM] Notify relevant stakeholders and legal/compliance teams.",
        "4. [SHORT-TERM] Conduct full threat hunt across the environment.",
        "5. [SHORT-TERM] Reset credentials for all potentially-affected accounts.",
        "6. [MEDIUM-TERM] Patch identified vulnerabilities across all systems.",
        "7. [MEDIUM-TERM] Review and harden security monitoring and alerting.",
        "8. [LONG-TERM] Conduct post-incident lessons-learned review.",
    ]


def generate_report(
    incident_description: str,
    affected_assets: list[str],
    analyst_name: str = "CyberAdapt-LLM",
    organization: str = "[Organization]",
    top_k: int = 3,
) -> dict:
    """Generate a full structured security report."""
    enforce_safety(incident_description)

    t_start = time.perf_counter()

    # RAG context
    sources, context, sufficient = _fetch_rag_context(incident_description, top_k=top_k)

    # Executive summary
    exec_result = run_analysis(
        input_text=incident_description,
        prompt_template=_EXEC_TEMPLATE,
        field_labels=["Executive Summary"],
        max_tokens=150,
        temperature=0.3,
        rag_top_k=top_k,
    )
    exec_lines = [l.strip() for l in exec_result.raw_output.splitlines() if l.strip()]
    exec_summary = exec_lines[0] if exec_lines else (
        f"A security incident was detected affecting {', '.join(affected_assets) or 'systems'}. "
        "This report summarises the threat analysis and recommended defensive actions."
    )
    exec_summary = exec_summary[:500]

    # Threat description
    threat_result = run_analysis(
        input_text=incident_description,
        prompt_template=_THREAT_TEMPLATE,
        field_labels=["Threat Description", "Impact"],
        max_tokens=200,
        temperature=0.3,
        rag_top_k=top_k,
    )
    threat_lines = [l.strip() for l in threat_result.raw_output.splitlines() if l.strip()]
    threat_desc = " ".join(threat_lines[:3]) if threat_lines else incident_description[:400]
    threat_desc = threat_desc[:600]

    # Derived fields
    indicators    = _extract_iocs_from_text(incident_description)
    mitre_mapping = _map_mitre(incident_description)
    risk          = _assess_risk(incident_description)
    confidence    = compute_confidence(sources)
    recommendations = _default_recommendations()

    limitations = (
        "This report was generated by CyberAdapt-LLM, an AI system trained on a small "
        "cybersecurity corpus. Findings should be verified by qualified security professionals. "
        "The LLM used (distilgpt2) is not instruction-tuned and may produce incomplete "
        "analysis. Do not rely solely on this report for critical security decisions."
    )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "report_id":          f"RPT-{uuid.uuid4().hex[:8].upper()}",
        "generated_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "analyst":            analyst_name,
        "organization":       organization,
        "executive_summary":  exec_summary,
        "threat_description": threat_desc,
        "affected_assets":    affected_assets,
        "indicators":         indicators,
        "risk_assessment":    risk,
        "mitre_mapping":      mitre_mapping,
        "recommendations":    recommendations,
        "evidence":           sources,
        "limitations":        limitations,
        "confidence":         confidence,
        "latency_ms":         latency_ms,
        "model":              exec_result.model,
    }
