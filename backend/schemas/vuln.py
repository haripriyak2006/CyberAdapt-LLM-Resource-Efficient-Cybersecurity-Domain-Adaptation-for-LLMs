"""backend/schemas/vuln.py — Schemas for POST /api/v1/vuln/analyze"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class VulnRequest(BaseModel):
    description: str = Field(
        ..., min_length=5, max_length=5000,
        description="CVE ID (e.g. CVE-2021-44228) or vulnerability description.",
        examples=["CVE-2021-44228 — Apache Log4j2 JNDI injection vulnerability allowing RCE."],
    )
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = {"json_schema_extra": {"example": {
        "description": "CVE-2021-44228 — Apache Log4j2 remote code execution via JNDI injection.",
        "top_k": 3,
    }}}


class VulnResponse(BaseModel):
    vulnerability_summary: str = Field(..., description="Concise vulnerability description.")
    affected_component: str = Field(..., description="Affected software, library, or system component.")
    severity: str = Field(..., description="Severity rating: Critical | High | Medium | Low | Informational.")
    attack_vector: str = Field(..., description="How the vulnerability is exploited (Network/Adjacent/Local/Physical).")
    potential_impact: str = Field(..., description="Potential impact if exploited.")
    mitigation: str = Field(..., description="Recommended mitigation or patch.")
    evidence: list[dict] = Field(default_factory=list, description="Retrieved supporting cybersecurity references.")
    evidence_sufficient: bool = Field(...)
    confidence: str = Field(...)
    latency_ms: float = Field(...)
    model: str = Field(...)
    disclaimer: str = Field(
        default="This analysis is for defensive security purposes only.",
        description="Safety disclaimer."
    )
