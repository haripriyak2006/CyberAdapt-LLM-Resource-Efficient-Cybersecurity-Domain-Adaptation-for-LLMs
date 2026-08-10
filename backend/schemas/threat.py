"""backend/schemas/threat.py — Schemas for POST /api/v1/threat/analyze"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ThreatRequest(BaseModel):
    description: str = Field(
        ..., min_length=10, max_length=5000,
        description="Security incident or threat description to analyze.",
        examples=["Suspicious outbound traffic detected from endpoint to unknown IP 185.234.x.x on port 443."],
    )
    top_k: int = Field(default=3, ge=1, le=10, description="RAG retrieval top-K chunks.")

    model_config = {"json_schema_extra": {"example": {
        "description": "Endpoint detected making repeated DNS queries to newly-registered domains and exfiltrating data via HTTPS.",
        "top_k": 3,
    }}}


class ThreatResponse(BaseModel):
    threat_type: str = Field(..., description="Identified threat category (e.g. APT, ransomware, insider threat).")
    indicators: list[str] = Field(default_factory=list, description="Indicators of Compromise (IoCs) extracted from input.")
    potential_impact: str = Field(..., description="Estimated potential impact if threat is not contained.")
    attack_technique: str = Field(..., description="Likely MITRE ATT&CK tactic/technique.")
    defensive_actions: list[str] = Field(default_factory=list, description="Recommended defensive actions.")
    confidence: str = Field(..., description="Analysis confidence: low | medium | high.")
    evidence: list[dict] = Field(default_factory=list, description="Retrieved supporting evidence sources.")
    evidence_sufficient: bool = Field(..., description="Whether retrieved evidence was sufficient for analysis.")
    latency_ms: float = Field(..., description="End-to-end analysis latency in milliseconds.")
    model: str = Field(..., description="LLM model used.")
    disclaimer: str = Field(
        default="This analysis is for defensive purposes only. Do not use to conduct offensive operations.",
        description="Safety disclaimer."
    )
