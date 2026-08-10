"""backend/schemas/document_analysis.py — Schemas for POST /api/v1/document/analyze"""
from __future__ import annotations
from pydantic import BaseModel, Field


class DocumentAnalysisRequest(BaseModel):
    content: str = Field(
        ..., min_length=50, max_length=10000,
        description="Plain-text content of the security document to analyze.",
    )
    filename: str = Field(default="document.txt", description="Optional filename for context.")
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = {"json_schema_extra": {"example": {
        "content": "This security assessment identified SQL injection vulnerabilities in the login form...",
        "filename": "pentest_report.txt",
        "top_k": 3,
    }}}


class DocumentAnalysisResponse(BaseModel):
    summary: str = Field(..., description="Concise summary of the document's security content.")
    threats: list[str] = Field(default_factory=list, description="Threat types identified in the document.")
    vulnerabilities: list[str] = Field(default_factory=list, description="Vulnerabilities mentioned or implied.")
    suspicious_indicators: list[str] = Field(default_factory=list, description="Suspicious indicators or IoCs found.")
    recommendations: list[str] = Field(default_factory=list, description="Defensive recommendations extracted or inferred.")
    evidence: list[dict] = Field(default_factory=list)
    evidence_sufficient: bool = Field(...)
    confidence: str = Field(...)
    char_count: int = Field(..., description="Character count of the analyzed document.")
    latency_ms: float = Field(...)
    model: str = Field(...)
    disclaimer: str = Field(default="Analysis is for defensive purposes only.")
