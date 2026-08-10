"""backend/schemas/cyber_chat.py — Schemas for POST /api/v1/cyber/chat"""
from __future__ import annotations
from pydantic import BaseModel, Field


class CyberChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=3, max_length=2000,
        description="A natural-language cybersecurity question.",
        examples=["What is the difference between a virus and a worm?"],
    )
    top_k: int = Field(default=3, ge=1, le=10)
    max_new_tokens: int = Field(default=200, ge=10, le=512)

    model_config = {"json_schema_extra": {"example": {
        "message": "What is the principle of least privilege and why does it matter?",
        "top_k": 3,
        "max_new_tokens": 200,
    }}}


class CyberChatResponse(BaseModel):
    answer: str = Field(..., description="Cybersecurity-focused answer.")
    sources: list[dict] = Field(default_factory=list, description="Retrieved cybersecurity references used.")
    evidence_sufficient: bool = Field(..., description="Whether retrieved evidence was sufficient.")
    confidence: str = Field(..., description="Response confidence: low | medium | high.")
    latency_ms: float = Field(...)
    model: str = Field(...)
    disclaimer: str = Field(
        default="This answer is for educational/defensive purposes only. Always verify with authoritative sources.",
    )
