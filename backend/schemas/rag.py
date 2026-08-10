"""
backend/schemas/rag.py
Pydantic schemas for the POST /api/v1/rag/query endpoint.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Incoming RAG query request."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="The cybersecurity question to answer using retrieved evidence.",
        examples=["What is a SQL injection attack and how can I prevent it?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (default: 5).",
    )
    min_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score to consider evidence sufficient (default: 0.3).",
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=512,
        description="Maximum new tokens to generate for the answer.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is a SQL injection attack and how can I prevent it?",
                "top_k": 5,
                "min_score": 0.3,
                "max_new_tokens": 256,
            }
        }
    }


class SourceReference(BaseModel):
    """A single retrieved source document reference."""

    source: str = Field(..., description="Document source (e.g. OWASP, NIST, MITRE).")
    topic: str  = Field(..., description="Document topic.")
    document_type: str = Field(..., description="Document type (guideline, cve, cwe, etc.).")
    license: str = Field(..., description="Document license.")
    score: float = Field(..., description="Cosine similarity score [0, 1].")
    text_preview: str = Field(..., description="First 200 characters of the retrieved chunk.")
    was_sanitised: bool = Field(
        default=False,
        description="True if the chunk contained and redacted prompt-injection patterns.",
    )


class RAGResponse(BaseModel):
    """RAG query response."""

    answer: str = Field(
        ...,
        description="Generated answer grounded in retrieved evidence.",
    )
    evidence_sufficient: bool = Field(
        ...,
        description="True if retrieved evidence met the relevance threshold.",
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Retrieved source references used to generate the answer.",
    )
    num_chunks_retrieved: int = Field(
        ...,
        description="Total number of chunks retrieved from the vector store.",
    )
    latency_ms: float = Field(
        ...,
        description="End-to-end RAG pipeline latency in milliseconds.",
    )
    model: str = Field(
        ...,
        description="LLM model ID used for generation.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "SQL injection is an attack where malicious SQL code...",
                "evidence_sufficient": True,
                "sources": [
                    {
                        "source": "OWASP",
                        "topic": "sql injection",
                        "document_type": "guideline",
                        "license": "CC-BY-4.0",
                        "score": 0.82,
                        "text_preview": "SQL Injection flaws are introduced when...",
                        "was_sanitised": False,
                    }
                ],
                "num_chunks_retrieved": 3,
                "latency_ms": 1243.5,
                "model": "distilgpt2",
            }
        }
    }
