"""
backend/schemas/chat.py
Pydantic request/response models for the /api/v1/chat endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user message / prompt to send to the model.",
        examples=["Explain what a SQL injection attack is."],
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. 0 = greedy / deterministic. "
            "Higher values → more creative but less predictable output."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is a buffer overflow vulnerability?",
                "max_tokens": 256,
                "temperature": 0.7,
            }
        }
    }


class ChatResponse(BaseModel):
    """Outgoing chat response payload."""

    response: str = Field(..., description="Generated text from the base model.")
    model: str = Field(..., description="HuggingFace model ID that produced the response.")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "A buffer overflow occurs when a program writes more data...",
                "model": "distilgpt2",
                "latency_ms": 843.21,
            }
        }
    }
