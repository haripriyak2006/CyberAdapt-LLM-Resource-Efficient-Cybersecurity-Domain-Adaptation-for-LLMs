"""
tests/test_chat.py
Tests for POST /api/v1/chat endpoint.

Strategy:
  - The LLM service is mocked in ALL tests — no model is downloaded or loaded.
  - Tests focus on request validation, response schema, and error propagation.

Run: pytest tests/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# ── Shared mock result ─────────────────────────────────────────────────────────

_MOCK_RESULT = {
    "response": "A SQL injection attack inserts malicious SQL into a query.",
    "model": "distilgpt2",
    "latency_ms": 123.45,
}


# ── Request validation tests ───────────────────────────────────────────────────

class TestChatRequestValidation:
    """Test that invalid requests are rejected before reaching the service."""

    def test_empty_message_returns_422(self) -> None:
        """An empty string message must be rejected with 422."""
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_missing_message_returns_422(self) -> None:
        """Omitting 'message' entirely must be rejected with 422."""
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_message_too_long_returns_422(self) -> None:
        """A message exceeding 4096 chars must be rejected with 422."""
        response = client.post("/api/v1/chat", json={"message": "x" * 4097})
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_max_tokens_too_low_returns_422(self) -> None:
        """max_tokens=0 (below minimum of 1) must be rejected."""
        response = client.post(
            "/api/v1/chat", json={"message": "hello", "max_tokens": 0}
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_max_tokens_too_high_returns_422(self) -> None:
        """max_tokens=1025 (above maximum of 1024) must be rejected."""
        response = client.post(
            "/api/v1/chat", json={"message": "hello", "max_tokens": 1025}
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_temperature_negative_returns_422(self) -> None:
        """temperature=-0.1 (below minimum of 0.0) must be rejected."""
        response = client.post(
            "/api/v1/chat", json={"message": "hello", "temperature": -0.1}
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_temperature_too_high_returns_422(self) -> None:
        """temperature=2.1 (above maximum of 2.0) must be rejected."""
        response = client.post(
            "/api/v1/chat", json={"message": "hello", "temperature": 2.1}
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"


# ── Success path tests ─────────────────────────────────────────────────────────

class TestChatSuccess:
    """Test successful chat requests with a mocked LLM service."""

    @patch("backend.api.chat.generate_response", return_value=_MOCK_RESULT)
    def test_valid_request_returns_200(self, mock_gen: MagicMock) -> None:
        """A well-formed request must return HTTP 200."""
        response = client.post("/api/v1/chat", json={"message": "What is SQL injection?"})
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    @patch("backend.api.chat.generate_response", return_value=_MOCK_RESULT)
    def test_response_schema(self, mock_gen: MagicMock) -> None:
        """Response body must have 'response', 'model', 'latency_ms'."""
        response = client.post("/api/v1/chat", json={"message": "What is SQL injection?"})
        data = response.json()
        assert "response" in data, "'response' key missing"
        assert "model" in data, "'model' key missing"
        assert "latency_ms" in data, "'latency_ms' key missing"
        assert isinstance(data["response"], str)
        assert isinstance(data["model"], str)
        assert isinstance(data["latency_ms"], float)

    @patch("backend.api.chat.generate_response", return_value=_MOCK_RESULT)
    def test_default_params_accepted(self, mock_gen: MagicMock) -> None:
        """Request with only 'message' (defaults for the rest) must succeed."""
        response = client.post("/api/v1/chat", json={"message": "Explain XSS."})
        assert response.status_code == 200

    @patch("backend.api.chat.generate_response", return_value=_MOCK_RESULT)
    def test_custom_params_accepted(self, mock_gen: MagicMock) -> None:
        """Request with all params specified must pass validation."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Explain CSRF.", "max_tokens": 128, "temperature": 0.5},
        )
        assert response.status_code == 200

    @patch("backend.api.chat.generate_response", return_value=_MOCK_RESULT)
    def test_model_field_is_string(self, mock_gen: MagicMock) -> None:
        """'model' in the response must be a non-empty string."""
        response = client.post("/api/v1/chat", json={"message": "Hello"})
        assert response.json()["model"], "model field should not be empty"


# ── Error path tests ───────────────────────────────────────────────────────────

class TestChatErrors:
    """Test that errors from the LLM service are handled gracefully."""

    @patch(
        "backend.api.chat.generate_response",
        side_effect=__import__(
            "backend.services.llm_service", fromlist=["LLMServiceError"]
        ).LLMServiceError("Model not loaded"),
    )
    def test_llm_service_error_returns_503(self, mock_gen: MagicMock) -> None:
        """LLMServiceError must result in HTTP 503."""
        response = client.post("/api/v1/chat", json={"message": "hello"})
        assert response.status_code == 503, f"Expected 503, got {response.status_code}"

    @patch(
        "backend.api.chat.generate_response",
        side_effect=RuntimeError("Something broke internally"),
    )
    def test_unexpected_error_returns_500(self, mock_gen: MagicMock) -> None:
        """Unexpected runtime errors must result in HTTP 500 (no trace exposed)."""
        response = client.post("/api/v1/chat", json={"message": "hello"})
        assert response.status_code == 500, f"Expected 500, got {response.status_code}"

    @patch(
        "backend.api.chat.generate_response",
        side_effect=RuntimeError("internal details"),
    )
    def test_500_response_does_not_expose_internal_error(self, mock_gen: MagicMock) -> None:
        """The 500 response body must NOT contain raw exception details."""
        response = client.post("/api/v1/chat", json={"message": "hello"})
        body = response.text
        assert "internal details" not in body, "Internal error message leaked to response!"
        assert "RuntimeError" not in body, "Exception class name leaked to response!"
