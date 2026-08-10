"""
tests/test_health.py
Tests for GET /health endpoint.

Run: pytest tests/ -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test suite for the /health liveness probe."""

    def test_health_returns_200(self) -> None:
        """Health endpoint must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )

    def test_health_response_schema(self) -> None:
        """Response body must conform to the HealthResponse schema."""
        response = client.get("/health")
        data = response.json()

        # Required fields
        assert data["status"] == "ok", f"status should be 'ok', got: {data.get('status')}"
        assert data["service"] == "CyberAdapt-LLM", f"Unexpected service name: {data.get('service')}"
        assert isinstance(data["version"], str), "version must be a string"
        assert isinstance(data["phase"], int), "phase must be an integer"
        assert isinstance(data["uptime_seconds"], float), "uptime_seconds must be a float"
        assert isinstance(data["environment"], str), "environment must be a string"

    def test_health_phase_is_1(self) -> None:
        """Phase 1 — phase field must equal 1."""
        response = client.get("/health")
        data = response.json()
        assert data["phase"] == 1, f"Expected phase=1, got {data.get('phase')}"

    def test_health_content_type_is_json(self) -> None:
        """Response Content-Type must be application/json."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", ""), (
            f"Expected JSON content-type, got: {response.headers.get('content-type')}"
        )
