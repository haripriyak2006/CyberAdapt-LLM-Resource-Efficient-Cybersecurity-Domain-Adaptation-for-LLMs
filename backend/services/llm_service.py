"""
backend/services/llm_service.py
LLM service layer — bridges the FastAPI routes and the ModelLoader.

Design:
  - One ModelLoader singleton per process (lazy, thread-safe)
  - generate_response() is the single public function for all chat calls
  - All model errors are caught here; callers receive structured dicts or
    raise LLMServiceError with a safe public message
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


# ── Custom exception ──────────────────────────────────────────────────────────

class LLMServiceError(Exception):
    """
    Raised when the LLM service cannot fulfil a request.
    The message is safe to expose to API users (no stack traces).
    """


# ── Singleton loader ──────────────────────────────────────────────────────────

_loader = None  # type: ignore[assignment]
_loader_error: Optional[str] = None  # set if the loader failed to initialise


def _get_loader():
    """Return the shared ModelLoader, initialising it on first call."""
    global _loader, _loader_error

    if _loader is not None:
        return _loader

    if _loader_error is not None:
        raise LLMServiceError(_loader_error)

    try:
        from models.model_loader import ModelLoader

        settings = get_settings()
        logger.info("Initialising ModelLoader with model: %s", settings.base_model_name)

        _loader = ModelLoader(
            model_name=settings.base_model_name,
            cache_dir=settings.model_cache_dir,
            device=settings.device,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )
        return _loader

    except Exception as exc:
        _loader_error = f"Model loader failed to initialise: {type(exc).__name__}"
        logger.exception("ModelLoader init failed: %s", exc)
        raise LLMServiceError(_loader_error) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def generate_response(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Generate a model response for the given prompt.

    Args:
        prompt:      The user's text prompt.
        max_tokens:  Override max new tokens for this request.
        temperature: Override sampling temperature for this request.

    Returns:
        dict with keys: ``response``, ``model``, ``latency_ms``

    Raises:
        LLMServiceError: If the model is unavailable or generation fails.
    """
    if not prompt or not prompt.strip():
        raise LLMServiceError("Prompt must not be empty.")

    loader = _get_loader()

    t0 = time.perf_counter()
    try:
        response_text = loader.generate(
            prompt=prompt.strip(),
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        logger.exception("Generation failed for prompt (first 80 chars): %.80s", prompt)
        raise LLMServiceError(
            "The model encountered an error during generation. Please try again."
        ) from exc
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        "Generated %s chars in %s ms | model=%s | device=%s",
        len(response_text),
        latency_ms,
        loader.model_name,
        loader.device,
    )

    return {
        "response": response_text,
        "model": loader.model_name,
        "latency_ms": latency_ms,
    }


def get_model_status() -> dict:
    """
    Return current model loading status.
    Safe to call at any time (never triggers a load).
    """
    global _loader, _loader_error
    if _loader_error:
        return {"loaded": False, "error": _loader_error, "model": None, "device": None}
    if _loader is None:
        return {"loaded": False, "error": None, "model": None, "device": None}
    return {
        "loaded": _loader.is_loaded,
        "error": None,
        "model": _loader.model_name,
        "device": _loader.device,
    }
