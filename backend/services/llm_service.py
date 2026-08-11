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

_loaders: dict[str, "ModelLoader"] = {}
_loader_errors: dict[str, str] = {}


def _get_loader(model_name: Optional[str] = None):
    """Return the shared ModelLoader for a specific model, initialising it on first call."""
    global _loaders, _loader_errors
    
    settings = get_settings()
    target_model = model_name or settings.base_model_name

    if target_model in _loaders:
        return _loaders[target_model]

    if target_model in _loader_errors:
        raise LLMServiceError(_loader_errors[target_model])

    try:
        from models.model_loader import ModelLoader

        logger.info("Initialising ModelLoader with model: %s", target_model)

        loader = ModelLoader(
            model_name=target_model,
            cache_dir=settings.model_cache_dir,
            device=settings.device,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )
        _loaders[target_model] = loader
        return loader

    except Exception as exc:
        err = f"Model loader failed to initialise for {target_model}: {type(exc).__name__}"
        _loader_errors[target_model] = err
        logger.exception("ModelLoader init failed for %s: %s", target_model, exc)
        raise LLMServiceError(err) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def generate_response(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
) -> dict:
    """
    Generate a model response for the given prompt.

    Args:
        prompt:      The user's text prompt.
        max_tokens:  Override max new tokens for this request.
        temperature: Override sampling temperature for this request.
        model_name:  Optional specific model to load (defaults to base_model_name).

    Returns:
        dict with keys: ``response``, ``model``, ``latency_ms``

    Raises:
        LLMServiceError: If the model is unavailable or generation fails.
    """
    if not prompt or not prompt.strip():
        raise LLMServiceError("Prompt must not be empty.")

    loader = _get_loader(model_name)

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


def get_model_status(model_name: Optional[str] = None) -> dict:
    """
    Return current model loading status.
    Safe to call at any time (never triggers a load).
    """
    global _loaders, _loader_errors
    settings = get_settings()
    target_model = model_name or settings.base_model_name

    if target_model in _loader_errors:
        return {"loaded": False, "error": _loader_errors[target_model], "model": None, "device": None}
    
    loader = _loaders.get(target_model)
    if loader is None:
        return {"loaded": False, "error": None, "model": None, "device": None}
    
    return {
        "loaded": loader.is_loaded,
        "error": None,
        "model": loader.model_name,
        "device": loader.device,
    }
