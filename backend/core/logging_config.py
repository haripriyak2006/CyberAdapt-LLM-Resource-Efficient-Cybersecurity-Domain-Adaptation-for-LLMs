"""
backend/core/logging_config.py
Structured logging setup for CyberAdapt-LLM.

Features:
  - JSON-structured output in production, human-readable in development
  - Automatic secret redaction for common sensitive key names
  - Single call to configure_logging() sets up the entire application
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

# ── Secret Redaction ──────────────────────────────────────────────────────────

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(api[_-]?key\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(token\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(password\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(secret\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(authorization\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),  # OpenAI-style keys
]


def _redact(text: str) -> str:
    """Replace secret values in a log message with [REDACTED]."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\g<1>[REDACTED]" if pattern.groups else "[REDACTED]", text)
    return text


class _RedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from log records."""

    def _redact_arg(self, arg: Any) -> Any:
        if isinstance(arg, str):
            return _redact(arg)
        return arg

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.msg = _redact(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact_arg(v) for k, v in record.args.items()}
            else:
                record.args = tuple(self._redact_arg(a) for a in record.args)
        return True


# ── Formatters ────────────────────────────────────────────────────────────────

_DEV_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DEV_DATE_FORMAT = "%H:%M:%S"


class _DevFormatter(logging.Formatter):
    """Colourised human-readable formatter for development."""

    _COLOURS: dict[int, str] = {
        logging.DEBUG: "\033[36m",    # cyan
        logging.INFO: "\033[32m",     # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",    # red
        logging.CRITICAL: "\033[35m", # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self._COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname}{self._RESET}"
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for production / structured logging."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        import json
        import time

        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# ── Public API ────────────────────────────────────────────────────────────────

def configure_logging(log_level: str = "INFO", env: str = "development") -> None:
    """
    Configure the root logger once at application startup.

    Args:
        log_level: Logging level string (DEBUG / INFO / WARNING / ERROR).
        env: 'development' uses colourised text; anything else uses JSON.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.addFilter(_RedactingFilter())

    if env == "development":
        formatter = _DevFormatter(fmt=_DEV_FORMAT, datefmt=_DEV_DATE_FORMAT)
    else:
        formatter = _JsonFormatter()

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate log lines
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s env=%s", log_level.upper(), env
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — returns a named logger."""
    return logging.getLogger(name)
