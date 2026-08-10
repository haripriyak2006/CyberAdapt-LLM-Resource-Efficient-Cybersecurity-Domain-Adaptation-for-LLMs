"""
backend/core/limits.py
Configurable request limits for CyberAdapt-LLM Phase 9.

All limits are read from environment variables with sensible defaults.
Services and routes import from here — never hard-code limits inline.
"""

from __future__ import annotations

import os

# ── Text input limits ─────────────────────────────────────────────────────────
MAX_CHAT_CHARS:       int = int(os.getenv("MAX_CHAT_CHARS",       "4096"))
MAX_QUESTION_CHARS:   int = int(os.getenv("MAX_QUESTION_CHARS",   "2048"))
MAX_INCIDENT_CHARS:   int = int(os.getenv("MAX_INCIDENT_CHARS",   "5000"))
MAX_DOCUMENT_CHARS:   int = int(os.getenv("MAX_DOCUMENT_CHARS",   "10000"))

# ── File upload limits ────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES:  int = int(os.getenv("MAX_FILE_SIZE_BYTES",  str(2 * 1024 * 1024)))  # 2 MB
MAX_FILE_SIZE_MB:     float = MAX_FILE_SIZE_BYTES / (1024 * 1024)

# Allowed MIME types for document upload
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/pdf",             # PDF — content extracted server-side
    "application/octet-stream",    # generic binary (allow, validated by extension)
})

# Allowed file extensions (secondary check)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".text", ".log",
    ".json", ".jsonl",
    ".pdf",
    ".csv",
})

# ── Generation limits ─────────────────────────────────────────────────────────
MAX_NEW_TOKENS_CHAT:  int = int(os.getenv("MAX_NEW_TOKENS_CHAT",  "512"))
MAX_NEW_TOKENS_RAG:   int = int(os.getenv("MAX_NEW_TOKENS_RAG",   "512"))
MAX_NEW_TOKENS_ANALYSIS: int = int(os.getenv("MAX_NEW_TOKENS_ANALYSIS", "512"))

# ── Retrieval limits ──────────────────────────────────────────────────────────
MAX_RAG_TOP_K:        int = int(os.getenv("MAX_RAG_TOP_K",        "20"))

# ── Report limits ─────────────────────────────────────────────────────────────
MAX_AFFECTED_ASSETS:  int = int(os.getenv("MAX_AFFECTED_ASSETS",  "50"))
