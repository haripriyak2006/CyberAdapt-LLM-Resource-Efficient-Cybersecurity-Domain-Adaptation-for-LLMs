"""
training/utils/__init__.py
Shared constants and types for the CyberAdapt-LLM training utilities.
"""

from __future__ import annotations

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_DATASETS_DIR: Path = PROJECT_ROOT / "data" / "datasets"

# ── Corpus record field names ─────────────────────────────────────────────────
FIELD_TEXT = "text"
FIELD_SOURCE = "source"
FIELD_DOCUMENT_TYPE = "document_type"
FIELD_TOPIC = "topic"
FIELD_LICENSE = "license"

# ── Pipeline defaults ─────────────────────────────────────────────────────────
DEFAULT_MIN_CHARS: int = 80          # discard paragraphs shorter than this
DEFAULT_MIN_WORDS: int = 10          # discard paragraphs with fewer words
DEFAULT_NEAR_DUP_BITS: int = 3       # SimHash Hamming distance threshold
DEFAULT_MAX_SIMHASH_WINDOW: int = 5000  # how many recent hashes to compare against
TOKENS_PER_CHAR_ESTIMATE: float = 0.25  # ≈ 4 chars per BPE token
