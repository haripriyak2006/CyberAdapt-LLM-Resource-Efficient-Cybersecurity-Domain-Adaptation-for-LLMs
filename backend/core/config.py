"""
backend/core/config.py
Configuration loader for CyberAdapt-LLM.

Priority (highest → lowest):
  1. Environment variables
  2. .env file
  3. configs/base.yaml
  4. Hard-coded defaults
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env as early as possible so env vars are available everywhere
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; return empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _yaml_value(key_path: str, default: Any = None) -> Any:
    """
    Retrieve a dotted key from configs/base.yaml.
    e.g. 'app.port' → yaml['app']['port']
    """
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data = _load_yaml(_PROJECT_ROOT / "configs" / "base.yaml")
    keys = key_path.split(".")
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


# ── Settings model ────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Central settings object.
    All fields are read from environment variables first, then .env, then YAML.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default_factory=lambda: _yaml_value("app.name", "CyberAdapt-LLM"))
    app_version: str = Field(default_factory=lambda: _yaml_value("app.version", "0.1.0"))
    app_phase: int = Field(default_factory=lambda: int(_yaml_value("app.phase", 1)))
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV") or _yaml_value("app.env", "development"))
    app_host: str = Field(default_factory=lambda: os.getenv("APP_HOST") or _yaml_value("app.host", "0.0.0.0"))
    app_port: int = Field(default_factory=lambda: int(os.getenv("APP_PORT") or _yaml_value("app.port", 8000)))
    app_log_level: str = Field(default_factory=lambda: os.getenv("APP_LOG_LEVEL") or _yaml_value("app.log_level", "INFO"))

    # ── API ───────────────────────────────────────────────────────────────────
    api_prefix: str = Field(default_factory=lambda: _yaml_value("api.prefix", "/api/v1"))
    cors_origins: list[str] = Field(
        default_factory=lambda: _yaml_value("api.cors_origins", ["http://localhost:3000"])
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # BASE_MODEL_NAME is the primary env var (Phase 2+).
    # Falls back to BASE_MODEL_ID for backward compat, then YAML, then default.
    base_model_name: str = Field(
        default_factory=lambda: (
            os.getenv("BASE_MODEL_NAME")
            or os.getenv("BASE_MODEL_ID")
            or _yaml_value("model.base_model_name", "distilgpt2")
        )
    )
    model_cache_dir: str = Field(
        default_factory=lambda: os.getenv("MODEL_CACHE_DIR") or _yaml_value("model.model_cache_dir", "./models/base")
    )
    device: str = Field(
        default_factory=lambda: os.getenv("DEVICE") or _yaml_value("model.device", "auto")
    )
    max_new_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS") or _yaml_value("model.max_new_tokens", 256))
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE") or _yaml_value("model.temperature", 0.7))
    )
    use_half_precision: bool = Field(
        default_factory=lambda: str(os.getenv("USE_HALF_PRECISION", "false")).lower() == "true"
    )

    # ── Tokenization (Phase 4) ─────────────────────────────────────────────────
    max_sequence_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SEQUENCE_LENGTH") or _yaml_value("tokenization.max_sequence_length", 1024))
    )
    train_split_ratio: float = Field(
        default_factory=lambda: float(os.getenv("TRAIN_SPLIT_RATIO") or _yaml_value("tokenization.train_split_ratio", 0.9))
    )
    tokenize_seed: int = Field(
        default_factory=lambda: int(os.getenv("TOKENIZE_SEED") or _yaml_value("tokenization.seed", 42))
    )
    tokenized_data_dir: str = Field(
        default_factory=lambda: os.getenv("TOKENIZED_DATA_DIR") or _yaml_value("tokenization.tokenized_data_dir", "./data/datasets/tokenized")
    )
    corpus_file: str = Field(
        default_factory=lambda: os.getenv("CORPUS_FILE") or _yaml_value("tokenization.corpus_file", "./data/processed/cybersecurity_corpus.jsonl")
    )

    # ── RAG ───────────────────────────────────────────────────────────────────
    vector_store_path: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE_PATH") or _yaml_value("rag.vector_store_path", "./data/datasets/faiss_index")
    )
    embedding_model_id: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_ID") or _yaml_value("rag.embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2")
    )
    retrieval_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K") or _yaml_value("rag.retrieval_top_k", 5))
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_output_dir: str = Field(
        default_factory=lambda: os.getenv("EVAL_OUTPUT_DIR") or _yaml_value("evaluation.output_dir", "./evaluation/results")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
