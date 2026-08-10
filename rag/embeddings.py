"""
rag/embeddings.py
Embedding model wrapper for CyberAdapt-LLM RAG pipeline.

Uses sentence-transformers for dense embeddings.
Embeddings are L2-normalised so inner-product == cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wraps a SentenceTransformer model for batch text encoding.

    All embeddings are L2-normalised on output so that inner-product
    search in FAISS is equivalent to cosine similarity, with scores in [0, 1].
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.cache_dir  = cache_dir
        self.batch_size = batch_size
        self._device    = device
        self._model     = None

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", self.model_name)
        kwargs: dict = {}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        if self._device:
            kwargs["device"] = self._device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        dim = self._model.get_embedding_dimension()
        logger.info(
            "Embedding model loaded | dim=%s | device=%s",
            dim,
            self._model.device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads the model if needed)."""
        self._load()
        return self._model.get_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of strings and return L2-normalised embeddings.

        Returns
        -------
        np.ndarray of shape (len(texts), dim), dtype float32
        """
        self._load()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalise → cosine = inner product
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (1, dim)."""
        return self.encode([query])
