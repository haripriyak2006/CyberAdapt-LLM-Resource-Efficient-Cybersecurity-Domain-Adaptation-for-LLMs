"""
rag/vector_store.py
FAISS vector store for CyberAdapt-LLM RAG pipeline.

Uses IndexFlatIP (exact inner-product search).
With L2-normalised embeddings this equals cosine similarity, scores in [0, 1].

Persists to disk as two files:
  <path>/index.faiss    — the FAISS index binary
  <path>/metadata.json  — chunk texts + metadata list
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from rag.chunking import TextChunk

logger = logging.getLogger(__name__)

_INDEX_FILE    = "index.faiss"
_METADATA_FILE = "metadata.json"


class VectorStore:
    """
    FAISS-backed vector store with persistent metadata.

    Supports:
    - add(chunks, embeddings)  — bulk index
    - search(query_vec, top_k) — return (chunks, scores)
    - save(path)               — persist to disk
    - load(path)               — restore from disk
    """

    def __init__(self, dimension: int) -> None:
        import faiss
        self.dimension = dimension
        # IndexFlatIP: exact inner-product (= cosine for normalised vecs)
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[dict] = []   # serialisable metadata

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the index.

        Parameters
        ----------
        chunks     : list of TextChunk objects (metadata preserved)
        embeddings : float32 array of shape (len(chunks), dimension)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) count mismatch"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != store dimension {self.dimension}"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)

        for chunk in chunks:
            self._chunks.append({
                "text":          chunk.text,
                "chunk_index":   chunk.chunk_index,
                "total_chunks":  chunk.total_chunks,
                "char_count":    chunk.char_count,
                "source":        chunk.source,
                "document_type": chunk.document_type,
                "topic":         chunk.topic,
                "license":       chunk.license,
                "doc_id":        chunk.doc_id,
            })

        logger.info(
            "Added %s chunks to index (total: %s)",
            len(chunks), self._index.ntotal
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[dict, float]]:
        """
        Search for the top-K most similar chunks.

        Parameters
        ----------
        query_embedding : float32 array of shape (1, dimension) or (dimension,)
        top_k           : number of results to return

        Returns
        -------
        list of (chunk_metadata_dict, similarity_score) sorted by score DESC
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty — no results.")
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        q = np.ascontiguousarray(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))

        return results  # already sorted DESC by FAISS

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Persist index and metadata to ``directory``."""
        import faiss
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path    = directory / _INDEX_FILE
        metadata_path = directory / _METADATA_FILE

        faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(
            json.dumps(
                {
                    "dimension":  self.dimension,
                    "num_chunks": len(self._chunks),
                    "chunks":     self._chunks,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Vector store saved: %s chunks → %s",
            len(self._chunks), directory,
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load a previously saved vector store from ``directory``."""
        import faiss
        directory = Path(directory)
        index_path    = directory / _INDEX_FILE
        metadata_path = directory / _METADATA_FILE

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {directory}. "
                "Run: python scripts/ingest_cybersec.py"
            )

        data = json.loads(metadata_path.read_text("utf-8"))
        dimension = data["dimension"]

        store = cls(dimension=dimension)
        store._index  = faiss.read_index(str(index_path))
        store._chunks = data["chunks"]

        logger.info(
            "Vector store loaded: %s chunks from %s",
            store._index.ntotal, directory,
        )
        return store

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return self._index.ntotal

    def stats(self) -> dict:
        sources = {}
        for c in self._chunks:
            src = c.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "num_chunks": self.num_chunks,
            "dimension":  self.dimension,
            "sources":    sources,
        }
