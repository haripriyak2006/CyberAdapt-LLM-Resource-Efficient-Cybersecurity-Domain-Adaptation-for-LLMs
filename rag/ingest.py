"""
rag/ingest.py
Document ingestion pipeline for CyberAdapt-LLM RAG — Phase 7.

Pipeline
--------
  Documents (TXT / JSONL / JSON)
  ↓  load_documents()
  ↓  TextChunker.chunk()
  ↓  EmbeddingModel.encode()
  ↓  VectorStore.add()
  ↓  VectorStore.save()

Supports the raw data files already in data/raw/:
  - *.txt    — plain text (OWASP, NIST)
  - *.jsonl  — one record per line (CVE, MITRE CWE)
  - *.json   — single JSON array or object

Usage
-----
  python scripts/ingest_cybersec.py                     # ingest data/raw/
  python scripts/ingest_cybersec.py --source-dir my/docs --output data/datasets/faiss_index
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Raw document representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    text: str
    source: str         # e.g. "OWASP", "NIST", "NVD", "MITRE"
    document_type: str  # e.g. "guideline", "cve", "cwe", "glossary"
    topic: str          # e.g. "sql_injection", "buffer_overflow"
    license: str        # e.g. "CC-BY-4.0", "Public Domain"
    doc_id: str         # unique identifier


# ─────────────────────────────────────────────────────────────────────────────
# Document loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> list[RawDocument]:
    """Load a plain-text file as one or more documents (split on blank lines)."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    # Guess metadata from filename
    name = path.stem.lower()
    if "owasp" in name:
        source, doc_type, license_ = "OWASP", "guideline", "CC-BY-4.0"
    elif "nist" in name:
        source, doc_type, license_ = "NIST", "glossary", "Public Domain"
    else:
        source, doc_type, license_ = path.stem, "text", "Unknown"

    topic = name.replace("_sample", "").replace("_", " ")

    return [RawDocument(
        text=text,
        source=source,
        document_type=doc_type,
        topic=topic,
        license=license_,
        doc_id=f"{path.stem}_0",
    )]


def _load_jsonl(path: Path) -> list[RawDocument]:
    """Load a JSONL file — one JSON record per line."""
    docs: list[RawDocument] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s line %s", path.name, line_no)
                continue

            text = _extract_text_from_record(record)
            if not text or len(text) < 20:
                continue

            # Guess source + type from filename and record keys
            name = path.stem.lower()
            if "cve" in name:
                source = "NVD"
                doc_type = "cve"
                license_ = "Public Domain"
                topic = record.get("type", "vulnerability")
                doc_id = record.get("id", f"{path.stem}_{line_no}")
            elif "cwe" in name or "mitre" in name:
                source = "MITRE"
                doc_type = "cwe"
                license_ = "CC-BY-4.0"
                topic = record.get("Name", record.get("topic", "weakness"))
                doc_id = f"CWE-{record.get('ID', line_no)}"
            else:
                source = path.stem
                doc_type = "json"
                license_ = "Unknown"
                topic = record.get("topic", path.stem)
                doc_id = f"{path.stem}_{line_no}"

            docs.append(RawDocument(
                text=text,
                source=source,
                document_type=doc_type,
                topic=str(topic)[:80],
                license=license_,
                doc_id=str(doc_id),
            ))
    return docs


def _extract_text_from_record(record: dict) -> str:
    """Extract the most useful text fields from a JSON record."""
    candidates = [
        # NVD CVE format
        "description", "cve_description",
        # MITRE CWE
        "Description", "Extended_Description", "Background_Details",
        # Generic
        "text", "content", "body", "summary", "abstract",
        "name", "Name",
    ]
    parts: list[str] = []
    for key in candidates:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    sub = item.get("value", item.get("text", ""))
                    if isinstance(sub, str) and sub.strip():
                        parts.append(sub.strip())
    return " ".join(parts)


def _load_json(path: Path) -> list[RawDocument]:
    """Load a JSON file (array or object)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        logger.warning("Malformed JSON: %s", path)
        return []

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        return []

    docs: list[RawDocument] = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        text = _extract_text_from_record(record)
        if not text or len(text) < 20:
            continue
        docs.append(RawDocument(
            text=text,
            source=path.stem,
            document_type="json",
            topic=record.get("topic", path.stem),
            license="Unknown",
            doc_id=f"{path.stem}_{i}",
        ))
    return docs


def load_documents(source_dir: Path) -> list[RawDocument]:
    """
    Recursively load all supported documents from ``source_dir``.
    Supports .txt, .jsonl, .json
    """
    docs: list[RawDocument] = []
    extensions = {".txt": _load_txt, ".jsonl": _load_jsonl, ".json": _load_json}

    for path in sorted(source_dir.rglob("*")):
        loader = extensions.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            loaded = loader(path)
            logger.info("  %s → %s raw document(s)", path.name, len(loaded))
            docs.extend(loaded)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    logger.info("Loaded %s raw documents from %s", len(docs), source_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Ingest pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ingest(
    source_dir: Path,
    output_dir: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_cache_dir: Optional[str] = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict:
    """
    Full ingestion pipeline:
    load → chunk → embed → index → save

    Returns a stats dict.
    """
    from rag.chunking import TextChunker
    from rag.embeddings import EmbeddingModel
    from rag.vector_store import VectorStore

    t_start = time.perf_counter()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("=== RAG Ingestion Pipeline ===")
    logger.info("Source : %s", source_dir)
    logger.info("Output : %s", output_dir)
    raw_docs = load_documents(source_dir)
    if not raw_docs:
        logger.error("No documents found in %s", source_dir)
        return {"status": "error", "message": "no documents found"}

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info("Chunking (size=%s, overlap=%s) ...", chunk_size, chunk_overlap)
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in raw_docs:
        chunks = chunker.chunk(
            text=doc.text,
            source=doc.source,
            document_type=doc.document_type,
            topic=doc.topic,
            license=doc.license,
            doc_id=doc.doc_id,
        )
        all_chunks.extend(chunks)

    logger.info("Produced %s chunks from %s documents", len(all_chunks), len(raw_docs))
    if not all_chunks:
        logger.error("Chunking produced 0 chunks.")
        return {"status": "error", "message": "chunking produced 0 chunks"}

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    logger.info("Embedding with %s ...", embedding_model_name)
    embedding_model = EmbeddingModel(
        model_name=embedding_model_name,
        cache_dir=embedding_cache_dir,
        batch_size=batch_size,
    )
    texts = [c.text for c in all_chunks]
    embeddings = embedding_model.encode(texts, show_progress=show_progress)
    logger.info("Embeddings: shape=%s", embeddings.shape)

    # ── Step 4: Index ─────────────────────────────────────────────────────────
    logger.info("Building FAISS index ...")
    store = VectorStore(dimension=embedding_model.dimension)
    store.add(all_chunks, embeddings)

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    store.save(output_dir)

    elapsed = time.perf_counter() - t_start
    stats = {
        "status": "ok",
        "num_documents": len(raw_docs),
        "num_chunks": len(all_chunks),
        "embedding_dim": embedding_model.dimension,
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(output_dir),
        "source_distribution": store.stats()["sources"],
    }

    logger.info("")
    logger.info("-- Ingestion Complete --")
    logger.info("  Documents  : %s", stats["num_documents"])
    logger.info("  Chunks     : %s", stats["num_chunks"])
    logger.info("  Embed dim  : %s", stats["embedding_dim"])
    logger.info("  Time       : %.2f s", elapsed)
    logger.info("  Sources    : %s", stats["source_distribution"])
    logger.info("  Saved to   : %s", output_dir)
    return stats
