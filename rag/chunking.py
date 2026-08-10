"""
rag/chunking.py
Sentence-aware text chunker for CyberAdapt-LLM RAG pipeline.

Splits text into overlapping chunks that respect sentence boundaries.
Chunk size is measured in characters (not tokens) for speed;
the embedding model handles its own token limit internally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Sentence boundary splitter
# ─────────────────────────────────────────────────────────────────────────────

# Matches sentence-ending punctuation followed by space/newline and a capital.
# Conservative — avoids splitting on "e.g.", "U.S.", "Fig. 1", etc.
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight regex."""
    # First normalise newlines
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Split on paragraph breaks first, then sentence boundary
    paragraphs = re.split(r'\n\n+', text)
    sentences: list[str] = []
    for para in paragraphs:
        parts = _SENT_RE.split(para.strip())
        sentences.extend(s.strip() for s in parts if s.strip())
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Chunk data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single text chunk with its source metadata."""
    text: str
    chunk_index: int
    total_chunks: int       # filled in after all chunks are produced
    char_count: int
    # Inherited from source document
    source: str = ""
    document_type: str = ""
    topic: str = ""
    license: str = ""
    doc_id: str = ""        # unique ID of the source document


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits text into overlapping chunks that respect sentence boundaries.

    Parameters
    ----------
    chunk_size    : Target maximum characters per chunk.
    chunk_overlap : Characters of overlap between consecutive chunks.
    min_chunk_size: Chunks shorter than this are merged into the next chunk.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        text: str,
        source: str = "",
        document_type: str = "",
        topic: str = "",
        license: str = "",
        doc_id: str = "",
    ) -> list[TextChunk]:
        """
        Split ``text`` into overlapping chunks and attach metadata.
        Returns an empty list for empty/whitespace-only input.
        """
        text = text.strip()
        if not text:
            return []

        sentences = _split_into_sentences(text)
        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_len: int = 0

        def _flush(idx: int) -> TextChunk:
            chunk_text = " ".join(current_sentences).strip()
            return TextChunk(
                text=chunk_text,
                chunk_index=idx,
                total_chunks=0,         # patched below
                char_count=len(chunk_text),
                source=source,
                document_type=document_type,
                topic=topic,
                license=license,
                doc_id=doc_id,
            )

        for sent in sentences:
            sent_len = len(sent)

            # If adding this sentence would exceed chunk_size, flush first
            if current_len + sent_len > self.chunk_size and current_sentences:
                chunks.append(_flush(len(chunks)))

                # Build overlap: keep trailing sentences that fit in overlap budget
                overlap_sentences: list[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_sentences):
                    if overlap_len + len(prev_sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sent)
            current_len += sent_len

        # Flush remainder
        if current_sentences:
            remainder = " ".join(current_sentences).strip()
            if remainder:
                # Merge very short remainders into previous chunk if possible
                if len(remainder) < self.min_chunk_size and chunks:
                    prev = chunks[-1]
                    merged = prev.text + " " + remainder
                    chunks[-1] = TextChunk(
                        text=merged,
                        chunk_index=prev.chunk_index,
                        total_chunks=0,
                        char_count=len(merged),
                        source=prev.source,
                        document_type=prev.document_type,
                        topic=prev.topic,
                        license=prev.license,
                        doc_id=prev.doc_id,
                    )
                else:
                    chunks.append(_flush(len(chunks)))

        # Patch total_chunks
        n = len(chunks)
        for c in chunks:
            c.total_chunks = n

        return chunks
