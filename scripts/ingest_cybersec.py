"""
scripts/ingest_cybersec.py
CLI wrapper to run the RAG ingestion pipeline.

Usage
-----
  python scripts/ingest_cybersec.py                          # ingest data/raw/
  python scripts/ingest_cybersec.py --source-dir data/raw   # explicit source
  python scripts/ingest_cybersec.py --chunk-size 256 --chunk-overlap 32
  python scripts/ingest_cybersec.py --output data/datasets/faiss_index
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
# Silence noisy third-party loggers
for _noisy in ("transformers", "sentence_transformers", "huggingface_hub", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "datasets" / "faiss_index"
DEFAULT_EMBED  = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE  = str(PROJECT_ROOT / "models" / "base")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberAdapt-LLM Phase 7 — Ingest cybersecurity documents into RAG vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/ingest_cybersec.py\n"
            "  python scripts/ingest_cybersec.py --chunk-size 256 --chunk-overlap 32\n"
            "  python scripts/ingest_cybersec.py --source-dir data/raw --output data/datasets/faiss_index\n"
        ),
    )
    parser.add_argument("--source-dir",    type=Path, default=DEFAULT_SOURCE,
                        help=f"Directory with documents to ingest (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output",        type=Path, default=DEFAULT_OUTPUT,
                        help=f"Where to save the FAISS index (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED,
                        help=f"Sentence-transformer model name (default: {DEFAULT_EMBED})")
    parser.add_argument("--cache-dir",     default=DEFAULT_CACHE,
                        help="HuggingFace model cache directory")
    parser.add_argument("--chunk-size",    type=int, default=512,
                        help="Target max characters per chunk (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64,
                        help="Overlap characters between chunks (default: 64)")
    parser.add_argument("--batch-size",    type=int, default=64,
                        help="Embedding batch size (default: 64)")
    args = parser.parse_args()

    if not args.source_dir.exists():
        logger.error("Source directory not found: %s", args.source_dir)
        sys.exit(1)

    from rag.ingest import ingest
    stats = ingest(
        source_dir=args.source_dir,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model_name=args.embedding_model,
        embedding_cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Save stats
    stats_path = args.output / "ingest_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Stats saved: %s", stats_path)

    if stats.get("status") == "ok":
        print()
        print("=" * 55)
        print("  RAG Ingestion Complete")
        print("=" * 55)
        print(f"  Documents  : {stats['num_documents']}")
        print(f"  Chunks     : {stats['num_chunks']}")
        print(f"  Embed dim  : {stats['embedding_dim']}")
        print(f"  Time       : {stats['elapsed_seconds']} s")
        print(f"  Sources    : {stats['source_distribution']}")
        print(f"  Index at   : {stats['output_dir']}")
        print()
        print("  Next step: python scripts/test_rag.py")
        print()
    else:
        logger.error("Ingestion failed: %s", stats.get("message"))
        sys.exit(1)


if __name__ == "__main__":
    main()
