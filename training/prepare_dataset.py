"""
training/prepare_dataset.py
Cybersecurity corpus preparation pipeline — Phase 3.

Pipeline stages (in order):
  1. Document loading        — TXT, JSON, JSONL, PDF (optional)
  2. Text extraction         — pull text from each format
  3. Unicode normalization   — NFC, smart-quote replacement
  4. Whitespace normalization
  5. Formatting noise removal
  6. Paragraph segmentation  — each paragraph becomes a corpus record
  7. Short/empty filtering   — discard below MIN_CHARS / MIN_WORDS
  8. Exact deduplication     — SHA-256 of normalised text
  9. Near-deduplication      — 64-bit SimHash, Hamming distance ≤ threshold
  10. Metadata preservation  — source, document_type, topic, license
  11. Statistics             — per-source counts, overall totals

Output: data/processed/cybersecurity_corpus.jsonl
        data/processed/dataset_stats.json

Usage:
    python training/prepare_dataset.py
    python training/prepare_dataset.py --input-dir data/raw --min-chars 100
    python training/prepare_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.utils import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_MAX_SIMHASH_WINDOW,
    DEFAULT_MIN_CHARS,
    DEFAULT_MIN_WORDS,
    DEFAULT_NEAR_DUP_BITS,
    TOKENS_PER_CHAR_ESTIMATE,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Source metadata registry ──────────────────────────────────────────────────
# Maps filename stem prefixes → (source_label, topic, license)
_SOURCE_REGISTRY: dict[str, tuple[str, str, str]] = {
    "nist":        ("NIST",         "cybersecurity",    "public-domain"),
    "owasp":       ("OWASP",        "web-security",     "CC BY-SA 3.0"),
    "cve":         ("NVD/CVE",      "vulnerabilities",  "public-domain"),
    "mitre_cwe":   ("MITRE CWE",    "weaknesses",       "public-domain"),
    "mitre_attack":("MITRE ATT&CK", "threat-intel",     "CC BY 4.0"),
    "ietf_rfc":    ("IETF RFC",     "protocols",        "IETF Trust"),
    "wikipedia":   ("Wikipedia",    "cybersecurity",    "CC BY-SA 4.0"),
    "cisa":        ("CISA",         "cybersecurity",    "public-domain"),
}

def _lookup_source(stem: str) -> tuple[str, str, str]:
    """Return (source, topic, license) for a filename stem."""
    stem_lower = stem.lower()
    for prefix, meta in _SOURCE_REGISTRY.items():
        if stem_lower.startswith(prefix):
            return meta
    return ("unknown", "cybersecurity", "unknown")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    """Represents a document as loaded from disk, before cleaning."""
    text: str
    file_path: str
    document_type: str   # txt | json | jsonl | pdf
    source: str = "unknown"
    topic: str = "cybersecurity"
    license: str = "unknown"


@dataclass
class CorpusRecord:
    """A single record in the output JSONL corpus."""
    text: str
    source: str
    document_type: str
    topic: str
    license: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Text Cleaner ──────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Applies a deterministic sequence of text normalization steps.
    All methods are stateless and can be called independently.
    """

    # ── Regex patterns (compiled once) ────────────────────────────────────────
    _CTRL_CHARS       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _PAGE_NUMBER      = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.MULTILINE)
    _HEADER_FOOTER    = re.compile(
        r"^(Page\s+\d+|DRAFT|Draft|Confidential|CONFIDENTIAL|"
        r"Copyright\s+\d{4}|All Rights Reserved).*$",
        re.MULTILINE | re.IGNORECASE,
    )
    _URL_ONLY_LINE    = re.compile(r"^\s*https?://\S+\s*$", re.MULTILINE)
    _EMAIL_ONLY_LINE  = re.compile(r"^\s*[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\s*$", re.MULTILINE)
    _DASHES_ONLY      = re.compile(r"^\s*[-=_*#]{4,}\s*$", re.MULTILINE)
    _MULTI_SPACE      = re.compile(r"[ \t]{2,}")
    _MULTI_NEWLINE    = re.compile(r"\n{3,}")

    # Smart-quote / typographic substitutions
    _SMART_QUOTES = str.maketrans({
        "\u2018": "'", "\u2019": "'",   # left/right single quotation marks
        "\u201c": '"', "\u201d": '"',   # left/right double quotation marks
        "\u2013": "-", "\u2014": "--",  # en-dash, em-dash
        "\u2026": "...",               # ellipsis
        "\u00a0": " ",                 # non-breaking space
    })

    @classmethod
    def clean(cls, text: str) -> str:
        """Full cleaning pipeline. Returns cleaned text."""
        if not text:
            return ""

        # 1. Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # 2. Replace typographic characters with ASCII equivalents
        text = text.translate(cls._SMART_QUOTES)

        # 3. Strip control characters (keep \n, \t)
        text = cls._CTRL_CHARS.sub("", text)

        # 4. Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 5. Remove formatting noise lines
        text = cls._PAGE_NUMBER.sub("", text)
        text = cls._HEADER_FOOTER.sub("", text)
        text = cls._URL_ONLY_LINE.sub("", text)
        text = cls._EMAIL_ONLY_LINE.sub("", text)
        text = cls._DASHES_ONLY.sub("", text)

        # 6. Normalize whitespace within lines
        text = cls._MULTI_SPACE.sub(" ", text)

        # 7. Collapse excessive blank lines
        text = cls._MULTI_NEWLINE.sub("\n\n", text)

        return text.strip()


# ── Deduplicator ──────────────────────────────────────────────────────────────

class Deduplicator:
    """
    Two-stage deduplication:
      Stage 1 — Exact: SHA-256 of normalised text (O(1) lookup via set)
      Stage 2 — Near:  64-bit SimHash, Hamming distance ≤ threshold

    SimHash is O(n·w) per document where w is the vocabulary window.
    To keep memory bounded, near-dup checks compare against a sliding
    window of the most recent `max_window` SimHashes.
    """

    def __init__(
        self,
        near_dup_threshold: int = DEFAULT_NEAR_DUP_BITS,
        max_window: int = DEFAULT_MAX_SIMHASH_WINDOW,
    ) -> None:
        self._exact_hashes: set[str] = set()
        self._simhashes: list[int] = []
        self._threshold = near_dup_threshold
        self._max_window = max_window
        self.exact_dup_count = 0
        self.near_dup_count = 0

    @staticmethod
    def _normalise(text: str) -> str:
        """Produce a canonical form for hashing: lower-cased, whitespace collapsed."""
        return " ".join(text.lower().split())

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _simhash(text: str) -> int:
        """
        Compute a 64-bit SimHash.
        Feature set: word unigrams. Each word is hashed with MD5.
        The sign of each bit position is accumulated and finalised.
        """
        words = text.lower().split()
        vector = [0] * 64
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8", errors="replace")).hexdigest(), 16)
            for i in range(64):
                vector[i] += 1 if (h >> i) & 1 else -1
        result = 0
        for i in range(64):
            if vector[i] > 0:
                result |= 1 << i
        return result

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def is_duplicate(self, text: str) -> tuple[bool, str]:
        """
        Returns (is_dup, reason).
        ``reason`` is one of 'exact', 'near', or '' (not a duplicate).
        Side-effect: registers the text if it is NOT a duplicate.
        """
        norm = self._normalise(text)

        # Stage 1 — exact
        h = self._sha256(norm)
        if h in self._exact_hashes:
            self.exact_dup_count += 1
            return True, "exact"

        # Stage 2 — near (sliding window)
        sh = self._simhash(norm)
        window = self._simhashes[-self._max_window:]
        for existing in window:
            if self._hamming(sh, existing) <= self._threshold:
                self.near_dup_count += 1
                return True, "near"

        # Not a duplicate — register
        self._exact_hashes.add(h)
        self._simhashes.append(sh)
        return False, ""

    @property
    def stats(self) -> dict:
        return {
            "exact_duplicates_removed": self.exact_dup_count,
            "near_duplicates_removed": self.near_dup_count,
            "unique_seen": len(self._exact_hashes),
        }


# ── Document Loader ───────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Loads documents from disk.
    Supported formats: .txt, .json, .jsonl, .pdf (optional — requires pypdf).

    For JSON/JSONL the ``text_field`` parameter names the key that holds
    the main text content.  Extra keys are merged into metadata.
    """

    # Fields to try (in priority order) when auto-detecting the text field
    _AUTO_TEXT_FIELDS = ["text", "description", "extended_description",
                         "content", "body", "abstract"]

    def __init__(
        self,
        text_field: str = "description",
        encoding: str = "utf-8",
    ) -> None:
        self._text_field = text_field
        self._encoding = encoding

    # ── Private helpers ───────────────────────────────────────────────────────

    def _meta_from_path(self, path: Path) -> tuple[str, str, str]:
        """Derive (source, topic, license) from the file stem."""
        return _lookup_source(path.stem)

    def _detect_text_field(self, record: dict) -> Optional[str]:
        """Return the first matching text-field key found in a dict."""
        for key in self._AUTO_TEXT_FIELDS:
            if key in record and isinstance(record[key], str):
                return key
        # Fall back: first string-valued key
        for key, val in record.items():
            if isinstance(val, str) and len(val) > 20:
                return key
        return None

    def _extract_from_record(self, record: dict, fallback_field: str) -> Optional[str]:
        """Extract the text from a JSON record dict."""
        # Explicit field first
        if fallback_field in record and isinstance(record[fallback_field], str):
            return record[fallback_field]
        # Auto-detect
        field = self._detect_text_field(record)
        if field:
            # Concatenate extra text fields
            parts = [record[field]]
            for extra in self._AUTO_TEXT_FIELDS:
                if extra != field and extra in record and isinstance(record[extra], str):
                    parts.append(record[extra])
            return " ".join(parts)
        return None

    # ── Loaders by format ─────────────────────────────────────────────────────

    def _load_txt(self, path: Path) -> Iterator[RawDocument]:
        source, topic, license_ = self._meta_from_path(path)
        try:
            text = path.read_text(encoding=self._encoding, errors="replace")
            yield RawDocument(
                text=text,
                file_path=str(path),
                document_type="txt",
                source=source,
                topic=topic,
                license=license_,
            )
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)

    def _load_json(self, path: Path) -> Iterator[RawDocument]:
        source, topic, license_ = self._meta_from_path(path)
        try:
            data = json.loads(path.read_text(encoding=self._encoding, errors="replace"))
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
            else:
                logger.warning("Unsupported JSON structure in %s", path)
                return
            for record in records:
                text = self._extract_from_record(record, self._text_field)
                if text:
                    # Merge source/topic/license from record if present
                    rec_source = record.get("source", source)
                    rec_topic  = record.get("topic", topic)
                    rec_license= record.get("license", license_)
                    yield RawDocument(
                        text=text,
                        file_path=str(path),
                        document_type="json",
                        source=rec_source,
                        topic=rec_topic,
                        license=rec_license,
                    )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cannot parse %s: %s", path, exc)

    def _load_jsonl(self, path: Path) -> Iterator[RawDocument]:
        source, topic, license_ = self._meta_from_path(path)
        try:
            for line_no, line in enumerate(
                path.open(encoding=self._encoding, errors="replace"), start=1
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSONL line %d in %s", line_no, path)
                    continue
                if not isinstance(record, dict):
                    continue
                text = self._extract_from_record(record, self._text_field)
                if text:
                    rec_source  = record.get("source", source)
                    rec_topic   = record.get("topic", topic)
                    rec_license = record.get("license", license_)
                    yield RawDocument(
                        text=text,
                        file_path=str(path),
                        document_type="jsonl",
                        source=rec_source,
                        topic=rec_topic,
                        license=rec_license,
                    )
        except OSError as exc:
            logger.warning("Cannot open %s: %s", path, exc)

    def _load_pdf(self, path: Path) -> Iterator[RawDocument]:
        """Load PDF using pypdf (optional dependency)."""
        try:
            import pypdf  # type: ignore[import]
        except ImportError:
            logger.warning(
                "pypdf not installed — skipping %s. "
                "Install with: pip install pypdf",
                path,
            )
            return

        source, topic, license_ = self._meta_from_path(path)
        try:
            reader = pypdf.PdfReader(str(path))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
            text = "\n\n".join(pages)
            if text.strip():
                yield RawDocument(
                    text=text,
                    file_path=str(path),
                    document_type="pdf",
                    source=source,
                    topic=topic,
                    license=license_,
                )
        except Exception as exc:
            logger.warning("Cannot parse PDF %s: %s", path, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_file(self, path: Path) -> Iterator[RawDocument]:
        """Yield RawDocument objects from a single file."""
        suffix = path.suffix.lower()
        if suffix == ".txt":
            yield from self._load_txt(path)
        elif suffix == ".json":
            yield from self._load_json(path)
        elif suffix == ".jsonl":
            yield from self._load_jsonl(path)
        elif suffix == ".pdf":
            yield from self._load_pdf(path)
        else:
            logger.debug("Skipping unsupported file type: %s", path)

    def load_directory(self, directory: Path) -> Iterator[RawDocument]:
        """Recursively yield RawDocuments from all supported files in a directory."""
        supported = {".txt", ".json", ".jsonl", ".pdf"}
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in supported)
        logger.info("Found %d supported files in %s", len(files), directory)
        for path in files:
            logger.debug("Loading: %s", path.name)
            yield from self.load_file(path)


# ── Paragraph Segmenter ───────────────────────────────────────────────────────

class ParagraphSegmenter:
    """
    Splits document text into paragraphs (double-newline boundaries).
    Applies a minimum length filter.
    """

    _PARA_SPLIT = re.compile(r"\n{2,}")

    def __init__(
        self,
        min_chars: int = DEFAULT_MIN_CHARS,
        min_words: int = DEFAULT_MIN_WORDS,
    ) -> None:
        self._min_chars = min_chars
        self._min_words = min_words

    def segment(self, text: str) -> list[str]:
        """
        Split ``text`` into paragraphs and filter short ones.
        Returns a list of clean paragraph strings.
        """
        paragraphs = self._PARA_SPLIT.split(text)
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) < self._min_chars:
                continue
            word_count = len(para.split())
            if word_count < self._min_words:
                continue
            result.append(para)
        return result


# ── Pipeline Statistics ───────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    """Accumulates statistics across the entire pipeline run."""
    files_loaded: int = 0
    raw_documents: int = 0
    paragraphs_extracted: int = 0
    paragraphs_filtered_short: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    records_written: int = 0

    total_chars: int = 0
    total_words: int = 0

    source_counts: dict = field(default_factory=lambda: defaultdict(int))
    doc_type_counts: dict = field(default_factory=lambda: defaultdict(int))
    topic_counts: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def estimated_tokens(self) -> int:
        return int(self.total_chars * TOKENS_PER_CHAR_ESTIMATE)

    def to_dict(self) -> dict:
        return {
            "files_loaded": self.files_loaded,
            "raw_documents": self.raw_documents,
            "paragraphs_extracted": self.paragraphs_extracted,
            "paragraphs_filtered_short": self.paragraphs_filtered_short,
            "exact_duplicates_removed": self.exact_duplicates,
            "near_duplicates_removed": self.near_duplicates,
            "records_written": self.records_written,
            "total_characters": self.total_chars,
            "total_words": self.total_words,
            "estimated_tokens": self.estimated_tokens,
            "source_distribution": dict(self.source_counts),
            "document_type_distribution": dict(self.doc_type_counts),
            "topic_distribution": dict(self.topic_counts),
        }


# ── Corpus Builder ────────────────────────────────────────────────────────────

class CorpusBuilder:
    """
    Orchestrates the full pipeline:
    Load → Clean → Segment → Filter → Deduplicate → Write
    """

    def __init__(
        self,
        input_dir: Path = DATA_RAW_DIR,
        output_file: Path = DATA_PROCESSED_DIR / "cybersecurity_corpus.jsonl",
        min_chars: int = DEFAULT_MIN_CHARS,
        min_words: int = DEFAULT_MIN_WORDS,
        near_dup_threshold: int = DEFAULT_NEAR_DUP_BITS,
        dry_run: bool = False,
        text_field: str = "description",
    ) -> None:
        self.input_dir = input_dir
        self.output_file = output_file
        self.min_chars = min_chars
        self.min_words = min_words
        self.dry_run = dry_run

        self._loader = DocumentLoader(text_field=text_field)
        self._cleaner = TextCleaner()
        self._segmenter = ParagraphSegmenter(min_chars=min_chars, min_words=min_words)
        self._deduplicator = Deduplicator(near_dup_threshold=near_dup_threshold)
        self._stats = PipelineStats()

    # ── Internal pipeline steps ───────────────────────────────────────────────

    def _process_document(self, doc: RawDocument) -> Iterator[CorpusRecord]:
        """Clean, segment, and filter a single raw document."""
        self._stats.raw_documents += 1

        clean_text = self._cleaner.clean(doc.text)
        if not clean_text:
            return

        paragraphs = self._segmenter.segment(clean_text)
        self._stats.paragraphs_extracted += len(paragraphs)

        for para in paragraphs:
            is_dup, reason = self._deduplicator.is_duplicate(para)
            if is_dup:
                if reason == "exact":
                    self._stats.exact_duplicates += 1
                else:
                    self._stats.near_duplicates += 1
                logger.debug("Dropped %s-duplicate (%.60s...)", reason, para)
                continue

            self._stats.total_chars += len(para)
            self._stats.total_words += len(para.split())
            self._stats.source_counts[doc.source] += 1
            self._stats.doc_type_counts[doc.document_type] += 1
            self._stats.topic_counts[doc.topic] += 1
            self._stats.records_written += 1

            yield CorpusRecord(
                text=para,
                source=doc.source,
                document_type=doc.document_type,
                topic=doc.topic,
                license=doc.license,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> PipelineStats:
        """
        Execute the full pipeline.

        Returns:
            PipelineStats with all counts populated.
        """
        logger.info("=== CyberAdapt-LLM Corpus Pipeline — Phase 3 ===")
        logger.info("Input  : %s", self.input_dir)
        logger.info("Output : %s", self.output_file)
        logger.info("Dry run: %s", self.dry_run)

        if not self.input_dir.exists():
            logger.error("Input directory does not exist: %s", self.input_dir)
            sys.exit(1)

        if not self.dry_run:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            out_handle = self.output_file.open("w", encoding="utf-8")
        else:
            out_handle = None

        seen_files: set[str] = set()

        try:
            for doc in self._loader.load_directory(self.input_dir):
                file_key = doc.file_path
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    self._stats.files_loaded += 1

                for record in self._process_document(doc):
                    if out_handle:
                        out_handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        finally:
            if out_handle:
                out_handle.close()

        # Sync dedup stats
        self._stats.exact_duplicates = self._deduplicator.exact_dup_count
        self._stats.near_duplicates  = self._deduplicator.near_dup_count

        self._print_summary()
        return self._stats

    def _print_summary(self) -> None:
        s = self._stats
        logger.info("")
        logger.info("── Pipeline Summary ──────────────────────────────────")
        logger.info("  Files loaded          : %s", s.files_loaded)
        logger.info("  Raw documents         : %s", s.raw_documents)
        logger.info("  Paragraphs extracted  : %s", s.paragraphs_extracted)
        logger.info("  Exact dups removed    : %s", s.exact_duplicates)
        logger.info("  Near-dups removed     : %s", s.near_duplicates)
        logger.info("  Records written       : %s", s.records_written)
        logger.info("  Total characters      : %s", f"{s.total_chars:,}")
        logger.info("  Total words           : %s", f"{s.total_words:,}")
        logger.info("  Estimated tokens      : %s", f"{s.estimated_tokens:,}")
        logger.info("  Source distribution   : %s", dict(s.source_counts))
        logger.info("  Doc-type distribution : %s", dict(s.doc_type_counts))
        logger.info("─────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CyberAdapt-LLM Phase 3 — Cybersecurity Corpus Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/prepare_dataset.py\n"
            "  python training/prepare_dataset.py --input-dir data/raw --dry-run\n"
            "  python training/prepare_dataset.py --min-chars 150 --near-dup-bits 2\n"
        ),
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help=f"Directory containing raw source documents (default: {DATA_RAW_DIR})",
    )
    p.add_argument(
        "--output-file",
        type=Path,
        default=DATA_PROCESSED_DIR / "cybersecurity_corpus.jsonl",
        help="Path for the output JSONL corpus file",
    )
    p.add_argument(
        "--stats-file",
        type=Path,
        default=DATA_PROCESSED_DIR / "dataset_stats.json",
        help="Path for the pipeline statistics JSON file",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum characters per paragraph (default: {DEFAULT_MIN_CHARS})",
    )
    p.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Minimum words per paragraph (default: {DEFAULT_MIN_WORDS})",
    )
    p.add_argument(
        "--near-dup-bits",
        type=int,
        default=DEFAULT_NEAR_DUP_BITS,
        help=f"SimHash Hamming distance threshold (default: {DEFAULT_NEAR_DUP_BITS})",
    )
    p.add_argument(
        "--text-field",
        default="description",
        help="JSON/JSONL field name containing the main text (default: description)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but do not write output (useful for validation)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    builder = CorpusBuilder(
        input_dir=args.input_dir,
        output_file=args.output_file,
        min_chars=args.min_chars,
        min_words=args.min_words,
        near_dup_threshold=args.near_dup_bits,
        dry_run=args.dry_run,
        text_field=args.text_field,
    )

    stats = builder.run()

    # Write stats JSON
    if not args.dry_run:
        args.stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_dict = stats.to_dict()
        args.stats_file.write_text(
            json.dumps(stats_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Stats written to: %s", args.stats_file)
        logger.info("Corpus written to: %s", args.output_file)
    else:
        logger.info("[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
