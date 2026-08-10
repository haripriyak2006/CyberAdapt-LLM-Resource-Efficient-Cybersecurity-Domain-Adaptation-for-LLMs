"""
training/tokenize_dataset.py
Tokenization pipeline for CyberAdapt-LLM — Phase 4.

Strategy: Pack-concatenation (no padding, maximally efficient)
============================================================
  1. Read corpus JSONL  →  list of text records
  2. Shuffle with fixed seed  →  reproducible ordering
  3. Split at RECORD level (90/10)  →  zero val-leakage into train
  4. For each split independently:
       a. Concatenate texts separated by EOS token
       b. Tokenize the full concatenated string in one call
       c. Chunk the flat token list into MAX_SEQUENCE_LENGTH windows
          (last incomplete window is discarded — no padding)
       d. Attach attention_mask (all-ones) and labels (copy of input_ids)
  5. Save each split as a HuggingFace Arrow dataset
  6. Write tokenization_stats.json next to the datasets

Output layout:
  data/datasets/tokenized/
  ├── train/            ← HuggingFace Arrow dataset
  ├── val/              ← HuggingFace Arrow dataset
  └── tokenization_stats.json

Configurable via environment variables or configs/base.yaml:
  MAX_SEQUENCE_LENGTH   (default 1024)
  TRAIN_SPLIT_RATIO     (default 0.9)
  TOKENIZE_SEED         (default 42)
  BASE_MODEL_NAME       (default distilgpt2)
  CORPUS_FILE           (default data/processed/cybersecurity_corpus.jsonl)
  TOKENIZED_DATA_DIR    (default data/datasets/tokenized)

Usage:
  python training/tokenize_dataset.py
  python training/tokenize_dataset.py --max-seq-len 512 --seed 0
  python training/tokenize_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ── Force UTF-8 on Windows terminals ─────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Suppress the false-positive "sequence length > model max" warning from
# transformers. We deliberately tokenize the entire concatenated corpus and
# chunk it ourselves — the warning is irrelevant in this packing strategy.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ── Statistics dataclass ──────────────────────────────────────────────────────

@dataclass
class TokenizationStats:
    """All metrics produced by one tokenization run."""
    # Configuration
    model_name: str
    max_sequence_length: int
    train_split_ratio: float
    seed: int
    corpus_file: str
    output_dir: str
    timestamp: str

    # Corpus
    total_records: int
    train_records: int
    val_records: int

    # Token counts
    total_raw_tokens: int       # tokens before chunking (train + val combined)
    train_raw_tokens: int       # tokens in the train partition before chunking
    val_raw_tokens: int         # tokens in the val partition before chunking
    tokens_used_train: int      # tokens that end up in complete sequences (train)
    tokens_used_val: int        # tokens that end up in complete sequences (val)
    tokens_discarded_train: int # trailing tokens that don't form a full sequence (train)
    tokens_discarded_val: int   # trailing tokens that don't form a full sequence (val)

    # Sequences
    train_sequences: int
    val_sequences: int
    total_sequences: int

    # Averages
    avg_sequence_length: float  # should be == max_sequence_length for packed data
    vocab_size: int

    # Derived
    @property
    def token_utilisation_pct(self) -> float:
        total_used = self.tokens_used_train + self.tokens_used_val
        total_raw = self.total_raw_tokens
        return round(100.0 * total_used / total_raw, 2) if total_raw else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["token_utilisation_pct"] = self.token_utilisation_pct
        return d


# ── Core pipeline ─────────────────────────────────────────────────────────────

class TokenizationPipeline:
    """
    Loads, splits, tokenizes, and saves the cybersecurity corpus.

    Attributes
    ----------
    corpus_file       : Path to the input JSONL corpus
    output_dir        : Where to write the train/ and val/ Arrow datasets
    max_seq_len       : Fixed sequence length for each training example
    train_ratio       : Fraction of records assigned to training
    seed              : RNG seed for reproducibility
    model_name        : HuggingFace tokenizer model ID
    model_cache_dir   : Local directory to cache downloaded tokenizer files
    dry_run           : If True, tokenize but don't write to disk
    """

    def __init__(
        self,
        corpus_file: Path,
        output_dir: Path,
        max_seq_len: int = 1024,
        train_ratio: float = 0.9,
        seed: int = 42,
        model_name: str = "distilgpt2",
        model_cache_dir: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.corpus_file = corpus_file
        self.output_dir = output_dir
        self.max_seq_len = max_seq_len
        self.train_ratio = train_ratio
        self.seed = seed
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.dry_run = dry_run

        self._tokenizer = None  # loaded lazily

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer
        logger.info("Loading tokenizer: %s", self.model_name)
        tok = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.model_cache_dir,
            trust_remote_code=False,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        logger.info(
            "Tokenizer ready | vocab_size=%s | eos_token=%s",
            tok.vocab_size,
            repr(tok.eos_token),
        )
        self._tokenizer = tok
        return tok

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_records(self) -> list[dict]:
        """Read all records from the corpus JSONL."""
        if not self.corpus_file.exists():
            logger.error("Corpus file not found: %s", self.corpus_file)
            logger.error("Run first: python training/prepare_dataset.py")
            sys.exit(1)

        records = []
        with self.corpus_file.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %s", line_no)

        logger.info("Loaded %s records from corpus", len(records))
        return records

    # ── Splitting (at record level to prevent leakage) ────────────────────────

    def _split_records(
        self, records: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Shuffle then split records into train / val.
        The split is performed BEFORE tokenization to guarantee
        that no validation text appears in the training set.
        """
        rng = random.Random(self.seed)
        shuffled = records.copy()
        rng.shuffle(shuffled)

        n_train = max(1, int(len(shuffled) * self.train_ratio))
        train_records = shuffled[:n_train]
        val_records = shuffled[n_train:]

        # Edge case: ensure val is non-empty even for tiny corpora
        if not val_records and len(train_records) > 1:
            val_records = [train_records.pop()]

        logger.info(
            "Split: %s train records | %s val records (seed=%s)",
            len(train_records), len(val_records), self.seed,
        )
        return train_records, val_records

    # ── Tokenization: pack + chunk ────────────────────────────────────────────

    def _pack_and_chunk(
        self, records: list[dict], split_name: str
    ) -> tuple[list[list[int]], int, int, int]:
        """
        Concatenate all texts with EOS separator, tokenize once,
        then slice into fixed-length non-overlapping windows.

        Returns
        -------
        sequences       : list of token-ID lists, each of length max_seq_len
        raw_token_count : total tokens before chunking
        tokens_used     : tokens that land in complete sequences
        tokens_discarded: trailing tokens in the incomplete last window
        """
        tok = self._load_tokenizer()
        eos_id = tok.eos_token_id

        # Build one long string with EOS between documents
        separator = tok.eos_token or "<|endoftext|>"
        full_text = separator.join(r.get("text", "") for r in records)

        logger.info("[%s] Tokenizing %s chars ...", split_name, f"{len(full_text):,}")
        t0 = time.perf_counter()

        # Tokenize without truncation — we handle chunking ourselves
        token_ids: list[int] = tok.encode(
            full_text,
            add_special_tokens=False,
        )

        elapsed = time.perf_counter() - t0
        raw_count = len(token_ids)
        logger.info(
            "[%s] %s raw tokens in %.2fs",
            split_name, f"{raw_count:,}", elapsed,
        )

        # Chunk into fixed-length windows (last incomplete window discarded)
        sequences = [
            token_ids[i : i + self.max_seq_len]
            for i in range(0, raw_count - self.max_seq_len + 1, self.max_seq_len)
        ]

        tokens_used = len(sequences) * self.max_seq_len
        tokens_discarded = raw_count - tokens_used

        logger.info(
            "[%s] %s sequences x %s tokens | discarded=%s trailing tokens (%.1f%%)",
            split_name,
            len(sequences),
            self.max_seq_len,
            tokens_discarded,
            100.0 * tokens_discarded / raw_count if raw_count else 0,
        )

        return sequences, raw_count, tokens_used, tokens_discarded

    # ── Dataset building ──────────────────────────────────────────────────────

    def _build_hf_dataset(self, sequences: list[list[int]]):
        """
        Wrap sequences in a HuggingFace Dataset.
        Each example has:
          input_ids      : list[int]  — token IDs
          labels         : list[int]  — same as input_ids (causal LM convention)
          attention_mask : list[int]  — all-ones (no padding in packed data)
        """
        from datasets import Dataset

        n = len(sequences)
        ones = [1] * self.max_seq_len

        data = {
            "input_ids":      sequences,
            "labels":         [seq.copy() for seq in sequences],
            "attention_mask": [ones.copy() for _ in range(n)],
        }
        return Dataset.from_dict(data)

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self) -> TokenizationStats:
        """
        Execute the full pipeline and return statistics.
        """
        import datetime

        logger.info("=== CyberAdapt-LLM Tokenization Pipeline - Phase 4 ===")
        logger.info("Corpus    : %s", self.corpus_file)
        logger.info("Output    : %s", self.output_dir)
        logger.info("Model     : %s", self.model_name)
        logger.info("Max len   : %s tokens", self.max_seq_len)
        logger.info("Split     : %.0f%% train / %.0f%% val", self.train_ratio * 100, (1 - self.train_ratio) * 100)
        logger.info("Seed      : %s", self.seed)
        logger.info("Dry run   : %s", self.dry_run)

        # Step 1: Load
        records = self._load_records()

        # Step 2: Split at record level (prevents leakage)
        train_records, val_records = self._split_records(records)

        # Step 3: Tokenize each split independently
        train_seqs, train_raw, train_used, train_disc = self._pack_and_chunk(train_records, "train")
        val_seqs,   val_raw,   val_used,   val_disc   = self._pack_and_chunk(val_records,   "val")

        tok = self._load_tokenizer()
        total_seqs = len(train_seqs) + len(val_seqs)
        all_lengths = [self.max_seq_len] * total_seqs  # all packed sequences are exactly max_seq_len
        avg_len = self.max_seq_len if total_seqs > 0 else 0.0

        logger.info("Total sequences: %s train + %s val = %s", len(train_seqs), len(val_seqs), total_seqs)

        # Step 4: Save datasets
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            train_path = self.output_dir / "train"
            val_path   = self.output_dir / "val"

            if train_seqs:
                logger.info("Saving train dataset (%s sequences) ...", len(train_seqs))
                train_ds = self._build_hf_dataset(train_seqs)
                train_ds.save_to_disk(str(train_path))
                logger.info("Train dataset saved: %s", train_path)
            else:
                logger.warning("No train sequences to save (corpus too small for max_seq_len=%s)", self.max_seq_len)

            if val_seqs:
                logger.info("Saving val dataset (%s sequences) ...", len(val_seqs))
                val_ds = self._build_hf_dataset(val_seqs)
                val_ds.save_to_disk(str(val_path))
                logger.info("Val dataset saved: %s", val_path)
            else:
                logger.warning("No val sequences to save (val partition too small)")
        else:
            logger.info("[DRY RUN] Skipping dataset write.")

        # Step 5: Collect stats
        stats = TokenizationStats(
            model_name=self.model_name,
            max_sequence_length=self.max_seq_len,
            train_split_ratio=self.train_ratio,
            seed=self.seed,
            corpus_file=str(self.corpus_file),
            output_dir=str(self.output_dir),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            total_records=len(records),
            train_records=len(train_records),
            val_records=len(val_records),
            total_raw_tokens=train_raw + val_raw,
            train_raw_tokens=train_raw,
            val_raw_tokens=val_raw,
            tokens_used_train=train_used,
            tokens_used_val=val_used,
            tokens_discarded_train=train_disc,
            tokens_discarded_val=val_disc,
            train_sequences=len(train_seqs),
            val_sequences=len(val_seqs),
            total_sequences=total_seqs,
            avg_sequence_length=float(avg_len),
            vocab_size=tok.vocab_size,
        )

        # Step 6: Save stats JSON
        stats_path = self.output_dir / "tokenization_stats.json"
        if not self.dry_run:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(
                json.dumps(stats.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Stats written: %s", stats_path)

        self._print_summary(stats)
        return stats

    def _print_summary(self, s: TokenizationStats) -> None:
        logger.info("")
        logger.info("-- Tokenization Summary ---------------------------------------")
        logger.info("  Tokenizer          : %s (vocab=%s)", s.model_name, s.vocab_size)
        logger.info("  Max seq length     : %s tokens", s.max_sequence_length)
        logger.info("  Total records      : %s", s.total_records)
        logger.info("  Train records      : %s  Val records : %s", s.train_records, s.val_records)
        logger.info("  Total raw tokens   : %s", f"{s.total_raw_tokens:,}")
        logger.info("  Train raw tokens   : %s", f"{s.train_raw_tokens:,}")
        logger.info("  Val raw tokens     : %s", f"{s.val_raw_tokens:,}")
        logger.info("  Train sequences    : %s", s.train_sequences)
        logger.info("  Val sequences      : %s", s.val_sequences)
        logger.info("  Total sequences    : %s", s.total_sequences)
        logger.info("  Avg seq length     : %s tokens", s.avg_sequence_length)
        logger.info("  Token utilisation  : %s%%", s.token_utilisation_pct)
        logger.info("---------------------------------------------------------------")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CyberAdapt-LLM Phase 4 - Tokenization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/tokenize_dataset.py\n"
            "  python training/tokenize_dataset.py --max-seq-len 512\n"
            "  python training/tokenize_dataset.py --dry-run\n"
            "  python training/tokenize_dataset.py --model distilgpt2 --seed 99\n"
        ),
    )

    from backend.core.config import get_settings
    cfg = get_settings()

    p.add_argument("--corpus", type=Path,
                   default=Path(cfg.corpus_file),
                   help=f"Input JSONL corpus (default: {cfg.corpus_file})")
    p.add_argument("--output-dir", type=Path,
                   default=Path(cfg.tokenized_data_dir),
                   help=f"Output directory for tokenized datasets (default: {cfg.tokenized_data_dir})")
    p.add_argument("--model", default=cfg.base_model_name,
                   help=f"HuggingFace tokenizer model ID (default: {cfg.base_model_name})")
    p.add_argument("--cache-dir", default=cfg.model_cache_dir,
                   help="Local directory to cache tokenizer files")
    p.add_argument("--max-seq-len", type=int, default=cfg.max_sequence_length,
                   help=f"Tokens per training sequence (default: {cfg.max_sequence_length})")
    p.add_argument("--train-ratio", type=float, default=cfg.train_split_ratio,
                   help=f"Fraction of records for training (default: {cfg.train_split_ratio})")
    p.add_argument("--seed", type=int, default=cfg.tokenize_seed,
                   help=f"RNG seed for reproducibility (default: {cfg.tokenize_seed})")
    p.add_argument("--dry-run", action="store_true",
                   help="Tokenize but do not write dataset to disk")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = TokenizationPipeline(
        corpus_file=args.corpus,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        seed=args.seed,
        model_name=args.model,
        model_cache_dir=args.cache_dir,
        dry_run=args.dry_run,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
