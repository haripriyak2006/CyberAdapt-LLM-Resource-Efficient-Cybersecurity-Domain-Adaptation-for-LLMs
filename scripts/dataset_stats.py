"""
scripts/dataset_stats.py
Generate a human-readable statistics report from the corpus JSONL.

Reads:  data/processed/cybersecurity_corpus.jsonl
Writes: data/processed/dataset_stats.json
Prints: formatted summary to stdout

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --corpus data/processed/cybersecurity_corpus.jsonl
    python scripts/dataset_stats.py --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.utils import (
    DATA_PROCESSED_DIR,
    TOKENS_PER_CHAR_ESTIMATE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(n: int | float, decimals: int = 0) -> str:
    if decimals:
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def _bar(value: int, total: int, width: int = 30) -> str:
    """ASCII progress bar."""
    if total == 0:
        return "-" * width
    filled = int(width * value / total)
    return "#" * filled + "." * (width - filled)



# ── Core stats computation ────────────────────────────────────────────────────

def compute_stats(corpus_path: Path) -> dict:
    """Read the JSONL corpus and compute all statistics."""
    if not corpus_path.exists():
        print(f"ERROR: Corpus file not found: {corpus_path}", file=sys.stderr)
        print("  Run first: python training/prepare_dataset.py", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    char_counts: list[int] = []
    word_counts: list[int] = []
    source_counter: Counter = Counter()
    doc_type_counter: Counter = Counter()
    topic_counter: Counter = Counter()
    license_counter: Counter = Counter()

    with corpus_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: Malformed JSON at line {line_no}", file=sys.stderr)
                continue

            text = record.get("text", "")
            chars = len(text)
            words = len(text.split())

            char_counts.append(chars)
            word_counts.append(words)
            source_counter[record.get("source", "unknown")] += 1
            doc_type_counter[record.get("document_type", "unknown")] += 1
            topic_counter[record.get("topic", "unknown")] += 1
            license_counter[record.get("license", "unknown")] += 1
            records.append(record)

    n = len(records)
    if n == 0:
        return {"error": "Corpus is empty", "num_records": 0}

    total_chars = sum(char_counts)
    total_words = sum(word_counts)
    total_tokens_est = int(total_chars * TOKENS_PER_CHAR_ESTIMATE)

    avg_chars = total_chars / n
    avg_words = total_words / n

    # Percentile helper
    def pct(lst: list[int], p: float) -> int:
        sorted_lst = sorted(lst)
        idx = min(int(len(sorted_lst) * p / 100), len(sorted_lst) - 1)
        return sorted_lst[idx]

    return {
        "num_records": n,
        "total_characters": total_chars,
        "total_words": total_words,
        "estimated_tokens": total_tokens_est,
        "avg_chars_per_record": round(avg_chars, 1),
        "avg_words_per_record": round(avg_words, 1),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "p25_chars": pct(char_counts, 25),
        "p50_chars": pct(char_counts, 50),
        "p75_chars": pct(char_counts, 75),
        "p95_chars": pct(char_counts, 95),
        "source_distribution": dict(source_counter.most_common()),
        "document_type_distribution": dict(doc_type_counter.most_common()),
        "topic_distribution": dict(topic_counter.most_common()),
        "license_distribution": dict(license_counter.most_common()),
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(stats: dict, top_n: int = 10) -> None:
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return

    n         = stats["num_records"]
    chars     = stats["total_characters"]
    words     = stats["total_words"]
    tokens    = stats["estimated_tokens"]

    print()
    print("=" * 62)
    print("  CyberAdapt-LLM — Dataset Statistics Report")
    print("=" * 62)

    print("\n[Overview]")
    print(f"  Records (paragraphs)  : {_fmt(n)}")
    print(f"  Total characters      : {_fmt(chars)}")
    print(f"  Total words           : {_fmt(words)}")
    print(f"  Estimated tokens      : {_fmt(tokens)}  (~4 chars/token)")

    print("\n[Record length distribution]")
    print(f"  Min chars             : {_fmt(stats['min_chars'])}")
    print(f"  P25 chars             : {_fmt(stats['p25_chars'])}")
    print(f"  Median chars          : {_fmt(stats['p50_chars'])}")
    print(f"  P75 chars             : {_fmt(stats['p75_chars'])}")
    print(f"  P95 chars             : {_fmt(stats['p95_chars'])}")
    print(f"  Max chars             : {_fmt(stats['max_chars'])}")
    print(f"  Avg chars / record    : {_fmt(stats['avg_chars_per_record'], 1)}")
    print(f"  Avg words / record    : {_fmt(stats['avg_words_per_record'], 1)}")

    def _section(title: str, dist: dict) -> None:
        print(f"\n[{title}]")
        total = sum(dist.values())
        items = sorted(dist.items(), key=lambda x: -x[1])[:top_n]
        for label, count in items:
            pct = 100 * count / total if total else 0
            bar = _bar(count, total)
            print(f"  {label:<28} {bar} {_fmt(count):>6} ({pct:5.1f}%)")

    _section("Source distribution",        stats["source_distribution"])
    _section("Document-type distribution", stats["document_type_distribution"])
    _section("Topic distribution",         stats["topic_distribution"])
    _section("License distribution",       stats["license_distribution"])

    print()
    print("=" * 62)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberAdapt-LLM — Dataset Statistics Reporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/dataset_stats.py\n"
            "  python scripts/dataset_stats.py --corpus data/processed/cybersecurity_corpus.jsonl\n"
            "  python scripts/dataset_stats.py --no-save\n"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DATA_PROCESSED_DIR / "cybersecurity_corpus.jsonl",
        help="Path to the JSONL corpus file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PROCESSED_DIR / "dataset_stats.json",
        help="Path for the output JSON stats file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top N entries per distribution category (default: 10)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print report but do not write the JSON stats file",
    )
    args = parser.parse_args()

    print(f"Reading corpus: {args.corpus}")
    stats = compute_stats(args.corpus)
    print_report(stats, top_n=args.top)

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Stats saved to: {args.output}")


if __name__ == "__main__":
    main()
