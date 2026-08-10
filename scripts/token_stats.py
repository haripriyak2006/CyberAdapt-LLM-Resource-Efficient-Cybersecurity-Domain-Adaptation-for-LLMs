"""
scripts/token_stats.py
Display tokenization statistics for the CyberAdapt-LLM training dataset.

Reads:  data/datasets/tokenized/tokenization_stats.json
Also inspects the Arrow datasets directly to verify integrity.

Usage:
  python scripts/token_stats.py
  python scripts/token_stats.py --dataset-dir data/datasets/tokenized
  python scripts/token_stats.py --no-verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _fmt(n: int | float, decimals: int = 0) -> str:
    if decimals:
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def _bar(value: int, total: int, width: int = 28) -> str:
    if total == 0:
        return "." * width
    filled = int(width * value / total)
    return "#" * filled + "." * (width - filled)


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "N/A"


# ── Stats from JSON file ──────────────────────────────────────────────────────

def load_stats(dataset_dir: Path) -> dict:
    stats_file = dataset_dir / "tokenization_stats.json"
    if not stats_file.exists():
        print(f"ERROR: Stats file not found: {stats_file}")
        print("  Run first: python training/tokenize_dataset.py")
        sys.exit(1)
    return json.loads(stats_file.read_text("utf-8"))


# ── Arrow dataset integrity check ─────────────────────────────────────────────

def verify_datasets(dataset_dir: Path) -> dict:
    """Load the Arrow datasets and verify their structure."""
    info = {}
    for split in ("train", "val"):
        split_path = dataset_dir / split
        if not split_path.exists():
            info[split] = {"exists": False}
            continue
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(split_path))
            sample = ds[0] if len(ds) > 0 else {}
            info[split] = {
                "exists": True,
                "num_examples": len(ds),
                "columns": ds.column_names,
                "seq_len_check": len(sample.get("input_ids", [])) if sample else 0,
                "has_labels": "labels" in ds.column_names,
                "has_attention_mask": "attention_mask" in ds.column_names,
                "labels_match_input": (
                    sample.get("input_ids") == sample.get("labels")
                    if sample else None
                ),
            }
        except Exception as exc:
            info[split] = {"exists": True, "error": str(exc)}
    return info


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(stats: dict, verify_info: dict) -> None:
    print()
    print("=" * 62)
    print("  CyberAdapt-LLM - Tokenization Statistics (Phase 4)")
    print("=" * 62)

    # Config
    print("\n[Configuration]")
    print(f"  Tokenizer model     : {stats['model_name']}")
    print(f"  Vocab size          : {_fmt(stats['vocab_size'])}")
    print(f"  Max sequence length : {_fmt(stats['max_sequence_length'])} tokens")
    print(f"  Train/val split     : {stats['train_split_ratio']*100:.0f}% / {(1-stats['train_split_ratio'])*100:.0f}%")
    print(f"  Random seed         : {stats['seed']}")
    print(f"  Timestamp           : {stats.get('timestamp', 'N/A')}")

    # Records
    print("\n[Corpus Records]")
    total_rec = stats['total_records']
    print(f"  Total records       : {_fmt(total_rec)}")
    bar_t = _bar(stats['train_records'], total_rec)
    bar_v = _bar(stats['val_records'], total_rec)
    print(f"  Train records       : {bar_t}  {_fmt(stats['train_records'])} ({_pct(stats['train_records'], total_rec)})")
    print(f"  Val records         : {bar_v}  {_fmt(stats['val_records'])} ({_pct(stats['val_records'], total_rec)})")

    # Tokens
    print("\n[Token Counts]")
    total_raw = stats['total_raw_tokens']
    print(f"  Total raw tokens    : {_fmt(total_raw)}")
    print(f"  Train raw tokens    : {_fmt(stats['train_raw_tokens'])}")
    print(f"  Val raw tokens      : {_fmt(stats['val_raw_tokens'])}")
    print(f"  Tokens used (train) : {_fmt(stats['tokens_used_train'])}")
    print(f"  Tokens used (val)   : {_fmt(stats['tokens_used_val'])}")
    disc_total = stats['tokens_discarded_train'] + stats['tokens_discarded_val']
    print(f"  Tokens discarded    : {_fmt(disc_total)} ({_pct(disc_total, total_raw)})  [trailing partial window]")
    print(f"  Token utilisation   : {stats.get('token_utilisation_pct', 'N/A')}%")

    # Sequences
    print("\n[Training Sequences]")
    total_seq = stats['total_sequences']
    print(f"  Total sequences     : {_fmt(total_seq)}")
    bar_ts = _bar(stats['train_sequences'], total_seq)
    bar_vs = _bar(stats['val_sequences'], total_seq)
    print(f"  Train sequences     : {bar_ts}  {_fmt(stats['train_sequences'])} ({_pct(stats['train_sequences'], total_seq)})")
    print(f"  Val sequences       : {bar_vs}  {_fmt(stats['val_sequences'])} ({_pct(stats['val_sequences'], total_seq)})")
    print(f"  Avg sequence length : {_fmt(stats['avg_sequence_length'])} tokens")
    print(f"  Sequence length     : fixed {_fmt(stats['max_sequence_length'])} (packed, no padding)")

    # Dataset verification
    if verify_info:
        print("\n[Dataset Integrity Checks]")
        for split, info in verify_info.items():
            if not info.get("exists"):
                print(f"  {split:<6}: NOT FOUND")
                continue
            if "error" in info:
                print(f"  {split:<6}: ERROR - {info['error']}")
                continue
            checks = []
            checks.append(f"examples={_fmt(info['num_examples'])}")
            checks.append(f"seq_len={info['seq_len_check']}")
            checks.append(f"cols={info['columns']}")
            checks.append(f"labels_match={'YES' if info.get('labels_match_input') else 'NO'}")
            status = "OK  " if info.get("labels_match_input") and info["num_examples"] > 0 else "WARN"
            print(f"  {split:<6}: [{status}]  {', '.join(checks)}")

    # Notes for small datasets
    print()
    if total_seq < 10:
        print("  NOTE: Very few sequences generated. This is expected for the")
        print("  sample dataset. Add more documents to data/raw/ and re-run")
        print("  training/prepare_dataset.py  then  training/tokenize_dataset.py")
        print("  For reference: GPT-2 was trained on ~40B tokens.")
        print()

    print("=" * 62)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberAdapt-LLM - Tokenization Statistics Reporter",
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "tokenized",
        help="Directory containing tokenized datasets (default: data/datasets/tokenized)",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip loading Arrow datasets (faster, no datasets import needed)",
    )
    args = parser.parse_args()

    stats = load_stats(args.dataset_dir)

    verify_info = {}
    if not args.no_verify:
        try:
            verify_info = verify_datasets(args.dataset_dir)
        except ImportError:
            print("WARNING: 'datasets' package not available for verification. Skipping.")

    print_report(stats, verify_info)


if __name__ == "__main__":
    main()
