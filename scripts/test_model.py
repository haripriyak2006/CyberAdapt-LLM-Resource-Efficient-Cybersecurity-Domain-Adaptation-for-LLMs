"""
scripts/test_model.py
CLI script to test base model inference end-to-end.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --prompt "What is a firewall?"
    python scripts/test_model.py --prompt "Explain XSS" --max-tokens 128 --temperature 0.5
    python scripts/test_model.py --model distilgpt2
    python scripts/test_model.py --list-devices

This script loads the model directly (no FastAPI server required).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def list_devices() -> None:
    """Print available compute devices."""
    try:
        import torch
        print("\n-- Compute Devices --------------------------------")
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device     : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  MPS available   : {torch.backends.mps.is_available()}")
        print("---------------------------------------------------\n")
    except ImportError:
        print("ERROR: torch is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


def run_inference(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: str,
    cache_dir: str,
) -> None:
    """Load the model and run a single inference pass."""
    from models.model_loader import ModelLoader

    print(f"\n{'='*60}")
    print(f"  BASE MODEL TEST  —  CyberAdapt-LLM Phase 2")
    print(f"{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Device     : {device} (auto-selected if 'auto')")
    print(f"  Max tokens : {max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Cache dir  : {cache_dir}")
    print(f"{'='*60}\n")

    loader = ModelLoader(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    print("Loading model (first run downloads weights ~330 MB for distilgpt2)...")
    t_load = time.perf_counter()
    try:
        loader.load()
    except Exception as exc:
        print(f"\nERROR: Failed to load model '{model_name}'")
        print(f"  Reason : {exc}")
        print(f"\n  Tip: Check BASE_MODEL_NAME in .env or try a different model with --model")
        sys.exit(1)
    load_time = time.perf_counter() - t_load
    print(f"Model loaded in {load_time:.2f}s on device: {loader.device}\n")

    print(f"Prompt:\n  {prompt}\n")
    print("Generating...")

    t_gen = time.perf_counter()
    try:
        response = loader.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        print(f"\nERROR: Generation failed: {exc}")
        sys.exit(1)
    gen_time = (time.perf_counter() - t_gen) * 1000

    print(f"\n-- Response {'─'*48}")
    print(response if response else "(model returned empty output)")
    print(f"{'─'*60}")
    print(f"\nLatency : {gen_time:.0f} ms")
    print(f"Tokens  : ~{len(response.split())} words generated")
    print(f"\nOK - Base model inference test passed.\n")


def main() -> None:
    from backend.core.config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="CyberAdapt-LLM Phase 2 — Base Model CLI Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/test_model.py\n"
            "  python scripts/test_model.py --prompt 'What is a buffer overflow?'\n"
            "  python scripts/test_model.py --model distilgpt2 --temperature 0\n"
            "  python scripts/test_model.py --list-devices\n"
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Explain what a SQL injection attack is in simple terms:",
        help="Text prompt to send to the model (default: SQL injection question)",
    )
    parser.add_argument(
        "--model",
        default=settings.base_model_name,
        help=f"HuggingFace model ID (default: {settings.base_model_name})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.max_new_tokens,
        help=f"Max tokens to generate (default: {settings.max_new_tokens})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.temperature,
        help=f"Sampling temperature — 0=greedy (default: {settings.temperature})",
    )
    parser.add_argument(
        "--device",
        default=settings.device,
        choices=["auto", "cpu", "cuda", "mps"],
        help=f"Compute device (default: {settings.device})",
    )
    parser.add_argument(
        "--cache-dir",
        default=settings.model_cache_dir,
        help=f"Model cache directory (default: {settings.model_cache_dir})",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available compute devices and exit.",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    run_inference(
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
