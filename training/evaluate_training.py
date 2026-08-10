"""
training/evaluate_training.py
Post-training evaluation for CyberAdapt-LLM Phase 5.

Reads an experiment directory and produces:
  - Perplexity from final validation loss
  - Loss curve summary (from training_log.jsonl)
  - Training time and efficiency stats
  - A printed report + evaluation_report.json

Usage:
  python training/evaluate_training.py --exp-dir models/adapted/exp_20240810_123456_ab1
  python training/evaluate_training.py                            # auto-finds latest
  python training/evaluate_training.py --list                     # list all experiments
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ADAPTED_DIR = PROJECT_ROOT / "models" / "adapted"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(n: float | int, decimals: int = 4) -> str:
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.{decimals}f}"


def _ppl(loss: float) -> float:
    return round(math.exp(loss), 4)


def _find_all_experiments(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("exp_")],
        key=lambda p: p.stat().st_mtime,
    )


def _find_latest_experiment(root: Path) -> Path | None:
    exps = _find_all_experiments(root)
    return exps[-1] if exps else None


# ─────────────────────────────────────────────────────────────────────────────
# Loader functions
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment_meta(exp_dir: Path) -> dict:
    meta_path = exp_dir / "experiment_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text("utf-8"))


def load_training_log(exp_dir: Path) -> list[dict]:
    """Load training_log.jsonl — our custom per-step log."""
    log_path = exp_dir / "training_log.jsonl"
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_trainer_state(exp_dir: Path) -> dict:
    """Load HuggingFace trainer_state.json (checkpointed by Trainer)."""
    # Try experiment root first, then checkpoint subdirs
    for candidate in [
        exp_dir / "trainer_state.json",
        *sorted((exp_dir / "checkpoints").glob("checkpoint-*/trainer_state.json"),
                key=lambda p: int(p.parent.name.split("-")[-1]), reverse=True),
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text("utf-8"))
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_training_log(rows: list[dict]) -> dict:
    """Compute summary statistics from the custom training log."""
    if not rows:
        return {}

    train_rows = [r for r in rows if r.get("train_loss") is not None]
    eval_rows  = [r for r in rows if r.get("eval_loss")  is not None]

    result: dict = {
        "total_log_rows": len(rows),
        "total_train_rows": len(train_rows),
        "total_eval_rows": len(eval_rows),
    }

    if train_rows:
        losses = [r["train_loss"] for r in train_rows]
        result["train_loss_initial"]  = losses[0]
        result["train_loss_final"]    = losses[-1]
        result["train_loss_min"]      = min(losses)
        result["train_loss_max"]      = max(losses)
        result["train_loss_delta"]    = round(losses[-1] - losses[0], 6)
        result["train_loss_improved"] = losses[-1] < losses[0]

    if eval_rows:
        e_losses = [r["eval_loss"] for r in eval_rows]
        result["eval_loss_initial"] = e_losses[0]
        result["eval_loss_final"]   = e_losses[-1]
        result["eval_loss_min"]     = min(e_losses)
        result["ppl_initial"]       = _ppl(e_losses[0])
        result["ppl_final"]         = _ppl(e_losses[-1])
        result["ppl_min"]           = _ppl(min(e_losses))

    # Tokens processed
    tok_rows = [r for r in rows if r.get("tokens_processed") is not None]
    if tok_rows:
        result["tokens_processed"] = max(r["tokens_processed"] for r in tok_rows)

    # Elapsed
    elapsed_rows = [r for r in rows if r.get("elapsed_s") is not None]
    if elapsed_rows:
        result["training_seconds"] = max(r["elapsed_s"] for r in elapsed_rows)

    return result


def analyse_trainer_state(state: dict) -> dict:
    """Extract key info from HF trainer_state.json."""
    if not state:
        return {}

    log_history = state.get("log_history", [])
    eval_logs   = [e for e in log_history if "eval_loss" in e]
    train_logs  = [e for e in log_history if "loss" in e and "eval_loss" not in e]

    result: dict = {
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "best_eval_loss": state.get("best_metric"),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
    }

    if result["best_eval_loss"] is not None:
        result["best_perplexity"] = _ppl(result["best_eval_loss"])

    if eval_logs:
        result["final_eval_loss"] = eval_logs[-1]["eval_loss"]
        result["final_perplexity"] = _ppl(eval_logs[-1]["eval_loss"])

    if train_logs:
        result["final_train_loss"] = train_logs[-1]["loss"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    exp_dir: Path,
    meta: dict,
    log_analysis: dict,
    state_analysis: dict,
) -> None:
    print()
    print("=" * 66)
    print("  CyberAdapt-LLM - Training Evaluation Report (Phase 5)")
    print("=" * 66)

    # Experiment info
    print(f"\n[Experiment]")
    print(f"  ID              : {meta.get('experiment_id', exp_dir.name)}")
    print(f"  Directory       : {exp_dir}")
    print(f"  Status          : {meta.get('status', 'unknown')}")
    print(f"  Started         : {meta.get('started_at', 'N/A')}")
    print(f"  Completed       : {meta.get('completed_at', 'N/A')}")

    # Model
    model_info = meta.get("model", {})
    print(f"\n[Model]")
    print(f"  Base model      : {model_info.get('base_model_name', 'N/A')}")
    print(f"  Adaptation type : {model_info.get('adaptation_type', 'N/A')}")
    specialized = model_info.get("is_cybersecurity_specialized", False)
    print(f"  Cybersec-adapted: {'YES (training complete)' if specialized else 'NO (training pending or smoke-test)'}")
    if model_info.get("final_model_path"):
        print(f"  Final model     : {model_info['final_model_path']}")

    # Hyperparameters
    hp = meta.get("hyperparameters", {})
    if hp:
        print(f"\n[Hyperparameters]")
        print(f"  Learning rate   : {hp.get('learning_rate', 'N/A')}")
        print(f"  Epochs          : {hp.get('num_train_epochs', 'N/A')}")
        print(f"  Batch size      : {hp.get('per_device_train_batch_size', 'N/A')}  x{hp.get('gradient_accumulation_steps', 1)} grad_accum = effective {hp.get('effective_batch_size', 'N/A')}")
        print(f"  Seq length      : {hp.get('max_sequence_length', 'N/A')} tokens")
        print(f"  Scheduler       : {hp.get('lr_scheduler_type', 'N/A')}")
        print(f"  Warmup ratio    : {hp.get('warmup_ratio', 'N/A')}")
        print(f"  Weight decay    : {hp.get('weight_decay', 'N/A')}")
        prec = "fp16" if hp.get("fp16") else "bf16" if hp.get("bf16") else "fp32 (CPU)"
        print(f"  Precision       : {prec}")
        print(f"  Seed            : {hp.get('seed', 42)}")

    # Training results
    # Prefer trainer_state analysis (more authoritative), fallback to log analysis
    best_loss = state_analysis.get("best_eval_loss") or log_analysis.get("eval_loss_min")
    best_ppl  = state_analysis.get("best_perplexity") or log_analysis.get("ppl_min")
    final_train_loss = (state_analysis.get("final_train_loss")
                        or log_analysis.get("train_loss_final")
                        or meta.get("results", {}).get("final_train_loss"))
    final_eval_loss  = (state_analysis.get("final_eval_loss")
                        or log_analysis.get("eval_loss_final")
                        or meta.get("results", {}).get("final_eval_loss"))
    final_ppl = _ppl(final_eval_loss) if final_eval_loss else meta.get("results", {}).get("final_perplexity")

    print(f"\n[Loss & Perplexity]")
    if log_analysis.get("train_loss_initial") is not None:
        print(f"  Train loss (start) : {_fmt(log_analysis['train_loss_initial'])}")
    if final_train_loss is not None:
        print(f"  Train loss (final) : {_fmt(final_train_loss)}")
    if log_analysis.get("eval_loss_initial") is not None:
        print(f"  Val loss   (start) : {_fmt(log_analysis['eval_loss_initial'])}")
        print(f"  PPL        (start) : {_fmt(log_analysis['ppl_initial'], 2)}")
    if final_eval_loss is not None:
        print(f"  Val loss   (final) : {_fmt(final_eval_loss)}")
    if final_ppl is not None:
        print(f"  PPL        (final) : {_fmt(final_ppl, 2)}")
    if best_loss is not None:
        print(f"  Best val loss      : {_fmt(best_loss)}")
    if best_ppl is not None:
        print(f"  Best PPL           : {_fmt(best_ppl, 2)}")
    if log_analysis.get("train_loss_improved") is not None:
        improved = log_analysis["train_loss_improved"]
        delta    = log_analysis.get("train_loss_delta", 0)
        print(f"  Loss improved?     : {'YES (delta=' + _fmt(delta) + ')' if improved else 'NO'}")

    # Efficiency
    print(f"\n[Training Efficiency]")
    steps = state_analysis.get("global_step") or meta.get("results", {}).get("total_steps")
    epoch = state_analysis.get("epoch") or meta.get("hyperparameters", {}).get("num_train_epochs")
    secs  = log_analysis.get("training_seconds") or meta.get("results", {}).get("training_seconds")
    tokens = log_analysis.get("tokens_processed") or meta.get("results", {}).get("tokens_processed")

    if steps:  print(f"  Total steps       : {_fmt(steps, 0)}")
    if epoch:  print(f"  Epoch reached     : {_fmt(epoch, 2)}")
    if secs:   print(f"  Training time     : {_fmt(secs, 1)} s  ({secs/60:.1f} min)")
    if tokens: print(f"  Tokens processed  : {_fmt(tokens, 0)}")
    bcp = state_analysis.get("best_model_checkpoint")
    if bcp:    print(f"  Best checkpoint   : {bcp}")

    # Loss curve (mini text plot)
    custom_log = load_training_log(exp_dir)
    eval_rows = [r for r in custom_log if r.get("eval_loss") is not None]
    if len(eval_rows) >= 2:
        print(f"\n[Val Loss Curve (every eval checkpoint)]")
        for r in eval_rows:
            loss = r["eval_loss"]
            ppl  = _ppl(loss)
            bar_w = 30
            # normalise: lower is better, so invert for bar
            min_l = min(x["eval_loss"] for x in eval_rows)
            max_l = max(x["eval_loss"] for x in eval_rows)
            span  = max_l - min_l if max_l > min_l else 1
            filled = int(bar_w * (1 - (loss - min_l) / span))
            bar    = "#" * filled + "." * (bar_w - filled)
            print(f"  step {r['step']:>5}  [{bar}]  loss={loss:.4f}  ppl={ppl:.2f}")

    print()
    print("=" * 66)
    print()

    # Interpretation note
    print("  INTERPRETATION NOTE:")
    print("  Perplexity measures how surprised the model is by the val text.")
    print("  Lower = better. A decrease from initial to final PPL indicates")
    print("  the model is learning the cybersecurity domain.")
    print()
    if not specialized:
        print("  CAUTION: This model has NOT completed full domain adaptation.")
        print("  Do not use it as a cybersecurity-specialized model until")
        print("  training/train.py completes with is_cybersecurity_specialized=true.")
    else:
        print("  Training marked as complete.")
        print("  The adapted model is in:", model_info.get("final_model_path", "unknown"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def list_experiments(root: Path) -> None:
    exps = _find_all_experiments(root)
    if not exps:
        print(f"No experiments found in: {root}")
        return
    print(f"\nFound {len(exps)} experiment(s) in {root}:\n")
    for exp in exps:
        meta = load_experiment_meta(exp)
        status = meta.get("status", "unknown")
        ts     = meta.get("started_at", "?")[:19]
        model  = meta.get("model", {}).get("base_model_name", "?")
        res    = meta.get("results", {})
        ppl_str = f"  ppl={res['final_perplexity']:.2f}" if res.get("final_perplexity") else ""
        print(f"  {exp.name}  [{status}]  started={ts}  model={model}{ppl_str}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberAdapt-LLM - Training Evaluation Report",
    )
    parser.add_argument("--exp-dir", type=Path, default=None,
                        help="Experiment directory to evaluate (default: latest)")
    parser.add_argument("--adapted-dir", type=Path, default=ADAPTED_DIR,
                        help=f"Root directory for all experiments (default: {ADAPTED_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    parser.add_argument("--save", action="store_true",
                        help="Save evaluation_report.json to the experiment directory")
    args = parser.parse_args()

    if args.list:
        list_experiments(args.adapted_dir)
        return

    # Resolve experiment directory
    if args.exp_dir:
        exp_dir = Path(args.exp_dir).resolve()
    else:
        exp_dir = _find_latest_experiment(args.adapted_dir)

    if exp_dir is None:
        print("ERROR: No experiment directories found.")
        print(f"  Run first: python training/train.py")
        sys.exit(1)

    if not exp_dir.exists():
        print(f"ERROR: Experiment directory does not exist: {exp_dir}")
        sys.exit(1)

    print(f"Evaluating experiment: {exp_dir}")

    meta          = load_experiment_meta(exp_dir)
    custom_log    = load_training_log(exp_dir)
    trainer_state = load_trainer_state(exp_dir)

    log_analysis   = analyse_training_log(custom_log)
    state_analysis = analyse_trainer_state(trainer_state)

    print_report(exp_dir, meta, log_analysis, state_analysis)

    if args.save:
        report = {
            "experiment_dir": str(exp_dir),
            "log_analysis": log_analysis,
            "state_analysis": state_analysis,
        }
        out = exp_dir / "evaluation_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
