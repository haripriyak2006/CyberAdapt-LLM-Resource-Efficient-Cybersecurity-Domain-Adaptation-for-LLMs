"""
training/train.py
Domain-Adaptive Continuous Pretraining (DACP) — CyberAdapt-LLM Phase 5.

Goal
----
Fine-tune a base causal LM on the cybersecurity corpus using standard
Causal Language Modeling (CLM) loss.  This produces a domain-adapted
base model — NOT a chat or instruction-tuned model.

Pipeline
--------
  1. Detect hardware  (CUDA / MPS / CPU)
  2. Load config from configs/training.yaml + env vars + CLI args
  3. Locate tokenized datasets  (data/datasets/tokenized/train|val)
  4. Create a unique experiment directory  (models/adapted/exp_YYYYMMDD_HHMMSS_<hex>/)
  5. Write experiment_meta.json
  6. Build TrainingArguments
  7. Load base model + tokenizer
  8. Resume from checkpoint if requested / auto-detected
  9. Train with HuggingFace Trainer + custom callbacks
  10. Save final model, tokenizer, and training metadata

Usage
-----
  # Full run (uses config defaults)
  python training/train.py

  # CPU smoke-test (5 steps, proves the loop works)
  python training/train.py --smoke-test

  # Custom hyperparams
  python training/train.py --lr 5e-5 --epochs 2 --batch-size 4 --fp16

  # Resume from checkpoint
  python training/train.py --resume models/adapted/exp_20240810_123456_ab1/checkpoint-500

  # Different experiment output
  python training/train.py --output-dir models/adapted/my_experiment
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ── UTF-8 on Windows ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Quiet noisy third-party loggers before importing transformers
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    """
    Detect available compute device and return a summary dict.
    Prints a clear warning when no GPU is available.
    """
    import torch

    info: dict = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": False,
        "mps_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_memory_gb": None,
        "num_gpus": 0,
    }

    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["device"] = "cuda"
        info["num_gpus"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = round(mem / (1024 ** 3), 2)

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon MPS"

    # Print summary
    logger.info("=== Hardware Detection ===")
    logger.info("  Platform    : %s", info["platform"])
    logger.info("  Python      : %s", info["python_version"])
    logger.info("  PyTorch     : %s", info["torch_version"])

    if info["cuda_available"]:
        logger.info("  Device      : CUDA  (%s GPU(s))", info["num_gpus"])
        logger.info("  GPU         : %s", info["gpu_name"])
        logger.info("  VRAM        : %s GB", info["gpu_memory_gb"])
    elif info["mps_available"]:
        logger.info("  Device      : MPS (Apple Silicon)")
    else:
        # ── GPU not available — print prominent warning ─────────────────────
        logger.warning("")
        logger.warning("=" * 65)
        logger.warning("  WARNING: No GPU detected — training will run on CPU.")
        logger.warning("")
        logger.warning("  Expected training time on CPU:")
        logger.warning("    - Smoke-test (5 steps)  :  ~2–5 min")
        logger.warning("    - 1 epoch on sample data:  ~30–60 min")
        logger.warning("    - Full 3-epoch training :  hours–days")
        logger.warning("")
        logger.warning("  Recommendations:")
        logger.warning("    1. Use --smoke-test to verify the pipeline works.")
        logger.warning("    2. Use Google Colab (free T4 GPU) for real training.")
        logger.warning("    3. Set fp16=false, bf16=false (already default for CPU).")
        logger.warning("=" * 65)
        logger.warning("")

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_training_yaml() -> dict:
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "training.yaml"
    if not cfg_path.exists():
        logger.warning("configs/training.yaml not found — using defaults.")
        return {}
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("training", {})


def _yaml_val(cfg: dict, key: str, default):
    return cfg.get(key, default)


# ─────────────────────────────────────────────────────────────────────────────
# Unique experiment directory
# ─────────────────────────────────────────────────────────────────────────────

def make_experiment_dir(root: Path, resume: Optional[str] = None) -> Path:
    """
    Create a unique experiment directory under ``root``.
    If ``resume`` points to a checkpoint *inside* an existing experiment dir,
    that parent dir is returned unchanged (no new dir created).
    """
    if resume:
        resume_path = Path(resume).resolve()
        # Walk up until we find a dir that starts with 'exp_' or equals root
        candidate = resume_path
        for _ in range(5):
            if candidate.parent == root or candidate.name.startswith("exp_"):
                return candidate.parent if candidate.name.startswith("checkpoint") else candidate
            candidate = candidate.parent
        # If we can't resolve it, create a new one anyway

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hex_suffix = uuid.uuid4().hex[:4]
    exp_name = f"exp_{timestamp}_{hex_suffix}"
    exp_dir = root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Experiment directory: %s", exp_dir)
    return exp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Experiment metadata
# ─────────────────────────────────────────────────────────────────────────────

def write_experiment_meta(
    exp_dir: Path,
    args: argparse.Namespace,
    hw_info: dict,
    cfg: dict,
    dataset_stats: Optional[dict],
) -> dict:
    """Write experiment_meta.json to the experiment directory."""
    meta = {
        "experiment_id": exp_dir.name,
        "experiment_dir": str(exp_dir),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed_at": None,  # updated at end of training
        "status": "running",

        "model": {
            "base_model_name": args.model,
            "model_cache_dir": args.cache_dir,
            "adaptation_type": "causal_lm_continuous_pretraining",
            "is_cybersecurity_specialized": False,  # True only after full training
        },

        "hyperparameters": {
            "learning_rate": args.lr,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lr_scheduler_type": args.scheduler,
            "max_sequence_length": args.max_seq_len,
            "seed": args.seed,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "smoke_test": args.smoke_test,
        },

        "data": {
            "train_dataset": args.train_data,
            "val_dataset": args.val_data,
            "tokenized_stats": dataset_stats,
        },

        "hardware": hw_info,
        "phase": 5,
        "note": (
            "Domain-Adaptive Continuous Pretraining on cybersecurity corpus. "
            "Model is NOT considered cybersecurity-specialized until training completes."
        ),
    }

    meta_path = exp_dir / "experiment_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Experiment metadata: %s", meta_path)
    return meta


def update_experiment_meta(exp_dir: Path, updates: dict) -> None:
    """Patch experiment_meta.json with final stats."""
    meta_path = exp_dir / "experiment_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text("utf-8"))
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Custom logging callback
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogCallback:
    """
    HuggingFace Trainer callback that logs:
      step | epoch | train_loss | eval_loss | lr | tokens_processed | elapsed
    Also appends each row to training_log.jsonl in the experiment directory.
    """

    def __init__(self, exp_dir: Path, total_tokens_per_step: int):
        from transformers import TrainerCallback

        self._exp_dir = exp_dir
        self._log_file = exp_dir / "training_log.jsonl"
        self._tokens_per_step = total_tokens_per_step
        self._start_time = time.time()
        self._rows: list[dict] = []

        # Create the TrainerCallback dynamically (avoids circular import)
        outer = self

        class _Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                elapsed = time.time() - outer._start_time
                step = state.global_step
                tokens = step * outer._tokens_per_step

                row = {
                    "step": step,
                    "epoch": round(state.epoch or 0, 4),
                    "train_loss": logs.get("loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "tokens_processed": tokens,
                    "elapsed_s": round(elapsed, 1),
                }

                train_loss = row["train_loss"]
                eval_loss  = row["eval_loss"]
                lr         = row["learning_rate"]

                parts = [f"step={step:>6}  epoch={row['epoch']:.2f}"]
                if train_loss is not None:
                    parts.append(f"loss={train_loss:.4f}")
                if eval_loss is not None:
                    ppl = round(2.718281828 ** eval_loss, 2)
                    parts.append(f"eval_loss={eval_loss:.4f}  ppl={ppl:.2f}")
                if lr is not None:
                    parts.append(f"lr={lr:.2e}")
                parts.append(f"tokens={tokens:,}  elapsed={elapsed:.0f}s")

                logger.info("  ".join(parts))

                outer._rows.append(row)
                # Append to JSONL log
                with outer._log_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.callback = _Callback()


# ─────────────────────────────────────────────────────────────────────────────
# Training stats helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokenization_stats(tokenized_dir: Path) -> Optional[dict]:
    stats_file = tokenized_dir / "tokenization_stats.json"
    if stats_file.exists():
        return json.loads(stats_file.read_text("utf-8"))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        default_data_collator,
        set_seed,
    )

    # ── Step 1: Hardware ──────────────────────────────────────────────────────
    hw_info = detect_hardware()
    device = hw_info["device"]

    # Auto-disable precision flags on non-CUDA devices
    if device != "cuda":
        if args.fp16:
            logger.warning("fp16 requested but device=%s — disabling fp16.", device)
            args.fp16 = False
        if args.bf16:
            logger.warning("bf16 requested but device=%s — disabling bf16.", device)
            args.bf16 = False

    set_seed(args.seed)

    # ── Step 2: Load datasets ─────────────────────────────────────────────────
    train_path = Path(args.train_data)
    val_path   = Path(args.val_data)

    if not train_path.exists():
        logger.error("Train dataset not found: %s", train_path)
        logger.error("Run first: python training/tokenize_dataset.py")
        sys.exit(1)
    if not val_path.exists():
        logger.error("Val dataset not found: %s", val_path)
        logger.error("Run first: python training/tokenize_dataset.py")
        sys.exit(1)

    logger.info("Loading train dataset: %s", train_path)
    train_ds = load_from_disk(str(train_path))
    logger.info("Loading val dataset  : %s", val_path)
    val_ds   = load_from_disk(str(val_path))

    logger.info("Train examples : %s", len(train_ds))
    logger.info("Val examples   : %s", len(val_ds))

    if len(train_ds) == 0:
        logger.error("Train dataset is empty. Re-run tokenize_dataset.py.")
        sys.exit(1)

    tok_stats = _load_tokenization_stats(train_path.parent)
    tokens_per_step = args.batch_size * args.max_seq_len

    # ── Step 3: Experiment directory ──────────────────────────────────────────
    root_dir = Path(args.output_dir)
    exp_dir  = make_experiment_dir(root_dir, resume=args.resume)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir = exp_dir / "final"

    # ── Step 4: Metadata ──────────────────────────────────────────────────────
    meta = write_experiment_meta(exp_dir, args, hw_info, {}, tok_stats)

    # Suppress harmless distilgpt2 weight-tying warning (lm_head.weight)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # ── Step 5: Smoke-test overrides ──────────────────────────────────────────
    yaml_cfg = _load_training_yaml()
    max_steps = -1
    if args.smoke_test:
        logger.info("*** SMOKE-TEST MODE — training for %s steps only ***",
                    _yaml_val(yaml_cfg, "smoke_test_max_steps", 5))
        args.epochs   = _yaml_val(yaml_cfg, "smoke_test_epochs", 1)
        args.batch_size = _yaml_val(yaml_cfg, "smoke_test_batch_size", 1)
        args.grad_accum = _yaml_val(yaml_cfg, "smoke_test_grad_accum", 1)
        max_steps     = _yaml_val(yaml_cfg, "smoke_test_max_steps", 5)

    # ── Step 6: TrainingArguments ─────────────────────────────────────────────
    # Compute warmup_steps from ratio (warmup_ratio deprecated in transformers v5)
    _effective_steps = max_steps if max_steps > 0 else (
        max(1, len(train_ds) // (args.batch_size * args.grad_accum)) * args.epochs
    )
    _warmup_steps = max(1, int(_effective_steps * args.warmup_ratio))
    logger.info("Warmup: %s steps out of ~%s total", _warmup_steps, _effective_steps)

    training_args = TrainingArguments(
        # Directories
        output_dir=str(checkpoint_dir),

        # Epochs / steps
        num_train_epochs=args.epochs,
        max_steps=max_steps,

        # Batch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Optimizer
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_steps=_warmup_steps,   # replaces deprecated warmup_ratio
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=float(_yaml_val(yaml_cfg, "adam_beta1", 0.9)),
        adam_beta2=float(_yaml_val(yaml_cfg, "adam_beta2", 0.999)),
        adam_epsilon=float(_yaml_val(yaml_cfg, "adam_epsilon", 1e-8)),

        # Precision
        fp16=args.fp16,
        bf16=args.bf16,

        # Evaluation
        eval_strategy=_yaml_val(yaml_cfg, "evaluation_strategy", "steps"),
        eval_steps=_yaml_val(yaml_cfg, "eval_steps", 100),

        # Checkpointing
        save_strategy="steps",
        save_steps=_yaml_val(yaml_cfg, "save_steps", 100),
        save_total_limit=_yaml_val(yaml_cfg, "save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=_yaml_val(yaml_cfg, "logging_steps", 10),
        logging_first_step=True,
        report_to="none",

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        # Dataloader (Windows-safe)
        dataloader_num_workers=0,
        dataloader_pin_memory=(device == "cuda"),

        # Misc
        remove_unused_columns=False,  # keep input_ids, labels, attention_mask
        prediction_loss_only=True,
        # no_cuda / use_mps_device were removed in transformers >=4.40.
        # CPU/MPS selection is automatic based on available hardware.
        use_cpu=(device == "cpu"),
    )

    # ── Step 7: Load tokenizer + model ───────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=False,
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters : %s total | %s trainable", f"{param_count:,}", f"{trainable:,}")

    # ── Step 8: Callback ─────────────────────────────────────────────────────
    cb = TrainingLogCallback(exp_dir, tokens_per_step)

    # ── Step 9: Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=[cb.callback],
    )

    # ── Step 10: Resume ───────────────────────────────────────────────────────
    resume_from = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists() and (resume_path / "trainer_state.json").exists():
            resume_from = str(resume_path)
            logger.info("Resuming from checkpoint: %s", resume_from)
        else:
            logger.warning("Resume path not found or not a valid checkpoint: %s", args.resume)
    else:
        # Auto-detect: find the most recent checkpoint in this experiment's checkpoint dir
        existing_checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"),
                                       key=lambda p: int(p.name.split("-")[-1]))
        if existing_checkpoints:
            resume_from = str(existing_checkpoints[-1])
            logger.info("Auto-resuming from: %s", resume_from)

    # ── Step 11: Train ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Starting Training ===")
    logger.info("  Experiment     : %s", exp_dir.name)
    logger.info("  Epochs         : %s", args.epochs)
    logger.info("  Batch size     : %s (x%s grad_accum = effective %s)",
                args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    logger.info("  LR             : %s  (%s scheduler)", args.lr, args.scheduler)
    logger.info("  Precision      : %s",
                "fp16" if args.fp16 else "bf16" if args.bf16 else "fp32")
    logger.info("  Device         : %s", device)
    logger.info("  Smoke-test     : %s", args.smoke_test)
    logger.info("")

    t_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t_start

    logger.info("Training complete in %.1f seconds (%.2f min)", elapsed, elapsed / 60)

    # ── Step 12: Save final model ─────────────────────────────────────────────
    logger.info("Saving final model to: %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Final model saved.")

    # ── Step 13: Save trainer state ───────────────────────────────────────────
    trainer.save_state()
    # Copy trainer_state.json to experiment root for easy access
    state_src = checkpoint_dir / "trainer_state.json"
    state_dst = exp_dir / "trainer_state.json"
    if state_src.exists():
        import shutil
        shutil.copy2(str(state_src), str(state_dst))

    # ── Step 14: Final metrics ────────────────────────────────────────────────
    metrics = train_result.metrics
    val_metrics = trainer.evaluate()

    final_train_loss = metrics.get("train_loss", None)
    final_eval_loss  = val_metrics.get("eval_loss", None)
    final_ppl = None
    if final_eval_loss is not None:
        import math
        final_ppl = round(math.exp(final_eval_loss), 4)

    logger.info("")
    logger.info("-- Final Results ------------------------------------------")
    logger.info("  Train loss      : %s", f"{final_train_loss:.4f}" if final_train_loss else "N/A")
    logger.info("  Val loss        : %s", f"{final_eval_loss:.4f}"  if final_eval_loss  else "N/A")
    logger.info("  Perplexity      : %s", f"{final_ppl:.2f}"        if final_ppl        else "N/A")
    logger.info("  Training time   : %.1f s (%.2f min)", elapsed, elapsed / 60)
    logger.info("  Model saved to  : %s", final_dir)
    logger.info("-----------------------------------------------------------")

    # ── Step 15: Update metadata ──────────────────────────────────────────────
    update_experiment_meta(exp_dir, {
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "completed" if not args.smoke_test else "smoke_test_completed",
        "results": {
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "final_perplexity": final_ppl,
            "training_seconds": round(elapsed, 1),
            "total_steps": train_result.global_step,
            "total_flos": metrics.get("total_flos"),
        },
        "model": {
            "base_model_name": args.model,
            "adaptation_type": "causal_lm_continuous_pretraining",
            # Only true after a full training run — smoke-test does not count
            "is_cybersecurity_specialized": (
                not args.smoke_test and args.epochs >= 1
            ),
            "final_model_path": str(final_dir),
        },
    })

    logger.info("")
    logger.info("Experiment dir  : %s", exp_dir)
    logger.info("Final model     : %s", final_dir)
    logger.info("Training log    : %s", exp_dir / "training_log.jsonl")
    logger.info("Experiment meta : %s", exp_dir / "experiment_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    yaml_cfg = _load_training_yaml()

    from backend.core.config import get_settings
    cfg = get_settings()

    p = argparse.ArgumentParser(
        description="CyberAdapt-LLM Phase 5 — Domain-Adaptive Continuous Pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/train.py --smoke-test          # CPU sanity check (5 steps)\n"
            "  python training/train.py                       # full run with defaults\n"
            "  python training/train.py --lr 5e-5 --epochs 2\n"
            "  python training/train.py --fp16                # requires NVIDIA GPU\n"
            "  python training/train.py --resume models/adapted/exp_XXX/checkpoints/checkpoint-500\n"
        ),
    )

    # Data
    tok_dir = Path(_yaml_val(yaml_cfg, "tokenized_data_dir", cfg.tokenized_data_dir) or cfg.tokenized_data_dir)
    p.add_argument("--train-data", default=str(tok_dir / "train"),
                   help="Path to HF Arrow train dataset")
    p.add_argument("--val-data", default=str(tok_dir / "val"),
                   help="Path to HF Arrow val dataset")

    # Model
    p.add_argument("--model", default=cfg.base_model_name,
                   help=f"Base model name (default: {cfg.base_model_name})")
    p.add_argument("--cache-dir", default=cfg.model_cache_dir,
                   help="Local directory to cache model weights")

    # Output
    p.add_argument("--output-dir",
                   default=_yaml_val(yaml_cfg, "output_dir", "./models/adapted"),
                   help="Root directory for all experiments")

    # Hyperparameters
    p.add_argument("--lr", type=float,
                   default=_yaml_val(yaml_cfg, "learning_rate", 2e-5),
                   help="Peak learning rate")
    p.add_argument("--epochs", type=int,
                   default=_yaml_val(yaml_cfg, "num_train_epochs", 3),
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int,
                   default=_yaml_val(yaml_cfg, "per_device_train_batch_size", 1),
                   help="Per-device train/eval batch size")
    p.add_argument("--grad-accum", type=int,
                   default=_yaml_val(yaml_cfg, "gradient_accumulation_steps", 8),
                   help="Gradient accumulation steps")
    p.add_argument("--max-seq-len", type=int,
                   default=_yaml_val(yaml_cfg, "max_sequence_length", cfg.max_sequence_length),
                   help="Max sequence length (must match tokenization)")
    p.add_argument("--warmup-ratio", type=float,
                   default=_yaml_val(yaml_cfg, "warmup_ratio", 0.06),
                   help="Fraction of total steps used for LR warmup")
    p.add_argument("--weight-decay", type=float,
                   default=_yaml_val(yaml_cfg, "weight_decay", 0.01),
                   help="AdamW weight decay")
    p.add_argument("--max-grad-norm", type=float,
                   default=_yaml_val(yaml_cfg, "max_grad_norm", 1.0),
                   help="Maximum gradient norm (clipping)")
    p.add_argument("--scheduler", default=_yaml_val(yaml_cfg, "lr_scheduler_type", "cosine"),
                   choices=["cosine", "linear", "constant", "constant_with_warmup",
                            "cosine_with_restarts", "polynomial"],
                   help="LR scheduler type")

    # Precision
    p.add_argument("--fp16", action="store_true",
                   default=_yaml_val(yaml_cfg, "fp16", False),
                   help="Enable FP16 mixed precision (requires NVIDIA GPU)")
    p.add_argument("--bf16", action="store_true",
                   default=_yaml_val(yaml_cfg, "bf16", False),
                   help="Enable BF16 mixed precision (requires Ampere+ GPU)")

    # Reproducibility
    p.add_argument("--seed", type=int,
                   default=_yaml_val(yaml_cfg, "seed", 42),
                   help="Random seed")

    # Modes
    p.add_argument("--smoke-test", action="store_true",
                   help="Run only 5 training steps (CPU sanity check)")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT_PATH",
                   help="Resume training from this checkpoint directory")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve tokenized_data_dir from the stats file if available
    tok_stats_path = (
        Path(args.train_data).parent / "tokenization_stats.json"
    )
    if tok_stats_path.exists():
        tok_stats = json.loads(tok_stats_path.read_text("utf-8"))
        auto_seq_len = tok_stats.get("max_sequence_length")
        if auto_seq_len and auto_seq_len != args.max_seq_len:
            logger.info(
                "max_seq_len auto-set to %s from tokenization_stats.json "
                "(was %s)", auto_seq_len, args.max_seq_len
            )
            args.max_seq_len = auto_seq_len

    train(args)


if __name__ == "__main__":
    main()
