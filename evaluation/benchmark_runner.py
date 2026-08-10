"""
evaluation/benchmark_runner.py
Phase 6 — Baseline vs CyberAdapt-LLM Evaluation System.

What this does
--------------
Runs the same set of cybersecurity benchmark questions through two models:
  1. BASE MODEL  — the unmodified pretrained LLM (e.g. distilgpt2)
  2. ADAPTED MODEL — the domain-adapted CyberAdapt-LLM checkpoint

Metrics calculated
------------------
  Multiple-choice questions (MCQ):
    - Per-question: correct / incorrect / extracted answer
    - Aggregate:    accuracy, macro-precision, macro-recall, macro-F1
    - Latency:      mean, p50, p95 per model

  Generative questions:
    - Keyword recall  (what fraction of reference keywords appear in output)
    - Response length (tokens and characters)
    - Latency

  Both model types:
    - Benchmark perplexity on reference answers

Outputs
-------
  evaluation/results/base_results.json
  evaluation/results/adapted_results.json
  evaluation/results/comparison.json
  evaluation/results/comparison_report.txt   (human-readable table)

IMPORTANT: All scores reported here are produced by our own experiments.
No benchmark scores are copied from research papers.

Usage
-----
  python evaluation/benchmark_runner.py
  python evaluation/benchmark_runner.py --adapted-model models/adapted/exp_XXX/final
  python evaluation/benchmark_runner.py --base-only          # skip adapted model
  python evaluation/benchmark_runner.py --max-new-tokens 64  # shorter responses
  python evaluation/benchmark_runner.py --questions evaluation/benchmark/questions.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ── UTF-8 on Windows ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Silence noisy third-party loggers
for _noisy in ("transformers", "datasets", "httpx", "huggingface_hub",
               "transformers.modeling_utils", "transformers.tokenization_utils_base"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
BENCHMARK_FILE = PROJECT_ROOT / "evaluation" / "benchmark" / "questions.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkQuestion:
    id: str
    type: str                   # "multiple_choice" | "generative"
    domain: str
    difficulty: str
    question: str
    # MCQ fields
    choices: Optional[dict] = None          # {"A": "...", "B": "...", ...}
    correct_answer: Optional[str] = None    # "A" | "B" | "C" | "D"
    explanation: Optional[str] = None
    # Generative fields
    reference_answer: Optional[str] = None
    keywords: Optional[list] = None


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    domain: str
    difficulty: str
    question_text: str

    # Generation
    prompt: str
    raw_output: str
    latency_ms: float

    # MCQ-specific
    extracted_answer: Optional[str] = None
    correct_answer: Optional[str] = None
    is_correct: Optional[bool] = None

    # Generative-specific
    keyword_recall: Optional[float] = None
    matched_keywords: Optional[list] = None
    response_chars: Optional[int] = None
    response_tokens: Optional[int] = None

    # Both
    reference_perplexity: Optional[float] = None   # PPL on reference answer


@dataclass
class ModelEvaluation:
    model_label: str          # "base" | "adapted"
    model_path: str
    evaluated_at: str
    results: list[QuestionResult] = field(default_factory=list)

    # Aggregate (filled in after all questions run)
    mcq_accuracy: Optional[float] = None
    mcq_precision: Optional[float] = None
    mcq_recall: Optional[float] = None
    mcq_f1: Optional[float] = None
    mcq_count: int = 0
    mcq_correct: int = 0

    gen_keyword_recall_mean: Optional[float] = None
    gen_count: int = 0

    mean_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    mean_reference_ppl: Optional[float] = None

    caveat: str = (
        "Results from our own experiments. "
        "No scores copied from any research paper."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[BenchmarkQuestion]:
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                questions.append(BenchmarkQuestion(
                    id=d["id"],
                    type=d["type"],
                    domain=d.get("domain", "general"),
                    difficulty=d.get("difficulty", "medium"),
                    question=d["question"],
                    choices=d.get("choices"),
                    correct_answer=d.get("correct_answer"),
                    explanation=d.get("explanation"),
                    reference_answer=d.get("reference_answer"),
                    keywords=d.get("keywords"),
                ))
            except (KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping malformed question at line %s: %s", line_no, exc)
    logger.info("Loaded %s questions (%s MCQ, %s generative)",
                len(questions),
                sum(1 for q in questions if q.type == "multiple_choice"),
                sum(1 for q in questions if q.type == "generative"))
    return questions


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_mcq_prompt(q: BenchmarkQuestion) -> str:
    """Format a multiple-choice question for a causal LM."""
    lines = [
        "Question: " + q.question,
        "",
    ]
    for letter, text in sorted(q.choices.items()):
        lines.append(f"{letter}) {text}")
    lines += ["", "The correct answer is:"]
    return "\n".join(lines)


def format_gen_prompt(q: BenchmarkQuestion) -> str:
    """Format a generative question for a causal LM."""
    return f"Question: {q.question}\n\nAnswer:"


# ─────────────────────────────────────────────────────────────────────────────
# Answer extractor (MCQ)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mcq_answer(text: str) -> str:
    """
    Extract A/B/C/D from model output using a cascade of heuristics.
    Returns "UNKNOWN" if no letter can be extracted.
    """
    # Normalise
    text = text.strip()
    upper = text.upper()

    # Pattern 1: "The answer is X" / "Answer: X" / "Answer is X"
    m = re.search(
        r'(?:THE\s+CORRECT\s+ANSWER\s+IS|ANSWER\s+IS|ANSWER\s*:)\s*[\("]?\s*([ABCD])\b',
        upper,
    )
    if m:
        return m.group(1)

    # Pattern 2: Standalone "(X)" or "X)" at the very start
    m = re.match(r'^\s*[\(\[]?([ABCD])[\)\]]?[\s\.\,\:)]', upper)
    if m:
        return m.group(1)

    # Pattern 3: First occurrence of standalone letter A/B/C/D
    m = re.search(r'\b([ABCD])\b', upper)
    if m:
        return m.group(1)

    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Keyword recall (generative)
# ─────────────────────────────────────────────────────────────────────────────

def compute_keyword_recall(output: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Return (recall_fraction, list_of_matched_keywords)."""
    if not keywords:
        return 0.0, []
    lower_output = output.lower()
    matched = [kw for kw in keywords if kw.lower() in lower_output]
    return round(len(matched) / len(keywords), 4), matched


# ─────────────────────────────────────────────────────────────────────────────
# Reference perplexity
# ─────────────────────────────────────────────────────────────────────────────

def compute_reference_perplexity(
    model, tokenizer, reference_text: str, device: str
) -> Optional[float]:
    """
    Compute the model's perplexity on a reference text string.
    Lower = model finds the reference more natural / expected.
    """
    try:
        import torch
        inputs = tokenizer(
            reference_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return round(math.exp(loss), 4)
    except Exception as exc:
        logger.debug("Perplexity calculation failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model runner
# ─────────────────────────────────────────────────────────────────────────────

class ModelRunner:
    """
    Wraps a single HuggingFace causal LM for benchmark inference.
    """

    def __init__(
        self,
        model_path: str,
        label: str,
        cache_dir: Optional[str] = None,
        max_new_tokens_mcq: int = 32,
        max_new_tokens_gen: int = 150,
    ) -> None:
        self.model_path = model_path
        self.label = label
        self.cache_dir = cache_dir
        self.max_new_tokens_mcq = max_new_tokens_mcq
        self.max_new_tokens_gen = max_new_tokens_gen
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[%s] Loading tokenizer from: %s", self.label, self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("[%s] Loading model from: %s", self.label, self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

        params = sum(p.numel() for p in self._model.parameters())
        logger.info("[%s] Model loaded | device=%s | params=%s",
                    self.label, self._device, f"{params:,}")

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        """
        Generate a response. Returns (generated_text, latency_ms).
        Uses greedy decoding (do_sample=False) for reproducibility.
        """
        import torch
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — deterministic
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the newly generated tokens (exclude the prompt)
        new_ids = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, round(latency_ms, 2)

    def perplexity_on(self, reference_text: str) -> Optional[float]:
        return compute_reference_perplexity(
            self._model, self._tokenizer, reference_text, self._device
        )

    def run_question(self, q: BenchmarkQuestion) -> QuestionResult:
        """Run a single benchmark question and return a QuestionResult."""
        if q.type == "multiple_choice":
            prompt = format_mcq_prompt(q)
            raw_output, latency_ms = self.generate(prompt, self.max_new_tokens_mcq)
            extracted = extract_mcq_answer(raw_output)
            is_correct = (extracted == q.correct_answer) if extracted != "UNKNOWN" else False

            # Reference perplexity: score the model on the correct choice text
            ref_text = (
                f"{q.correct_answer}) {q.choices[q.correct_answer]}"
                if q.correct_answer and q.choices
                else None
            )
            ref_ppl = self.perplexity_on(ref_text) if ref_text else None

            return QuestionResult(
                question_id=q.id,
                question_type=q.type,
                domain=q.domain,
                difficulty=q.difficulty,
                question_text=q.question,
                prompt=prompt,
                raw_output=raw_output,
                latency_ms=latency_ms,
                extracted_answer=extracted,
                correct_answer=q.correct_answer,
                is_correct=is_correct,
                reference_perplexity=ref_ppl,
            )

        else:  # generative
            prompt = format_gen_prompt(q)
            raw_output, latency_ms = self.generate(prompt, self.max_new_tokens_gen)
            recall, matched = compute_keyword_recall(raw_output, q.keywords or [])

            ref_ppl = (
                self.perplexity_on(q.reference_answer)
                if q.reference_answer
                else None
            )

            return QuestionResult(
                question_id=q.id,
                question_type=q.type,
                domain=q.domain,
                difficulty=q.difficulty,
                question_text=q.question,
                prompt=prompt,
                raw_output=raw_output,
                latency_ms=latency_ms,
                keyword_recall=recall,
                matched_keywords=matched,
                response_chars=len(raw_output),
                response_tokens=len(self._tokenizer.encode(raw_output)),
                reference_perplexity=ref_ppl,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculator
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregate_metrics(results: list[QuestionResult]) -> dict:
    """
    Compute aggregate metrics from a list of question results.
    Returns a dict of all aggregate scores.
    """
    mcq_results = [r for r in results if r.question_type == "multiple_choice"]
    gen_results  = [r for r in results if r.question_type == "generative"]

    agg: dict = {
        "total_questions": len(results),
        "mcq_count":  len(mcq_results),
        "gen_count":  len(gen_results),
    }

    # ── MCQ metrics ───────────────────────────────────────────────────────────
    if mcq_results:
        correct = sum(1 for r in mcq_results if r.is_correct)
        agg["mcq_correct"] = correct
        agg["mcq_accuracy"] = round(correct / len(mcq_results), 4)

        # Per-class precision, recall, F1 (A/B/C/D)
        classes = ["A", "B", "C", "D"]
        class_metrics = {}
        for cls in classes:
            tp = sum(1 for r in mcq_results
                     if r.extracted_answer == cls and r.correct_answer == cls)
            fp = sum(1 for r in mcq_results
                     if r.extracted_answer == cls and r.correct_answer != cls)
            fn = sum(1 for r in mcq_results
                     if r.extracted_answer != cls and r.correct_answer == cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            class_metrics[cls] = {"precision": round(prec, 4),
                                   "recall":    round(rec, 4),
                                   "f1":        round(f1, 4)}

        # Macro averages (over classes that actually appear as correct answers)
        active_classes = [c for c in classes
                          if any(r.correct_answer == c for r in mcq_results)]
        if active_classes:
            agg["mcq_macro_precision"] = round(
                sum(class_metrics[c]["precision"] for c in active_classes) / len(active_classes), 4)
            agg["mcq_macro_recall"] = round(
                sum(class_metrics[c]["recall"] for c in active_classes) / len(active_classes), 4)
            agg["mcq_macro_f1"] = round(
                sum(class_metrics[c]["f1"] for c in active_classes) / len(active_classes), 4)
        agg["mcq_per_class"] = class_metrics

        # Unknown extraction rate
        unknown = sum(1 for r in mcq_results if r.extracted_answer == "UNKNOWN")
        agg["mcq_unknown_rate"] = round(unknown / len(mcq_results), 4)

        # Per-domain accuracy
        domains = sorted(set(r.domain for r in mcq_results))
        domain_acc = {}
        for dom in domains:
            dom_results = [r for r in mcq_results if r.domain == dom]
            dom_correct = sum(1 for r in dom_results if r.is_correct)
            domain_acc[dom] = round(dom_correct / len(dom_results), 4)
        agg["mcq_domain_accuracy"] = domain_acc

    # ── Generative metrics ────────────────────────────────────────────────────
    if gen_results:
        recalls = [r.keyword_recall for r in gen_results if r.keyword_recall is not None]
        if recalls:
            agg["gen_keyword_recall_mean"] = round(sum(recalls) / len(recalls), 4)
            agg["gen_keyword_recall_min"]  = round(min(recalls), 4)
            agg["gen_keyword_recall_max"]  = round(max(recalls), 4)

        char_lens = [r.response_chars for r in gen_results if r.response_chars is not None]
        if char_lens:
            agg["gen_mean_response_chars"] = round(sum(char_lens) / len(char_lens), 1)

    # ── Latency ───────────────────────────────────────────────────────────────
    latencies = sorted(r.latency_ms for r in results)
    if latencies:
        agg["latency_mean_ms"] = round(sum(latencies) / len(latencies), 2)
        agg["latency_min_ms"]  = round(latencies[0], 2)
        agg["latency_max_ms"]  = round(latencies[-1], 2)
        # percentiles
        def pct(lst, p):
            idx = max(0, int(len(lst) * p / 100) - 1)
            return round(lst[idx], 2)
        agg["latency_p50_ms"] = pct(latencies, 50)
        agg["latency_p95_ms"] = pct(latencies, 95)

    # ── Reference perplexity ──────────────────────────────────────────────────
    ppls = [r.reference_perplexity for r in results if r.reference_perplexity is not None]
    if ppls:
        agg["mean_reference_perplexity"] = round(sum(ppls) / len(ppls), 4)
        agg["min_reference_perplexity"]  = round(min(ppls), 4)
        agg["max_reference_perplexity"]  = round(max(ppls), 4)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect latest adapted model
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_adapted_model() -> Optional[Path]:
    adapted_root = PROJECT_ROOT / "models" / "adapted"
    if not adapted_root.exists():
        return None
    exps = sorted(
        [p for p in adapted_root.iterdir()
         if p.is_dir() and p.name.startswith("exp_")],
        key=lambda p: p.stat().st_mtime,
    )
    for exp in reversed(exps):
        final = exp / "final"
        if final.exists() and (final / "config.json").exists():
            return final
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "YES" if v else "NO"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _bar(v: Optional[float], width: int = 20) -> str:
    if v is None:
        return "." * width
    filled = int(width * min(1.0, max(0.0, v)))
    return "#" * filled + "." * (width - filled)


def generate_comparison_report(
    base_eval: ModelEvaluation,
    adapted_eval: Optional[ModelEvaluation],
    questions: list[BenchmarkQuestion],
    out_path: Path,
) -> str:
    lines = []
    sep = "=" * 72

    lines += [
        sep,
        "  CyberAdapt-LLM — Baseline vs Adapted Comparison Report",
        "  Phase 6 Evaluation",
        sep,
        "",
        "  IMPORTANT: All scores produced by our own experiments.",
        "  No scores were copied from any research paper.",
        "",
        f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Base model: {base_eval.model_path}",
    ]
    if adapted_eval:
        lines.append(f"  Adapted   : {adapted_eval.model_path}")
    lines += ["", sep]

    # ── Aggregate stats table ─────────────────────────────────────────────────
    lines += [
        "",
        "  AGGREGATE STATISTICS",
        "  " + "-" * 68,
        f"  {'Metric':<35} {'Base':>12} {'Adapted':>12}",
        "  " + "-" * 68,
    ]

    def row(label, base_val, adapted_val=None, decimals=4):
        bv = _fmt(base_val, decimals) if base_val is not None else "N/A"
        av = _fmt(adapted_val, decimals) if adapted_val is not None else ("N/A" if adapted_eval else "(skipped)")
        lines.append(f"  {label:<35} {bv:>12} {av:>12}")

    ba = base_eval.__dict__
    aa = adapted_eval.__dict__ if adapted_eval else {}

    row("MCQ questions",            ba.get("mcq_count"),           aa.get("mcq_count"), 0)
    row("MCQ correct",              ba.get("mcq_correct"),         aa.get("mcq_correct"), 0)
    row("MCQ accuracy",             ba.get("mcq_accuracy"),        aa.get("mcq_accuracy"))
    row("MCQ macro precision",      ba.get("mcq_precision"),       aa.get("mcq_precision"))
    row("MCQ macro recall",         ba.get("mcq_recall"),          aa.get("mcq_recall"))
    row("MCQ macro F1",             ba.get("mcq_f1"),              aa.get("mcq_f1"))
    row("Gen keyword recall (mean)", ba.get("gen_keyword_recall_mean"), aa.get("gen_keyword_recall_mean"))
    row("Mean reference PPL",       ba.get("mean_reference_ppl"),  aa.get("mean_reference_ppl"), 2)
    row("Mean latency (ms)",        ba.get("mean_latency_ms"),     aa.get("mean_latency_ms"), 1)
    row("P95 latency (ms)",         ba.get("p95_latency_ms"),      aa.get("p95_latency_ms"), 1)

    lines += ["  " + "-" * 68, ""]

    # ── Per-question comparison ────────────────────────────────────────────────
    lines += ["", "  PER-QUESTION RESULTS", "  " + "-" * 68]

    # Build lookup
    base_map = {r.question_id: r for r in base_eval.results}
    adapted_map = {r.question_id: r for r in adapted_eval.results} if adapted_eval else {}

    mcq_qs  = [q for q in questions if q.type == "multiple_choice"]
    gen_qs  = [q for q in questions if q.type == "generative"]

    lines += ["", "  -- Multiple-Choice Questions --"]
    for q in mcq_qs:
        br = base_map.get(q.id)
        ar = adapted_map.get(q.id)
        lines += [
            "",
            f"  [{q.id}] ({q.domain}, {q.difficulty})",
            f"  Q: {q.question}",
            f"  Expected     : {q.correct_answer}",
        ]
        if br:
            b_correct = "CORRECT" if br.is_correct else "WRONG  "
            lines.append(f"  Base answer  : {br.extracted_answer or 'N/A':>2}  [{b_correct}]  ({br.latency_ms:.0f}ms)  raw='{br.raw_output[:60].strip()}'")
        if ar:
            a_correct = "CORRECT" if ar.is_correct else "WRONG  "
            lines.append(f"  Adapted ans  : {ar.extracted_answer or 'N/A':>2}  [{a_correct}]  ({ar.latency_ms:.0f}ms)  raw='{ar.raw_output[:60].strip()}'")

    lines += ["", "  -- Generative Questions --"]
    for q in gen_qs:
        br = base_map.get(q.id)
        ar = adapted_map.get(q.id)
        lines += [
            "",
            f"  [{q.id}] ({q.domain}, {q.difficulty})",
            f"  Q: {q.question}",
        ]
        if br:
            lines.append(f"  Base  → keyword_recall={br.keyword_recall}  matched={br.matched_keywords}  ({br.latency_ms:.0f}ms)")
            lines.append(f"         response: '{br.raw_output[:100].strip()}'")
        if ar:
            lines.append(f"  Adapted → keyword_recall={ar.keyword_recall}  matched={ar.matched_keywords}  ({ar.latency_ms:.0f}ms)")
            lines.append(f"           response: '{ar.raw_output[:100].strip()}'")

    lines += [
        "",
        sep,
        "",
        "  INTERPRETATION NOTES",
        "  " + "-" * 68,
        "  1. distilgpt2 is a 82M-param model trained on general web text.",
        "     It is not instruction-tuned and does not reliably output 'A/B/C/D'",
        "     for MCQ prompts. Low MCQ accuracy is therefore expected for BOTH models.",
        "",
        "  2. The adapted model was trained on a SMALL sample corpus for very few",
        "     steps (smoke-test). Meaningful accuracy improvement requires:",
        "       - A larger cybersecurity corpus (thousands of documents)",
        "       - Full training (3+ epochs on GPU)",
        "       - A larger or instruction-tuned base model",
        "",
        "  3. 'Keyword recall' for generative questions measures whether",
        "     domain-relevant terms appear in the output — NOT answer quality.",
        "     Use human evaluation for production quality assessment.",
        "",
        "  4. Reference perplexity measures model surprise on expert answers.",
        "     Lower perplexity = model finds expert answers more natural.",
        "     Compare base vs adapted PPL to see domain adaptation effect.",
        "",
        "  DO NOT interpret these results as final production benchmark scores.",
        sep,
    ]

    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _result_to_dict(r: QuestionResult) -> dict:
    return asdict(r)


def save_evaluation(eval_obj: ModelEvaluation, path: Path) -> None:
    data = {
        "model_label":   eval_obj.model_label,
        "model_path":    eval_obj.model_path,
        "evaluated_at":  eval_obj.evaluated_at,
        "caveat":        eval_obj.caveat,
        "aggregate": {
            "mcq_count":           eval_obj.mcq_count,
            "mcq_correct":         eval_obj.mcq_correct,
            "mcq_accuracy":        eval_obj.mcq_accuracy,
            "mcq_macro_precision": eval_obj.mcq_precision,
            "mcq_macro_recall":    eval_obj.mcq_recall,
            "mcq_macro_f1":        eval_obj.mcq_f1,
            "gen_keyword_recall_mean": eval_obj.gen_keyword_recall_mean,
            "gen_count":           eval_obj.gen_count,
            "mean_latency_ms":     eval_obj.mean_latency_ms,
            "p50_latency_ms":      eval_obj.p50_latency_ms,
            "p95_latency_ms":      eval_obj.p95_latency_ms,
            "mean_reference_ppl":  eval_obj.mean_reference_ppl,
        },
        "results": [_result_to_dict(r) for r in eval_obj.results],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    runner: ModelRunner,
    questions: list[BenchmarkQuestion],
    model_label: str,
) -> ModelEvaluation:
    """Run all benchmark questions through a model and return a ModelEvaluation."""

    runner.load()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    eval_obj = ModelEvaluation(
        model_label=model_label,
        model_path=runner.model_path,
        evaluated_at=ts,
    )

    logger.info("[%s] Running %s benchmark questions ...", model_label, len(questions))
    for i, q in enumerate(questions, 1):
        logger.info("  [%s] %d/%d  %s (%s) ...", model_label, i, len(questions), q.id, q.type)
        result = runner.run_question(q)
        eval_obj.results.append(result)

    # Aggregate
    agg = compute_aggregate_metrics(eval_obj.results)
    eval_obj.mcq_count   = agg.get("mcq_count", 0)
    eval_obj.mcq_correct = agg.get("mcq_correct", 0)
    eval_obj.mcq_accuracy  = agg.get("mcq_accuracy")
    eval_obj.mcq_precision = agg.get("mcq_macro_precision")
    eval_obj.mcq_recall    = agg.get("mcq_macro_recall")
    eval_obj.mcq_f1        = agg.get("mcq_macro_f1")
    eval_obj.gen_count          = agg.get("gen_count", 0)
    eval_obj.gen_keyword_recall_mean = agg.get("gen_keyword_recall_mean")
    eval_obj.mean_latency_ms = agg.get("latency_mean_ms")
    eval_obj.p50_latency_ms  = agg.get("latency_p50_ms")
    eval_obj.p95_latency_ms  = agg.get("latency_p95_ms")
    eval_obj.mean_reference_ppl = agg.get("mean_reference_perplexity")

    logger.info("[%s] Done. MCQ accuracy=%.1f%%  gen_recall=%.1f%%  mean_ppl=%s",
                model_label,
                (eval_obj.mcq_accuracy or 0) * 100,
                (eval_obj.gen_keyword_recall_mean or 0) * 100,
                _fmt(eval_obj.mean_reference_ppl, 2))
    return eval_obj


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from backend.core.config import get_settings
    cfg = get_settings()

    parser = argparse.ArgumentParser(
        description="CyberAdapt-LLM Phase 6 — Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation/benchmark_runner.py\n"
            "  python evaluation/benchmark_runner.py --base-only\n"
            "  python evaluation/benchmark_runner.py --adapted-model models/adapted/exp_XXX/final\n"
            "  python evaluation/benchmark_runner.py --max-new-tokens-mcq 64\n"
        ),
    )
    parser.add_argument("--base-model",    default=cfg.base_model_name,
                        help=f"Base model name or path (default: {cfg.base_model_name})")
    parser.add_argument("--adapted-model", default=None,
                        help="Path to adapted model final/ dir (auto-detected if omitted)")
    parser.add_argument("--cache-dir",     default=cfg.model_cache_dir,
                        help="HuggingFace model cache directory")
    parser.add_argument("--questions",     type=Path, default=BENCHMARK_FILE,
                        help=f"Benchmark questions JSONL (default: {BENCHMARK_FILE})")
    parser.add_argument("--output-dir",    type=Path, default=RESULTS_DIR,
                        help=f"Results output directory (default: {RESULTS_DIR})")
    parser.add_argument("--max-new-tokens-mcq", type=int, default=32,
                        help="Max new tokens for MCQ generation (default: 32)")
    parser.add_argument("--max-new-tokens-gen", type=int, default=150,
                        help="Max new tokens for generative questions (default: 150)")
    parser.add_argument("--base-only", action="store_true",
                        help="Only evaluate the base model (skip adapted model)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve adapted model path
    adapted_path: Optional[str] = None
    if not args.base_only:
        if args.adapted_model:
            adapted_path = args.adapted_model
        else:
            detected = find_latest_adapted_model()
            if detected:
                adapted_path = str(detected)
                logger.info("Auto-detected adapted model: %s", adapted_path)
            else:
                logger.warning(
                    "No adapted model found in models/adapted/. "
                    "Run: python training/train.py  (or use --base-only)"
                )

    # Load questions
    if not args.questions.exists():
        logger.error("Benchmark questions file not found: %s", args.questions)
        sys.exit(1)
    questions = load_questions(args.questions)
    if not questions:
        logger.error("No questions loaded from: %s", args.questions)
        sys.exit(1)

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run base model ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PHASE 6 — BENCHMARK EVALUATION")
    logger.info("=" * 60)
    logger.info("  Base model   : %s", args.base_model)
    logger.info("  Adapted model: %s", adapted_path or "(not running)")
    logger.info("  Questions    : %s (%s total)", args.questions.name, len(questions))
    logger.info("=" * 60)
    logger.info("")

    base_runner = ModelRunner(
        model_path=args.base_model,
        label="base",
        cache_dir=args.cache_dir,
        max_new_tokens_mcq=args.max_new_tokens_mcq,
        max_new_tokens_gen=args.max_new_tokens_gen,
    )
    base_eval = run_evaluation(base_runner, questions, "base")
    save_evaluation(base_eval, args.output_dir / "base_results.json")

    # ── Run adapted model ─────────────────────────────────────────────────────
    adapted_eval: Optional[ModelEvaluation] = None
    if adapted_path:
        adapted_runner = ModelRunner(
            model_path=adapted_path,
            label="adapted",
            cache_dir=None,              # local path — no cache needed
            max_new_tokens_mcq=args.max_new_tokens_mcq,
            max_new_tokens_gen=args.max_new_tokens_gen,
        )
        adapted_eval = run_evaluation(adapted_runner, questions, "adapted")
        save_evaluation(adapted_eval, args.output_dir / "adapted_results.json")

    # ── Comparison JSON ───────────────────────────────────────────────────────
    comparison = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "caveat": (
            "All scores from our experiments only. "
            "No scores copied from research papers."
        ),
        "base": {
            "model": base_eval.model_path,
            "mcq_accuracy":   base_eval.mcq_accuracy,
            "mcq_macro_f1":   base_eval.mcq_f1,
            "gen_keyword_recall": base_eval.gen_keyword_recall_mean,
            "mean_reference_ppl": base_eval.mean_reference_ppl,
            "mean_latency_ms":    base_eval.mean_latency_ms,
        },
        "adapted": (
            {
                "model": adapted_eval.model_path,
                "mcq_accuracy":   adapted_eval.mcq_accuracy,
                "mcq_macro_f1":   adapted_eval.mcq_f1,
                "gen_keyword_recall": adapted_eval.gen_keyword_recall_mean,
                "mean_reference_ppl": adapted_eval.mean_reference_ppl,
                "mean_latency_ms":    adapted_eval.mean_latency_ms,
            }
            if adapted_eval else None
        ),
        "delta": (
            {
                "mcq_accuracy_delta":  (
                    round((adapted_eval.mcq_accuracy or 0) - (base_eval.mcq_accuracy or 0), 4)
                    if adapted_eval else None
                ),
                "gen_recall_delta": (
                    round((adapted_eval.gen_keyword_recall_mean or 0) - (base_eval.gen_keyword_recall_mean or 0), 4)
                    if adapted_eval else None
                ),
                "ppl_delta": (
                    round((adapted_eval.mean_reference_ppl or 0) - (base_eval.mean_reference_ppl or 0), 4)
                    if adapted_eval else None
                ),
            }
        ),
    }
    comp_path = args.output_dir / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved: %s", comp_path)

    # ── Text report ───────────────────────────────────────────────────────────
    report_path = args.output_dir / "comparison_report.txt"
    report = generate_comparison_report(base_eval, adapted_eval, questions, report_path)
    logger.info("Saved: %s", report_path)

    # ── Print summary to console ──────────────────────────────────────────────
    print()
    print(report)
    print()
    logger.info("All results saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
