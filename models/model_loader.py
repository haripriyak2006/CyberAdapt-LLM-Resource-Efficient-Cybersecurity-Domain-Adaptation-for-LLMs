"""
models/model_loader.py
Thread-safe, lazy HuggingFace causal LM loader for CyberAdapt-LLM.

Responsibilities:
  - Auto device selection  (CUDA → MPS → CPU)
  - Tokenizer + model loading with configurable cache dir
  - Optional float16 (CUDA only)
  - Text generation with configurable sampling parameters
  - Per-request parameter override without mutating shared state

Default model: distilgpt2  (82 M params, CPU-friendly, ~330 MB)
Override via env var BASE_MODEL_NAME or configs/base.yaml model.base_model_name
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Thread-safe lazy loader for a HuggingFace causal language model.

    The model is NOT loaded at construction time — it loads on the first
    call to ``generate()`` or an explicit ``load()`` call.

    Example::

        loader = ModelLoader(model_name="distilgpt2")
        text = loader.generate("Explain SQL injection in one sentence:")
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        cache_dir: Optional[str] = None,
        device: str = "auto",
        use_half_precision: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or None
        self._device_pref = device
        self.use_half_precision = use_half_precision

        # Default generation params (can be overridden per-call)
        self._default_max_new_tokens = max_new_tokens
        self._default_temperature = temperature
        self._default_top_p = top_p

        self._tokenizer = None
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._loaded = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True after the model has been successfully loaded."""
        return self._loaded

    @property
    def device(self) -> Optional[str]:
        """String representation of the torch device in use, or None if not loaded."""
        return str(self._device) if self._device is not None else None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_device(self):
        """Return the best available torch.device given the preference string."""
        import torch

        pref = self._device_pref.lower()

        if pref == "cpu":
            logger.info("Device forced to CPU.")
            return torch.device("cpu")

        if pref == "cuda":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
                return dev
            logger.warning("CUDA requested but unavailable — falling back to CPU.")
            return torch.device("cpu")

        if pref == "mps":
            if torch.backends.mps.is_available():
                logger.info("Using Apple MPS device.")
                return torch.device("mps")
            logger.warning("MPS requested but unavailable — falling back to CPU.")
            return torch.device("cpu")

        # auto: CUDA → MPS → CPU
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("Auto-selected CUDA: %s", torch.cuda.get_device_name(0))
            return dev
        if torch.backends.mps.is_available():
            logger.info("Auto-selected Apple MPS.")
            return torch.device("mps")
        logger.info("Auto-selected CPU (no GPU detected).")
        return torch.device("cpu")

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the tokenizer and model (thread-safe, idempotent).

        Raises:
            Exception: Any error from the HuggingFace hub download or
                       model instantiation is propagated to the caller.
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:  # double-checked locking
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading tokenizer + model: %s", self.model_name)
            t_start = time.perf_counter()

            # Device
            self._device = self._select_device()

            # ── Tokenizer ─────────────────────────────────────────────────────
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False,
            )
            # Many causal LMs have no pad token — use eos as fallback
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.debug("pad_token set to eos_token (%s)", self._tokenizer.eos_token)

            # ── Model ─────────────────────────────────────────────────────────
            load_kwargs: dict = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": False,
            }

            # float16 only saves memory on CUDA; skip on CPU/MPS
            if self.use_half_precision and self._device.type == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
                logger.info("Loading in float16 (CUDA half-precision).")

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            self._model.eval()
            self._model = self._model.to(self._device)

            elapsed = time.perf_counter() - t_start
            param_count = sum(p.numel() for p in self._model.parameters())
            logger.info(
                "Model ready in %.2fs | params=%.1fM | device=%s",
                elapsed,
                param_count / 1e6,
                self._device,
            )
            self._loaded = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a completion for ``prompt``.

        Args:
            prompt:         The input text prompt.
            max_new_tokens: Override the default max-new-tokens limit.
            temperature:    Override the default sampling temperature.
                            0.0 → greedy (deterministic).
            top_p:          Override the default nucleus-sampling probability.

        Returns:
            The generated text, with the original prompt stripped.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if not self._loaded:
            self.load()

        import torch

        _max_tokens = max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens
        _temp = temperature if temperature is not None else self._default_temperature
        _top_p = top_p if top_p is not None else self._default_top_p

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        ).to(self._device)

        input_length = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": _max_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        if _temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = _temp
            gen_kwargs["top_p"] = _top_p
        else:
            gen_kwargs["do_sample"] = False  # greedy

        with torch.no_grad():
            output_ids = self._model.generate(inputs["input_ids"], **gen_kwargs)

        # Decode only the newly generated tokens
        new_token_ids = output_ids[0][input_length:]
        response = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return response.strip()
