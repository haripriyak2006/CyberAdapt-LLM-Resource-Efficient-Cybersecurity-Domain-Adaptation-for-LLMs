"""
backend/api/model_info.py
GET /api/model/info — Model and RAG status.
"""
from __future__ import annotations
import logging
from fastapi import APIRouter
from backend.schemas.common import ModelInfoResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Returns the loaded LLM model details and RAG vector store status.",
)
async def model_info() -> ModelInfoResponse:
    # ── LLM info ──────────────────────────────────────────────────────────────
    model_loaded        = False
    model_id            = "not loaded"
    param_count_m       = None
    device              = None
    half_precision      = None
    load_time_s         = None
    adapted_model_path  = None

    try:
        from backend.services.llm_service import _loader
        if _loader is not None and _loader._model is not None:
            model_loaded   = True
            model_id       = _loader.model_name
            device         = str(_loader._model.device) if hasattr(_loader._model, "device") else None
            half_precision = _loader.use_half_precision if hasattr(_loader, "use_half_precision") else False
            load_time_s    = getattr(_loader, "_load_time_s", None)
            # Parameter count
            try:
                n_params = sum(p.numel() for p in _loader._model.parameters())
                param_count_m = round(n_params / 1_000_000, 1)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Could not read LLM loader state: %s", exc)

    # ── Adapted model ─────────────────────────────────────────────────────────
    try:
        from backend.core.config import get_settings
        import glob, os
        from pathlib import Path
        adapted_dirs = sorted(
            glob.glob(str(Path(get_settings().model_cache_dir).parent / "adapted" / "exp_*" / "final"))
        )
        if adapted_dirs:
            adapted_model_path = adapted_dirs[-1]
    except Exception:
        pass

    # ── RAG info ──────────────────────────────────────────────────────────────
    embed_id    = None
    rag_chunks  = None
    try:
        from backend.services.rag_service import _vector_store, _embedding_model
        if _vector_store is not None:
            rag_chunks = _vector_store.num_chunks
        if _embedding_model is not None:
            embed_id = _embedding_model.model_name
        if embed_id is None:
            from backend.core.config import get_settings
            embed_id = get_settings().embedding_model_id
    except Exception as exc:
        logger.debug("Could not read RAG state: %s", exc)

    from backend.core.config import get_settings
    cfg = get_settings()
    if not model_loaded:
        model_id = cfg.base_model_name

    return ModelInfoResponse(
        model_id=model_id,
        model_loaded=model_loaded,
        parameter_count_m=param_count_m,
        device=device,
        half_precision=half_precision,
        load_time_s=load_time_s,
        embedding_model_id=embed_id or cfg.embedding_model_id,
        rag_chunks=rag_chunks,
        adapted_model_path=adapted_model_path,
    )
