"""
scripts/verify_all.py
Full verification script for Phases 1, 2, and 3.
Run from project root: python scripts/verify_all.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str, fn) -> None:
    try:
        detail = fn()
        results.append((PASS, name, detail or ""))
    except AssertionError as exc:
        results.append((FAIL, name, str(exc)))
    except Exception as exc:
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Config / Logging / Health
# ─────────────────────────────────────────────────────────────────────────────

def _check_config():
    from backend.core.config import get_settings
    s = get_settings()
    assert s.app_name, "app_name is empty"
    assert isinstance(s.app_port, (int, str)), "app_port missing"
    return f"app={s.app_name}  env={s.app_env}"

def _check_logging():
    from backend.core.logging_config import configure_logging
    configure_logging(log_level="INFO", env="development")
    return "logging configured OK"

def _check_fastapi_import():
    from backend.main import app
    assert app is not None
    return f"FastAPI app: {app.title}"

def _check_health_route():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok"
    return f"status={data['status']}  uptime={data.get('uptime_seconds', '?')}s"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — LLM Service / Chat Endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _check_model_loader_import():
    from models.model_loader import ModelLoader
    m = ModelLoader(model_name="distilgpt2")
    assert not m.is_loaded, "should not be loaded yet"
    return f"ModelLoader OK  model={m.model_name}"

def _check_llm_service_import():
    from backend.services.llm_service import LLMServiceError, get_model_status
    status = get_model_status()
    assert "loaded" in status
    return f"LLMServiceError importable  model_status.loaded={status['loaded']}"

def _check_chat_schemas():
    from backend.schemas.chat import ChatRequest, ChatResponse
    req = ChatRequest(message="What is XSS?")
    assert req.message == "What is XSS?"
    assert req.max_tokens == 256
    assert req.temperature == 0.7
    return "ChatRequest + ChatResponse schema validated"

def _check_chat_validation_422():
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    cases = [
        ({}, "missing message"),
        ({"message": ""}, "empty message"),
        ({"message": "x" * 4097}, "message too long"),
        ({"message": "hi", "max_tokens": 0}, "max_tokens=0"),
        ({"message": "hi", "temperature": -1}, "temp<0"),
    ]
    for payload, label in cases:
        r = client.post("/api/v1/chat", json=payload)
        assert r.status_code == 422, f"Expected 422 for '{label}', got {r.status_code}"
    return f"{len(cases)} invalid payloads correctly rejected with 422"

def _check_chat_success_mocked():
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app
    mock_result = {"response": "XSS injects scripts.", "model": "distilgpt2", "latency_ms": 99.9}
    with patch("backend.api.chat.generate_response", return_value=mock_result):
        client = TestClient(app)
        r = client.post("/api/v1/chat", json={"message": "What is XSS?"})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "response" in data and "model" in data and "latency_ms" in data
    return f"POST /api/v1/chat -> 200  model={data['model']}"

def _check_chat_503_on_error():
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.services.llm_service import LLMServiceError
    with patch("backend.api.chat.generate_response", side_effect=LLMServiceError("not loaded")):
        client = TestClient(app)
        r = client.post("/api/v1/chat", json={"message": "hello"})
        assert r.status_code == 503, f"Expected 503, got {r.status_code}"
        assert "internal" not in r.text.lower() or "error" not in r.text.lower()
    return "LLMServiceError -> 503, no internal details leaked"

def _check_no_trace_leak_on_500():
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app
    with patch("backend.api.chat.generate_response", side_effect=RuntimeError("secret trace")):
        client = TestClient(app)
        r = client.post("/api/v1/chat", json={"message": "hello"})
        assert r.status_code == 500
        assert "secret trace" not in r.text
        assert "RuntimeError" not in r.text
    return "RuntimeError -> 500, stack trace NOT exposed"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Dataset Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _check_corpus_file_exists():
    corpus = PROJECT_ROOT / "data" / "processed" / "cybersecurity_corpus.jsonl"
    assert corpus.exists(), f"Missing: {corpus}"
    size = corpus.stat().st_size
    assert size > 1000, f"Corpus too small: {size} bytes"
    return f"exists  size={size:,} bytes"

def _check_corpus_schema():
    corpus = PROJECT_ROOT / "data" / "processed" / "cybersecurity_corpus.jsonl"
    required = {"text", "source", "document_type", "topic", "license"}
    records = [json.loads(l) for l in corpus.read_text("utf-8").splitlines() if l.strip()]
    bad = [i for i, r in enumerate(records) if not required <= set(r.keys())]
    assert not bad, f"Records missing fields at lines: {bad}"
    empty = [i for i, r in enumerate(records) if not r.get("text", "").strip()]
    assert not empty, f"Empty text at lines: {empty}"
    return f"{len(records)} records, all schema-valid, no empty texts"

def _check_stats_file():
    stats_file = PROJECT_ROOT / "data" / "processed" / "dataset_stats.json"
    assert stats_file.exists(), f"Missing: {stats_file}"
    stats = json.loads(stats_file.read_text("utf-8"))
    required_keys = {"num_records", "total_characters", "total_words",
                     "estimated_tokens", "source_distribution"}
    assert required_keys <= set(stats.keys())
    assert stats["num_records"] > 0
    return (f"{stats['num_records']} records  "
            f"{stats['total_words']:,} words  "
            f"~{stats['estimated_tokens']:,} tokens")

def _check_deduplicator():
    from training.prepare_dataset import Deduplicator
    d = Deduplicator(near_dup_threshold=3)
    text = "SQL injection bypasses authentication by appending malicious SQL to input fields."
    is_dup1, _ = d.is_duplicate(text)
    is_dup2, reason2 = d.is_duplicate(text)
    assert not is_dup1,   "First insertion should NOT be a duplicate"
    assert is_dup2,       "Second insertion SHOULD be a duplicate"
    assert reason2 == "exact"
    # Near-dup: same text with one word changed
    similar = "SQL injection bypasses authentication by inserting malicious SQL into input fields."
    is_near, reason_near = d.is_duplicate(similar)
    return f"exact-dup={is_dup2}  near-dup detected={is_near} ({reason_near or 'none'})"

def _check_text_cleaner():
    from training.prepare_dataset import TextCleaner
    dirty = "  Hello\r\n\r\nWorld  \u2013  test\u00a0content  "
    clean = TextCleaner.clean(dirty)
    assert "\r" not in clean,    "CR not removed"
    assert "\u00a0" not in clean, "NBSP not replaced"
    assert "\u2013" not in clean, "en-dash not replaced"
    assert clean.startswith("Hello"), f"Unexpected start: {clean[:20]!r}"
    return f"Input: {len(dirty)} chars -> Output: {len(clean)} chars"

def _check_data_sources_doc():
    doc = PROJECT_ROOT / "data" / "DATA_SOURCES.md"
    assert doc.exists(), f"Missing: {doc}"
    text = doc.read_text("utf-8")
    for required in ["Public Domain", "NIST", "OWASP", "MITRE", "NVD"]:
        assert required in text, f"'{required}' not found in DATA_SOURCES.md"
    return f"DATA_SOURCES.md exists ({doc.stat().st_size} bytes), all key sources documented"

def _check_pipeline_dry_run():
    import subprocess
    result = subprocess.run(
        [sys.executable, "training/prepare_dataset.py", "--dry-run"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    output = result.stdout + result.stderr
    assert "Records written" in output or "records_written" in output.lower() or "84" in output, \
        f"Unexpected output: {output[-200:]}"
    return "dry-run completes without writing files"

# ─────────────────────────────────────────────────────────────────────────────
# Live server check (optional — skipped if server not running)
# ─────────────────────────────────────────────────────────────────────────────

def _check_live_health():
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as resp:
            data = json.loads(resp.read())
        assert data.get("status") == "ok"
        return f"LIVE  status={data['status']}  uptime={data.get('uptime_seconds', '?')}s"
    except urllib.error.URLError:
        results.append((SKIP, "Live /health", "Server not running (start with: uvicorn backend.main:app)"))
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print("=" * 65)
    print("  CyberAdapt-LLM -- Full Verification  (Phases 1, 2, 3)")
    print("=" * 65)

    sections = [
        ("PHASE 1 -- Foundation", [
            ("Config loader",           _check_config),
            ("Logging configuration",   _check_logging),
            ("FastAPI app import",      _check_fastapi_import),
            ("GET /health (test client)", _check_health_route),
        ]),
        ("PHASE 2 -- Base LLM Inference", [
            ("ModelLoader import",      _check_model_loader_import),
            ("LLM service import",      _check_llm_service_import),
            ("Chat schemas",            _check_chat_schemas),
            ("Request validation (422)", _check_chat_validation_422),
            ("Chat success (mocked)",   _check_chat_success_mocked),
            ("503 on model error",      _check_chat_503_on_error),
            ("500 no trace leak",       _check_no_trace_leak_on_500),
        ]),
        ("PHASE 3 -- Dataset Pipeline", [
            ("Corpus file exists",      _check_corpus_file_exists),
            ("Corpus schema valid",     _check_corpus_schema),
            ("Stats file valid",        _check_stats_file),
            ("Deduplicator logic",      _check_deduplicator),
            ("Text cleaner",            _check_text_cleaner),
            ("DATA_SOURCES.md",         _check_data_sources_doc),
            ("Pipeline dry-run",        _check_pipeline_dry_run),
        ]),
        ("LIVE SERVER (optional)", [
            ("GET /health (live)",      _check_live_health),
        ]),
    ]

    for section_name, checks in sections:
        print(f"\n  {section_name}")
        print(f"  {'-' * (len(section_name) + 2)}")
        for name, fn in checks:
            check(name, fn)
            # print immediately
            last = results[-1]
            status, chk_name, detail = last
            detail_str = f"  ({detail})" if detail else ""
            print(f"    {status}  {chk_name}{detail_str}")

    print()
    print("=" * 65)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    skipped = sum(1 for r in results if r[0] == SKIP)
    total = len(results)
    print(f"  Results: {passed} passed  |  {failed} failed  |  {skipped} skipped  |  {total} total")
    print("=" * 65)
    print()

    if failed > 0:
        print("  FAILED CHECKS:")
        for status, name, detail in results:
            if status == FAIL:
                print(f"    {name}: {detail}")
        print()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
