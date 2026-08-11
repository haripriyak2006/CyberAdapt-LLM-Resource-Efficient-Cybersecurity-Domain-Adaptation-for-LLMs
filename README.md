# CyberAdapt-LLM
## Resource-Efficient Cybersecurity Domain Adaptation for Large Language Models

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-11%20Hackathon-orange)]()

---

## Overview

**CyberAdapt-LLM** is a research platform that demonstrates **domain-adaptive pre-training (DAP)** and **retrieval-augmented generation (RAG)** applied to cybersecurity. A general-purpose LLM (distilgpt2) is adapted toward cybersecurity tasks — threat intelligence, vulnerability analysis, and incident response — then compared head-to-head against its unmodified baseline in a live web dashboard.

**Key Results**
- ↓ Perplexity on cybersecurity text (domain vocabulary internalised)
- ↑ Keyword recall on generative cybersecurity tasks
- ↑ MCQ accuracy on cybersecurity multiple-choice questions
- RAG evidence grounding with source attribution (CVE, MITRE, NIST, OWASP)

> ⚠️ **Disclaimer:** This is a research demonstration. All outputs are AI-generated and must be verified by a qualified cybersecurity professional before any operational use.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CyberAdapt-LLM Platform                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Next.js Dashboard (Port 3000)                                        │   │
│  │  Dashboard │ Chat │ Threat │ Vuln │ Document │ Report │ Comparison    │   │
│  └────────────────────────────┬─────────────────────────────────────────┘   │
│                                │ HTTP proxy → :8000                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Backend (Port 8000)                                          │   │
│  │  /health  /api/chat  /api/rag/query  /api/threat/analyze             │   │
│  │  /api/vulnerability/analyze  /api/document/upload                    │   │
│  │  /api/report/generate  /api/demo/compare  /api/metrics               │   │
│  └────────────────┬──────────────────────┬──────────────────────────────┘   │
│                   │                      │                                   │
│     ┌─────────────▼──────────┐  ┌───────▼────────────────────────────┐     │
│     │  LLM Service           │  │  RAG Pipeline                       │     │
│     │  Base Model (distilgpt2│  │  Embeddings → FAISS → Retriever     │     │
│     │  Adapted Model (DAP)   │  │  CVE · ATT&CK · NIST · OWASP       │     │
│     └────────────────────────┘  └────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Domain Adaptation Pipeline**
```
Raw Corpus (CVE/MITRE/NIST/OWASP)
         ↓ chunking + cleaning
   Cybersecurity Corpus JSONL
         ↓ tokenisation (Phase 4)
   Tokenised HuggingFace Dataset
         ↓ causal LM training (Phase 5)
         ↓ AdamW + LR warmup + gradient accumulation
   Adapted Model (models/adapted/)
```

---

## Project Status

| Phase | Description                          | Status      |
|-------|--------------------------------------|-------------|
| 1     | Project Foundation & Config          | ✅ Complete |
| 2     | LLM Service & Base Chat API          | ✅ Complete |
| 3     | Cybersecurity Corpus Preparation     | ✅ Complete |
| 4     | Dataset Tokenisation                 | ✅ Complete |
| 5     | Domain-Adaptive Pre-Training (DAP)   | ✅ Complete |
| 6     | Evaluation Framework & Benchmarks    | ✅ Complete |
| 7     | RAG Pipeline (FAISS + Embeddings)    | ✅ Complete |
| 8     | Threat & Vulnerability Analysis APIs | ✅ Complete |
| 9     | Production FastAPI Backend           | ✅ Complete |
| 10    | Next.js SOC Dashboard                | ✅ Complete |
| 11    | Live Comparison & Hackathon Demo     | ✅ Complete |

---

## Quick Start (One Command)

```bash
python start.py
```

This starts the backend on **:8000** and the frontend on **:3000** with health polling.

```bash
# Backend only
python start.py --backend

# Frontend only
python start.py --frontend

# Health check
python start.py --check
```

---

## Manual Setup

### 1. Clone

```bash
git clone https://github.com/haripriyak2006/CyberAdapt-LLM-Resource-Efficient-Cybersecurity-Domain-Adaptation-for-LLMs.git
cd CyberAdapt-LLM-Resource-Efficient-Cybersecurity-Domain-Adaptation-for-LLMs
```

### 2. Python Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set BASE_MODEL_NAME
```

### 4. Ingest RAG Knowledge Base

```bash
python scripts/ingest_cybersec.py
```

### 5. (Optional) Train Adapted Model

```bash
python training/train.py
```

> **Note:** Training requires CUDA or Apple MPS for practical speeds. CPU is supported but will be slow. Skip this step for demo — the system falls back to the base model gracefully.

### 6. Start Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

---

## Docker (Optional)

```bash
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

> Model weights and data are **bind-mounted** — they are never baked into the image.

---

## Repository Layout

```
CyberAdapt-LLM/
├── backend/            # FastAPI application (Phases 2, 8, 9, 11)
│   ├── api/            # Route handlers
│   ├── core/           # Config, middleware, safety, limits
│   ├── schemas/        # Pydantic models
│   └── services/       # Business logic
├── configs/            # YAML configuration defaults
├── data/               # Corpus (gitignored — see .gitignore)
│   ├── raw/            # Source datasets (gitignored)
│   ├── processed/      # Cleaned corpus JSONL (gitignored)
│   └── datasets/       # Tokenised + FAISS index (gitignored)
├── docs/               # Architecture, methodology, demo script
├── evaluation/         # Benchmark runner + result JSON (Phase 6)
├── frontend/           # Next.js dashboard (Phase 10)
├── models/             # Model weights (gitignored)
│   ├── base/           # Cached HuggingFace weights
│   └── adapted/        # Fine-tuned checkpoint
├── rag/                # RAG pipeline (Phase 7)
├── scripts/            # Utility scripts (ingest, verify, health-check)
├── tests/              # pytest suite
├── training/           # DAP training (Phase 5)
├── .env.example        # Environment variable template
├── .gitignore          # Excludes weights, data, secrets
├── docker-compose.yml  # Docker orchestration
├── Dockerfile.backend  # Backend container
├── pyproject.toml      # Python package + tool config
├── requirements.txt    # Python dependencies
└── start.py            # One-command launcher
```

---

## API Reference

| Method | Endpoint                    | Description                        |
|--------|-----------------------------|------------------------------------|
| GET    | `/health`                   | Service health check               |
| POST   | `/api/chat`                 | Base LLM chat                      |
| POST   | `/api/cyber/chat`           | RAG-grounded cybersecurity Q&A     |
| POST   | `/api/rag/query`            | Raw RAG retrieval + generation     |
| POST   | `/api/threat/analyze`       | Threat analysis from incident text |
| POST   | `/api/vulnerability/analyze`| CVE / vulnerability analysis       |
| POST   | `/api/document/upload`      | Upload TXT/PDF for analysis        |
| POST   | `/api/report/generate`      | Generate formal security report    |
| POST   | `/api/demo/compare`         | Live model comparison              |
| GET    | `/api/demo/questions`       | Pre-designed demo questions        |
| GET    | `/api/model/info`           | Model metadata                     |
| GET    | `/api/metrics`              | Request metrics                    |
| GET    | `/api/evaluation/results`   | Benchmark results                  |
| GET    | `/docs`                     | Interactive Swagger UI             |

---

## Configuration

All settings are controlled via environment variables (highest priority) or YAML files in `configs/`.

| Variable              | Default        | Description                    |
|-----------------------|---------------|--------------------------------|
| `BASE_MODEL_NAME`     | `distilgpt2`  | HuggingFace model ID           |
| `DEVICE`              | `auto`        | cpu / cuda / mps / auto        |
| `MAX_NEW_TOKENS`      | `256`         | Max generation tokens          |
| `TEMPERATURE`         | `0.7`         | Sampling temperature           |
| `EMBEDDING_MODEL_ID`  | `all-MiniLM-L6-v2` | Sentence embedding model |
| `RETRIEVAL_TOP_K`     | `5`           | RAG top-K retrieval            |
| `APP_ENV`             | `development` | development / production       |

See `.env.example` for the full list.

---

## Running Tests

```bash
# Fast tests (no model loading)
pytest tests/ -m "not slow" -v

# All tests including model loading (requires downloaded model)
pytest tests/ -v
```

---

## Known Limitations

1. **Base model is small (82M params)** — distilgpt2 is chosen for accessibility. Production use needs Llama 3 / Mistral.
2. **Static knowledge base** — RAG is not updated in real-time. Operational deployment needs live CVE feeds.
3. **CPU inference is slow** — ~5-30s per request on CPU. Use CUDA for production.
4. **Hallucination risk** — AI outputs must be verified by a qualified cybersecurity professional.
5. **Not adversarially evaluated** — no red-team testing has been performed.
6. **English only** — corpus and prompts are English. Multilingual CVEs are not handled.

---

## Security & Ethics

- The system is designed **exclusively for defensive cybersecurity analysis**.
- It will **not** generate exploit code, malware, or payloads.
- All outputs include a disclaimer and confidence indicator.
- A safety filter blocks requests containing exploit generation language.
- No credentials, API keys, or model weights are committed to this repository.

---

## Research Attribution

- **Domain-Adaptive Pre-Training:** Gururangan et al., "Don't Stop Pretraining" (ACL 2020)
- **RAG:** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- **MITRE ATT&CK:** MITRE Corporation — CC BY 4.0
- **NIST NVD:** National Vulnerability Database — Public Domain
- **OWASP:** Open Web Application Security Project — CC BY-SA 4.0
- **Sentence Transformers:** Reimers & Gurevych (2019) — Apache 2.0
- **FAISS:** Meta AI — MIT License

---

## Dataset Licenses

| Dataset         | Source              | License         |
|-----------------|---------------------|-----------------|
| CVE records     | MITRE / NVD         | Public Domain   |
| MITRE ATT&CK    | MITRE Corporation   | CC BY 4.0       |
| NIST Glossary   | NIST                | Public Domain   |
| OWASP Top 10    | OWASP Foundation    | CC BY-SA 4.0    |
| CWE             | MITRE Corporation   | Public Domain   |

---

## Model Licenses

| Model                          | License    |
|-------------------------------|------------|
| distilgpt2 (base)             | MIT        |
| all-MiniLM-L6-v2 (embeddings) | Apache 2.0 |
| Adapted weights (from distilgpt2) | MIT    |

---

## License

Code: [MIT](LICENSE)  
Datasets and model weights retain their original licenses — see table above.

---

## 🔬 For the Hackathon Demo

See [`docs/demo-script.md`](docs/demo-script.md) for the complete 3–5 minute presentation script including:
- Narrative flow
- Live demo prompts
- Backup questions
- Limitations talking points
- Q&A preparation
