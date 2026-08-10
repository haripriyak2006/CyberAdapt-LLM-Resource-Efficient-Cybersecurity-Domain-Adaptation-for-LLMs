# CyberAdapt-LLM
## Resource-Efficient Cybersecurity Domain Adaptation for Large Language Models

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-1%20Foundation-orange)]()

---

## Overview

CyberAdapt-LLM is a research project that applies **domain-adaptive pre-training (DAP)**
and **retrieval-augmented generation (RAG)** to adapt a general-purpose LLM for
cybersecurity tasks — threat intelligence, vulnerability analysis, and incident response —
while keeping compute requirements accessible on a single consumer GPU.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Foundation | ✅ Complete |
| 2 | LLM Service & Chat API | ⏳ Pending |
| 3 | Corpus Preparation | ⏳ Pending |
| 4 | Dataset Tokenisation | ⏳ Pending |
| 5 | Domain-Adaptive Pre-Training | ⏳ Pending |
| 6 | Evaluation Framework | ⏳ Pending |
| 7 | RAG Pipeline | ⏳ Pending |
| 8 | Threat & Vulnerability APIs | ⏳ Pending |
| 9 | Report Generation | ⏳ Pending |
| 10 | Frontend Dashboard | ⏳ Pending |

---

## Repository Layout

```
CyberAdapt-LLM/
├── configs/            # YAML configuration defaults
├── data/               # Raw → processed → tokenised corpus
├── training/           # DAP training scripts
├── models/             # Model cache (gitignored weights)
├── rag/                # Retrieval-Augmented Generation pipeline
├── evaluation/         # Benchmark runner + metrics
├── backend/            # FastAPI application
├── frontend/           # React/Next.js dashboard (Phase 10)
├── tests/              # pytest test suite
├── scripts/            # Standalone utility scripts
└── docs/               # Architecture & methodology docs
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/haripriyak2006/CyberAdapt-LLM-Resource-Efficient-Cybersecurity-Domain-Adaptation-for-LLMs.git
cd CyberAdapt-LLM-Resource-Efficient-Cybersecurity-Domain-Adaptation-for-LLMs
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Health check

```bash
# Via browser / curl
curl http://localhost:8000/health

# Via the health-check script
python scripts/health_check.py
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Expected Output

**`GET /health`**
```json
{
  "status": "ok",
  "service": "CyberAdapt-LLM",
  "version": "0.1.0",
  "phase": 1
}
```

---

## Configuration

All settings live in `configs/` YAML files and can be overridden via environment variables
(see `.env.example`).  The loader priority is:

```
Environment variables  >  .env file  >  configs/*.yaml  >  hard-coded defaults
```

---

## License

Code: [MIT](LICENSE).  
Model weights and datasets retain their original licenses — see `models/README.md`.
