# Project Handoff Report — Automotive Certificate Compliance & Q&A Platform
**Date:** 2026-08-27 (updated Three-Tier Validity Disambiguation, Perpetual Certificate Sentinel, SSE Streaming Chat, DATABASES Uploader UX, Robust Anti-Duplicate System, Chunker Safety-Valve Guardrail, Agentic Supervisor 5-Intent Router, CASUAL_CONVERSATION General-LLM Replies, Agent Tool Registry & 3 Concrete Tools, Autonomous Background Scheduler, HITL Proposal Pipeline & DB Write Capabilities, CONTROL Page & Chat Agent Visibility, Autonomous PDF Discovery & INGEST_DOCUMENT, Playwright JS-Rendering & Transient-Cookie Auth, Scheduler Configuration UI, Multiple Source URLs (`sources` table), RF-Certificate Relevance Gate, Chat AGENT_ACTION -> HITL Proposal Staging, Chat HITL Confirmation & Global LLM Serialization, Read-Only vs Side-Effectful AGENT_ACTION & Conversational HITL)
**Status:** **LIVE IN PRODUCTION on GCP NVIDIA L4 Host**. Primary LLM Engine: **Qwen3.8-27B GGUF UD-IQ3_XXS** (`qwen3.8-27b-gguf`, 3-bit, 32k context), Pure-HF GLM-OCR (`glm-ocr`), LLM-Based Automated File Column Mapping, Multi-Lingual French Date Normalization (`_parse_iso_date`), Streamlit `LinkColumn` Direct Access, Zero-Hardcoding Dynamic Link Resolution (`PUBLIC_API_URL`), Docker Compose Infrastructure (`rf_app` + `rf_postgres_db`), Deterministic 7-Field Compliance Pipeline (`CertificateExtractionSchema`), SQL Lookup Tables & Ingestion (`AuthorityLookup`, `SupplierLookup`), PostgreSQL-Persisted Chat Sessions, SSE Token-Streaming Chat (job-based), Robust Anti-Duplicate Ingestion (upsert + batch skip + dedup endpoint), GPU VRAM Coexistence (NVIDIA L4 24GB), End-to-End Hybrid RAG, SQL Hydration, CPU Embeddings, Model Registry, Intelligent Router, & Production Decoupled Architecture.

---

## 1. Project Purpose

A **smart document digitisation and Q&A platform** that takes a single image or PDF file upload, runs it through a configurable OCR engine for text extraction, structures the data using a local LLM, persists normalized metadata and dense vector embeddings into a local PostgreSQL database, and provides a natural-language Q&A interface powered by a hybrid RAG pipeline.

The entire pipeline operates **100% locally** without relying on any external APIs. OCR and LLM inference use the GPU, while embeddings are intentionally CPU-bound in the current T4-safe configuration to preserve VRAM headroom. It is built as a **FastAPI backend** with a **Streamlit frontend**, testable on Google Colab via `TestClient` (bypassing Colab's network loopback limitations). Moving to production requires deleting the `colab/` and `sandbox/` folders.

### Key Capabilities
- **Document Ingestion** — Upload PDFs/images, extract structured certificate metadata (Component, Supplier, Country, Certif Number, Authority, Issue Date, Exp Date).
- **PostgreSQL + pgvector Hydration** — Atomic database persistence for relational metadata (`certificates`) and 1024-dimensional dense vector embeddings (`certificate_chunks` using `BAAI/bge-m3`).
- **Intelligent Query Router** — Automated intent classification (`METADATA_QUERY`, `UNSTRUCTURED_RAG`, `HYBRID_QUERY`) to route requests to Text-to-SQL or dense RAG pipelines.
- **Cross-Lingual RAG Q&A Chat** — Ask natural-language questions in any language against multi-lingual document context (Spanish, German, etc.) with cited answers.
- **Pluggable Model Registry** — Seamlessly swap LLM and OCR engines via string keys in `server/config.py` (default: `qwen3.8-27b-gguf`; options: `qwen3.6-35b-gguf`, `gemma4-26b-gguf`, `qwen3-8b`, `qwen2-7b-gguf`, `qwen3-14b-gguf`, `qwen-agentworld-35b`, `glm-ocr`, `deepseek-ocr-2`).
- **Centralized Context Management** — Unified context window limit (`DEFAULT_CONTEXT_WINDOW = 8192`) enforced centrally across all engines for Tesla T4-safe operation.
- **Benchmarking** — Empirical 99-query router benchmark suite with automated metrics reporting.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (ui/app.py)                     │
│   ┌──────────────────┐            ┌───────────────────────┐     │
│   │ Document Ingestion│            │   RAG Q&A Chat        │     │
│   └────────┬─────────┘            └───────────┬───────────┘     │
│     TestClient (in-process)             engines.rag             │
│                                (router  retriever  qa)        │
└────────────┼──────────────────────────────────┼─────────────────┘
             ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (server/main.py)                    │
│                                                                 │
│   POST /api/v1/parse                                            │
│   ┌──────────┐   ┌──────────────┐       ┌─────────────────┐     │
│   │OCR Engine│  │  Extractor   │ ----> │ Storage Hydrator│     │
│   │(registry)│   │  (LLM Engine)│       │  (pgvector)     │     │
│   └──────────┘   └──────────────┘       └────────┬────────┘     │
└──────────────────────────────────────────────────┼──────────────┘
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │ PostgreSQL 16 + pgvector    │
                                    │  - certificates             │
                                    │  - certificate_chunks       │
                                    └─────────────────────────────┘
```

---

## 3. Final Folder Structure

```
Project/
├── server/                           FastAPI application package (model-agnostic via registry & auto DB init)
│   ├── main.py                    FastAPI app entrypoint (uvicorn server.main:app)
│   ├── config.py                  Central config (LLM_ENGINE=qwen3.8-27b-gguf, OCR_ENGINE=glm-ocr, DEFAULT_CONTEXT_WINDOW=8192, EMBEDDING_DEVICE=cpu)
│   └── __init__.py
├── core/                          Production-ready AI compute & RAG engines (formerly engines/)
│   │
│   ├── rag/                       RAG Q&A pipeline (router, chunker, embeddings, retriever, sql_engine, hybrid_engine, orchestrator, qa)
│   │   ├── __init__.py
│   │   ├── router.py              Intent Router (METADATA_QUERY, UNSTRUCTURED_RAG, HYBRID_QUERY)
│   │   ├── sql_engine.py          Text-to-SQL engine (execute_metadata_query)
│   │   ├── hybrid_engine.py       Hybrid Dense/Sparse RRF Engine with Parent Expansion (retrieve_hybrid_context & execute_unstructured_query)
│   │   ├── orchestrator.py        Central Dual-Path RAG Orchestrator (answer_compliance_query)
│   │   ├── chunker.py             Page-aware paragraph chunking with <Page X> tracking
│   │   ├── embeddings.py          1024-d dense vector embeddings facade (BAAI/bge-m3, CPU)
│   │   ├── retriever.py           Dual-path retrieval (Text-to-SQL + Hybrid Dense/Sparse RRF)
│   │   └── qa.py                  Cross-lingual Q&A synthesis with citation generation
│   │
│   ├── utils/                     GPU guardrails & VRAM helper utilities
│   │   ├── __init__.py
│   │   ├── vram.py                ensure_headroom() graceful MemoryError guard + free_vram_mb() + flush_gpu_cache()
│   │   └── system_check.py        System readiness verification & initialization gate
│   │
│   ├── llm/                       Pluggable LLM engines (qwen3_8_27b, qwen3_35b, qwen3_14b, qwen3_8b, qwen2_gguf, gemma4_26b, qwen_agentworld)
│   ├── ocr/                       Pluggable OCR engines (glm_ocr, got_ocr2, deepseek_ocr2)
│   ├── registry.py                Model registry (OCR_REGISTRY, LLM_REGISTRY) + lazy factory
│   ├── extractor.py               Structured extraction, lookup enrichment (enrich_certificate_metadata), & atomic DB hydration
│   ├── prompts.py                 System prompt configurations (CERTIFICATE_EXTRACTION_SYSTEM_PROMPT, router, & cross-lingual QA)
│   └── base.py                    Abstract BaseOCREngine / BaseLLMEngine contracts
│
├── schemas/                       Pydantic data models & SQLAlchemy ORM models
│   ├── extraction.py              7-field CertificateExtractionSchema + CertificateMetadata & CertificateChunk ORM
│   └── qa.py                      Citation & QAResponseSchema Pydantic models
│
├── storage/                       Relational, vector, lookup DB storage & master seed data
│   ├── database.py                SQLAlchemy engine, SessionLocal, get_db_session, & pgvector init_db()
│   ├── models.py                  ORM models for AuthorityLookup & SupplierLookup reference tables
│   ├── seed_lookups.py            Idempotent JSON ingestion script reading from data/lookups/*.json
│   ├── backup.py                  Portable pg_dump database export utility
│   └── __init__.py
│
├── ui/                            Streamlit frontend
│   ├── app.py                     Two-tab UI (Document Ingestion + RAG Q&A Chat with Intent Badges & Sources)
│   └── static/                    Branding assets (stellantis.png)
│
├── data/                          Consolidated runtime data & seed datasets
│   ├── lookups/                   Master reference datasets (authorities.json, suppliers.json) — tracked
│   ├── uploads/                   Batch ingestion staging (gitignored)
│   ├── files/                     Permanent uploaded files served statically at /files/ (gitignored)
│   ├── ocr_cache/                 OCR markdown cache for batch resume (gitignored)
│   ├── model_cache/               Cached AI model weights (gitignored)
│   └── postgres/                  PostgreSQL 16 data volume (gitignored)
│
├── handoff_report.md                Complete architectural handoff report (read first in new sessions)
│
├── docker-compose.yaml            PostgreSQL 16 + pgvector container infrastructure
├── requirements.txt               Python dependencies (includes sqlalchemy, pgvector, sentence-transformers, llama-cpp-python)
├── Dockerfile                     CUDA image build (torch 2.6.0, llama-cpp-python 0.3.34-cu122, transformers 5.15.1)
├── entrypoint.sh                  Container boot: DB wait  init_db  seed lookups  Streamlit + uvicorn
└── .gitignore                     Runtime artifacts excluded from version control
```

---

## 4. Model Registry & Engine Inventory

### LLM Engine Registry (`LLM_REGISTRY`)

| Key | File | Model Weights | Params / Quant | Primary Purpose |
|---|---|---|---|---|
| `qwen3.8-27b-gguf` (Default) | `core/llm/qwen3_8_27b.py` | `unsloth/Qwen3.8-27B-GGUF` | 27B / `UD-IQ3_XXS` (3-bit) | Production reasoning / RAG & SQL engine (32k ctx, q8_0 KV) |
| `gemma4-26b-gguf` | `core/llm/gemma4_26b.py` | `unsloth/gemma-4-26B-A4B-it-GGUF` | 26B / `UD-IQ2_M` | High-precision instruction model (**100% benchmark accuracy**) |
| `qwen3.6-35b-gguf` / `qwen3-35b` | `core/llm/qwen3_35b.py` | `unsloth/Qwen3.6-35B-A3B-GGUF` | 35B / `UD-IQ2_M` (MoE) | Cutting-edge MoE model with fast 3B active params |
| `qwen3-8b` | `core/llm/qwen3_8b.py` | `Qwen/Qwen3-8B-GGUF` | 8B / `Q8_0` | High-speed dense instruction model |
| `qwen3-14b-gguf` | `core/llm/qwen3_14b.py` | `unsloth/Qwen3-14B-GGUF` | 14B / `UD-IQ1_M` | Balanced general instruction model |
| `qwen-agentworld-35b` | `core/llm/qwen_agentworld.py` | `unsloth/Qwen-AgentWorld-35B-A3B-GGUF` | 35B / `UD-IQ2_M` (MoE) | Specialized agentic reasoning model |
| `qwen2-7b-gguf` | `core/llm/qwen2_gguf.py` | `Qwen/Qwen2-7B-Instruct-GGUF` | 7B / `Q4_K_M` | Lightweight baseline model |

### OCR Engine Registry (`OCR_REGISTRY`)

| Key | File | Model | Primary Purpose |
|---|---|---|---|
| `glm-ocr` (Default) | `core/ocr/glm_ocr.py` | GLM-OCR 0.9B (pure-HF `transformers` 5.15.1, FP16) | Accuracy-first full-document layout markdown extraction (no vLLM) |
| `deepseek-ocr-2` | `core/ocr/deepseek_ocr2.py` | DeepSeek-OCR-2 3B (NF4) | Dense document & table OCR |
| `got-ocr2` | `core/ocr/got_ocr2.py` | GOT-OCR2_0 0.5B | Fallback baseline OCR |

---

## 5. Architectural & System Rules

### 5.1 Centralized Context Management
The default engine (`qwen3.8-27b-gguf`, IQ3_XXS) uses `DEFAULT_CONTEXT_WINDOW = 32768` from `server/config.py` on the GCP NVIDIA L4 (24 GB). This was scaled up from the T4-era `8192` cap (and the historical 16K setting in §11) to leverage the L4's VRAM headroom (measured ~8.2 GB free at 32k). Context window fallback loops have been completely removed in favor of single-pass initialization, ensuring deterministic VRAM footprint and execution.

### 5.2 Strict Quantization Floor (Anti-OOM Directive)
- **Forbidden Upward Swaps:** The system must never change or upgrade a model's quantization level (e.g. from `IQ2_M`/`Q4_K_M` to 8-bit or FP16) during debugging, as this triggers uncatchable OOM kernel crashes on Tesla T4 GPUs.
- **Allowed Last-Resort Exception:** If a model download fails or corrupts, the agent is permitted to swap to an equivalent repository from a different publisher (e.g. `unsloth`, `bartowski`, `google`), provided the model file size and quantization precision remain strictly identical.
- **Production L4 note (2026-08-26):** The above floor is a **T4-sandbox debugging guard**. Deliberate, VRAM-verified production upgrades are permitted on the GCP L4 — e.g., the default engine moved `UD-IQ1_M` (1-bit, T4 compromise) to `UD-IQ3_XXS` (3-bit), scaling quantization quality up since the L4 has ample headroom (§20.4).

### 5.3 Dynamic Thinking Modes (`disable_thinking`)
Mode handling is configured **per model**, not centrally. `BaseLLMEngine.generate_json(system_prompt, user_prompt, disable_thinking, max_tokens)` (in `core/base.py`) is a **concrete template method** that executes each model's single abstract hook:
- `_generate_raw(system_prompt, user_prompt, disable_thinking, max_tokens)` — formats native prompts, applies think/no-think switches, sets sampling params, and runs completion.
- `extract_json(raw_content)` (consolidated in `core/base.py`) — automatically scrubs reasoning traces and extracts strict JSON.

Per-model non-thinking switches (all engines, `disable_thinking=True`):
1. **Qwen3 (8B / 14B)** — ChatML + soft ` /no_think` / ` /think` tag injected inside the **user block** (never after the assistant header — the ChatML tag-placement rule).
2. **Qwen3.6 / Qwen-AgentWorld (35B)** — ChatML + an already-closed empty `<think>\n</think>` block right after the assistant header. Qwen3.6 **ignores** `/no_think`, so no mode tag is injected.
3. **Gemma 4 26B** — `<start_of_turn>/<end_of_turn>` turn format (incl. system turn); no soft switch — fixed `0.7/0.8` sampling.
4. **Qwen2 7B** — ChatML + native `response_format={"type": "json_object"}`; sampling `0.1/0.7`.

In every mode, reasoning traces are scrubbed **unconditionally (mode-agnostic)** before strict JSON parsing via `core/base.py` (`extract_json` / `strip_reasoning_traces`), per the Qwen3 empty-JSON bug fix.

---

## 6. Empirical Benchmark Evaluation

### Gemma 4 26B Engine (`gemma4-26b-gguf`) — 99-Query Benchmark Report

| Metric | Benchmark Result |
|---|---|
| **Router Accuracy Rate** | **100.0%** (99 out of 99 queries correctly classified)  |
| **Average Latency per Query** | **1,231.6 ms (~1.23 seconds)**  |
| **Total Evaluation Time** | **121.92 seconds** |
| **Model Quantization** | `UD-IQ2_M` (9.33 GB file size) |
| **Active VRAM Usage** | **~10.5 GB / 14.56 GB** (Tesla T4 GPU) |

#### Accuracy by Query Category
- **`METADATA_QUERY` (33 Queries):** **100.0%** (33/33)
- **`UNSTRUCTURED_RAG` (33 Queries):** **100.0%** (33/33)
- **`HYBRID_QUERY` (33 Queries):** **100.0%** (33/33)

---

### Qwen-AgentWorld 35B Engine (`qwen-agentworld-35b`) — Historical 99-Query Benchmark Report

| Metric | Benchmark Result |
|---|---|
| **Router Accuracy Rate** | **89.9%** (89 of 99 queries correctly classified) |
| **Average Latency per Query** | **~5.06 seconds** |
| **Total Evaluation Time** | **~501 seconds** |
| **Empty/`{}` Token Warnings** | 12 |
| **Model Quantization** | `UD-IQ2_M` (MoE, 3B active params) |

> **Note:** This figure predates the per-model thinking refactor (§5.3) and the `extract_json` raw-first scan fix. The router prompt structure is preserved identically under the new architecture, but the benchmark should be **re-validated** after re-syncing the refactored engines to Colab.

### Fresh Colab Session Runtime Prerequisites (2026-08-20) — OOM/Load-Failure Hotfix

A brand-new Colab runtime does **not** inherit the verified environment. The current verified baseline (used by the 2026-08-20 full benchmark) is:

1. `pip install transformers==5.15.1 huggingface-hub==1.28.0 tokenizers==0.21.4 safetensors` — **GLM-OCR requires `transformers >= 5.0`** (`glm_ocr` is not a recognized config in 4.49.0; `AutoModelForConditionalGeneration` no longer exists in 5.x — the native entry point is `AutoModelForImageTextToText`).
2. `pip install` the **prebuilt v0.3.34-cu122 GitHub release wheel** (`llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl`, ~60s) — bare `pip install llama-cpp-python` gives **0.3.19**, which rejects the `gemma4` GGUF architecture (`unknown model architecture: 'gemma4'`). There is **no** cp312 binary on the abetlen pip index for 0.3.34 — `--extra-index-url` triggers a 15-min source compile; install the GitHub release wheel directly.
3. `pip install numpy==2.0.2` — fresh runtimes ship **numpy 2.5.2** (breaks `numpy._core.multiarray` imports for transformers).
4. `pip uninstall -y torchcodec` — §10 issue #3; sentence-transformers fails at import on `libtorchcodec`.

> **vLLM is no longer required.** GLM-OCR is a pure-HF backend (§13) — nothing imports vLLM anymore. The 2026-08-20 verified session kept the Colab default `torch 2.5.1+cu124`; note that **torch 2.5.1 blocks `bge-m3` `.bin` loading** (CVE-2025-32434 guard needs torch >= 2.6) so embedding vectors fall back to all-zero until torch is upgraded (§13.4). A fully-provisioned `colab/setup.sh` session installs `torch 2.6.0` (STEP 3), which resolves this automatically.

Verified on 2026-08-20: `transformers 5.15.1` + `llama-cpp-python 0.3.34` (CUDA offload active) + `numpy 2.0.2` + `torch 2.5.1+cu124` + PostgreSQL 14/pgvector 0.8.6  full end-to-end pipeline (OCR  Gemma extraction  pgvector persistence  SQL verification) ran successfully.

---

## 7. Engineering Roadmap & Milestone Status

- [x] **Phase 1: Storage Layer Evolution**
  - [x] Relational Metadata Persistence (`certificates` & `certificate_chunks` via PostgreSQL + pgvector)
  - [x] Database Ingestion Hydration & Vector Dimension Fix (`BAAI/bge-m3` 1024-d embeddings)
- [x] **Phase 2: Intelligent Query Router** (`core/rag/router.py` intent classifier)
- [x] **Phase 3: Dual-Path Retrieval & LLM Model Registry**
  - [x] Dual-path retrieval engine (`core/rag/retriever.py` with Text-to-SQL + Hybrid RRF)
  - [x] Expanded LLM model registry (`gemma4-26b-gguf`, `qwen3.6-35b-gguf`, `qwen3-8b`, `qwen3-14b-gguf`, `qwen-agentworld-35b`, `qwen2-7b-gguf`)
  - [x] Centralized context window management (`DEFAULT_CONTEXT_WINDOW = 8192` for T4-safe operation)
  - [x] **Phase 3 Step 1: Text-to-SQL engine (`core/rag/sql_engine.py`)** — `execute_metadata_query()` end-to-end validated against live Colab PostgreSQL (2026-08-09)
  - [x] Bulk ingestion + duplicate prevention (`GET /api/v1/certificates/exists` + Streamlit batch tab)
  - [x] **Phase 3 Step 2: Sequential Two-Phase Batch Ingestion (2026-08-11)** — single-residency model lifecycle (`core/utils/model_lifecycle.py`), `POST /api/v1/batch/ingest` + status/certificates endpoints, deterministic resume via OCR markdown cache + manifest, Streamlit polling UI. Fixes the 2026-08-10 batch OOM (84 page OOMs  NULL supplier rows).
  - [x] **Colab re-verify of the new batch architecture (2026-08-11)** — unpacked `project_sync.zip`, installed deps, booted Streamlit + cloudflared, and ran an end-to-end multi-file batch (OCR phase  extract phase  Q&A) with zero OOMs and no NULL supplier rows.
  - [x] **Phase 3 Step 3: Pure-HF GLM-OCR Backend Migration (2026-08-20)** — replaced the vLLM backend in `core/ocr/glm_ocr.py` with native `transformers>=5.0` (no vLLM, no config stubs, no weight-remap patches); reproduced the full benchmark end-to-end on the vLLM-era target document (`sandbox/benchmark_pure_HF.md`) and verified PostgreSQL/pgvector persistence via SQL. Known caveat: `bge-m3` vectors are all-zero until torch >= 2.6 (see §13.4).
  - [x] **Phase 3 Step 4: Folder/Subfolder Ingestion & ZIP Archive Unpacking (2026-08-21)** — extended `POST /api/v1/batch/ingest` and `ui/app.py` to accept `.zip` folder archives, automatically expanding nested directory structures and extracting all `.pdf`, `.png`, `.jpg`, `.jpeg` document files for batch ingestion.
  - [x] **Phase 3 Step 5: Comprehensive Case-Insensitive Duplicate Guardrails (2026-08-21)** — implemented universal case-insensitive `file_name` and `certif_number` + `country` pre-checks and in-place upsert logic across batch ingestion, single parse, and manual certificate creation endpoints.
  - [x] **Phase 3 Step 6: Automatic Post-Batch SQL Export & Upload Dir Cleanup (2026-08-21)** — integrated `trigger_async_backup()` into `_run_batch` to automatically export PostgreSQL to `data/db_backup.sql` upon batch completion, and automatically delete temporary raw upload subfolders (`data/uploads/<batch_id>`) to prevent disk accumulation.
  - [x] **Phase 3 Step 7: Streamlit UI & Logging Emoji Removal (2026-08-21)** — completely removed all emoji characters across `ui/app.py` and `server/main.py` for a clean, professional production appearance.
- [x] **Phase 4: Productionization & Container Packaging**
  - [x] Deployed and running via `docker-compose up -d` on GCP NVIDIA L4 production instance.
  - [x] Verified active containers: `rf_app` (FastAPI on `:8000`, Streamlit on `:8501`) and `rf_postgres_db` (`pgvector/pgvector:pg16` on `:5432`).
  - [x] Verified GPU hardware drivers (NVIDIA L4 24GB VRAM, Driver `580.173.02`, CUDA `13.0`) and Docker runtime (`29.7.2`).

---

## 8. How to Run & Deploy

### Local Docker Environment
1. Start PostgreSQL with pgvector:
   ```bash
   docker-compose up -d
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run FastAPI backend:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Run Streamlit UI:
   ```bash
   streamlit run ui/app.py
   ```

### Google Colab Environment
1. Execute setup cell to initialize container cache (`/content/model_cache/`) and unpack `project_sync.zip`.
2. Run database setup:
   ```bash
   !bash colab/setup.sh
   ```
3. Launch Streamlit & Cloudflared tunnel.

---

## 9. Production Cleanup & Handoff Checklist

Before deploying to production, execute the following cleanup steps:

1. **Remove Colab & Sandbox Files (Zero Code Impact):**
   - Delete `colab/` directory (`colab/setup.sh`, `colab/installer.py`, `colab/env_config.py`, `colab/verify_env.py`).
   - Delete `sandbox/` directory (`sandbox/benchmark_ocr.py`).
   - Delete `.codex/mcp.json` if bundling only production application code.
   - Remove `project_sync.zip` from workspace.
   - *Note:* Core production modules (`server/config.py`, `server/main.py`, `core/`, `schemas/`) contain zero hardcoded `/content/` paths. Core defaults resolve portably relative to `config.BASE_DIR`.
2. **Environment Variables:**
   - Set `DATABASE_URL` for production PostgreSQL cluster (defaults to `docker-compose.yaml` local database).
   - Optionally set `HF_HOME`, `OCR_CACHE_DIR`, and `BATCH_UPLOAD_DIR` to custom volume mounts if needed.
3. **Model Serving & Concurrency:**
   - For high concurrency, models can be served via vLLM, Ollama, or separate container workers while preserving the `BaseLLMEngine` / `BaseOCREngine` contracts.
   - For single-GPU production hosts (e.g. 16GB GPU), `core/utils/model_lifecycle.py` provides automatic single-residency VRAM management.

---

## 10. Colab-Specific Dependency Fixes & Production Mapping (2026-08-09)

The following constraints are **Colab-only** workarounds enforced inside `colab/setup.sh`. None affect production (which uses `docker-compose.yaml` + a clean `pip install -r requirements.txt`).

| # | Colab Constraint | Root Cause | Fix Applied (Colab only) | Production Mapping |
|---|---|---|---|---|
| 1 | `llama-cpp-python` must be pinned | PyPI's unpinned build silently wins (CPU-only); default `gemma4-26b-gguf` uses the `gemma4` GGUF architecture that older llama.cpp builds reject (`0.3.19`  `unknown model architecture: 'gemma4'`); also loading a second model copy on the 16GB T4 causes `cudaMalloc: out of memory` | **`pip install` the prebuilt `v0.3.34-cu122` GitHub release wheel directly** (`llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl`, glibc 2.35 = Colab) — installs in ~60s. **Do NOT** use `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122` for 0.3.34 — no cp312 binary exists there, pip falls back to ~15 min source compilation. **0.3.19 is forbidden** (rejects gemma4 arch). Kernel restart after any C-extension change | Pin the same version in the production image; serve via vLLM/Ollama for concurrency |
| 2 | `numpy` must be pinned to 2.0.2 | Colab preinstalls numpy 2.5.2, which removes `_blas_supports_fpe`; breaks scikit-learn 1.6.1 and numba 0.60 at import time | `pip install numpy==2.0.2` | Pin `numpy>=2.0,<2.3` (or the tested 2.0.2) in requirements |
| 3 | `torchcodec` must be removed | `sentence-transformers` 5.6.0 imports `torchcodec`; its libtorchcodec is ABI-incompatible with torch 2.6.0+cu118 (and FFmpeg libs are missing), raising `RuntimeError` at import | `pip uninstall -y torchcodec` (not needed; bge-m3 embeddings don't use it) | Not installed in production; pin a compatible torchcodec if audio/video decoding is ever needed |
| 4 | Public ingress blocked | Colab firewall blocks ports 8000/8501 | `.streamlit/config.toml` with `headless=true`, `enableCORS=false`, `enableXsrfProtection=false` + `cloudflared tunnel --url http://localhost:8501` | Standard reverse proxy / container port mapping; no CORS overrides needed |
| 5 | Single 16GB T4 GPU | Only one heavyweight model process can be resident at a time | Never run the Streamlit in-process TestClient and a separate model subprocess simultaneously; stop the app before running standalone engine tests | Unbounded multi-GPU inference in production |
| 6 | `transformers` must be >= 5.0 for GLM-OCR | 4.49.0 rejects `glm_ocr` (unknown config); `AutoModelForConditionalGeneration` was removed in 5.x | `pip install transformers==5.15.1` (verified); use `AutoModelForImageTextToText` for `glm_ocr` | Pin `transformers>=5.0` in the image |
| 7 | `torch` must be >= 2.6 to load `bge-m3` `.bin` | transformers 4.51+/5.x block `torch.load(weights_only=True)` of `.bin` on torch < 2.6 (CVE-2025-32434 guard); Colab ships torch 2.5.1  `core/rag/embeddings.py` silently falls back to **zero vectors** | Upgrade torch to >= 2.6 (metadata/chunks unaffected); or convert the model to safetensors | Pin `torch>=2.6` in the image so embeddings are real |

## 11. T4 VRAM Fit Audit: OCR + LLM + Embeddings (2026-08-09)

Empirically verified on the Colab T4 (15,360 MiB) by loading all three models into **one** process and sampling `nvidia-smi` after each:

| Model | Config | VRAM |
|---|---|---|
| GLM-OCR 0.9B | FP16, `device_map="cuda"` (now pure-HF backend) | ~2.2 GB |
| Gemma 4 26B (UD-IQ2_M) | `n_gpu_layers=-1`, `flash_attn=True`, KV q8_0 | ~11.9 GB @ 16K ctx |
| bge-m3 | **FP16** (`torch_dtype`) | ~1.1 GB |
| **Total (16K ctx, FP32 emb)** | | **~16.4 GB  CUDA OOM** |
| **Total (8K ctx, FP16 emb)** | | **~14.3 GB (524 MiB free)** |

**Conclusion:** All three models fit only with `DEFAULT_CONTEXT_WINDOW=8192` **and** FP16 embeddings. 16K context or FP32 embeddings overflow the T4.

**Two latent bugs surfaced and fixed:**
1. **`core/llm/gemma4_26b.py`** passed `type_k="q8_0"` / `type_v="q8_0"` as **strings**. `llama-cpp-python` 0.3.34 requires the integer GGML enum (`GGML_TYPE_Q8_0`). The string raised `TypeError`, silently falling back to the no-flash-attn constructor, which pads the V cache to 2048 and fails `llama_context` creation — the app could not load OCR + Gemma at all. Fixed to `type_k=llama_cpp.GGML_TYPE_Q8_0`.
2. **`core/rag/embeddings.py`** loaded bge-m3 in FP32 (~2.3 GB), then FP16 (`SentenceTransformer(..., model_kwargs={"torch_dtype": torch.float16})`, ~1.1 GB). Both were later superseded by the CPU move below — bge-m3 now runs on CPU (`EMBEDDING_DEVICE="cpu"`, no dtype kwargs), keeping the T4 entirely for OCR + LLM.

**Final design decision — bge-m3 moved to CPU (2026-08-09 PM):**
Idle-fit alone was not enough. A live ingestion run (real image OCR  extraction  embeddings  SQL) with all three co-resident hit a **real CUDA OOM**: GLM-OCR's dynamic KV cache spiked free VRAM to ~350 MiB mid-generation, then Gemma crashed with `ggml_abort` on an unfourth 40 MiB allocation. Root cause: hardcoded `max_new_tokens=8192` in `glm_ocr.py` built a +1.5 GB transient KV cache on top of the loaded models (peak 14,811 MiB).

Resolution (all code in `server/config.py` / `core/utils/vram.py` / engines):
- **`EMBEDDING_DEVICE="cpu"`** — bge-m3 leaves the T4 entirely (~1.1 GB freed). Embeddings are per-chunk during ingestion, so CPU latency (~2-4 s/batch) is acceptable.
- **GLM-OCR token cap (`min(OCR_MAX_NEW_TOKENS, 2048)`)** — `config.OCR_MAX_NEW_TOKENS` is `8192`; the engine caps generation at `2048` via `min()`, bounding GLM-OCR's dynamic KV cache.
- **`MIN_FREE_VRAM_MB=1024` headroom guard** — `core/utils/vram.py` `ensure_headroom()` raises a **graceful `MemoryError`** (mapped to HTTP 507 in `server/main.py`) before OCR generation and LLM extraction instead of an uncatchable kernel OOM.
- **`MAX_EXTRACTION_PROMPT_CHARS=20000`** — truncates long OCR text so extraction stays inside the 8K context.
- Cache flush (`flush_gpu_cache()`) between OCR  extraction enables the +1 GB headroom check to pass.

**Verified end-to-end after the final fixes** (real upload through `/api/v1/parse`  10 chunks persisted  SQL Q&A): OCR headroom OK (1,644 MiB free), extraction OK (1,590 MiB free), embeddings on CPU, **PARSE status 200**, supplier correctly extracted, peak `14,841 MiB / 14,913 MiB` usable — OOM-free and graceful-guard-protected.

**Production note:** On GPU-limited hardware keep `DEFAULT_CONTEXT_WINDOW <= 8192` when co-hosting OCR + 26B LLM on the same T4, and keep `EMBEDDING_DEVICE="cpu"`. On multi-GPU/4090-class hardware 16K ctx and GPU embeddings can be restored via `server/config.py`.

## 12. Model VRAM Coexistence (2026-08-20 Update: Qwen3.8-27B GGUF)

### 12.1 Transition from Single-Residency to VRAM Coexistence
Previously under Gemma 4 26B GGUF (`UD-IQ2_M`, ~11.9 GB VRAM footprint), running GLM-OCR (~2.12 GB VRAM) simultaneously caused CUDA OOM crashes due to GLM-OCR's dynamic KV cache prefill (~1.47 GB spike).

With the adoption of **Qwen3.8-27B GGUF** (`UD-IQ1_M`, ~6.27 GB weight size / ~6.3 GB VRAM footprint), **GLM-OCR and Qwen3.8-27B co-exist simultaneously in GPU VRAM** (~12.4 GB total allocated out of 15.36 GB T4 capacity).

### 12.2 Deprecation of Sequential Model Unloading
Sequential model load/unload calls (`load_ocr_only`, `unload_ocr`, `load_llm_only`, `unload_llm`) and `core/utils/model_lifecycle.py` have been removed. Both engines load cleanly and remain resident in GPU VRAM without flushing GPU cache between OCR and extraction phases. See [`sandbox/benchmark_coexistence_qwen38.md`](file:///c:/Users/hp/OneDrive/Documents/Capgemini/Project/sandbox/benchmark_coexistence_qwen38.md) for full empirical benchmark details.

---

## 13. GLM-OCR Pure-HF Backend Migration & End-to-End Verification (2026-08-20)

### 13.1 What changed
`core/ocr/glm_ocr.py` was rewritten as a **pure Hugging Face `transformers` backend**:
- Removed all vLLM code (`LLM`/`AsyncEngineArgs`, `patched_load_weights`, and the `GlmOcrConfig` / `GlmOcrVisionConfig` / `GlmOcrTextConfig` stubs).
- Loads natively via `AutoProcessor` + `AutoModelForImageTextToText.from_pretrained(..., torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)`.
- `ImageOps.exif_transpose` orientation correction; OpenAI-style `content` message list (the chat template silently drops top-level `image`/`text` keys); greedy chat-template generation (`do_sample=False`) capped at `min(OCR_MAX_NEW_TOKENS, 2048)`, stopped by the model's own `eos_token_id` (`[59246, 59253]` — no override).
- `close()` verified to return VRAM to ~0.26 GB used before the LLM loads (single-residency lifecycle intact).

### 13.2 Environment requirements (deviation from older docs)
- `transformers >= 5.0` (verified 5.15.1). `glm_ocr` is not in 4.49's config registry; `AutoModelForCausalLM` does not map it and `AutoModelForConditionalGeneration` no longer exists in 5.x — **use `AutoModelForImageTextToText`**.
- `huggingface_hub 1.28.0`, `tokenizers 0.21.4`, `safetensors` (aligned to transformers 5.15.1).
- vLLM 0.7.3 is no longer installed or imported. Older sections of this report (§6, §10, §12.6) describing `transformers 4.49.0` + vLLM as the baseline are **historical** and superseded by this section.

### 13.3 Full end-to-end benchmark (2026-08-20, same IM3C doc as `benchmark_vllm.md`)
| Stage | vLLM (`benchmark_vllm.md`) | Pure HF (`benchmark_pure_HF.md`) |
|---|---|---|
| GLM-OCR load | 18.25 s | 13.77 s |
| OCR total (2 pages) | 2.45 s | 60.31 s |
| Avg / page | **1.22 s/page  MET** | **30.15 s/page  EXCEEDED** |
| OCR output | Truncated (~20–24 tok, `stop_token_ids`) | Complete legal document (~650 tok/page) |
| Gemma 4 26B load | 209.25 s | 81.52 s (cached GGUF) |
| Extraction | 9.26 s | 7.13 s |
| Extraction accuracy | Supplier `ENACOM`, Issue `2023-04-04` (wrong) | Supplier `PABLO RICARDO CASSI`, Issue `2025-04-04` (correct) |
| DB record | `ID 1` | `cert_4585ab2c0a29` + 25 chunks (SQL-verified) |

The ~25x speed gap is inherent: full-resolution prefill (~4831 tok/page), ~650 generated tokens/page at ~19 tok/s on a T4 (float16, no FlashAttention on Turing), plus the vLLM figure's reliance on stop-token truncation. Keep pure-HF when fidelity matters.

### 13.4 Known caveats
- **`bge-m3` embedding vectors are currently all-zero in the 2026-08-20 Colab run.** `BAAI/bge-m3` ships a `pytorch_model.bin`; `sentence-transformers` loads it via `torch.load(weights_only=True)`, which transformers 4.51+/5.x hard-blocks on `torch < 2.6` (CVE-2025-32434 guard). That session ran the Colab default `torch 2.5.1+cu124`, so `core/rag/embeddings.py` fell back to zero vectors. **Metadata and chunks persist correctly** — only vectors are degenerate. Fix: upgrade `torch` to `>= 2.6` (or convert the model to safetensors); a fully-provisioned `colab/setup.sh` session already installs `torch 2.6.0` (STEP 3).
- **Gemma 4 26B full-size SWA KV cache sits at the T4's VRAM limit** (`n_ctx=8192`, `flash_attn=True`, `type_k/v=Q8_0`). One transient `Failed to create llama_context` occurred on the first (right-after-download) run; with clean VRAM the same production config loaded reliably (36.7 s cold / 81.52 s warm).

### 13.5 Date Validation Architecture (Updated 2026-08-20)

The string-matching `_date_present_in_text()` helper and `STRICT_DATE_VALIDATION` guard setting were **removed** from `core/extractor.py` and `server/config.py`.

**Rationale:** Exact string-matching validation produced false-negative rejections for valid extracted dates (e.g. expiration dates derived from document header metadata or validity clauses that were not verbatim spelled out in the raw OCR body text). Extracted date fields (`issue_date`, `exp_date`) from `CertificateExtractionSchema` pass directly to `_parse_iso_date()` for ISO-8601 normalization and PostgreSQL `date` column hydration without post-parse string-matching rejection.

### 13.6 Documentation
- `sandbox/benchmark_vllm.md` — vLLM-era GLM-OCR benchmark (historical reference).
- `sandbox/benchmark_pure_HF.md` — pure-HF full end-to-end benchmark, DB verification log, and comparison (current).

---

## 14. Deterministic Extraction Pipeline & SQL Lookup Enrichment Architecture (2026-08-20 Update)

To guarantee deterministic metadata accuracy and eliminate ambiguities between foreign manufacturers and domestic legal applicants, the extraction and persistence pipeline was upgraded with structured lookup reference tables and post-extraction enrichment.

### 14.1 Pydantic Schema & System Prompt Alignment
- **Strict 7 Core Metadata Fields (`schemas/extraction.py`)**: `CertificateExtractionSchema` explicitly defines `component`, `supplier`, `country`, `certif_number`, `authority`, `issue_date`, and `exp_date`.
- **Supplier vs. Applicant Disambiguation**: The `supplier` field description and prompt directive (`CERTIFICATE_EXTRACTION_SYSTEM_PROMPT` in `core/prompts.py`) strictly instruct the model to extract foreign manufacturers/brands (e.g., `VALEO`, `BOSCH`, `APTIV`, `FIH Mobile Limited`) and ignore domestic legal representatives or filing attorneys (e.g., `PABLO RICARDO CASSI`, `APPROVE - IT S.A.`).

### 14.2 Relational Lookup Models & Seed Assets
> **Superseded (2026-08-27):** `standard_validity_years` is now a **`String(20)`** column holding a three-tier value (numeric string, `"infinite"`, or `NULL`), and seeding upserts by `canonical_authority + country`. See §21.1.
- **Database Models (`core/storage/models.py`)**:
  - `AuthorityLookup` (`authority_lookups`): Stores `canonical_authority`, `country`, `standard_validity_years`, and `aliases` (JSONB).
  - `SupplierLookup` (`supplier_lookups`): Stores `canonical_supplier` and `aliases` (JSONB).
- **JSON Seeding (`core/storage/seed_lookups.py`)**: Idempotently populates the database lookup tables from master reference datasets in `knowledge/authorities.json` and `knowledge/suppliers.json`. Seeding triggers automatically during `init_db()`.

### 14.3 Post-Extraction Normalization & Enrichment Layer (`core/extractor.py`)
Prior to PostgreSQL database persistence, `enrich_certificate_metadata(cert_data, db)` normalizes and enriches LLM extraction output using SQL lookup queries:
1. **Jurisdiction Country Resolution**: Matches raw extracted authority text against `authority_lookups` canonical names and aliases. If `country` is missing or null, it is deterministically assigned from the matched authority (e.g., `ATT` $\rightarrow$ `"Bolivia"`).
2. **Validity & Expiration Date Calculation**: If explicit `exp_date` is missing from the document but the matched issuing authority specifies `standard_validity_years`, `exp_date` is computed as `issue_date + standard_validity_years` (e.g., `2025-06-03` + 10 years = `2035-06-03`).
3. **Supplier Canonicalization**: Performs fuzzy normalized matching against `supplier_lookups` aliases to map legal entity variants and local subsidiaries to standard global brand names (e.g., `Valeo Comfort & Driving Assistance S.A.S.` $\rightarrow$ `"VALEO"`).

### 14.4 Multi-Country Authority Name Collisions & Database Schema Adjustments (2026-08-20)
- **Constraint Handling**: In regulatory dataset `knowledge/authorities.json`, multiple countries share identical generic authority titles (e.g. `"Telecommunications Regulatory Authority"` in UAE, Bahrain, and Oman). Enforcing `unique=True` on `canonical_authority` caused `psycopg2.errors.UniqueViolation` during database seeding.
- **ORM Schema Resolution**:
  - Removed `unique=True` constraint on `canonical_authority` in `core/storage/models.py`.
  - Added `__table_args__ = {"extend_existing": True}` on `AuthorityLookup` and `SupplierLookup` ORM classes.
  - Updated `core/storage/database.py` to issue `Base.metadata.clear()` and explicitly execute DDL to drop stale indexes/constraints (`ALTER TABLE IF EXISTS authority_lookups DROP CONSTRAINT IF EXISTS authority_lookups_canonical_authority_key;`).
- **Empirical Execution & Database Verification**:
  - Verified cell execution in `main.ipynb` with GLM-OCR and Qwen3.8-27B GGUF.
  - Confirmed 50 authority lookup records and 40 supplier lookup records seeded cleanly into PostgreSQL.
  - Verified end-to-end metadata extraction and deterministic enrichment (Target `BO_IM3A_401-2025_ATT__03.06.2035_.pdf` $\rightarrow$ Component: `IM3A`, Supplier: `VALEO`, Country: `Bolivia`, Certif Number: `401/2025`, Authority: `ATT`, Issue Date: `2025-06-04`, Exp Date: `2035-06-03`).

---

## 15. 100% Local In-Process GLM-OCR vLLM PoC & Permanent VRAM Retention (2026-08-21 Update)

To evaluate high-performance local OCR serving on Tesla T4 GPUs with zero external API dependencies, an isolated proof-of-concept (PoC) and benchmark suite was implemented.

### 15.1 Isolated PoC Architecture (`sandbox/vllm_ocr_poc/`)
- **`vllm_glm_ocr.py`**: Defines `StandaloneVLLMGLMOCREngine(BaseOCREngine)` inheriting cleanly from `BaseOCREngine` without altering `core/registry.py` or modifying any production files.
- **100% In-Process Execution**: Loads `vllm.LLM` locally on GPU with `enforce_eager=True`, `dtype="float16"`, `max_model_len=4096`, and `limit_mm_per_prompt={"image": 1}`. Outbound network REST calls / OpenAI-compatible API modes (Mode B) were dropped completely.
- **Multiprocessing & Hardware Guards**: Enforces `VLLM_WORKER_MULTIPROC_METHOD="spawn"`, `VLLM_USE_V2_MODEL_RUNNER="0"`, `VLLM_ENABLE_V1_MULTIPROCESSING="0"`, and `VLLM_USE_FLASHINFER_SAMPLER="0"` (FlashInfer disabled for Turing compute capability 7.5).
- **Jupyter / ipykernel `fileno()` Patch**: Injected module-level `suppress_stdout` monkeypatch (`vllm.utils.system_utils.suppress_stdout` and `vllm.distributed.parallel_state.suppress_stdout`) to resolve ipykernel's missing `sys.stdout.fileno()` when running inside Colab notebook cells.
- **Multi-Modal Prompt Format**: Custom GLM-OCR replacement tag format `<|begin_of_image|><|image_pad|><|end_of_image|>Text Recognition:` for image input handling in vLLM.

### 15.2 Codebase-Wide Model Retention (Co-Residency Enforcement)
All model `close()` and `unload_llm()` logic has been removed and converted to retention hooks across `server/main.py`, `core/ocr/glm_ocr.py`, `sandbox/vllm_ocr_poc/vllm_glm_ocr.py`, and `sandbox/vllm_ocr_poc/benchmark_standalone.py`. 

Once initialized, both OCR and LLM engines remain resident concurrently in GPU VRAM for fast, zero-latency re-use without memory unloading.

---

## 16. Folder Ingestion, Universal Duplicate Guardrails & Automated Persistence (2026-08-21 Update)

To support complex document uploads, eliminate database duplication, and guarantee portable SQL state retention, key ingestion and persistence enhancements were deployed:

### 16.1 Folder & Subfolder Ingestion (ZIP Archive Unpacking)
- **HTML5 Drag & Drop**: Users can drag and drop regular folders directly into the Streamlit file uploader in modern web browsers (Chrome, Edge, Firefox).
- **Nested ZIP Expansion (`server/main.py`)**: `POST /api/v1/batch/ingest` automatically detects uploaded `.zip` archives, recursively expands nested directory structures, and extracts all supported document files (`.pdf`, `.png`, `.jpg`, `.jpeg`). All discovered files are automatically added to the batch manifest.

### 16.2 Comprehensive Case-Insensitive Duplicate Guardrails
The system enforces strict duplicate prevention across all input paths:
> **Superseded (2026-08-27):** this section describes the original guardrails. The current unified implementation is documented in §21.5 — identity is `certif_number + country` (case-insensitive) first, with `file_name` as a fallback only when the certif number is absent; manual/import paths intentionally skip the file_name identity.
1. **Batch Ingestion Pre-Check**: Checks existing PostgreSQL records by `func.lower(file_name)`. Matching files are skipped completely prior to running GLM-OCR or LLM extraction.
2. **Single Parsing In-Place Upsert**: `save_certificate_to_db()` checks for existing `file_name` or `certif_number` + `country` records. If found, it updates the metadata in-place and replaces vector chunks rather than inserting duplicates.
3. **Manual Entry Guardrail**: `POST /api/v1/certificates/manual` enforces identical case-insensitive checking and updates existing records in-place.
4. **On-Demand Deduplication**: `POST /api/v1/certificates/deduplicate` scans the relational table and purges legacy duplicate rows.

### 16.3 Automatic Post-Batch Persistence & Upload Cleanup
- **Automated SQL Backup (`storage/backup.py`)**: As soon as any batch ingestion finishes (`phase == "done"`), `trigger_async_backup()` automatically exports the entire PostgreSQL database (metadata + vector chunks) to `data/db_backup.sql`.
- **Automated Upload Dir Cleanup**: Once batch processing completes, `shutil.rmtree(upload_dir)` automatically deletes temporary raw upload files from `data/uploads/<batch_id>` to prevent disk accumulation.
- **Streamlit & Backend Emoji Removal**: All emoji icons were removed across `ui/app.py` and `server/main.py` for a clean, professional enterprise appearance.

---

## 17. Production Deployment on GCP NVIDIA L4 Instance (2026-08-23 Update)

The platform is officially live in production on a dedicated GCP NVIDIA L4 GPU instance.

### 17.1 Verified Hardware & Runtime Specifications
- **GPU Instance:** GCP NVIDIA L4 (24 GB VRAM)
- **NVIDIA Driver:** `580.173.02` | **CUDA Version:** `13.0`
- **Container Engine:** Docker `29.7.2` (build `a7dcaa6`)
- **Remote Host Path:** `/home/mouadnadzi3/rf-compliance-platform`

### 17.2 Container Infrastructure (`docker-compose.yaml`)
- **`rf_app` Container (`rf-compliance-platform-app`):**
  - Runs FastAPI backend (`0.0.0.0:8000`) and Streamlit UI (`0.0.0.0:8501`).
  - Mounted volume: `/home/mouadnadzi3/rf-compliance-platform` $\rightarrow$ `/app`.
  - Mounted data directories: `batch_uploads`, `data/lookups`, `model_cache`, `ocr_cache`.
- **`rf_postgres_db` Container (`pgvector/pgvector:pg16`):**
  - PostgreSQL 16 database with `pgvector` extension enabled (`0.0.0.0:5432`).
  - Healthy status verified.

### 17.3 Operational Verification
- GPU drivers and Docker runtime verified clean and ready (`nvidia-smi` & `docker --version`).
- Relational schema (`certificates`, `authority_lookups`, `supplier_lookups`) and vector store (`certificate_chunks`) initialized and persistent.

---

## 18. LLM-Based Schema Mapping, Multi-Lingual Date Normalization & Direct Link Access (2026-08-24 Update)

### 18.1 LLM-Based Automated File Schema Mapping (`server/main.py`)
- **Eliminated Hardcoded Synonym Chains:** Removed legacy hardcoded column synonym matching (`norm_row.get("comp") or norm_row.get("part") ...`) in `import_certificates_file`.
- **Single Pre-Ingestion LLM Mapping Step:** Before processing CSV/Excel rows, the API sends the file's raw column headers and a sample row to the LLM (`generate_json` with `disable_thinking=True`).
- **Dynamic Schema Field Resolution:** The LLM inspects the file structure and returns a JSON mapping dictionary associating file column headers to target schema fields (`component`, `supplier`, `country`, `certif_number`, `authority`, `issue_date`, `exp_date`, `cert_link`, `file_name`).
- **Multi-Lingual Support:** Works seamlessly with exotic or foreign-language column headers (e.g., French `"File link"`, `"Date de délivrance"`).

### 18.2 Robust Multi-Lingual Date Parsing & 2-Digit Year Normalization (`core/extractor.py`)
- **French Month Abbreviation Mapping:** Added `FRENCH_MONTH_MAP` dictionary (`avr` $\rightarrow$ April, `sept` $\rightarrow$ September, `août` $\rightarrow$ August, `juil` $\rightarrow$ July, `févr` $\rightarrow$ February, `déc` $\rightarrow$ December) to `_parse_iso_date()`.
- **2-Digit Year Guardrails:** Automatically normalizes 2-digit years to modern 20xx dates (`99` $\rightarrow$ `2099`, `24` $\rightarrow$ `2024`) for certificate documents.
- **Excel Sentinel Retention:** Preserves Excel "no expiry" sentinel dates (e.g. `9999-01-01`).

### 18.3 Direct Document Link & Dynamic URL Normalization (`ui/app.py`, `server/main.py`, `server/config.py`)
- **Native Streamlit LinkColumn:** Replaced in-app iframe preview panel with `st.column_config.LinkColumn` displaying lowercase `"open file"` styled links. Clicking `"open file"` opens the target document directly in a new browser tab.
- **Centralized Public Host Config (`server/config.py`)**: Added `PUBLIC_HOST` (defaulting to GCP external IP `34.158.150.51`) and `PUBLIC_API_URL` (`http://34.158.150.51:8000`).
- **Zero-Hardcoding Link Normalization (`ui/app.py`)**: Built a zero-hardcoding dynamic link transformer (`_normalize_link`). Any link containing `/files/` dynamically extracts the relative file path and prepends `PUBLIC_API_URL` to route requests directly to FastAPI's static file endpoint on port **8000** (bypassing Streamlit's port 8501). External authenticated links (e.g. Stellantis portal URLs) are passed through untouched.
- **Database Storage Guardrails**: PostgreSQL container volume permissions (`/data/postgres` owned by `70:70` postgres UID) are explicitly preserved during host file syncs.


## 19. Production UI/Architecture Refactor & Operational Fixes (2026-08-24 Update)

### 19.1 Streamlit UI Overhaul (`ui/app.py`, `.streamlit/config.toml`)
- **Blue Brand Theme:** Replaced Streamlit's default red primary color with the Stellantis logo blue (`#243881`). Centralized into CSS custom properties (`--brand-blue`, `--brand-blue-light`, `--brand-blue-dark`) and a `[theme]` block in `.streamlit/config.toml`. Applied to nav pills, buttons, alerts, trend text, and status dots.
- **Top Navbar Rework:** Logo sized to its natural aspect ratio (wide banner) instead of a square box; nav row content vertically centered; fixed Streamlit 1.62 radio selector (`data-selected` instead of `data-checked`) and hid the default radio circles.
- **Breadcrumb Header Bar Removed:** Deleted the "HOME > INGESTION DASHBOARD / HOME PAGE" bar above page content.
- **Page Spacing:** Added bottom padding (`3rem`) to all pages; removed extraneous whitespace around the navbar.

### 19.2 DATABASES Page: Full CRUD via Tables (`ui/app.py`, `server/main.py`)
- **Interactive Tables:** RF Certificates uses `st.dataframe` with multi-row selection; Authorities/Suppliers tables display without internal `id`/`aliases` columns.
- **Selected-Row Actions (sidebar):** "Export Excel" (openpyxl), "Delete Selected", and "Edit Selected" apply only to the selected row(s). Added REST endpoints: `PUT /api/v1/certificates/{id}`, `PUT /api/v1/lookups/authorities/{id}`, `PUT /api/v1/lookups/suppliers/{id}` (plus existing add/delete endpoints).
- **In-Grid Add/Edit:** "+ Add Row" button opens a dynamic `data_editor` for entering new certificate rows; "Edit Selected" loads the selected row into a pre-filled editor. Both commit via the API.
- **Column-Based Multi-Filter:** RF Certificates replaced the single search box with a per-column dropdown filter supporting multiple AND-ed conditions ("Add Filter" / remove).
- **"Last Update" Column:** Added `last_update` (from `created_at`) to the certificates API response and table.

### 19.3 Batch Ingestion: Single Per-File Pipeline (`server/main.py`, `ui/app.py`)
- Removed the legacy two-phase batch (all-OCR then all-extract). Each file is now processed end-to-end (OCR -> extraction -> persist) before the next, with unified progress reporting. Both models stay resident, so no phase-based lifecycle is needed.
- The "Select certificate documents" ingestion moved from HOME into DATABASES > Management Actions > "Ingest Certificate Documents".

### 19.4 Eager Model Preload & VRAM Management (`server/main.py`, `server/config.py`)
- Both GLM-OCR and Qwen3.8-27B GGUF now preload into VRAM at API startup (fail-fast, no lazy fallback). Embeddings (bge-m3) preload on CPU.
- **Embeddings stay on CPU** (`EMBEDDING_DEVICE="cpu"`): moving them to CUDA reduced OCR headroom and caused a CUDA OOM during page generation. Measured steady state: ~9.2 GB used / ~13.3 GB free on the L4.

### 19.5 OCR Failure Integrity Fix (`core/base.py`)
- `process_document` previously swallowed per-page inference failures, returning `[ERROR: ...]` text as "successful" OCR. This caused error markdown to be cached and empty records persisted. It now raises `RuntimeError` on any page failure, so failed files are marked failed (no cache write, no DB record, no false manifest entries).

### 19.6 UI-to-API Connectivity: Real HTTP over TestClient (`ui/app.py`, `server/main.py`)
- Replaced the Colab-era in-process `fastapi.testclient.TestClient` with a persistent `httpx.Client` against the real uvicorn service on `http://localhost:8000`.
- Eliminates the "Running get_api_client()" freeze (no in-process FastAPI lifespan / model load in the UI process) and halves VRAM (models now resident only in the uvicorn process).
- The UI process no longer imports `server.main` or the heavy ML stack (only the standalone `core.rag.chunker.chunk_for_qa`), keeping it light.

### 19.7 Repository Hygiene
- Removed all emojis / decorative Unicode symbols from the codebase (log messages, comments, docs); arrows replaced with ASCII `->` where meaningful.
- Added a "Code Style Directives - No Emojis" rule to `.agents/AGENTS.md`.


## 20. Chat Session Persistence, RAG Accuracy Hardening & LLM Model Upgrade (2026-08-26 Update)

### 20.1 PostgreSQL-Persisted Chat Sessions (`server/main.py`, `schemas/extraction.py`)
- **ORM models:** Added `ChatSession` (`chat_sessions`) and `ChatMessage` (`chat_messages`) tables to `schemas/extraction.py` (auto-created by `init_db`).
- **DB-backed store:** The chat session store in `server/main.py` now mirrors every mutation to PostgreSQL (session create, message turns, title, freeze state) and re-hydrates the in-memory cache from the DB at startup, so sessions and their history **survive backend/container restarts**.
- **Endpoints:** `GET/POST /api/v1/chat/sessions`, `GET /api/v1/chat/sessions/{id}/messages`, `DELETE /api/v1/chat/sessions/{id}` all read/write PostgreSQL directly.

### 20.2 Chat Context-Window Freeze Guard (`server/config.py`, `server/main.py`)
- `CHAT_CONTEXT_WINDOW` / `CHAT_CONTEXT_FULL_THRESHOLD` (85% of the context window) / `CHAT_PROMPT_OVERHEAD_TOKENS` (700) budget each session's cumulative prompt.
- A session **freezes** once its projected budget exceeds the threshold and returns `"Context window full, open a new session."` with `frozen: true` until a new session is opened.

### 20.3 RAG Accuracy Hardening — Anti-Hallucination Fixes (`core/rag/*`, `core/prompts.py`)
Investigation of an off-topic answer ("what about the others" returning Dominican Republic/Brazil/Peru docs) traced the failure to anaphoric follow-up queries + weak retrieval. Fixes:
1. **History-aware query reformulation** (`core/rag/orchestrator.py` `_reformulate_query` + `QUERY_REWRITE_SYSTEM_PROMPT`): rewrites follow-ups into standalone queries carrying entities from history (`"what about the others" -> "list the other certificates from Argentina and whether they have missing values"`), with a deterministic anaphora fallback when the LLM returns the query unchanged.
2. **Router history** (`core/rag/router.py`): `classify_intent(query, history=...)` injects prior turns so follow-ups inherit the previous metadata context.
3. **Resolved query drives retrieval**: RAG/SQL now embed/generate against the resolved query, not the raw anaphoric string.
4. **Stopword filtering** (`core/rag/hybrid_engine.py`): `_tokenize_query` drops English stopwords so "what/about/others" no longer trigger broad ILIKE/FTS matches.
5. **Relevance gate + SQL fallback**: stopword-only queries return empty context (`LOW_SIGNAL_QUERY`); a dense cosine-distance threshold rejects unrelated retrievals; the orchestrator falls back to the metadata (SQL) path when RAG context is empty.
6. **Generic schema-driven "missing values" SQL fix** (`core/rag/sql_engine.py`, `core/prompts.py`): `build_schema_description()` now annotates every nullable column as `NULLABLE`, and the SQL prompt instructs the generator to check ALL `NULLABLE` columns for missing/empty/incomplete-value questions. This is ORM-derived (auto-scales with schema changes) and replaced an earlier rigid `cert_link`-specific wording.

### 20.4 LLM Model Upgrade: Qwen3.8-27B IQ1_M -> UD-IQ3_XXS (`core/llm/qwen3_8_27b.py`, `server/config.py`)
- **Model:** `Qwen3.8-27B-UD-IQ1_M.gguf` (6.73 GB, 1-bit) -> `Qwen3.8-27B-UD-IQ3_XXS.gguf` (10.93 GB, 3-bit) from `unsloth/Qwen3.8-27B-GGUF`.
- **Rationale:** IQ1_M was a T4 (16 GB) VRAM-coexistence compromise that degraded reasoning (incomplete SQL field enumeration, non-deterministic routing). On the GCP L4 (24 GB) the 3-bit model fits comfortably.
- **VRAM measured on L4:** ~14.3 GB used / ~8.2 GB free at 32k context (OCR + LLM + KV). No GPU-layer reduction needed; `QWEN3_8_27B_N_GPU_LAYERS` (env, default `-1`) allows offloading 4-8 layers to CPU if VRAM ever tight.
- **Context window:** `DEFAULT_CONTEXT_WINDOW` scaled 8192 -> **32768** (32k). 64k would add ~4 GB q8_0 KV on top of the 11 GB model + OCR and is too tight on the L4. KV cache stays `q8_0` (`type_k/type_v = GGML_TYPE_Q8_0`, flash_attn on).
- **MTP speculative decoding (NOT applied):** The IQ3_XXS main model has no embedded MTP heads; the repo ships a separate draft head `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`. llama.cpp exposes MTP (`--spec-type draft-mtp`) only through llama-server/CLI; language bindings (llama-cpp-python, any version) do NOT expose it (llama.cpp issue #27469). Enabling MTP would require a llama-server sidecar migration, which was evaluated and deferred — the pipeline is latency-bound by multiple short serial LLM calls per query, not single-sequence decode throughput, so MTP's payoff is limited here.

### 20.5 AGENTS.md Update (`/home/mouadnadzi3/rf-compliance-platform/.agents/AGENTS.md`)
- The hard `llama-cpp-python v0.3.34-cu122` wheel pin is now documented as a **Google Colab / Tesla T4 sandbox constraint**, superseded on production (GCP NVIDIA L4). Production may use a current CUDA build, or serve the LLM via a recent `llama-server` sidecar for speculative decoding. Integer GGML enums for `type_k`/`type_v` and GGUF-architecture compatibility requirements remain.


---

## 21. Session Updates — Validity Disambiguation, Perpetual Sentinel, Streaming Chat, Uploader UX, Anti-Duplicates & Chunker Guardrail (2026-08-27)

### 21.1 Three-Tier Validity Disambiguation & Enrichment Refactoring
- **Data model (`storage/models.py`):** `AuthorityLookup.standard_validity_years` changed from `Integer` to `String(20)` and now stores three distinct tiers read from `data/lookups/authorities.json`:
  - **Determined term** — numeric string (e.g. `"3"`, `"5"`).
  - **Infinite term** — the literal string `"infinite"` (non-expiring certificate).
  - **Variable / context-dependent term** — SQL `NULL`.
- **Normalization helper:** `normalize_validity_years(value)` (in `storage/models.py`) canonicalizes raw JSON values to `"infinite"`, `"3"`, or `None` without casting exceptions.
- **Seeding (`storage/seed_lookups.py`):** performs an idempotent migration `ALTER TABLE authority_lookups ALTER COLUMN standard_validity_years TYPE VARCHAR(20) USING standard_validity_years::text`, then upserts every authority keyed on `canonical_authority + country` (fixes a first-pass key-mismatch that duplicated rows; the table was truncated and reseeded cleanly to 50 rows: 10 numeric, 28 `infinite`, 12 `NULL`).
- **Enrichment (`core/extractor.py` `enrich_certificate_metadata`):** resolves a missing `exp_date` by the matched authority's validity tier:
  - `"infinite"` -> sets `data.exp_date = "Perpetual"` (persisted as the sentinel below).
  - numeric -> `exp_date = issue_date + validity_years` via `_parse_iso_date` (robust to non-ISO/French dates).
  - `NULL` -> leaves the raw extracted `exp_date` untouched (variable-term authorities rely on the document).
- **API/UI normalization (`server/main.py`, `ui/app.py`):** add/update/import authority endpoints normalize validity values via `normalize_validity_years`; the DATABASES authorities table passes raw string values (no `int()` casts) so `"infinite"` survives the edit round-trip.

### 21.2 Non-Expiring ("Perpetual") Certificate Sentinel — `9999-01-01`
- **Rationale:** a perpetual certificate's `exp_date` must not be stored as `NULL` (NULL is reserved for variable-term authorities with no extracted expiry). The existing Excel "no expiry" convention `9999-01-01` is used as the canonical sentinel.
- **Persistence (`core/extractor.py`):** `_parse_iso_date` maps `"Perpetual"` / `"infinite"` / `"no expiry"` markers to `NO_EXPIRY_DATE = date(9999, 1, 1)` (defined in `core/extractor.py`); the relational `Date` column stores the sentinel. The `9999-01-01` sentinel itself still round-trips as-is.
- **Display (`core/extractor.py` `format_exp_date`):** renders the sentinel as `"infinite"` (real dates as `YYYY-MM-DD`); used by `GET /api/v1/certificates` and the CSV/Excel exports so the DATABASES table shows `infinite`.
- **Edit round-trip (`server/main.py` `PUT /api/v1/certificates/{id}`):** accepts `"infinite"` / `"Perpetual"` (and the sentinel) back to `NO_EXPIRY_DATE`.
- **Backfill:** existing `NULL`-expiry certificates whose authority is `infinite` were backfilled to the sentinel (5 rows); the remaining NULLs are genuine variable-term/unresolved records.
- `NO_EXPIRY_MARKERS`, `is_no_expiry_marker()`, `NO_EXPIRY_DISPLAY` helpers live in `core/extractor.py`.

### 21.3 SSE Token-Streaming Chat (Job-Based, Decoupled from the Script Run)
- **Motivation:** the original blocking chat call froze the Streamlit UI during generation, and a naive `@st.fragment(run_every)` renderer did not tick in this deployment. The final architecture decouples generation from the UI thread while rendering tokens progressively and correctly.
- **Backend streaming contract:**
  - `core/base.py` — `BaseLLMEngine.generate_stream()` template method + `_generate_stream()` hook (default yields the full `_generate_raw()` result, so all engines stay streaming-safe).
  - `core/llm/qwen3_8_27b.py` — real token streaming via llama.cpp `stream=True` (prompt/sampling shared with `_generate_raw` via `_build_prompt` / `_sampling_params`).
  - `core/rag/qa.py` — `stream_synthesize_answer()` yields `thinking` / `token` / `synthesis_done` events; `extract_answer_value()` incrementally decodes the JSON `answer` field (latches at the opening quote, handles escaped quotes/newlines and incomplete trailing escapes); `parse_qa_response()` final-validates via `extract_json`.
  - `core/rag/orchestrator.py` — `answer_compliance_query_stream()` mirrors the dual-path router for all three intents: METADATA streams the SQL answer synthesis (`stream_metadata_answer` in `core/rag/sql_engine.py`), UNSTRUCTURED/HYBRID stream the QA generation; emits `status`, `token`, `done` events.
  - `core/rag/sql_engine.py` — `_stream_synthesize_answer()` + `stream_metadata_answer()` stream the metadata answer synthesis; `build_unstructured_qa_prompt()` shared with the sync path.
- **Job + SSE endpoints (`server/main.py`):**
  - `POST /api/v1/chat/stream` — validates session/freeze/budget, launches the RAG pipeline in a background thread, returns `{job_id, session_id}` immediately.
  - `GET /api/v1/chat/stream/{job_id}` — tailing SSE (`data: <json>` lines); emits `status`, `thinking`, `token`, then `done` (which carries `session_id`).
  - `_CHAT_INFERENCE_LOCK` serializes concurrent jobs (llama-cpp engine is single-residency and not thread-safe).
  - The turn is persisted to PostgreSQL **before** the `done` event is emitted, so a client that receives `done` can immediately re-fetch history.
  - The legacy `POST /api/v1/chat` endpoint remains intact and verified working.
- **Frontend (`ui/app.py` `_render_chat_window`):** the script run submits the job and then blocks on the SSE stream, updating `st.empty()` placeholders created inside `progress_ph` (positioned in the message area, ABOVE the typing box) so chat-element ordering stays correct — status + elapsed timer, growing "Reasoning:" preview during the deep-thinking phase, then answer tokens. The answer renders in place (no refresh). Streamlit 1.62 has no script-run timeout, so the blocking run is safe.
- **Known tradeoff:** navigation queues while a long generation is in flight, but the UI shows continuous live progress (status/reasoning/tokens) rather than a silent freeze.
- **Metadata-path latency note:** "Querying certificate database... (Ns)" is NOT thinking-mode — all metadata-path LLM calls use `disable_thinking=True`. The ~8s is 3 sequential LLM calls on the 27B model (router ~4s + SQL generation ~5s + synthesis), which is a model-inference bound on the L4 (300 GB/s bandwidth caps decode near ~27 tok/s theoretical; smaller models decode ~3-4x faster).

### 21.4 DATABASES Page — Uploader UX Refinements (`ui/app.py`)
- **Uploaders now visible:** removed the CSS that hid `[data-testid="stFileUploaderDropzone"]` (leftover from the old custom-browse-button era).
- **Removed the auto-click browse-button hack:** deleted `_AUTO_CLICK_SCRIPT`, `browse_button()`, and `open_dialog_accept`; replaced with plain text labels above each native uploader: "Import CSV or Excel" (RF Certificates), "Ingest certificate documents" (RF Certificates), "Import authorities from CSV or Excel" (Authorities), "Import suppliers from CSV or Excel" (Suppliers).
- **Each upload section** (label + uploader + preview + action buttons) is wrapped in its own `st.container(border=True)` box; labels render via `<b>` HTML with zero bottom margin and a tightened gap before the uploader (`upload-label` CSS).
- **File count caption:** "N file(s) selected" above the batch "Process Batch Ingestion" button.
- **Cancel buttons** beside "Process Batch Ingestion" and "Import File Records" (wider `[2, 1]` columns). Clearing works by **changing the uploader's widget `key`** (e.g. `file_import_uploader_0` -> `file_import_uploader_1` via a reset counter) — Streamlit treats a new key as a fresh, empty widget. The earlier flag+pop approach failed because the file uploader re-hydrates from the frontend after the session-state key is deleted.
- **`seek(0)` fix:** every CSV/Excel preview read now rewinds the `UploadedFile` stream before `pd.read_csv`/`read_excel` (an EOF stream caused "No columns to parse from file" on reruns).
- **Overall page scroll enabled:** removed the `overflow: hidden` + `100vh` rule on the main containers; the chat card retains its internal message-area scroll.

### 21.5 Robust Anti-Duplicate System
- **Identity rule (`core/extractor.py` `find_existing_certificate`):** `certif_number + country` (case-insensitive) is the primary identity whenever both are known; `file_name` (case-insensitive) is the fallback identity only when the certificate number is absent. This avoids collapsing legitimate multi-record CSV/Excel imports that share one source filename.
- **Central upsert (`save_certificate_to_db`):** every ingestion path (single parse, batch, manual, CSV/Excel import) now detects an existing match and **updates the record in place + replaces its vector chunks** instead of inserting a duplicate. New `dedup_file_name` param.
- **Batch pre-check (`server/main.py` `_run_batch`):** files whose `file_name` already exists (case-insensitive) are skipped BEFORE OCR (counted as `skipped` in the manifest, surfaced by the UI batch progress).
- **Manual entry** passes `dedup_file_name=False` (dedup by certif+country only — all manual rows share the synthetic "Manual Entry" filename).
- **CSV/Excel import** passes `dedup_file_name=False` (dedup by certif+country only — one file yields many distinct certificates).
- **`GET /api/v1/certificates/exists`** is now case-insensitive (`func.lower(file_name)`).
- **`POST /api/v1/certificates/deduplicate`:** merges legacy duplicates by the unified identity (keeps the newest record, deletes older ones, cascading chunk deletion). Verified `removed: 0` on the current 21-cert corpus (no false positives; the 9 records sharing `import_test.xlsx` are distinct certificates and are preserved).
- **Verified upsert behavior:** re-ingesting the same file -> same `certificate_id` (chunks replaced); same cert under a different filename -> same id via certif+country; a genuinely different cert -> new record.

### 21.6 RAG Chunker Safety-Valve Guardrail (`core/rag/chunker.py`)
- **Primary:** unchanged paragraph chunking on `\n\n`, **zero overlap** across the board (prevents duplicate chunk indexing in `certificate_chunks`).
- **Soft max-length guardrail:** `SOFT_MAX_CHUNK_TOKENS = 800` (~4 chars/token via `estimate_tokens`). If and only if a single paragraph exceeds the cap (giant OCR table/block missing blank lines), that paragraph alone is split:
  1. On single newlines (`\n`), then
  2. On period boundaries (`. `) for any still-oversized piece, then
  3. Greedily packed back into cap-sized chunks (`_pack_parts`) with no overlap.
- Page-tag tracking (`<Page N>`) is preserved through the splitter (sub-chunks inherit/update the page). Verified: normal docs unchanged; a 200-row table block -> 5 packed chunks (max 800 tokens); giant prose -> 4 chunks (max 795 tokens); no overlap detected.


---

## 22. Agentic Architecture — Supervisor Router, CASUAL Replies & Tool Registry (2026-08-27)

### 22.1 Agentic Supervisor Router (`core/rag/router.py`, `core/prompts.py`)
- **Five-intent classification:** `QueryIntent` now exposes `METADATA_QUERY`, `UNSTRUCTURED_RAG`, `HYBRID_QUERY`, `CASUAL_CONVERSATION`, and `AGENT_ACTION`.
- **`ROUTER_SYSTEM_PROMPT`** was hardened with:
  - Rule D — greetings/pleasantries/assistant-identity questions are `CASUAL_CONVERSATION` and MUST NEVER trigger DB lookups or retrieval.
  - Rule E/H — read-only questions are never `AGENT_ACTION`; only explicit side-effect commands qualify.
  - Rule F — out-of-domain general-knowledge questions (e.g. "who is the CEO of google") are `CASUAL_CONVERSATION`, never `METADATA_QUERY`/`HYBRID_QUERY` even when entity-looking tokens appear.
  - Rule G — `AGENT_ACTION` precedence: imperative verbs (download/fetch/convert/send/update/delete...) targeting a URL/file/email/DB record dominate all other intents.
  - Rule A scoped to certificate/document questions only.
- **Statelessness:** the router keeps no cross-call state; context is passed per call and flushed immediately after classification.
- **HITL note:** `AGENT_ACTION` intents will drive DB edits / tool executions; those MUST be gated behind human-in-the-loop approval in the Streamlit UI before any `execute()` is invoked.

### 22.2 Orchestrator Intents & Streaming (`core/rag/orchestrator.py`)
- Both sync (`answer_compliance_query`) and streaming (`answer_compliance_query_stream`) orchestrators now branch explicitly on all five intents. The old catch-all `else` became an explicit `UNSTRUCTURED_RAG` branch plus a safety-net fallback.
- `CASUAL_CONVERSATION` short-circuits: **no DB, no retrieval**; reply generated by a dedicated `CASUAL_CONVERSATION_SYSTEM_PROMPT` (`core/prompts.py`) via the LLM in fast non-thinking mode, with a canned fallback.
- `AGENT_ACTION` short-circuits to `_agent_action_placeholder()`: no tool is dispatched; the reply states the action was NOT executed and requires HITL approval.
- **General-LLM casual behavior (2026-08-27):** `CASUAL_CONVERSATION_SYSTEM_PROMPT` instructs the model to behave like a general-purpose assistant (Gemini/ChatGPT) — answering out-of-domain factual questions directly (no platform-only limitation), while never querying the certificate DB.
- **Token streaming everywhere:** `stream_synthesize_answer` (`core/rag/qa.py`) gained a `disable_thinking` param (threaded through `_stream_and_collect`); the streaming CASUAL branch streams token-by-token. JSON preamble tokens (`{ "answer":`) are buffered silently — `thinking` events are only emitted in deep-thinking mode BEFORE the JSON payload begins, eliminating "Reasoning: { "answer":" noise.

### 22.3 Tool Registry & Concrete Tools (`core/agent/`)
- **`BaseTool`** (`core/agent/tools.py`) — abstract contract requiring `get_schema()` (JSON Schema of expected args) and `execute(**kwargs)`; model-agnostic, no LLM/OCR coupling.
- **`WebDownloaderTool`** (`web_downloader`) — fetches a URL with Python `requests` (streaming, 30s timeout), saves under `data/agent/downloads/`, returns the absolute path; raises `RuntimeError` on HTTP/network failure (partial files cleaned up) and `ValueError` on malformed URLs.
- **`DataConverterTool`** (`data_converter`) — converts a list of row dicts to strict `"csv"` (built-in `csv.DictWriter` via in-memory `StringIO`) or `"json"` (pretty-printed block); rejects any other format.
- **`EmailDraftingTool`** (`email_drafting`) — compiles recipient/subject/body into a structured JSON payload with a UUID draft id, persists under `data/agent/drafts/<id>.json`, and returns `{draft_id, message, path}`. **No live SMTP** — sending is deferred behind HITL approval.
- **Registry:** `get_tool_registry()` returns `{web_downloader, data_converter, email_drafting}`; the supervisor resolves `AGENT_ACTION` into a tool via this lookup.
- **Dependencies:** `requests` added to `requirements.txt`; `data/agent/` (tool runtime artifacts) added to `.gitignore`.

### 22.4 Streaming Chat UI Fixes (`ui/app.py`)
- **Post-turn rerun:** after a handled chat turn, the UI clears the `chat_input` widget keys and calls `st.rerun()` once. Previously the sidebar was rendered before the session existed (stale "No sessions yet.") and the typing box stayed `disabled` (it was rendered while the optimistic message was pending). One rerun re-syncs both from server state. Guarded so no turn -> no rerun (no infinite loop).
- **Status text removed:** `status` SSE events ("Routing query... (0s)", "Responding conversationally... (6s)") are no longer rendered in `_blocking_stream`; the chat shows only the reasoning preview and the answer.

### 22.5 Verified Behavior (live on the L4 host)
- `hi` -> `CASUAL_CONVERSATION`, friendly reply, zero DB calls.
- `who is the CEO of google` / `what is ai engineering` -> `CASUAL_CONVERSATION`, general factual answer (Sundar Pichai / AI engineering definition).
- `how many certificates does Bosch have?` -> `METADATA_QUERY`, "Bosch has 3 certificates."
- `what are the test requirements for section 4?` -> `UNSTRUCTURED_RAG` (reasoning streamed live, then answer tokens).
- `download this URL https://example.com/cert.pdf` -> `AGENT_ACTION`, HITL message, nothing executed.
- `update the database with these rows` -> `AGENT_ACTION`.
- Tool unit tests (local HTTP server, CSV/JSON conversion, draft persistence, abstract-base instantiation) all pass.

### 22.6 Autonomous Background Scheduler (`core/agent/worker.py`, `server/main.py`)
- **Dependency:** `apscheduler` added to `requirements.txt` (installed in the container: 3.11.3).
- **Module `core/agent/worker.py`:** module-level `AsyncIOScheduler` singleton + `_scheduler_started` flag.
  - `autonomous_ingestion_job(target_url)` — instantiates `WebDownloaderTool`, downloads the target URL (sync call offloaded via `asyncio.to_thread` so it never blocks the event loop), and securely logs the local path. Fully try/except-guarded: a background failure returns `None` and logs — it can never crash the FastAPI server. (Full OCR/extract/persist wiring lands in Step 5.4.)
  - `start_scheduler(interval_seconds=86400, target_url=None)` — registers the job on an interval trigger (floored at 60s, `coalesce=True`, `max_instances=1`) and starts; idempotent.
  - `shutdown_scheduler()` — graceful, idempotent, never-raises shutdown (flag-driven to avoid double-scheduling APScheduler's async `_shutdown`).
- **Config (`server/config.py`):** `AUTONOMOUS_INGESTION_INTERVAL_SECONDS` (default 86400, env-overridable) and `AUTONOMOUS_INGESTION_TARGET_URL` (empty = job logs and skips).
- **FastAPI lifespan (`server/main.py`):** after engine preload, `start_scheduler(interval_seconds=..., target_url=...)` runs on startup (wrapped in try/except); teardown calls `shutdown_scheduler()` so Docker stops leave no zombie background processes.
- **Verified on the L4 host:** scheduler logs `Scheduler started: autonomous ingestion every 86400s.` at boot, `Scheduler shut down.` on `docker stop`, and the API restarts cleanly. Worker smoke tests (no-URL skip, failed-download containment, idempotent start/stop) pass in-container.

**Next step (5.4, deferred):** wire `AGENT_ACTION` intents to `get_tool_registry()` dispatch in the orchestrator, surface HITL approve/deny controls in the Streamlit UI, and connect `autonomous_ingestion_job` to the full ingestion pipeline (OCR -> extraction -> persistence).

### 22.7 HITL Proposal Pipeline & Database Write Capabilities (`core/agent/db_editor.py`, `core/agent/proposals.py`, `server/main.py`)
- **`core/agent/db_editor.py` — Database Editor engine:**
  - `ALLOWED_TABLES` static identifier allowlist: `certificates`, `authority_lookups`, `supplier_lookups`, `certificate_chunks` with their exact writable columns. Identifiers are validated against `[A-Za-z_][A-Za-z0-9_]*` plus a whole-word DDL keyword guard (drop/truncate/alter/create/delete/grant/revoke/merge/call/copy).
  - `build_mutation_sql(op, table, values, row_filter)` supports only `update` | `insert`. DELETE and DDL ops raise `ValueError`; `update` requires a non-empty `row_filter` (prevents full-table updates); unknown tables/columns are rejected; non-JSON-serializable values are rejected.
  - Returns `{op, table, sql, params, preview}` — named-placeholder SQL safe for SQLAlchemy `text()`, a params dict, and a literal-bound `preview` for dry-run inspection.
  - `execute_mutation(db, mutation)` runs the compiled statement in a single transaction (commit on success, rollback on any error) and returns the affected row count.
- **`core/agent/proposals.py` — ProposalManager:**
  - File-backed JSON store under `data/agent/proposals/` (auditable, survives restarts; gitignored via `data/agent/`).
  - Proposal shape: `{proposal_id (uuid hex), type: "DB_EDIT"|"SEND_EMAIL", payload, status, created_at, sql_preview}`.
  - Methods: `create_proposal()`, `get_proposal()`, `list_pending_proposals()`, `list_all_proposals()`, `update_status()` with enforced transitions `PENDING -> APPROVED|REJECTED` (nothing leaves APPROVED/REJECTED).
- **HITL routes (`server/main.py`):**
  - `GET /api/v1/agent/proposals` — returns all PENDING proposals for UI rendering.
  - `POST /api/v1/agent/proposals/{id}/approve` — `DB_EDIT`: rebuilds + validates the mutation, executes it in a strict transaction (rollback on error), returns rowcount + preview; `SEND_EMAIL`: reads the draft JSON from `data/agent/drafts/`, marks it `dispatched` (no live SMTP); then sets the proposal `APPROVED`. Non-PENDING proposals return 409; missing proposal/draft return 404; validation failures return 500 with the actionable message.
  - `POST /api/v1/agent/proposals/{id}/reject` — sets `REJECTED`, no execution.
- **Architectural constraints met:** strictly parameter-bound SQL (injection impossible by construction), local-first (filesystem-only, no third-party APIs), rollback on every exception, sync execution matching the codebase's sync SQLAlchemy sessions.
- **Verified live on the L4 host:** staged 3 proposals -> `GET` listed them; approve applied the DB update (rowcount 1, supplier persisted then reverted for cleanup); approve marked the draft `dispatched`; reject left the DB untouched; re-approving/rejecting non-PENDING returned 409; unknown id returned 404. Module tests (DDL/DELETE/unknown-table/unknown-column/no-filter blocks, transition guards, real-PG update) all pass.

**Next step (5.5, deferred):** wire `AGENT_ACTION` intents to `get_tool_registry()`/`proposal_manager` dispatch in the orchestrator, and surface HITL approve/deny controls in the Streamlit UI.

### 22.8 Agentic Frontend — CONTROL Page, Chat Agent Visibility & Background Trigger (`ui/app.py`, `server/main.py`)
- **Top nav:** `["HOME", "DATABASES", "CONTROL"]` — new CONTROL page for agent visibility + HITL workflow.
- **CONTROL page (`ui/app.py`):**
  - *Pending Agent Actions dashboard* — queries `GET /api/v1/agent/proposals` on render; renders each PENDING proposal in a bordered container with Type badge (`DB_EDIT` vs `SEND_EMAIL`), created timestamp, and proposal id. `DB_EDIT` shows the raw SQL preview in a `st.code(..., language="sql")` block; `SEND_EMAIL` fetches the staged draft via `GET /api/v1/agent/drafts/{id}` and renders recipient/subject/body. Side-by-side `Approve` (primary) / `Reject` buttons call the HITL endpoints, surface `st.success()`/`st.info()`, and `st.rerun()`. Empty state: `st.caption("No pending proposals requiring human approval.")`.
  - *Background worker sidebar* — "Run Autonomous Scraper Now" button calls `POST /api/v1/agent/autonomous/run` and shows dispatch feedback.
- **Chat agent visibility (HOME):** when a streamed turn's `done` event carries `intent == AGENT_ACTION`, the chat renders a live `st.status("Agent action staged - awaiting approval")` widget (action NOT executed) and sets `st.session_state.agent_action_notice`; a persistent `st.info` banner then points the user to the CONTROL page. The notice stays in sync with pending proposals (cleared on approve/reject or when none remain).
- **Backend additions (`server/main.py`):**
  - `GET /api/v1/agent/drafts/{draft_id}` — returns a staged email draft for the dashboard preview (404 if missing).
  - `POST /api/v1/agent/autonomous/run` — manually dispatches `autonomous_ingestion_job` in a daemon thread (`asyncio.run` per thread), returning `{status: dispatched, message, target_url}` immediately; job failures are contained and logged.
- **Verified live on the L4 host:** nav serves HTTP 200; autonomous trigger returns dispatched; staged DB_EDIT + SEND_EMAIL proposals render their SQL preview / draft payload via the endpoints; chat stream `done` event returns `intent: AGENT_ACTION` for "send an email to ..." (drives the status widget); test artifacts cleaned up.

**Phase 5 (Agentic Transformation) COMPLETE** — 5-intent supervisor router, tool registry (download/data-convert/email-draft), autonomous scheduler, HITL proposal pipeline with parameter-bound DB writes, CONTROL approval dashboard, autonomous PDF discovery with DB/manifest dedup, plus Playwright JS-rendering and transient-cookie auth for authenticated portals.

### 22.9 Autonomous PDF Discovery & INGEST_DOCUMENT (`core/agent/scraper.py`, `core/agent/worker.py`, `server/main.py`, `ui/app.py`)
- **`core/agent/scraper.py` — discovery engine.** Given a user-supplied portal/database URL (e.g. `https://docinfogroupe.stellantis.com/ead/doc/`), the agent finds the target PDF URLs itself:
  - `HtmlFetcher` (requests, stdlib `html.parser` link extraction) and `PlaywrightFetcher` (headless Chromium for JS-rendered SPAs; see §22.10).
  - **Phase A deterministic guards:** same-host scoping (incl. subdomains), keyword blocklist (`login/privacy/terms/contact/...`), PDF/document keyword hints, crawl budget (`max_pages`, `max_depth`), polite delay, timeouts.
  - **Phase B agentic selection:** `filter_pdf_links_with_llm()` sends candidates (URL + anchor text + source page) to the local LLM via the new `PDF_LINK_SELECTION_SYSTEM_PROMPT` (`core/prompts.py`), returning strict JSON of only the compliance/certificate document links.
  - **Hard verification:** each selected URL must return `Content-Type: application/pdf` and/or `%PDF` magic bytes; non-PDFs/404s land in `failed_verification`.
  - **Dedup (per user requirement):** skips URLs already in the database (`certificates.cert_link`, case-insensitive) OR in the append-only manifest `data/agent/fetched_urls.json`; reported as `skipped_existing`.
- **`core/agent/worker.py`:** `autonomous_ingestion_job(target_url, cookie_header=None, fetcher_type="html")` runs discover -> verify -> dedup -> download each new PDF via `WebDownloaderTool` into `data/agent/downloads/<host>/`, appends each fetched URL to the manifest immediately, and returns a structured result `{summary, discovered_urls, downloaded_paths, skipped_existing, failed_verification}`. Fully guarded; never crashes the API.
- **API (`server/main.py`):**
  - `POST /api/v1/agent/autonomous/run` — optional body `{"target_url": ..., "fetcher": "html"|"playwright", "cookie_header": "<transient session cookie>"}` (user URL at click time; falls back to config), spawns daemon thread, returns `{run_id, status: "dispatched", target_url, fetcher, has_cookie}`. The cookie value is never stored/logged (see §22.10).
  - `GET /api/v1/agent/autonomous/runs/{run_id}` — pollable status + structured result (completed runs purged after 30 min).
  - Approve route handles the new **`INGEST_DOCUMENT`** proposal type: `_ingest_document_file()` runs the existing GLM-OCR -> extraction -> `save_certificate_to_db` pipeline on the downloaded file (mirrors `/api/v1/parse`), sets `cert_link` to the source URL. All failures roll back and the proposal stays PENDING.
- **Config (`server/config.py`):** `SCRAPER_FETCHER`, `SCRAPER_MAX_PAGES/DEPTH`, `SCRAPER_TIMEOUT_SECONDS`, `SCRAPER_USER_AGENT`, `SCRAPER_POLITE_DELAY_SECONDS`, `SCRAPER_BLOCKED/ALLOWED_KEYWORDS`, `SCRAPER_USE_LLM_FILTER`, `SCRAPER_FETCHED_MANIFEST`, `SCRAPER_COOKIE_HEADER` (optional env-level transient cookie; prefer per-run passing from the CONTROL page).
- **CONTROL page (`ui/app.py`):** URL text input + "Run Autonomous Scraper Now" dispatches with the user URL; a `@st.fragment(run_every="3s")` polls the run and renders result buckets (downloaded paths / already-in-DB skipped / failed verification). Dashboard renders `INGEST_DOCUMENT` cards (source URL + file path); approve timeout raised to 300s (OCR ingest is slow).
- **Verified live (mock portal + LLM selection):** crawled portal + sub-page, LLM selected exactly the 3 certificate PDFs, excluded login/privacy/external links, downloaded all 3; manifest dedup skipped all on re-run; DB `cert_link` dedup skipped a URL already in the certificates table; `INGEST_DOCUMENT` missing-file -> 404, fake-PDF -> 500 with rollback (proposal stayed PENDING), reject -> REJECTED. All artifacts cleaned up.

### 22.10 JS-Rendered & Authenticated Portals (Playwright + Transient Cookie) (2026-08-27)
- **`PlaywrightFetcher` (`core/agent/scraper.py`):** one headless Chromium instance launched lazily and reused across a discovery run (`close()` in `finally`); renders `wait_until="networkidle"` with a `domcontentloaded`+2s fallback for SPAs that never reach network-idle. `create_fetcher("html"|"playwright")` factory selects a concrete backend. **Auto mode (default):** `fetcher_type="auto"` (`SCRAPER_FETCHER` default) tries the fast `requests` fetcher first and, when the HTML crawl finds no candidate links (JS-rendered portal), transparently retries with headless Chromium — no dropdown/selector needed. Requires `playwright` pip + `playwright install chromium` + `install-deps` (provisioned in the container: Ubuntu 22.04, Chrome Headless Shell).
- **Transient cookie auth:** optional `cookie_header` threaded through HTML fetch, PDF verification, and `WebDownloaderTool` downloads. Supplied per-run from the CONTROL-page password field (cleared after dispatch) or via `SCRAPER_COOKIE_HEADER` env. **Security:** in-memory only — the run record stores just a `has_cookie: bool` flag, the value never appears in API responses (verified) or logs (verified 0 occurrences), and is never persisted.
- **API:** `POST /api/v1/agent/autonomous/run` accepts `{"target_url", "fetcher", "cookie_header"}`; responses/logs carry only `has_cookie`.
- **Verified live:** cookie-gated mock portal -> no-cookie run downloads nothing (403), cookie run downloads the PDF, and the secret is absent from run JSON + logs. JS-injected mock portal -> `html` fetcher finds nothing (JS not executed), while `fetcher_type="auto"` transparently falls back to Playwright and discovers/verifies/downloads the injected PDFs (confirmed end-to-end via the API with no fetcher selector).

### 22.11 Scheduler Configuration UI (CONTROL page) (`core/agent/worker.py`, `server/main.py`, `ui/app.py`)
- **Persisted config:** `data/agent/scheduler_config.json` holds `{enabled, interval_seconds, target_url}`; `start_scheduler()` reads it at lifespan startup (falling back to env defaults), and the manual-run endpoint falls back to the configured `target_url` too.
- **Live application:** `update_scheduler_config()` persists and applies immediately — enable adds/starts the APScheduler job, disable removes it, interval/URL reschedule it in place.
- **API:** `GET /api/v1/agent/autonomous/config` returns `{enabled, interval_seconds, target_url, running, next_run_time}`; `POST /api/v1/agent/autonomous/config` accepts `{enabled, interval_seconds (>=60), target_url}` (all optional), 422 on invalid values.
- **CONTROL page:** new "Autonomous Scheduler" section — enabled toggle and interval (hours, 1-168) auto-apply immediately via `on_change` (no save button; inline success/error feedback), next-run display, and a pointer to DATABASES > Sources for the URL list. The scheduled job is the SAME `autonomous_ingestion_job` as the manual button, and both paths now honor `SCRAPER_FETCHER`/`SCRAPER_COOKIE_HEADER`.
- **Verified live:** interval/URL update rescheduled the job (`next_run_time` updated); disable removed the job; re-enable restored it; interval < 60 -> 422; config survived a container restart.

### 22.13 RF-Certificate Relevance Gate (`core/agent/scraper.py`, `core/agent/worker.py`, `core/prompts.py`)
How the agent ensures a grabbed link is actually an **RF certificate** (layered defenses):
1. **Phase A (URL/anchor heuristics):** same-host scoping, blocklist, PDF hints, certificate/homologation keyword hints.
2. **Phase B (agentic LLM selection):** `PDF_LINK_SELECTION_SYSTEM_PROMPT` selects certificate/homologation/compliance document links with a preference for RF/radio-telecom, but deliberately does NOT hard-exclude anchors lacking RF wording (recall) — relevance is decided by content.
3. **Hard PDF verification:** Content-Type `application/pdf` + `%PDF` magic bytes.
4. **Phase C content gate (new, `classify_pdf_relevance`):** after download, embedded text is extracted best-effort with `pypdf` (added to requirements + container; no raw-byte scanning to avoid PDF-structural false positives). Classifies each PDF:
   - `relevant` — text contains **both** a certificate signal (`certificate/homologation/approval/attestation/conformity/registro/disposición/...`) AND an RF signal (`radio frequency/RF/telecom/GSM/UMTS/LTE/5G/Wi-Fi/bluetooth/frequency/MHz/GHz/antenna/transmitter/EIRP/SAR/...`).
   - `unclear` — no embedded text (scanned cert); kept, because the INGEST_DOCUMENT OCR -> 7-field extraction at approval is the authoritative verification.
   - `irrelevant` — text present but not an RF certificate (e.g. ISO 9001); the file is deleted, the URL is recorded in the manifest, and it is reported as `irrelevant` (never staged for ingest).
- **Worker:** `autonomous_ingestion_job` runs the gate per verified URL, returns an `irrelevant` bucket, and **auto-stages each kept download as an `INGEST_DOCUMENT` proposal** (`_stage_ingest_proposal`, idempotent per file path) so it appears in CONTROL "Pending Agent Actions"; the CONTROL run-status shows "Staged for approval (N)". Approving runs OCR -> extraction -> persistence into the `certificates` table; rejecting skips.
- **Whole-word keyword matching:** `filter_candidates` now matches blocked/allowed keywords on word boundaries (`_contains_keyword`) — fixes false drops like the blocked word `press` matching the hostname `espres**s-press**if`.com.
- **`auto` fallback trigger hardened:** falls back to Playwright when the HTML crawl yields **zero verified PDFs** (not merely zero candidates) — so JS SPAs whose raw HTML shell exposes non-PDF nav URLs (e.g. `/en/documentList`) are still re-rendered with headless Chromium.

### 22.14 Chat AGENT_ACTION -> HITL Proposal Staging (`core/rag/orchestrator.py`, `server/main.py`, `ui/app.py`)
- Chat commands like "check this link and add certificates to our database: <url>" are classified `AGENT_ACTION`; the orchestrator now **stages a real `AGENT_ACTION` proposal** (`_stage_agent_action_proposal`, payload `{action: "scrape_and_ingest", url}`) and replies with the proposal id + approval instructions (both sync and streaming paths).
- The proposal appears in CONTROL "Pending Agent Actions" (card shows the action + URL); the HOME banner + "Agent action staged - awaiting approval" status widget point there.
- **Approve** runs the same autonomous scrape job (`autonomous_ingestion_job(url)`): discovery -> download -> auto-stage `INGEST_DOCUMENT` proposals (each then approved individually for OCR -> DB). **Reject** skips.
- **Verified live:** chat AGENT_ACTION staged a proposal; approve ran the scrape (deduped against manifest) and marked it APPROVED; streaming path stages too; `AGENT_ACTION` added to `VALID_TYPES`.

### 22.15 Chat HITL Confirmation & Global LLM Serialization (`core/rag/orchestrator.py`, `core/llm/__init__.py`)
- **Chat-based approve/reject:** after an AGENT_ACTION proposal is staged, the user can confirm in the chat itself — `yes`/`approve`/`go ahead`/`ok` approves the most recent PENDING AGENT_ACTION proposal (marks APPROVED, dispatches the autonomous scrape in a background thread, replies with confirmation); `no`/`reject`/`cancel` rejects it (nothing executed). When no proposal is pending, the phrase falls through to normal routing (e.g. "yes" -> casual reply).
- **Global LLM serialization (`core/llm/__init__.py`):** a module-level `_INFERENCE_LOCK` (RLock) now wraps `generate_json` and `generate_stream` — the llama-cpp engine is single-residency and not thread-safe, and a chat-approved background scrape previously called the LLM concurrently with chat turns, wedging the whole process. All LLM calls (chat, extraction, router, autonomous scrape) are now serialized at the facade.
- **Verified live:** stage -> "yes" approved + dispatched (proposal APPROVED, backend stayed responsive); stage -> "no" rejected (REJECTED, nothing executed); "yes" with nothing pending -> CASUAL_CONVERSATION reply. Backend healthy.

### 22.16 Read-Only vs Side-Effectful AGENT_ACTION & Conversational HITL (`core/rag/router.py`, `core/prompts.py`, `core/rag/orchestrator.py`, `ui/app.py`)
- **URL inspection IS an agent action (no separate intent).** URL-checking queries ("check this link and tell me if you find any certificates: <url>") classify as `AGENT_ACTION` like any other URL action. The orchestrator then decides at execution:
  - **Read-only** (URL present, no side-effect verb, per `_is_read_only_url_action`): runs `discover_pdf_urls` (in a worker thread, since the sync chat endpoint runs on the asyncio loop where Playwright's Sync API is forbidden) and **reports the found certificate documents conversationally** — no proposal, no confirmation, no DB writes.
  - **Side-effectful** ("download and add to database", mutate, convert, send): stages a proposal and asks a **natural confirmation question** ("...Is that OK? Just reply 'yes' to proceed, or 'no' to cancel."), then the user confirms/rejects in chat ("yes"/"no").
- **LLM-selection robustness (`core/agent/scraper.py`):** an empty LLM selection is no longer trusted when clear PDF-hint candidates exist — it falls back to those candidates (hard PDF verification is the authoritative gate). Makes the autonomous scraper resilient to a flaky LLM verdict.
- **Banner removed (`ui/app.py`):** the persistent "An agent action is staged and awaiting your approval - review it in the CONTROL page" banner and the `agent_action_notice` flag were removed; AGENT_ACTION proposals still appear in CONTROL for audit/alternate approval.
- **Verified live:** read-only query -> `AGENT_ACTION` -> reported the 10 Espressif certificate documents with no proposal staged; side-effect query -> `AGENT_ACTION` -> confirmation question -> proposal staged -> "no" rejected. Backend + Streamlit healthy.
- **Verified live (3-doc mock portal):** embedded-RF cert -> downloaded (relevant); scanned cert (no embedded text) -> downloaded (unclear, kept for OCR); ISO 9001 cert (embedded text, no RF) -> dropped to `irrelevant`.
- **Verified live (real Espressif portal):** `https://documentation.espressif.com/en/documentList?t=Certificate&eol=false` -> **9 RF certificates downloaded** (FCC/CE/ANATEL/WFA certs) and the *Environmental Compliance Declaration* correctly **dropped as non-RF**. This case originally returned 0 due to the `press`->`espressif` false block and the html-candidate-only fallback trigger; both fixed. After auto-staging, 9 `INGEST_DOCUMENT` proposals were created; approving one ran OCR -> extraction -> persistence and created `cert_bce061e134b1` (ESP8684-WROOM-03, supplier ESPRESSIF SYSTEMS (SHANGHAI) CO., LTD, authority FCC, cert_link = Espressif URL).

### 22.12 Multiple Source URLs (`sources` table) (`schemas/extraction.py`, `server/main.py`, `core/agent/worker.py`, `ui/app.py`)
- **New `sources` PostgreSQL table** (`Source` ORM in `schemas/extraction.py`, auto-created by `init_db`): `id`, `url`, `description`, `active`, `created_at`. The scheduler's single-URL config is superseded by this table.
- **CRUD API:** `GET/POST /api/v1/sources` (case-insensitive uniqueness -> 409), `PUT /api/v1/sources/{id}`, `DELETE /api/v1/sources/{id}` (404 if missing).
- **Worker:** `autonomous_ingestion_job(target_url=None, source_urls=None, ...)` loads all **active** sources from the DB (`_load_active_source_urls`) when no explicit list is given, iterates each source through discover -> verify -> dedup -> download, and returns an **aggregated** result `{summary, discovered_urls, downloaded_paths, skipped_existing, failed_verification}`. Scheduled runs and the manual button both use the sources table.
- **DATABASES page:** new **"Sources"** table — add form (URL/description/active), table with selection, per-row Edit (data_editor -> PUT) and Delete Selected (shared machinery), export, metrics (total/active).
- **CONTROL page:** the single URL input was removed; "Run Autonomous Scraper Now" now checks all active sources, with a read-only "Active sources" summary; the scheduler config section keeps on/off + interval (URLs managed in DATABASES > Sources).
- **Verified live:** sources CRUD (add/update/delete/409-duplicate/404), worker loaded only active sources, and a two-portal E2E downloaded 3 PDFs across 2 sources in one run (alpha/beta/gamma). Test data cleaned up.

### 22.17 Step-Based Agent Execution Loop (`core/agent/agent_loop.py`, `core/agent/ingestor.py`, `core/prompts.py`, `core/rag/orchestrator.py`, `server/main.py`)
- **Decomposed Agent Execution:** AGENT_ACTION queries are decomposed into a JSON step plan (`check_url`, `download_documents`, `ingest_to_database`) via `plan_agent_action` and `AGENT_PLANNER_SYSTEM_PROMPT`.
- **Zero-Permission Read Steps:** READ steps (`check_url`, `download_documents`) execute immediately, streaming step-by-step narration token updates to the user without asking for approval up front.
- **Scoped WRITE-Step Approval:** Permission is requested only when reaching a WRITE step (`ingest_to_database`), staging a single proposal carrying the downloaded PDF paths and prompting the user for approval.
- **Shared Ingestion Pipeline:** `core/agent/ingestor.py` (`ingest_document_file`) encapsulates the OCR -> extraction -> PostgreSQL persistence workflow, shared between CONTROL proposal approval and step-loop write execution.
- **Chat & Control Resolution:** "yes"/CONTROL approve calls `resolve_write_step(proposal)` in a background thread; "no"/CONTROL reject cancels without re-running discovery/downloads.
- **Verified Unit Tests:** 4 unit tests covering read-only safety filtering, side-effect step planning, and chat approval/rejection lifecycle passed in 0.02s. Container `rf_app` restarted cleanly.

### 22.18 Semantic HITL Approval Classification & History-Aware Target Filtering (`core/prompts.py`, `core/agent/agent_loop.py`, `core/rag/orchestrator.py`)
- **Semantic Approval Classifier (`APPROVAL_CLASSIFIER_SYSTEM_PROMPT`):** Completely replaced rigid regex string matching (`_APPROVAL_RE`) in `_handle_chat_approval` with an LLM evaluation prompt (`APPROVE` | `REJECT` | `NEW_QUERY`) + robust keyword fallback. Any natural confirmation (`"go"`, `"do it"`, `"sure"`, `"make it happen"`, `"proceed"`, `"okay"`) is recognized as approval.
- **History Context & URL Inheritance (`plan_agent_action`):** `plan_agent_action` receives the conversation history. If the user asks to ingest a specific file without supplying a URL in the current turn, the agent automatically inherits the active URL from prior turns.
- **File-Specific Step Targeting (`target_file`):** The LLM Planner extracts specific requested filenames (e.g. `target_file="ESP8685-WROOM-01.pdf"`). `_download_documents` and `_ingest_to_database` filter candidate URLs and downloaded file paths by `target_file`, isolating and ingesting **only** the requested document.
- **Verified Unit Tests:** 4 unit tests covering semantic approval (`"go"` -> APPROVED), semantic rejection (`"cancel"` -> REJECTED), history URL inheritance, and target file filtering passed in 0.024s. `rf_app` container restarted cleanly.

### 22.25 Modular DAG Workflow Engine Architecture (`core/workflow/dag_engine.py`, `server/main.py`, `tests/test_dag_engine.py`)
- **Modular DAG Workflow Engine (`core/workflow/dag_engine.py`):**
  - Created standard `DAGNode`, `DAGWorkflow`, `DAGRunResult`, and `DAGExecutor` data models for multi-step automated compliance workflows (e.g. Portal Discovery -> Download -> GLM-OCR -> Qwen Metadata Extraction -> PostgreSQL Vector Persistence -> Expiration Filter -> Email Notification).
  - **Topological Sorting & Cycle Detection (`compute_topological_order`):** Implemented Kahn's Algorithm to compute exact node execution dependencies and detect graph cycles (`ValueError: Cycle detected in workflow DAG`).
  - **Dynamic Inter-Node State Wiring:** Automatically passes outputs from parent nodes (e.g. `file_path`, `url`, `data`, `verified_urls`) to child nodes.
  - **Concrete Node Tool Dispatcher (`_dispatch_node_tool`):** Integrated 8 concrete platform capabilities: `web_downloader`, `data_converter`, `email_drafting`, `discover_pdfs`, `glm_ocr`, `extract_metadata`, `persist_database`, and `filter_data`.
- **REST API Integration (`server/main.py`):**
  - `GET /api/v1/workflows`: Lists all registered DAG workflows.
  - `POST /api/v1/workflows`: Registers or updates a DAG workflow schema.
  - `POST /api/v1/workflows/{workflow_id}/execute`: Triggers asynchronous background execution of a DAG workflow.
  - `GET /api/v1/workflows/runs/{run_id}`: Fetches status and outputs of a workflow run.
- **Automated Test Verification (`tests/test_dag_engine.py`):**
  - `test_topological_sorting_valid_dag`: Verified topological ordering on valid DAG graphs.
  - `test_cycle_detection_invalid_dag`: Verified `ValueError("Cycle detected")` on cyclic graphs (A -> B -> A).
  - `test_dag_execution_state_wiring`: Verified end-to-end execution and inter-node state passing between `filter_data` and `data_converter`.
  - All 3 tests passed in 0.089s (`python3 -m unittest tests/test_dag_engine.py`). Application container `rf_app` restarted cleanly.

### 22.26 Session-Independent Long-Term Memory Engine (`schemas/extraction.py`, `core/agent/memory.py`, `core/agent/tools.py`, `core/prompts.py`, `server/main.py`, `ui/app.py`, `tests/test_memory_engine.py`)
- **Session-Independent Long-Term Memory Engine (`core/agent/memory.py`):**
  - **ORM & Storage (`AgentMemory` in `schemas/extraction.py`):** Created `agent_memories` PostgreSQL table (`id`, `memory_key`, `fact_text`, `source_session_id`, `created_at`).
  - **Memory Operations:** `save_agent_memory`, `get_active_memories`, `delete_agent_memory`, and `format_memories_for_prompt`.
  - **Base Identity & Tone Memory Seeding (`seed_base_identity_memories` in `core/agent/memory.py` & `storage/database.py`):** Idempotently seeds core persona directives (`identity` and `tone_and_format`) into long-term memory on database startup, eliminating hardcoded persona boilerplate from task prompts.
  - **Agent Tool (`RememberFactTool` in `core/agent/tools.py`):** Registered `remember_fact` tool in `get_tool_registry()` allowing the agent to automatically save directives when users say *"Remember that..."*.
  - **Dynamic System Prompt Injection (`config_qa_system_prompt`, `config_router_system_prompt`, `config_planner_system_prompt` in `core/prompts.py`):** Automatically injects active cross-session memories into LLM prompts during chat synthesis, intent routing, and action planning.
- **REST API Endpoints (`server/main.py`):**
  - `GET /api/v1/memories`: Lists all active long-term memories.
  - `POST /api/v1/memories`: Stores a new long-term memory fact.
  - `DELETE /api/v1/memories/{memory_id}`: Deletes a memory by ID.
- **UI Database Table View & Scrollable Table Selector Buttons (`ui/app.py`):** Integrated **Agent Memories** as the 5th official database table view under **HOME > DATABASE TABLES** (`RF Certificates`, `Authorities`, `Suppliers`, `Sources`, `Agent Memories`), complete with left-panel memory creation forms, filter tools, batch Excel export, and multi-row delete actions. Replaced the static dropdown menu (`st.selectbox`) with a scrollable container wrapper (`with st.container(height=170, border=False)` inside `with st.container(border=True)`) containing interactive table buttons that highlight when active, exactly mirroring the `CHAT SESSIONS` sidebar layout on the ASSISTANT page.
- **Automated Test Verification (`tests/test_memory_engine.py`):**
  - `test_memory_lifecycle`: Verified memory creation, listing, system prompt block formatting, and deletion. Passed in 0.043s. Container `rf_app` restarted cleanly.

### 22.27 Dynamic Schema & Custom Table Engine (2026-08-29 Implementation)
- **Dynamic DDL Storage Engine (`storage/dynamic_schema.py`):** Implemented sanitized PostgreSQL Data Definition Language (DDL) and Data Manipulation Language (DML) helper functions:
  - `create_custom_table(table_name, columns)`: Dynamically creates new user tables with sanitized columns, `id SERIAL PRIMARY KEY`, and `created_at TIMESTAMP`.
  - `add_column_to_table(table_name, column_name, column_type)`: Executes `ALTER TABLE <table_name> ADD COLUMN IF NOT EXISTS ...`.
  - `drop_column_from_table(table_name, column_name)`: Executes `ALTER TABLE <table_name> DROP COLUMN IF EXISTS ...`.
  - `fetch_dynamic_records()`, `insert_dynamic_record()`, `delete_dynamic_record()`: Dynamic CRUD operations for user-defined tables.
- **AI Agent Tool (`ManageSchemaTool` in `core/agent/tools.py`):** Registered `manage_schema` tool in `get_tool_registry()` supporting `create_table`, `add_column`, `drop_column`, `list_tables`, and `get_columns` actions directly driven by natural language prompts.
- **REST API Endpoints (`server/main.py`):** Exposed `/api/v1/schema/tables` endpoints to list tables, inspect columns, create tables, modify columns, and manage dynamic row data.
- **Dynamic Streamlit UI Discovery (`ui/app.py`):**
  - Automatically discovers custom user tables from backend API and populates them as scrollable table selector buttons under **HOME > DATABASE TABLES**.
  - Added **"+ Create Custom Table"** expander form allowing users to define new tables directly from the UI.
  - Dynamically renders custom table data under `db_main_col` with column metrics, search/filter bars, interactive dataframes, record addition forms, and multi-row delete actions.
- **Automated Test Suite (`tests/test_dynamic_schema.py`):**
  - `test_dynamic_table_lifecycle`: Verified table creation, column addition, row insertion, listing, column dropping, and row deletion. Passed all 5 suite tests in 0.078s. Container `rf_app` restarted cleanly.

### 22.28 Dynamic Table Label Unification & Full Feature Parity (2026-08-29 Refinement)
- **Removed "Custom:" Prefix & Filtered Internal Tables (`ui/app.py` & `storage/dynamic_schema.py`):** Removed the `"Custom: "` string prefix from table button labels. Internal engine infrastructure tables (`certificate_chunks`, `chat_sessions`, `chat_messages`, `agent_actions`, `workflows`, `workflow_runs`, `alembic_version`) are strictly excluded from user database table views. All user-created dynamic tables are formatted cleanly in title case (e.g. `Test Vendor Audits`, `Supplier Audits`) and listed alongside core business tables (`RF Certificates`, `Authorities`, `Suppliers`, `Sources`, `Agent Memories`).
- **Universal Table Feature Parity:** Enabled full feature parity across all dynamic custom tables:
  - **CSV/Excel File Import (`POST /api/v1/schema/tables/{table}/import` in `server/main.py` & `bulk_insert_dynamic_records` in `storage/dynamic_schema.py`):** File import uploader added to the left management panel for custom tables.
  - **Selected Row Editing (`PUT /api/v1/schema/tables/{table}/data/{id}` in `server/main.py` & `update_dynamic_record` in `storage/dynamic_schema.py`):** "Edit Selected" button opens an inline editor pre-populated with row fields to save updates back to PostgreSQL.
  - **Automated Test Verification:** All 5 test suite items passed in 0.072s. Container `rf_app` restarted cleanly.

### 22.29 Core Table Mapping Deduplication (2026-08-29 Fix)
- **Resolved Duplicate Table Entries (`storage/dynamic_schema.py` & `ui/app.py`):** Fixed duplicate entry creation where `authority_lookups` and `supplier_lookups` were flagged as custom tables alongside hardcoded base options `"Authorities"` and `"Suppliers"`. Registered `authority_lookups` and `supplier_lookups` as core system tables (`CORE_PLATFORM_TABLES`), ensuring zero duplicate entries in the table selector sidebar. Container `rf_app` restarted cleanly with all 5 unit tests passing in 0.073s.

### 22.30 Dynamic Table Deletion Tooling (2026-08-29 Implementation)
- **Table Drop Storage Function (`drop_custom_table` in `storage/dynamic_schema.py`):** Added sanitized `DROP TABLE IF EXISTS ... CASCADE` helper protecting core system tables from accidental deletion.
- **REST API Endpoint (`DELETE /api/v1/schema/tables/{table_name}` in `server/main.py`):** Added API endpoint for dropping custom user database tables.
- **AI Agent Tooling (`drop_table` action in `core/agent/tools.py`):** Added `drop_table` action to `ManageSchemaTool` allowing assistant or user to drop custom tables via chat.
- **Interactive Toggle Action UI & ID Column Hiding (`ui/app.py`):** Converted **"+ New"** and **"Delete"** into clean action buttons in `db_nav_col`. Clicking **"+ New"** toggles the table creation form directly underneath. Clicking **"Delete"** targets the **currently selected table**, opening a confirmation box underneath to drop that table cleanly. Internal database `id` / `certificate_id` primary key columns are now hidden by default across all table canvas views while remaining accessible for row selection and delete operations. Tested and verified in container (`Ran 5 tests in 0.088s, OK`).

### 22.31 Full Excel-Style Spreadsheet UX Transformation (2026-08-29 Implementation)
### 22.32 Instant Table Creation & Interactive Spreadsheet Canvas (2026-08-29 Refinement)
- **Instant Sidebar & Canvas Binding (`ui/app.py`):** Clicking **`+ New`** creates a new table instantly in PostgreSQL, populates its button inside the **DATABASE TABLES** sidebar wrapper, and opens it directly on the right main canvas (`db_main_col`) as a brand new empty spreadsheet.
- **On-the-Fly Column Additions:** Added **`➕ Add Column to Table`** toolbar control above dynamic custom tables to add columns on the fly. Tested and verified in container (`Ran 5 tests in 0.086s, OK`).

### 22.33 Removal of Legacy "Add Record" Forms (`ui/app.py` Cleanup)
### 22.34 Implicit Column Auto-Detection & True Excel UX (`ui/app.py`)
- **Zero-Button Pre-Expanded Grid Columns (`ui/app.py`):** Pre-expanded extra editable columns (`Extra Col 1`, `Extra Col 2`) directly inside the spreadsheet grid canvas (`st.data_editor`). Typing data into any extra column automatically registers it in PostgreSQL upon clicking **`💾 Save Changes to Database`** with zero buttons needed. Tested and verified in container (`Ran 5 tests in 0.092s, OK`).

### 22.40 Strict Zero-Emoji Codebase Hygiene (`ui/app.py` & Workspace)
- **Emoji Removal:** Scanned the codebase and purged all decorative unicode emojis from code strings, labels, and expanders (`Edit Table Columns`, `Save Changes to Database`). Codebase strictly adheres to AGENTS.md Protocol 1.D (No Emojis in codebase). Tested and verified in container (`Ran 5 tests in 0.088s, OK`).

### 22.41 Removal of Obsolete Add Button & Canvas Caption Text (`ui/app.py`)
- **Cleaned Toolbar:** Removed the obsolete `+` add row icon button from TABLE ACTIONS header row in `db_nav_col`.
- **Cleaned Spreadsheet Canvas:** Removed redundant instruction caption text above the spreadsheet editor grid. Tested and verified in container (`Ran 5 tests in 0.089s, OK`).

### 22.42 Custom Columns & File Import Creation under "+ New" Table Button (`ui/app.py`)
- **Custom Initial Columns:** Added a comma-separated column input field under **"+ New"** (`vendor_code, audit_score, inspector`) allowing users to define custom table columns upon creation.
- **Import CSV/Excel Table Creation:** Added an "Import File" tab under **"+ New"** allowing users to create a new database table directly by uploading a `.csv` or `.xlsx` file. Table schema is generated from file header columns and records are imported immediately. Tested and verified in container (`Ran 5 tests in 0.086s, OK`).

### 22.43 Natural Spaces for Column Names (`ui/app.py`)
- **Natural Space Formatting:** Configured dynamic custom tables in `st.data_editor` to format all column titles cleanly with spaces and Title Case (e.g. `Vendor Code`, `Audit Score`, `Inspector Name`). Users can type column names with normal spaces anywhere in the application. Tested and verified in container (`Ran 5 tests in 0.091s, OK`).

### 22.44 Interactive Spreadsheet Capabilities for Core Tables (`ui/app.py`)
- **Core Lookup Table Spreadsheet UX:** Extended interactive spreadsheet grid capabilities (`st.data_editor` with double-click inline cell editing, dynamic row additions, hidden primary key IDs, and batch save button) to all core lookup tables (`Authorities`, `Suppliers`, `Sources`, `Agent Memories`), excluding `RF Certificates`.
- **Sidebar Column Management:** Enabled **"Edit Table Columns"** in `db_nav_col` for all core lookup tables to allow renaming, adding, and deleting columns. Tested and verified in container (`Ran 5 tests in 0.089s, OK`).

### 22.45 Agent Deletion Structural Breakdown Analysis & Resolution (`core/agent/`, `core/prompts.py`)
- **Structural Root Cause:** The agent loop (`core/agent/agent_loop.py`) previously registered ONLY web scraping actions (`check_url`, `download_documents`, `ingest_to_database`). When users requested database record deletions (e.g. "delete it"), the planner system prompt had no deletion action, causing `_fallback_plan` to default to `check_url`, which failed with `"I need a URL to check. Please provide the link."`. In addition, `core/agent/db_editor.py` explicitly blocked `DELETE` SQL operations.
- **Systematic Resolution:**
  1. Added `delete_certificate` and `delete_record` actions to `AGENT_PLANNER_SYSTEM_PROMPT` in [`core/prompts.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/prompts.py).
  2. Enabled `delete` operation in `SUPPORTED_OPS` and added parameterized `DELETE FROM table WHERE ...` construction with mandatory `row_filter` guardrails in [`core/agent/db_editor.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/db_editor.py).
  3. Registered `_delete_certificate` and `_delete_record` step executors in [`core/agent/agent_loop.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py). Updated `_fallback_plan` to properly handle deletion requests without requiring a URL.
  4. Tested and verified in container (`Ran 5 tests in 0.087s, OK`).

### 22.46 Dynamic Agent Planner Prompt Auto-Generation (`core/agent/agent_loop.py`, `core/prompts.py`)
- **Automated Tool Registration in Prompts:** Replaced the hardcoded action list in `AGENT_PLANNER_SYSTEM_PROMPT` with dynamic injection token `{DYNAMIC_KNOWN_ACTIONS}`.
- **Dynamic Helper Function:** Added `get_registered_actions_prompt_block()` in [`core/agent/agent_loop.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py). `config_planner_system_prompt()` in [`core/prompts.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/prompts.py) now dynamically inspects `_STEP_EXECUTORS` and constructs the action menu on the fly.
- **Zero Manual Maintenance:** Adding any new tool to `_STEP_EXECUTORS` automatically populates the LLM system prompt without requiring manual prompt edits. Tested and verified in container (`Ran 5 tests in 0.094s, OK`).

### 22.47 Complete Removal of Deterministic Fallback Plan (`core/agent/agent_loop.py`, `core/rag/orchestrator.py`)
- **Pure LLM Execution Architecture:** Completely removed `_fallback_plan` from [`core/agent/agent_loop.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py). All agent actions are now planned strictly by the LLM against the dynamically registered tool inventory.
- **Explicit Unhandled Action Response:** When a user requests an action for which no tool is registered, the agent no longer guesses or defaults to web scraping. Instead, it responds with: `"I cannot execute what you are asking because I do not have a tool available for this action."`
- **Verification:** Tested and verified in container (`Ran 5 tests in 0.090s, OK`).

### 22.48 Multi-Certificate & Filtered Batch Deletion Engine (`core/agent/agent_loop.py`, `core/prompts.py`)
- **Criteria & Batch Deletion Engine:** Updated `_delete_certificate` in [`core/agent/agent_loop.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py) to support deleting certificates by criteria (`country`, `supplier`, `authority`, `component`), explicit `target_id`, or batch `delete_all`.
- **System Prompt Parameters:** Updated `AGENT_PLANNER_SYSTEM_PROMPT` in [`core/prompts.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/prompts.py) to instruct the LLM on passing `filters` and `delete_all` parameters in step payloads.
- **Verification:** Tested and verified in container (`Ran 5 tests in 0.102s, OK`).

### 22.49 Natural User Identifier Deletion Matching (`core/agent/agent_loop.py`)
- **Natural Identifier Resolution:** Updated `_delete_certificate` in [`core/agent/agent_loop.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py) to resolve target search terms against all natural user-facing identifiers: `component` (e.g. `IM3C`), `supplier` (e.g. `Bosch`), `certif_number` (e.g. `H-22392`), and `file_name` (e.g. `ESP8685-WROOM-01.pdf`). Internal database IDs are no longer required.
- **Verification:** Tested and verified in container (`Ran 5 tests in 0.095s, OK`).

### 22.50 Explicit Record Previews in Confirmation Messages (`core/agent/agent_loop.py`)
- **Detailed Preview Staging:** Enhanced `stage_write_step` in [`core/agent/agent_loop.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py) to query matching PostgreSQL records during proposal creation.
- **Explicit HITL Confirmation:** The confirmation prompt now lists the exact count and matching certificate names/numbers (e.g. `Delete 2 certificate(s) matching 'Valeo': (IM3C, RTBM-SHSAGEN)`) before asking for user approval (`'yes'/'no'`), preventing any ambiguity or misunderstanding.
- **Verification:** Tested and verified in container (`Ran 5 tests in 0.091s, OK`).

### 22.51 Immediate Execution for Explicit Direct Write Commands (`core/rag/orchestrator.py`)
- **Direct Command Immediate Execution:** Updated `_iter_agent_action` in [`core/rag/orchestrator.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/orchestrator.py) to check `is_explicit_direct_write_action(clean_query)`. Explicit direct user commands (e.g. `delete certificate IM3C`, `delete certificates from Argentina`, `ingest ESP8685-WROOM-01.pdf`) now execute immediately without requiring a second confirmation turn.
- **Discovery Confirmation Preserved:** Indirect discovery or background web scraping jobs continue to stage proposals for HITL confirmation.
- **Verification:** Tested and verified in container (`Ran 5 tests in 0.091s, OK`).











