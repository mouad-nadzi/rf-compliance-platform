# Project Handoff Report — Automotive Certificate Compliance & Q&A Platform
**Date:** 2026-08-24 (updated Production Deployment on GCP NVIDIA L4 Instance, LLM Schema Mapping, Multi-Lingual Date Normalization & Direct Link Access)
**Status:** **LIVE IN PRODUCTION on GCP NVIDIA L4 Host**. Primary LLM Engine: **Qwen3.8-27B GGUF** (`qwen3.8-27b-gguf`), Pure-HF GLM-OCR (`glm-ocr`), LLM-Based Automated File Column Mapping, Multi-Lingual French Date Normalization (`_parse_iso_date`), Streamlit `LinkColumn` Direct Access, Zero-Hardcoding Dynamic Link Resolution (`PUBLIC_API_URL`), Docker Compose Infrastructure (`rf_app` + `rf_postgres_db`), Deterministic 7-Field Compliance Pipeline (`CertificateExtractionSchema`), SQL Lookup Tables & Ingestion (`AuthorityLookup`, `SupplierLookup`), GPU VRAM Coexistence (NVIDIA L4 24GB), End-to-End Hybrid RAG, SQL Hydration, CPU Embeddings, Model Registry, Intelligent Router, & Production Decoupled Architecture.

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
│                                (router → retriever → qa)        │
└────────────┼──────────────────────────────────┼─────────────────┘
             ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (server/main.py)                    │
│                                                                 │
│   POST /api/v1/parse                                            │
│   ┌──────────┐   ┌──────────────┐       ┌─────────────────┐     │
│   │OCR Engine│ → │  Extractor   │ ----> │ Storage Hydrator│     │
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
├── server/                          ← FastAPI application package (model-agnostic via registry & auto DB init)
│   ├── main.py                   ← FastAPI app entrypoint (uvicorn server.main:app)
│   ├── config.py                 ← Central config (LLM_ENGINE=qwen3.8-27b-gguf, OCR_ENGINE=glm-ocr, DEFAULT_CONTEXT_WINDOW=8192, EMBEDDING_DEVICE=cpu)
│   └── __init__.py
├── core/                         ← Production-ready AI compute & RAG engines (formerly engines/)
│   │
│   ├── rag/                      ← RAG Q&A pipeline (router, chunker, embeddings, retriever, sql_engine, hybrid_engine, orchestrator, qa)
│   │   ├── __init__.py
│   │   ├── router.py             ← Intent Router (METADATA_QUERY, UNSTRUCTURED_RAG, HYBRID_QUERY)
│   │   ├── sql_engine.py         ← Text-to-SQL engine (execute_metadata_query)
│   │   ├── hybrid_engine.py      ← Hybrid Dense/Sparse RRF Engine with Parent Expansion (retrieve_hybrid_context & execute_unstructured_query)
│   │   ├── orchestrator.py       ← Central Dual-Path RAG Orchestrator (answer_compliance_query)
│   │   ├── chunker.py            ← Page-aware paragraph chunking with <Page X> tracking
│   │   ├── embeddings.py         ← 1024-d dense vector embeddings facade (BAAI/bge-m3, CPU)
│   │   ├── retriever.py          ← Dual-path retrieval (Text-to-SQL + Hybrid Dense/Sparse RRF)
│   │   └── qa.py                 ← Cross-lingual Q&A synthesis with citation generation
│   │
│   ├── utils/                    ← GPU guardrails & VRAM helper utilities
│   │   ├── __init__.py
│   │   ├── vram.py               ← ensure_headroom() graceful MemoryError guard + free_vram_mb() + flush_gpu_cache()
│   │   └── system_check.py       ← System readiness verification & initialization gate
│   │
│   ├── llm/                      ← Pluggable LLM engines (qwen3_8_27b, qwen3_35b, qwen3_14b, qwen3_8b, qwen2_gguf, gemma4_26b, qwen_agentworld)
│   ├── ocr/                      ← Pluggable OCR engines (glm_ocr, got_ocr2, deepseek_ocr2)
│   ├── registry.py               ← Model registry (OCR_REGISTRY, LLM_REGISTRY) + lazy factory
│   ├── extractor.py              ← Structured extraction, lookup enrichment (enrich_certificate_metadata), & atomic DB hydration
│   ├── prompts.py                ← System prompt configurations (CERTIFICATE_EXTRACTION_SYSTEM_PROMPT, router, & cross-lingual QA)
│   └── base.py                   ← Abstract BaseOCREngine / BaseLLMEngine contracts
│
├── schemas/                      ← Pydantic data models & SQLAlchemy ORM models
│   ├── extraction.py             ← 7-field CertificateExtractionSchema + CertificateMetadata & CertificateChunk ORM
│   └── qa.py                     ← Citation & QAResponseSchema Pydantic models
│
├── storage/                      ← Relational, vector, lookup DB storage & master seed data
│   ├── database.py               ← SQLAlchemy engine, SessionLocal, get_db_session, & pgvector init_db()
│   ├── models.py                 ← ORM models for AuthorityLookup & SupplierLookup reference tables
│   ├── seed_lookups.py           ← Idempotent JSON ingestion script reading from data/lookups/*.json
│   ├── backup.py                 ← Portable pg_dump database export utility
│   └── __init__.py
│
├── ui/                           ← Streamlit frontend
│   ├── app.py                    ← Two-tab UI (Document Ingestion + RAG Q&A Chat with Intent Badges & Sources)
│   └── static/                   ← Branding assets (stellantis.png)
│
├── data/                         ← Consolidated runtime data & seed datasets
│   ├── lookups/                  ← Master reference datasets (authorities.json, suppliers.json) — tracked
│   ├── uploads/                  ← Batch ingestion staging (gitignored)
│   ├── files/                    ← Permanent uploaded files served statically at /files/ (gitignored)
│   ├── ocr_cache/                ← OCR markdown cache for batch resume (gitignored)
│   ├── model_cache/              ← Cached AI model weights (gitignored)
│   └── postgres/                 ← PostgreSQL 16 data volume (gitignored)
│
├── handoff_report.md               ← Complete architectural handoff report (read first in new sessions)
│
├── docker-compose.yaml           ← PostgreSQL 16 + pgvector container infrastructure
├── requirements.txt              ← Python dependencies (includes sqlalchemy, pgvector, sentence-transformers, llama-cpp-python)
├── Dockerfile                    ← CUDA image build (torch 2.6.0, llama-cpp-python 0.3.34-cu122, transformers 5.15.1)
├── entrypoint.sh                 ← Container boot: DB wait → init_db → seed lookups → Streamlit + uvicorn
└── .gitignore                    ← Runtime artifacts excluded from version control
```

---

## 4. Model Registry & Engine Inventory

### LLM Engine Registry (`LLM_REGISTRY`)

| Key | File | Model Weights | Params / Quant | Primary Purpose |
|---|---|---|---|---|
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
All LLM engines use `DEFAULT_CONTEXT_WINDOW = 8192` from `server/config.py` in the current T4-safe configuration. Context window fallback loops have been completely removed in favor of single-pass initialization, ensuring deterministic VRAM footprint and execution. The earlier 16K setting is documented in §11 as an historical fit limit; it should only be restored on larger GPU hardware or multi-GPU deployments.

### 5.2 Strict Quantization Floor (Anti-OOM Directive)
- **Forbidden Upward Swaps:** The system must never change or upgrade a model's quantization level (e.g. from `IQ2_M`/`Q4_K_M` to 8-bit or FP16) during debugging, as this triggers uncatchable OOM kernel crashes on Tesla T4 GPUs.
- **Allowed Last-Resort Exception:** If a model download fails or corrupts, the agent is permitted to swap to an equivalent repository from a different publisher (e.g. `unsloth`, `bartowski`, `google`), provided the model file size and quantization precision remain strictly identical.

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
| **Router Accuracy Rate** | **100.0%** (99 out of 99 queries correctly classified) 🏆 |
| **Average Latency per Query** | **1,231.6 ms (~1.23 seconds)** ⚡ |
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

Verified on 2026-08-20: `transformers 5.15.1` + `llama-cpp-python 0.3.34` (CUDA offload active) + `numpy 2.0.2` + `torch 2.5.1+cu124` + PostgreSQL 14/pgvector 0.8.6 → full end-to-end pipeline (OCR → Gemma extraction → pgvector persistence → SQL verification) ran successfully.

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
  - [x] **Phase 3 Step 2: Sequential Two-Phase Batch Ingestion (2026-08-11)** — single-residency model lifecycle (`core/utils/model_lifecycle.py`), `POST /api/v1/batch/ingest` + status/certificates endpoints, deterministic resume via OCR markdown cache + manifest, Streamlit polling UI. Fixes the 2026-08-10 batch OOM (84 page OOMs → NULL supplier rows).
  - [x] **Colab re-verify of the new batch architecture (2026-08-11)** — unpacked `project_sync.zip`, installed deps, booted Streamlit + cloudflared, and ran an end-to-end multi-file batch (OCR phase → extract phase → Q&A) with zero OOMs and no NULL supplier rows.
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
| 1 | `llama-cpp-python` must be pinned | PyPI's unpinned build silently wins (CPU-only); default `gemma4-26b-gguf` uses the `gemma4` GGUF architecture that older llama.cpp builds reject (`0.3.19` → `unknown model architecture: 'gemma4'`); also loading a second model copy on the 16GB T4 causes `cudaMalloc: out of memory` | **`pip install` the prebuilt `v0.3.34-cu122` GitHub release wheel directly** (`llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl`, glibc 2.35 = Colab) — installs in ~60s. **Do NOT** use `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122` for 0.3.34 — no cp312 binary exists there, pip falls back to ~15 min source compilation. **0.3.19 is forbidden** (rejects gemma4 arch). Kernel restart after any C-extension change | Pin the same version in the production image; serve via vLLM/Ollama for concurrency |
| 2 | `numpy` must be pinned to 2.0.2 | Colab preinstalls numpy 2.5.2, which removes `_blas_supports_fpe`; breaks scikit-learn 1.6.1 and numba 0.60 at import time | `pip install numpy==2.0.2` | Pin `numpy>=2.0,<2.3` (or the tested 2.0.2) in requirements |
| 3 | `torchcodec` must be removed | `sentence-transformers` 5.6.0 imports `torchcodec`; its libtorchcodec is ABI-incompatible with torch 2.6.0+cu118 (and FFmpeg libs are missing), raising `RuntimeError` at import | `pip uninstall -y torchcodec` (not needed; bge-m3 embeddings don't use it) | Not installed in production; pin a compatible torchcodec if audio/video decoding is ever needed |
| 4 | Public ingress blocked | Colab firewall blocks ports 8000/8501 | `.streamlit/config.toml` with `headless=true`, `enableCORS=false`, `enableXsrfProtection=false` + `cloudflared tunnel --url http://localhost:8501` | Standard reverse proxy / container port mapping; no CORS overrides needed |
| 5 | Single 16GB T4 GPU | Only one heavyweight model process can be resident at a time | Never run the Streamlit in-process TestClient and a separate model subprocess simultaneously; stop the app before running standalone engine tests | Unbounded multi-GPU inference in production |
| 6 | `transformers` must be >= 5.0 for GLM-OCR | 4.49.0 rejects `glm_ocr` (unknown config); `AutoModelForConditionalGeneration` was removed in 5.x | `pip install transformers==5.15.1` (verified); use `AutoModelForImageTextToText` for `glm_ocr` | Pin `transformers>=5.0` in the image |
| 7 | `torch` must be >= 2.6 to load `bge-m3` `.bin` | transformers 4.51+/5.x block `torch.load(weights_only=True)` of `.bin` on torch < 2.6 (CVE-2025-32434 guard); Colab ships torch 2.5.1 → `core/rag/embeddings.py` silently falls back to **zero vectors** | Upgrade torch to >= 2.6 (metadata/chunks unaffected); or convert the model to safetensors | Pin `torch>=2.6` in the image so embeddings are real |

## 11. T4 VRAM Fit Audit: OCR + LLM + Embeddings (2026-08-09)

Empirically verified on the Colab T4 (15,360 MiB) by loading all three models into **one** process and sampling `nvidia-smi` after each:

| Model | Config | VRAM |
|---|---|---|
| GLM-OCR 0.9B | FP16, `device_map="cuda"` (now pure-HF backend) | ~2.2 GB |
| Gemma 4 26B (UD-IQ2_M) | `n_gpu_layers=-1`, `flash_attn=True`, KV q8_0 | ~11.9 GB @ 16K ctx |
| bge-m3 | **FP16** (`torch_dtype`) | ~1.1 GB |
| **Total (16K ctx, FP32 emb)** | | **~16.4 GB → CUDA OOM** |
| **Total (8K ctx, FP16 emb)** | | **~14.3 GB (524 MiB free)** |

**Conclusion:** All three models fit only with `DEFAULT_CONTEXT_WINDOW=8192` **and** FP16 embeddings. 16K context or FP32 embeddings overflow the T4.

**Two latent bugs surfaced and fixed:**
1. **`core/llm/gemma4_26b.py`** passed `type_k="q8_0"` / `type_v="q8_0"` as **strings**. `llama-cpp-python` 0.3.34 requires the integer GGML enum (`GGML_TYPE_Q8_0`). The string raised `TypeError`, silently falling back to the no-flash-attn constructor, which pads the V cache to 2048 and fails `llama_context` creation — the app could not load OCR + Gemma at all. Fixed to `type_k=llama_cpp.GGML_TYPE_Q8_0`.
2. **`core/rag/embeddings.py`** loaded bge-m3 in FP32 (~2.3 GB), then FP16 (`SentenceTransformer(..., model_kwargs={"torch_dtype": torch.float16})`, ~1.1 GB). Both were later superseded by the CPU move below — bge-m3 now runs on CPU (`EMBEDDING_DEVICE="cpu"`, no dtype kwargs), keeping the T4 entirely for OCR + LLM.

**Final design decision — bge-m3 moved to CPU (2026-08-09 PM):**
Idle-fit alone was not enough. A live ingestion run (real image OCR → extraction → embeddings → SQL) with all three co-resident hit a **real CUDA OOM**: GLM-OCR's dynamic KV cache spiked free VRAM to ~350 MiB mid-generation, then Gemma crashed with `ggml_abort` on an unfourth 40 MiB allocation. Root cause: hardcoded `max_new_tokens=8192` in `glm_ocr.py` built a +1.5 GB transient KV cache on top of the loaded models (peak 14,811 MiB).

Resolution (all code in `server/config.py` / `core/utils/vram.py` / engines):
- **`EMBEDDING_DEVICE="cpu"`** — bge-m3 leaves the T4 entirely (~1.1 GB freed). Embeddings are per-chunk during ingestion, so CPU latency (~2-4 s/batch) is acceptable.
- **GLM-OCR token cap (`min(OCR_MAX_NEW_TOKENS, 2048)`)** — `config.OCR_MAX_NEW_TOKENS` is `8192`; the engine caps generation at `2048` via `min()`, bounding GLM-OCR's dynamic KV cache.
- **`MIN_FREE_VRAM_MB=1024` headroom guard** — `core/utils/vram.py` `ensure_headroom()` raises a **graceful `MemoryError`** (mapped to HTTP 507 in `server/main.py`) before OCR generation and LLM extraction instead of an uncatchable kernel OOM.
- **`MAX_EXTRACTION_PROMPT_CHARS=20000`** — truncates long OCR text so extraction stays inside the 8K context.
- Cache flush (`flush_gpu_cache()`) between OCR → extraction enables the +1 GB headroom check to pass.

**Verified end-to-end after the final fixes** (real upload through `/api/v1/parse` → 10 chunks persisted → SQL Q&A): OCR headroom OK (1,644 MiB free), extraction OK (1,590 MiB free), embeddings on CPU, **PARSE status 200**, supplier correctly extracted, peak `14,841 MiB / 14,913 MiB` usable — OOM-free and graceful-guard-protected.

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
| Avg / page | **1.22 s/page ✅ MET** | **30.15 s/page ❌ EXCEEDED** |
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



