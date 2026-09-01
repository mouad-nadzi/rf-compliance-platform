# Project Handoff Report — RF Compliance Platform
**Date:** 2026-09-01  
**Status:** LIVE IN PRODUCTION on GCP NVIDIA L4 Host  
**Primary LLM:** Qwen3.8-27B GGUF UD-IQ3_XXS (`qwen3.8-27b-gguf`, 3-bit, 32k context)  
**Primary OCR:** GLM-OCR (pure-HF transformers, `glm-ocr`)  
**Frontend:** React 18 + TypeScript SPA (Vite) — Streamlit UI fully decommissioned  
**Repository:** `mouad-nadzi/rf-compliance-platform` (branch: `main`, commit: `4ae99bd`)

---

## 1. Project Purpose

An AI-native compliance intelligence platform for Stellantis that centralizes RF (Radio Frequency) certificate management across all vehicle components and markets. The platform:

- **Ingests** PDF certificates (via GLM-OCR + LLM extraction) and Excel spreadsheets (via LLM column mapping).
- **Stores** normalized metadata + dense vector embeddings in PostgreSQL 16 + pgvector.
- **Enriches** records automatically using bidirectional Authority ↔ Country ↔ Supplier lookup tables.
- **Answers** natural-language Q&A queries via hybrid RAG (SQL + dense retrieval).
- **Discovers** new certificates autonomously by scraping configured supplier portals.
- **Notifies** users of every silent AI action (auto-fills, standardizations) through an in-app notification system.

The entire pipeline operates 100% locally — no external APIs. All inference runs on the GCP NVIDIA L4 GPU.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              React 18 SPA (TypeScript + Vite)                   │
│         Served as static files at http://<host>:8000            │
│   HOME (Databases) | ASSISTANT | AUTOMATIONS | SETTINGS        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ REST / SSE
┌──────────────────────────────▼──────────────────────────────────┐
│            FastAPI Backend (server/main.py)                     │
│            Uvicorn on 0.0.0.0:8000                              │
│                                                                 │
│  POST /api/v1/parse              — PDF ingestion                │
│  POST /api/v1/certificates/batch — Excel batch import           │
│  POST /api/v1/chat               — RAG Q&A (SSE streaming)      │
│  GET  /api/v1/notifications      — Notification feed            │
│  GET/POST /api/v1/databases/*    — Dynamic table management     │
│  GET/POST /api/v1/automations/*  — Background agent control     │
└────────────┬──────────────────────────────┬─────────────────────┘
             │                              │
     ┌───────▼──────────┐         ┌────────▼────────────┐
     │  AI Engine Layer │         │   PostgreSQL 16      │
     │                  │         │   + pgvector         │
     │  GLM-OCR (VRAM)  │         │                      │
     │  Qwen3.8-27B     │         │  certificates        │
     │  (VRAM)          │         │  certificate_chunks  │
     │  bge-m3 (CPU)    │         │  authority_lookups   │
     └──────────────────┘         │  supplier_lookups    │
                                  │  notifications       │
                                  │  agent_memories      │
                                  │  recycle_bin         │
                                  │  dynamic tables      │
                                  └─────────────────────┘
```

**No Streamlit. No Colab. No port 8501. No external tunnel.** The React SPA is served directly from FastAPI static file mounting on port 8000.

---

## 3. Current Folder Structure

```
rf-compliance-platform/
├── server/
│   ├── main.py           FastAPI application (30+ REST endpoints, SSE streaming)
│   ├── config.py         Central config (engines, context window, paths, secrets)
│   └── __init__.py
│
├── core/
│   ├── extractor.py      PDF→OCR→LLM extraction pipeline + notification creation
│   ├── prompts.py        System prompts (extraction, router, QA, agent planner)
│   ├── registry.py       Model registry (OCR_REGISTRY, LLM_REGISTRY) + lazy factory
│   ├── base.py           Abstract BaseOCREngine / BaseLLMEngine contracts
│   │
│   ├── agent/
│   │   ├── agent_loop.py      Agentic plan executor + HITL proposal staging
│   │   ├── ingestor.py        Document ingestion orchestrator
│   │   ├── memory.py          Long-term memory read/write (agent_memories table)
│   │   ├── proposals.py       HITL proposal schema and lifecycle
│   │   ├── scraper.py         Web scraper (HTML + optional Playwright fallback)
│   │   ├── tools.py           Registered agent tool registry
│   │   ├── worker.py          Background batch worker
│   │   ├── db_editor.py       Agent DB write operations (INSERT/UPDATE/DELETE)
│   │   └── __init__.py
│   │
│   ├── llm/
│   │   ├── qwen3_8_27b.py     Active production LLM (Qwen3.8-27B IQ3_XXS)
│   │   ├── qwen3_35b.py       Qwen3.6-35B MoE engine
│   │   ├── qwen3_14b.py       Qwen3-14B engine
│   │   ├── qwen3_8b.py        Qwen3-8B engine
│   │   ├── qwen2_gguf.py      Qwen2-7B baseline engine
│   │   ├── gemma4_26b.py      Gemma 4 26B engine (100% benchmark accuracy)
│   │   ├── qwen_agentworld.py Qwen-AgentWorld-35B specialized agent engine
│   │   └── __init__.py
│   │
│   ├── ocr/
│   │   ├── glm_ocr.py         Active production OCR (pure-HF GLM-OCR)
│   │   ├── deepseek_ocr2.py   DeepSeek-OCR-2 3B (NF4)
│   │   ├── got_ocr2.py        GOT-OCR2_0 0.5B fallback
│   │   └── __init__.py
│   │
│   ├── rag/
│   │   ├── orchestrator.py    Central dual-path RAG orchestrator
│   │   ├── router.py          Intent router (METADATA_QUERY / UNSTRUCTURED_RAG / HYBRID_QUERY)
│   │   ├── sql_engine.py      Text-to-SQL engine
│   │   ├── hybrid_engine.py   Hybrid Dense/Sparse RRF retrieval
│   │   ├── retriever.py       Dual-path retrieval (SQL + hybrid dense/sparse)
│   │   ├── chunker.py         Page-aware paragraph chunker with <Page X> tracking
│   │   ├── embeddings.py      1024-d dense vector embeddings (BAAI/bge-m3, CPU)
│   │   ├── qa.py              Cross-lingual Q&A synthesis with citations
│   │   ├── unified_agent.py   Unified agentic query handler
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── vram.py            VRAM headroom guard (ensure_headroom, flush_gpu_cache)
│   │   ├── system_check.py    System readiness verification
│   │   ├── history.py         Chat history management utilities
│   │   └── __init__.py
│   │
│   └── workflow/
│       ├── dag_engine.py      DAG-based workflow execution engine
│       └── __init__.py
│
├── schemas/
│   ├── extraction.py     SQLAlchemy ORM models (all tables — see §5)
│   ├── automation.py     AutomationConfig dataclass
│   └── qa.py             Citation & QAResponseSchema Pydantic models
│
├── storage/
│   ├── database.py       SQLAlchemy engine, SessionLocal, get_db_session, init_db()
│   ├── models.py         ORM models for AuthorityLookup & SupplierLookup
│   ├── seed_lookups.py   Idempotent JSON seed loader (data/lookups/*.json)
│   ├── dynamic_schema.py Dynamic table schema management
│   ├── backup.py         pg_dump export utility
│   └── __init__.py
│
├── frontend/                   React 18 + TypeScript SPA (Vite build)
│   ├── src/
│   │   ├── App.tsx             Root app with React Router
│   │   ├── api.ts              Typed API client (all backend endpoints)
│   │   ├── index.css           Global design system & tokens
│   │   ├── main.tsx            Vite entry point
│   │   ├── components/
│   │   │   ├── Navbar/         Navigation bar with notification bell
│   │   │   └── Sidebar/        Collapsible sidebar
│   │   └── views/
│   │       ├── DatabasesView.tsx   HOME — all compliance tables, import/export
│   │       ├── ChatView.tsx        ASSISTANT — RAG Q&A with SSE streaming
│   │       ├── ControlView.tsx     AUTOMATIONS — background agent control
│   │       └── SettingsView.tsx    SETTINGS — platform configuration
│   ├── public/                 Static assets (Stellantis logo, icons)
│   ├── package.json
│   └── vite.config.ts
│
├── data/
│   ├── lookups/
│   │   ├── authorities.json   183 global national RF regulatory authorities (seeded on boot)
│   │   └── suppliers.json     50+ registered component suppliers
│   ├── uploads/               Batch ingestion staging (gitignored)
│   ├── files/                 Permanent uploaded PDFs served at /files/ (gitignored)
│   ├── ocr_cache/             OCR markdown cache for batch resume (gitignored)
│   ├── model_cache/           Cached AI model weights (gitignored)
│   └── postgres/              PostgreSQL 16 data volume (gitignored)
│
├── tests/
│   ├── test_memory_engine.py
│   ├── test_dynamic_schema.py
│   └── test_dag_engine.py
│
├── handoff_report.md           This file — read first in every new session
├── docker-compose.yaml         2-container stack (rf_app + rf_postgres_db)
├── Dockerfile                  CUDA 12.1 image (torch 2.6.0, llama-cpp 0.3.34, transformers 5.15.1)
├── entrypoint.sh               Boot sequence: DB wait → init_db → seed_lookups → uvicorn
├── requirements.txt            Python dependencies (no streamlit)
└── .gitignore                  Runtime artifacts excluded
```

> **NOTE:** `ui/` (Streamlit), `.streamlit/`, `scripts/`, `batch_uploads/`, `model_cache/`, `ocr_cache/` (top-level) have all been permanently deleted. The legacy `colab/` and `sandbox/` directories never existed in production.

---

## 4. Database Schema (PostgreSQL + pgvector)

All models are defined in `schemas/extraction.py`.

| Table | Purpose |
|---|---|
| `certificates` | RF certificate records (component, supplier, country, certif_number, authority, issue_date, exp_date, pdf_link, file_name) |
| `certificate_chunks` | 1024-d pgvector embeddings for RAG retrieval |
| `authority_lookups` | 183 national RF regulatory authorities (country, canonical_authority, abbreviation, aliases, standard_validity_years) |
| `supplier_lookups` | Component supplier registry (canonical_supplier, aliases) |
| `sources` | Web sources configured for autonomous scraping |
| `agent_memories` | Long-term AI memory (key-value facts persisted across sessions) |
| `notifications` | In-app notification feed (title, message, category, is_read, created_at) |
| `recycle_bin_records` | Soft-deleted records (table_name, record_data, deleted_at) |
| `chat_sessions` | Persisted chat session contexts (id, messages JSON, context_token_count) |
| Dynamic tables | User-created tables via Databases UI (managed by `storage/dynamic_schema.py`) |

---

## 5. Model Registry & Active Engines

### Active Production Configuration (`server/config.py`)

```python
OCR_ENGINE = "glm-ocr"          # GLM-OCR 0.9B, pure-HF, FP16
LLM_ENGINE = "qwen3.8-27b-gguf" # Qwen3.8-27B, UD-IQ3_XXS (3-bit), 32k ctx
EMBEDDING_MODEL = "BAAI/bge-m3"  # 1024-d, CPU
EMBEDDING_DEVICE = "cpu"
DEFAULT_CONTEXT_WINDOW = 32768
```

### VRAM Footprint on GCP NVIDIA L4 (24 GB)

| Component | Model | VRAM Usage |
|---|---|---|
| Vision OCR | GLM-OCR (zai-org/GLM-OCR) | ~2.5 GB |
| Language Model | Qwen3.8-27B UD-IQ3_XXS | ~11 GB |
| Embedding Model | BAAI/bge-m3 | CPU only |
| **Total** | | **~13.5 GB / 24 GB** |

Both OCR and LLM are co-resident in VRAM simultaneously. Sequential load/unload was removed — see §7 for VRAM history.

### Full LLM Registry (`core/registry.py`)

| Key | File | Model | Notes |
|---|---|---|---|
| `qwen3.8-27b-gguf` **(default)** | `core/llm/qwen3_8_27b.py` | `unsloth/Qwen3.8-27B-GGUF` UD-IQ3_XXS | Active production engine, 32k ctx, q8_0 KV |
| `gemma4-26b-gguf` | `core/llm/gemma4_26b.py` | `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ2_M | 100% router benchmark accuracy |
| `qwen3.6-35b-gguf` / `qwen3-35b` | `core/llm/qwen3_35b.py` | `unsloth/Qwen3.6-35B-A3B-GGUF` UD-IQ2_M | MoE, 3B active params |
| `qwen3-8b` | `core/llm/qwen3_8b.py` | `Qwen/Qwen3-8B-GGUF` Q8_0 | Fast dense model |
| `qwen3-14b-gguf` | `core/llm/qwen3_14b.py` | `unsloth/Qwen3-14B-GGUF` UD-IQ1_M | Balanced model |
| `qwen-agentworld-35b` | `core/llm/qwen_agentworld.py` | `unsloth/Qwen-AgentWorld-35B-A3B-GGUF` UD-IQ2_M | Specialized agentic model |
| `qwen2-7b-gguf` | `core/llm/qwen2_gguf.py` | `Qwen/Qwen2-7B-Instruct-GGUF` Q4_K_M | Lightweight baseline |

### Full OCR Registry

| Key | File | Model |
|---|---|---|
| `glm-ocr` **(default)** | `core/ocr/glm_ocr.py` | GLM-OCR 0.9B (pure-HF FP16) |
| `deepseek-ocr-2` | `core/ocr/deepseek_ocr2.py` | DeepSeek-OCR-2 3B NF4 |
| `got-ocr2` | `core/ocr/got_ocr2.py` | GOT-OCR2_0 0.5B |

---

## 6. Key Platform Features

### 6.1 Certificate Ingestion Pipeline

**PDF via `POST /api/v1/parse`:**
1. File upload → temp staging → permanent copy to `data/files/parse/`.
2. GLM-OCR extracts full document text (capped at `min(OCR_MAX_NEW_TOKENS, 2048)` tokens).
3. Qwen3.8-27B maps OCR text to `CertificateExtractionSchema` (7 fields: component, supplier, country, certif_number, authority, issue_date, exp_date).
4. Bidirectional lookup enrichment: empty authority derived from country (or vice versa), empty supplier derived from component name.
5. Record + pgvector chunk embeddings persisted to PostgreSQL.
6. `NotificationItem` created for the ingestion event.

**Excel/CSV via `POST /api/v1/certificates/batch`:**
1. LLM maps non-standard column headers to platform fields (`POST /api/v1/certificates/map-headers`).
2. Rows normalized: dates parsed to ISO 8601, authority/country bidirectional enrichment applied per row.
3. Auto-fill notifications created for any derived fields.
4. Upsert logic: existing records updated, new records inserted.

### 6.2 Bidirectional Authority ↔ Country Auto-Fill

On every certificate save (PDF or Excel):

- **Authority → Country**: If `authority` is known (e.g., `ENACOM`) and `country` is empty → derived as `Argentina`.
- **Country → Authority**: If `country` is known (e.g., `Pakistan`) and `authority` is empty → derived as `PTA`.
- **Component → Supplier**: If `component` is known (e.g., `AIDA / IVI R1`) and `supplier` is empty → derived as `Magneti Marelli`.

All 183 national regulatory authorities are loaded into `authority_lookups` at startup from `data/lookups/authorities.json`. Matching is case-insensitive and alias-aware.

> **Empty cell invariance rule**: If a field cannot be derived, it remains `""` / `NULL`. No placeholder strings (`"N/A"`, `"Unknown"`, `"—"`) are ever written.

### 6.3 Notification System

Every silent AI action generates a `NotificationItem` in PostgreSQL:
- Bell icon in React navbar shows real-time unread count (polled every 10 seconds).
- Opening the panel calls `PUT /api/v1/notifications/read` — all notifications marked read, badge cleared.
- Categories: `standardization` (auto-fill), `ingestion` (PDF parse), `system`.

### 6.4 RAG Q&A (Chat)

- **Intent router** classifies queries into `METADATA_QUERY` (→ SQL), `UNSTRUCTURED_RAG` (→ dense retrieval), or `HYBRID_QUERY` (→ both).
- **SSE streaming**: responses stream token-by-token via `text/event-stream`.
- **Chat session persistence**: session contexts stored in `chat_sessions` table, reloaded on startup.
- **Long-term memory**: AI learns facts during sessions (authority mappings, supplier associations) and stores them in `agent_memories`.

### 6.5 Agentic Automations

- Background APScheduler job for autonomous certificate discovery.
- Scraper (`core/agent/scraper.py`) fetches HTML and optionally renders JS-heavy portals via Playwright.
- HITL (Human-in-the-Loop) proposal staging: agent-discovered records require user approval before DB commit.
- Dynamic tool registration — adding any tool to `_STEP_EXECUTORS` auto-populates the LLM planner prompt.

### 6.6 Multi-Database Management (Dynamic Tables)

- Users create new compliance tables via the Databases UI (no code required).
- `storage/dynamic_schema.py` manages schema migrations at runtime.
- Soft-delete: deleted records go to `recycle_bin_records`, recoverable via the UI.
- Import (Excel/CSV) and export available for every table.

---

## 7. Architectural Rules & Critical Directives

### 7.1 Model Quantization — DO NOT UPGRADE

- The active LLM (`Qwen3.8-27B UD-IQ3_XXS`, 3-bit) was deliberately selected for VRAM fit on the L4.
- Do NOT upgrade to IQ4, Q4_K_M, Q8_0, or FP16 without first measuring VRAM impact. A transient OCR spike (~1.5 GB) must fit within the remaining headroom.
- See §T4 VRAM history in legacy sections for background.

### 7.2 VRAM Guardrails (Still Active)

- `core/utils/vram.py` `ensure_headroom(MIN_FREE_VRAM_MB=1024)` raises a graceful `MemoryError` (HTTP 507) before heavy inference instead of letting the CUDA kernel OOM crash the process.
- GLM-OCR token generation capped at `min(OCR_MAX_NEW_TOKENS, 2048)` to bound dynamic KV cache growth.

### 7.3 `type_k` / `type_v` Must Be Integer GGML Enums

`llama-cpp-python` 0.3.34 requires:
```python
type_k=llama_cpp.GGML_TYPE_Q8_0  # integer enum
type_v=llama_cpp.GGML_TYPE_Q8_0  # NOT the string "q8_0"
```
Passing strings silently triggers the no-flash-attn constructor fallback, pads the V cache to 2048, and causes `llama_context` creation to fail. This is fixed in `core/llm/qwen3_8_27b.py`.

### 7.4 Thinking Mode Tags

For Qwen3 models, `/no_think` and `/think` mode tags must be injected **inside the user block** — never after `<|im_start|>assistant\n`. The string placement rule is model-specific:
- **Qwen3 8B/14B**: inject soft ` /no_think` at end of user block.
- **Qwen3.6/AgentWorld 35B**: inject already-closed `<think>\n</think>` after assistant header (Qwen3.6 ignores `/no_think`).
- **Gemma 4**: fixed 0.7/0.8 sampling, no thinking tag.
- **Qwen2 7B**: `response_format={"type": "json_object"}`.

Reasoning traces are stripped **unconditionally** before JSON parsing (`core/base.py extract_json()`).

### 7.5 Empty Cell Invariance

Never insert `"N/A"`, `"Unknown"`, `"—"`, or any other placeholder into certificate fields. Verified across: `create_or_save_certificate`, `batch_save_certificates`, `restore_recycle_bin_item` in `server/main.py`.

### 7.6 No Emojis in Code

Emojis and decorative Unicode symbols are forbidden in all source files, log messages, comments, and documentation (per AGENTS.md).

### 7.7 Git Commit Threshold

Commits and `git push` are reserved for major feature completions or milestone deliveries. Not for minor UI tweaks, styling fixes, or routine file syncs.

---

## 8. Container Infrastructure

```yaml
# docker-compose.yaml
services:
  rf_postgres_db:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    volumes: ["./data/postgres:/var/lib/postgresql/data"]

  rf_app:
    image: rf-compliance-platform-app
    ports: ["8000:8000"]          # FastAPI + React SPA (port 8501 removed)
    depends_on: [db]
    volumes:
      - .:/app                   # workspace bind-mounted directly into /app
      - ./data/lookups:/app/data/lookups
      - ./data/uploads:/app/data/uploads
      - ./data/ocr_cache:/app/data/ocr_cache
      - ./data/model_cache:/app/data/model_cache
      - ./data/files:/app/data/files
    deploy.resources.reservations.devices: [nvidia gpu]
```

### Boot Sequence (`entrypoint.sh`)

```bash
1. pg_isready wait loop
2. python3 -c "from storage.database import init_db; init_db()"
3. python3 -m storage.seed_lookups   # seeds data/lookups/authorities.json (183 authorities) + suppliers.json
4. exec uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### Container Operations

```bash
# Restart app container after code changes (workspace is bind-mounted — no rebuild needed)
sudo docker restart rf_app

# Check logs
sudo docker logs --tail 30 rf_app

# Run unit tests
sudo docker exec rf_app python3 -m unittest discover tests

# Check running containers
sudo docker ps

# Full stack restart
sudo docker compose down && sudo docker compose up -d
```

---

## 9. Production Environment

| Property | Value |
|---|---|
| **Host** | GCP NVIDIA L4 GPU Instance |
| **VRAM** | 24 GB |
| **Driver** | 580.173.02 |
| **CUDA** | 13.0 |
| **OS** | Ubuntu 22.04 |
| **Docker** | 29.7.2 |
| **Working Directory** | `/home/mouadnadzi3/rf-compliance-platform` |
| **App URL** | `http://34.158.150.51:8000` |
| **Public API URL** | `PUBLIC_API_URL = http://34.158.150.51:8000` |

The workspace directory is bind-mounted to `/app` inside `rf_app`. Edit files locally, restart the container — no rebuild required unless adding new Python packages.

---

## 10. Frontend Build

The React SPA must be rebuilt whenever `frontend/src/` changes are made:

```bash
cd frontend && npm run build
# Outputs to frontend/dist/ which is served by FastAPI's StaticFiles mount
sudo docker restart rf_app
```

FastAPI mounts `frontend/dist` at `/` — the SPA handles routing via React Router. All API calls go to the same origin at `/api/v1/*`.

---

## 11. How to Run

### Production (Docker, standard)
```bash
sudo docker compose up -d
```

### Development (no Docker — run backend directly)
```bash
# Start PostgreSQL separately (or use the docker compose db service)
python3 -m storage.seed_lookups
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, run the frontend dev server
cd frontend && npm run dev
```

---

## 12. Live System State (as of 2026-09-01)

| Metric | Value |
|---|---|
| RF Certificate Records | **1,580** |
| Regulatory Authorities | **183** (all countries seeded in `data/lookups/authorities.json`) |
| Suppliers | **50+** |
| Active Model Cache | Qwen3.8-27B (11 GB) + GLM-OCR (2.5 GB) + bge-m3 (4.3 GB) = **17.8 GB** |
| Tests | **5/5 passing** (`python3 -m unittest discover tests`) |
| Git Commit | `4ae99bd` — feat: React SPA integration + Streamlit purge |

---

## 13. Major Engineering Milestones (Cumulative)

| Milestone | Status |
|---|---|
| Core FastAPI backend + PostgreSQL + pgvector schema | ✅ |
| GLM-OCR pure-HF backend (replaced vLLM) | ✅ |
| Qwen3.8-27B GGUF production LLM | ✅ |
| OCR + LLM VRAM coexistence (no sequential unloading) | ✅ |
| Deterministic 7-field extraction pipeline | ✅ |
| Bidirectional Authority ↔ Country ↔ Supplier auto-fill | ✅ |
| 183 global regulatory authority knowledge base | ✅ |
| Empty cell invariance (zero placeholder strings) | ✅ |
| Excel bulk import with LLM header mapping | ✅ |
| RAG Q&A (intent router + SQL + hybrid dense retrieval) | ✅ |
| SSE token-streaming chat | ✅ |
| Chat session persistence (PostgreSQL) | ✅ |
| Long-term AI memory (agent_memories) | ✅ |
| HITL agentic proposal pipeline | ✅ |
| Autonomous PDF scraper + scheduler | ✅ |
| Dynamic multi-table database management | ✅ |
| Recycle bin with record recovery | ✅ |
| In-app AI notification system | ✅ |
| React 18 TypeScript SPA frontend | ✅ |
| Streamlit UI fully decommissioned | ✅ |
| Port 8501 released; port 8000 only | ✅ |
| Stale model cache purged (IQ1_M 6.3 GB removed) | ✅ |
| GitHub push (`4ae99bd`) | ✅ |

---

## 14. Next Steps & Roadmap

| Priority | Feature |
|---|---|
| High | Certificate expiry alert dashboard (30/60/90 day heatmap) |
| High | Role-based access control (RBAC) |
| Medium | Email / Teams notification integration for expiry warnings |
| Medium | Certificate renewal workflow tracking |
| Medium | Multi-program portfolio analytics |
| Low | Power BI / Tableau data connector |
| Low | Multi-language UI (French, Spanish, Italian, German) |

---

## 15. Critical Files Quick Reference

| File | Description |
|---|---|
| [`server/main.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/main.py) | All API endpoints, batch save, bidirectional auto-fill logic |
| [`server/config.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/config.py) | Central config (engine selection, VRAM limits, paths) |
| [`core/extractor.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/extractor.py) | PDF ingestion pipeline + notification creation |
| [`core/llm/qwen3_8_27b.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/llm/qwen3_8_27b.py) | Active production LLM engine |
| [`core/ocr/glm_ocr.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/ocr/glm_ocr.py) | Active production OCR engine |
| [`core/rag/orchestrator.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/orchestrator.py) | RAG Q&A entry point |
| [`schemas/extraction.py`](file:///home/mouadnadzi3/rf-compliance-platform/schemas/extraction.py) | All SQLAlchemy ORM models |
| [`storage/seed_lookups.py`](file:///home/mouadnadzi3/rf-compliance-platform/storage/seed_lookups.py) | Authority/supplier knowledge base seeder |
| [`data/lookups/authorities.json`](file:///home/mouadnadzi3/rf-compliance-platform/data/lookups/authorities.json) | 183 global regulatory authority definitions |
| [`frontend/src/views/DatabasesView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/DatabasesView.tsx) | Main compliance table UI |
| [`frontend/src/components/Navbar/Navbar.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/components/Navbar/Navbar.tsx) | Notification bell + unread badge |
| [`docker-compose.yaml`](file:///home/mouadnadzi3/rf-compliance-platform/docker-compose.yaml) | Container stack definition |
| [`entrypoint.sh`](file:///home/mouadnadzi3/rf-compliance-platform/entrypoint.sh) | Container boot sequence |

---

*Last updated: 2026-09-01 | Commit: `4ae99bd` | Branch: `main`*
