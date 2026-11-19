# Project Handoff Report — RF Compliance Platform
**Date:** 2026-11-18  
**Status:** LIVE IN PRODUCTION on GCP NVIDIA L4 Host  
**Primary LLM:** Qwen3.8-27B GGUF UD-IQ3_XXS (`qwen3.8-27b-gguf`, 3-bit, 32k context)  
**Primary OCR:** GLM-OCR (pure-HF transformers, `glm-ocr`)  
**Frontend:** React 18 + TypeScript SPA (Vite) — Streamlit UI fully decommissioned  
**Repository:** `mouad-nadzi/rf-compliance-platform` (branch: `main`)

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
│   ├── main.py           FastAPI application (40+ REST endpoints, RBAC middleware, SSE streaming)
│   ├── auth.py           PBKDF2-SHA256 hashing, HMAC-SHA256 tokens, RBAC & agent user context
│   ├── config.py         Central config (engines, context window, paths, SMTP, auth defaults)
│   └── __init__.py
│
├── core/
│   ├── extractor.py      PDF->OCR->LLM extraction pipeline + notification creation
│   ├── prompts.py        System prompts (extraction, router, QA, agent planner)
│   ├── registry.py       Model registry (OCR_REGISTRY, LLM_REGISTRY) + lazy factory
│   ├── base.py           Abstract BaseOCREngine / BaseLLMEngine contracts
│   │
│   ├── agent/
│   │   ├── agent_loop.py      Agentic plan executor + HITL proposal staging
│   │   ├── alerting.py        Certificate expiry alerting engine (30/60/90d + idempotency)
│   │   ├── ingestor.py        Document ingestion orchestrator
│   │   ├── mailer.py          Google OAuth 2.0 Gmail REST API + SMTP fallback dispatcher
│   │   ├── memory.py          Long-term memory read/write (agent_memories table)
│   │   ├── proposals.py       HITL proposal schema, draft hydration & lifecycle
│   │   ├── report_pdf.py      Stellantis-branded A4 PDF renewal report generator
│   │   ├── scraper.py         Web scraper (HTML + optional Playwright fallback)
│   │   ├── tools.py           Registered agent tool registry (incl. InsertRecordTool)
│   │   ├── worker.py          Background APScheduler (ingestion + expiry alert scanner)
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
│   │   ├── llama_server.py    HTTP client engine for llama-server sidecar
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
│   │   ├── App.tsx             Root app with React Router & AuthProvider guards
│   │   ├── api.ts              Typed API client (all backend endpoints)
│   │   ├── index.css           Global design system & tokens
│   │   ├── main.tsx            Vite entry point
│   │   ├── context/
│   │   │   └── AuthContext.tsx Role-aware auth state (admin / editor / viewer)
│   │   ├── components/
│   │   │   ├── Navbar/         Navigation bar with notification bell, Admin-only USERS tab & sign out
│   │   │   └── Sidebar/        Collapsible sidebar (Pending Actions, Chat History, Tables)
│   │   └── views/
│   │       ├── LoginView.tsx       Sign In & Invite-Token Password Setup
│   │       ├── UsersView.tsx       USERS (Admin-Only) — invite links, roles, user removal
│   │       ├── DatabasesView.tsx   HOME — all compliance tables, import/export
│   │       ├── ChatView.tsx        ASSISTANT — per-user RAG Q&A with SSE streaming
│   │       ├── ControlView.tsx     AUTOMATIONS — scraper & expiry alerting cards
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
│   ├── test_alerting_system.py
│   ├── test_auth_and_rbac.py
│   ├── test_memory_engine.py
│   ├── test_dynamic_schema.py
│   ├── test_dag_engine.py
│   ├── test_normalization.py
│   └── test_session_and_performance.py
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
| `llama-server` | `core/llm/llama_server.py` | External `llama-server` HTTP sidecar | Optional multi-slot / MTP sidecar |

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

### 6.7 Two-Phase Intelligent Data Import & AI Diffing Pipeline

A generalized, table-agnostic data onboarding architecture designed for importing spreadsheet data into both built-in and user-defined tables:

1. **Phase 1 — Semantic Header Mapping (`POST /api/v1/databases/map-headers` / `/api/v1/certificates/map-headers`):**
   - The LLM maps incoming spreadsheet column headers to target table fields based on semantics, not rigid regex.
   - **Zero-row fallback:** Uses the table's Long-Term Memory schema profile when 0 records exist in PostgreSQL.
   - **Unmapped column guardrail:** If incoming data contains columns outside `target_fields`, the system flags them (`unmapped_columns`) and requires explicit user permission (`allow_new_columns=True`) before expanding table schema memory.

2. **Phase 2 — AI Diffing Batch Standardization (`POST /api/v1/databases/standardize-batch`):**
   - **10x Performance Breakthrough:** Replaced full-row rewrites with a patch/diffing approach. The AI only returns modified cells keyed by `_temp_id`, reducing token generation by >90% and cutting processing time from ~10 minutes to <60 seconds for 88 rows.
   - **Strict Column Type Enforcement:** Enforces valid formats based on target column names (e.g. ISO 8601 dates for date columns, E.164 for phone numbers).
   - **Cross-Column Data Rescue:** If mismatched data is detected (e.g. an email address inside a `phone` column), the system rescues the value into the `email` column if present in the target table; otherwise it clears the invalid cell.

3. **Unified Long-Term Memory Architecture (`agent_memories` table):**
   - Long-Term Memory (`core/agent/memory.py::format_memories_for_prompt`) is the **single source of truth** across all 13 prompt builders (`config_*` functions in `core/prompts.py`).
   - All table schema entries use canonical physical PostgreSQL table names (`table_schema:certificates`, `table_schema:authority_lookups`, `table_schema:supplier_lookups`, `table_schema:sources`, `table_schema:agent_memories`, `table_schema:responsibles`, plus custom tables) and are dynamically enriched at prompt-generation time with live column names and PostgreSQL data types (`information_schema.columns`).
   - Legacy `sample_formatting` and duplicate display-name schema keys (`table_schema:rf certificates`, `table_schema:authorities`, etc.) were eliminated in favor of zero-shot typed schema truth.
   - Provides **Agent Context Awareness**: every prompt (routing, SQL generation, QA synthesis, casual conversation, action planning, action synthesis, approval classification, header mapping, batch standardization, scraper relevance, data conversion, query rewriting, extraction) shares the same authoritative knowledge of agent identity, capabilities, registered tools, and typed table schemas while keeping individual sub-prompts clean and task-specific.

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
- **Qwen3.8 27B**: emit `<think>\n\n</think>\n\n` after assistant header (`disable_thinking=True`) per GGUF `tokenizer.chat_template`.
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

## 12. Live System State (as of 2026-09-05)

| Metric | Value |
|---|---|
| RF Certificate Records | **1,580** (1,314 suppliers standardized to canonical names; 18 authority/country auto-filled) |
| Regulatory Authorities | **183** (all countries seeded in `data/lookups/authorities.json`) |
| Suppliers | **50+** (alias mappings pre-cached for 100% canonical resolution) |
| Active Model Cache | Qwen3.8-27B (11 GB) + GLM-OCR (2.5 GB) + bge-m3 (4.3 GB) = **17.8 GB** |
| Tests | **21/21 passing** (`python3 -m unittest discover tests`) |
| Table Schema Memories | Unified 5-key JSON schema across all 6 core & dynamic database tables in `agent_memories` |
| Import Performance | AI Diffing batch standardizer achieves **10x speedup** (<60s for 88 rows) |
| Git Status | Clean workspace, services fully containerized on GCP L4 host |

---

## 13. Recent Architecture Enhancements (2026-09-15)

### A. Raw Query Routing & Vector Rewriter Isolation
- **Orchestrator Routing Fix:** In [`core/rag/orchestrator.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/orchestrator.py), the raw query (`clean_query`) is passed directly to the Intent Router and Action Planner. `_reformulate_query` is strictly isolated to dense vector retrieval (`search_document_chunks`), eliminating context-poisoning loops where conversational rewriters hallucinated mutation verbs onto read-only requests.

### B. Bulk Schema Modifications & Multi-Column Operations
- **Atomic Multi-Column Operations:** [`storage/dynamic_schema.py`](file:///home/mouadnadzi3/rf-compliance-platform/storage/dynamic_schema.py) and [`core/agent/tools.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/tools.py) (`ManageSchemaTool`) support `drop_columns_from_table`, `add_columns_to_table`, and `combine_columns_in_table`.
- Single-transaction multi-column drop/add/merge queries eliminate single-column operational halts.
- Scoped `certificate_id` deletion protection strictly to the core `certificates` table, giving custom tables full customization flexibility while preserving platform core integrity.

### C. Non-Blocking Fast Semantic Profiling
- **Sub-Millisecond Table Loading:** Replaced synchronous 27B LLM column profiling in `GET /api/v1/schema/tables/{table_name}/data` with [`fast_profile_table_columns`](file:///home/mouadnadzi3/rf-compliance-platform/server/main.py), reducing database table view response latency from 15-30s hanging to **37 milliseconds**.

### E. Unified RF Certificate Normalization Engine (`core/normalization.py`)
- **Centralized Normalization Rule:** Consolidated all country canonicalization (`USA`/`U.S.A.` -> `United States`, `UK` -> `United Kingdom`, `UAE` -> `United Arab Emirates`, etc.), supplier canonicalization, authority abbreviation resolution, and date formatting into [`core/normalization.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/normalization.py).
- **Universal Pathway Integration:** Integrated into PDF extraction ([`core/extractor.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/extractor.py)), batch Excel/CSV import ([`server/main.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/main.py)), single certificate save ([`server/main.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/main.py)), recycle bin recovery, and startup database seeding ([`storage/seed_lookups.py`](file:///home/mouadnadzi3/rf-compliance-platform/storage/seed_lookups.py)).
- **Database Migration:** Migrated 48 historical database records to canonical standards with zero data loss.

### F. Systemic Prompt, Tooling, Scraper, & Database Mutation Architecture (2026-11-02)
- **Single-Source-of-Truth Prompt Architecture (`core/prompts.py`, `core/agent/memory.py`, `server/main.py`):**
  - Refactored all 13 prompt builders (`config_*` in [`core/prompts.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/prompts.py)) to inject Long-Term Memory via `_with_long_term_memory()`. Sub-prompts contain only their task-specific role and JSON output contract.
  - Canonicalized all `table_schema:<physical_table_name>` entries in `agent_memories`, purged legacy display-name duplicates and nonexistent columns (`last_update`), and enriched `format_memories_for_prompt()` with live PostgreSQL column names and data types from `information_schema.columns`.
- **Schema-Typed Literal Binding & SQL Delegation (`core/agent/db_editor.py`, `core/agent/tools.py`):**
  - Removed narrow regex heuristics (`_normalize_table_name`, `_normalize_column_value`, `_is_like_filter`, `_normalize_filter_item`) from [`core/agent/db_editor.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/db_editor.py).
  - Added [`_validate_typed_literal()`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/db_editor.py#L65-L105) to validate literal mutations strictly against live PostgreSQL column data types (`date`, `integer`/`numeric`, `text`) without mutating 4-digit strings in text columns (e.g. `component='2008'`).
  - Wired [`UpdateRecordTool`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/tools.py#L931-L986) and [`DeleteRecordTool`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/tools.py#L877-L929) to delegate non-literal or conditional transformations directly to [`ExecuteRawSQLTool`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/tools.py#L783-L874), which generates standard PostgreSQL against the live typed schema using `response_prefix='{"sql_query":'`.
- **Multi-Step Write Execution & Deterministic Error Propagation (`core/rag/orchestrator.py`, `core/agent/agent_loop.py`):**
  - Updated [`answer_compliance_query_stream()`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/orchestrator.py#L766-L888) and [`_handle_chat_approval_stream()`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/orchestrator.py#L275-L349) to execute all planned `write_steps` sequentially (supporting compound operations such as `combine_columns` followed by `drop_columns`), with exception-based failure tracking (`step_failed`) and full payload preservation in [`stage_write_step()`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py#L292-L379).
- **Headless Browser Portal Scraping & Session-Scoped Artifacts (`Dockerfile`, `core/agent/scraper.py`, `core/agent/tools.py`, `core/agent/agent_loop.py`):**
  - Installed Playwright Chromium (`playwright install --with-deps chromium`) in [`Dockerfile`](file:///home/mouadnadzi3/rf-compliance-platform/Dockerfile) for JS-rendered SPA portals.
  - Scoped `_PDF_HINT_RE` in [`core/agent/scraper.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/scraper.py) to URL path+query (excluding domain names), removed duplicate `WebDownloaderTool` from `get_tool_registry()`, and replaced the process-global `_SHARED_ARTIFACTS` singleton with session-scoped [`extract_artifacts_from_history()`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/agent_loop.py#L50-L96).
- **Excel Import Preview Modal, Scoped Chat Bubble CSS, & Uniform `response_prefix` (`core/base.py`, `core/llm/*`, `frontend/`):**
  - Restored the interactive Preview & Mapping modal in [`frontend/src/views/DatabasesView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/DatabasesView.tsx), added balanced JSON array/object extraction in [`core/base.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/base.py), and normalized bidirectional header mappings in `/api/v1/databases/map-headers`.
  - Replaced inline `isUser` style overrides in [`frontend/src/components/Chat/ChatMessage.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/components/Chat/ChatMessage.tsx) and [`frontend/src/views/ChatView.css`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/ChatView.css) with scoped CSS custom properties (`--msg-text-color`, `--msg-accent-color`, `--msg-code-bg`, `--msg-quote-bg`, `--msg-border-color`).
  - Standardized `response_prefix: str = ""` across `BaseLLMEngine` and all 7 LLM engine implementations in `core/llm/`.

### G. Inference Speed Optimization & Session-Independent Chat Lifecycle (2026-11-18)
- **Native Qwen3.8 ChatML Non-Thinking Switch (`core/llm/qwen3_8_27b.py`):**
  - Inspected the GGUF `tokenizer.chat_template` embedded in `Qwen3.8-27B-UD-IQ3_XXS.gguf` and replaced the legacy ` /no_think` user-block suffix (ignored by Qwen3.8 when `response_prefix` is empty) with the native `<|im_start|>assistant\n<think>\n\n</think>\n\n{response_prefix}` generation header for `disable_thinking=True`.
  - Eliminated hidden `<think>...</think>` generation during SQL generation, SQL result synthesis, and casual replies, and added an opt-in **Deep Think** toggle in [`frontend/src/views/ChatView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/ChatView.tsx).
- **Sequence-State Prefix Checkpointing for Hybrid Recurrent (`qwen35`) Layers (`core/llm/qwen3_8_27b.py`, `core/prompts.py`, `core/agent/memory.py`):**
  - Because `Qwen3.8-27B` uses `llama_memory_hybrid` (`llama_kv_cache` + `llama_memory_recurrent` Gated DeltaNet layers), standard `llama.cpp` partial KV truncation (`kv_cache_seq_rm` with `0 < p0 < n_tokens`) returns `false` after a completion finishes, previously forcing full prompt re-evaluation on every hop.
  - Reordered [`_with_long_term_memory()`](file:///home/mouadnadzi3/rf-compliance-platform/core/prompts.py#L11-L17) so `[AGENT LONG-TERM MEMORY]` sits at token 0 of every prompt, followed by append-only conversation history segments, and implemented sequence-0 state snapshotting (`llama_state_seq_get_data` / `llama_state_seq_set_data`) at the exact prefix boundary (`n_batch=2048, n_ubatch=2048`), pre-warmed at container startup (`1014` tokens, `183.3 MB` state).
  - Deduplicated `config_sql_system_prompt()` and `config_sql_generator_system_prompt()` so they reference the cached `TABLE_SCHEMA:*` definitions in `[AGENT LONG-TERM MEMORY]` instead of repeating 900 tokens of schema descriptions in the uncached suffix.
  - Added in-memory caching with deterministic DDL/memory invalidation (`invalidate_memory_cache()`, `invalidate_schema_cache()`, `invalidate_prefix_cache()`) and bounded history compaction in [`core/utils/history.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/utils/history.py).
- **Deterministic SQL Markdown Table Rendering & Per-Hop Hardware Telemetry (`core/rag/sql_engine.py`, `core/llm/__init__.py`):**
  - [`core/rag/sql_engine.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/sql_engine.py) now renders multi-row database results deterministically via `_build_markdown_table()` while the LLM generates only a concise 1-2 sentence executive summary (`max_tokens=180`, `response_prefix='{"answer": "'`), eliminating 20-30s of row-by-row LLM table transcription.
  - Added native `llama_perf_context` per-hop telemetry (`[LLM_TELEMETRY]` logs and `timings` array on `done` events) tracking `prompt_tokens`, `cached_prefix_tokens`, `eval_prompt_tokens`, `completion_tokens`, `prefill_ms`, `decode_ms`, and `tok_per_s`.
- **Verified 4-Turn Benchmark Results on GCP NVIDIA L4:**
  - Greeting (`"hi"`): **13.93 s -> 4.79 s** (TTFT **9.36 s -> 2.46 s**, **2.9x faster**).
  - Single SQL count (`"How many certificates are there from Brazil?"`): **30.35 s -> 7.31 s** (TTFT **26.71 s -> 6.22 s**, **4.15x faster**).
  - Contextual SQL follow-up (`"and from Argentina?"`): **31.51 s -> 6.79 s** (TTFT **27.79 s -> 5.70 s**, **4.64x faster**).
  - 5-row SQL table (`"list the 5 most recent ones from Argentina with supplier and expiry"`): **54.57 s -> 9.46 s** (TTFT **31.25 s -> 6.26 s**, **5.77x faster**).
  - Cumulative 4-turn session latency: **130.36 s -> 28.35 s (4.60x end-to-end speedup)**.
- **Session-Independent Chat Lifecycle & Scoped HITL Proposals (`server/main.py`, `core/agent/proposals.py`, `frontend/`):**
  - Persists user turns immediately at stream start (`_persist_streamed_user_turn`), tracks active jobs per session (`_SESSION_ACTIVE_JOBS`, `GET /api/v1/chat/sessions/{id}/active`, `from_index` resume on `/api/v1/chat/stream/{job_id}`), and scopes HITL proposals by `session_id`.
  - Lifted `EventSource` connections and per-session stream state into [`frontend/src/components/Layout/AppLayout.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/components/Layout/AppLayout.tsx) (`sessionStreams`) so switching sessions or navigating across views preserves live thinking, status, and streaming output without cross-session bleed.

### H. Certificate Expiry Alerting System, Live SMTP, `InsertRecordTool`, & 3-Tier RBAC Authentication (2026-11-18)
- **Automated Certificate Expiry Alerting Engine ([`core/agent/alerting.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/alerting.py), [`core/agent/worker.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/worker.py), [`schemas/extraction.py`](file:///home/mouadnadzi3/rf-compliance-platform/schemas/extraction.py)):**
  - Queries `certificates` as the canonical source of truth via `LEFT JOIN LATERAL` against `components`, `responsibles`, and `countries` for upcoming expirations (`exp_date >= CURRENT_DATE AND exp_date <= CURRENT_DATE + make_interval(days => :alert_days)`), excluding permanent `2999-01-01` records.
  - Deduplicates emitted alerts in `expiry_alert_logs` on `(certificate_id, exp_date, alert_window_days)`.
  - Emits in-app notifications (`category="expiry_alert"`), generates a Stellantis-branded multi-page A4 PDF renewal report ([`core/agent/report_pdf.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/report_pdf.py)) with the Stellantis header logo and structured certificate table, drafts concise renewal alert emails with the PDF attached, and stages HITL `SEND_EMAIL` proposals (hydrated in Pending Actions with recipient, subject, body, and clickable PDF preview via `GET /api/v1/agent/drafts/{draft_id}/attachment`).
  - Exposed via `GET /api/v1/agent/alerts/config`, `POST /api/v1/agent/alerts/config`, and `POST /api/v1/agent/alerts/run` (with `force_rescan` support), with `30d`/`60d`/`90d` presets, `1-365` day lead-time adjustment, and a **Force Rescan** toggle on the **Certificate Expiry Alerting System** card in [`frontend/src/views/ControlView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/ControlView.tsx).
- **Google OAuth 2.0 Gmail API & SMTP Fallback Dispatcher ([`core/agent/mailer.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/mailer.py)):**
  - Primary transport uses **Google OAuth 2.0 + Gmail REST API** (`https://gmail.googleapis.com/gmail/v1/users/me/messages/send`) with automatic `refresh_token` rotation, Admin-only OAuth setup (`GET /api/v1/agent/alerts/oauth/start`, `GET /api/v1/auth/google/callback`, `POST /api/v1/agent/alerts/oauth/disconnect`, auto-mapping raw IPv4 origins to `<ipv4>.sslip.io` for Google redirect validation).
  - Builds full RFC 2822 MIME messages with explicit `Date`, `Message-ID`, `multipart/alternative` (plain text + HTML), and `application/pdf` attachments, falling back to STARTTLS (`587`) / SSL (`465`) SMTP when OAuth is not connected.
- **`InsertRecordTool` & Relational Schema Awareness ([`core/agent/tools.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/tools.py), [`core/agent/db_editor.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/db_editor.py)):**
  - Registered `insert_record` (`InsertRecordTool`) supporting single/multi-row inserts, case-insensitive column binding, automatic `certificates.certificate_id` generation, idempotent upserts on `responsibles` (`email`) and `components` (`componentname`), and `CURRENT DATE` injection into `_llm_plan()`.
- **Authentication, 3-Tier RBAC & Per-User Chat Isolation ([`server/auth.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/auth.py), [`server/main.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/main.py), [`frontend/src/views/UsersView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/UsersView.tsx), [`frontend/src/views/LoginView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/LoginView.tsx)):**
  - Enforces 3 roles via centralized `rbac_auth_middleware` using PBKDF2-SHA256 passwords and HMAC-SHA256 `HttpOnly` cookie + Bearer tokens:
    - `admin` (Unique Administrator, bootstrapped as `mouadnadzi3@gmail.com`): exclusive `/users` management page, invite-link generation/emailing, user removal/role switching, and Admin-only Sender Email + SMTP App Password configuration.
    - `editor` (Allow to Edit): DB writes, ingestion, automations, and proposal approvals (blocked from `/api/v1/users*` and Sender Email credentials).
    - `viewer` (Read-Only): read-only table browsing and Q&A chat (blocked from all mutations and automation triggers).
  - Scopes `chat_sessions` per `user_id` and injects `[ACTIVE USER SESSION & PERMISSIONS]` at the tail of user prompts (preserving the 1014-token `[AGENT LONG-TERM MEMORY]` prefix checkpoint) with a deterministic `viewer` guard in `answer_compliance_query_stream()`.
  - Verified with **32/32 passing unit & integration tests** (`python3 -m unittest discover tests -v`).

---

## 14. Major Engineering Milestones (Cumulative)

| Milestone | Status |
|---|---|
| Core FastAPI backend + PostgreSQL + pgvector schema | Completed |
| GLM-OCR pure-HF backend (replaced vLLM) | Completed |
| Qwen3.8-27B GGUF production LLM | Completed |
| OCR + LLM VRAM coexistence (no sequential unloading) | Completed |
| Deterministic 7-field extraction pipeline | Completed |
| Bidirectional Authority / Country / Supplier auto-fill | Completed |
| 183 global regulatory authority knowledge base | Completed |
| Empty cell invariance (zero placeholder strings) | Completed |
| Excel bulk import with LLM header mapping & preview modal | Completed |
| RAG Q&A (intent router + SQL + hybrid dense retrieval) | Completed |
| SSE token-streaming chat | Completed |
| Chat session persistence (PostgreSQL) | Completed |
| Long-term AI memory (agent_memories) | Completed |
| HITL agentic proposal pipeline | Completed |
| Autonomous PDF scraper + Playwright headless browser + scheduler | Completed |
| Dynamic multi-table database management | Completed |
| Recycle bin with record recovery | Completed |
| In-app AI notification system | Completed |
| React 18 TypeScript SPA frontend | Completed |
| Streamlit UI fully decommissioned | Completed |
| Port 8501 released; port 8000 only | Completed |
| Stale model cache purged (IQ1_M 6.3 GB removed) | Completed |
| Transparent duplicate import UX reporting (Scenario A & B) | Completed |
| Unified supplier canonicalization (PDF + CSV/Excel parity) | Completed |
| Database cleanup migration (1,314 supplier records standardized) | Completed |
| Bidirectional Authority / Country DB auto-fill (18 records updated) | Completed |
| Long-Term Memory single source of truth across all 13 prompt builders | Completed |
| AI Diffing batch standardizer (`llm_standardize_batch`) — 10x throughput boost | Completed |
| Strict column type enforcement with cross-column data rescue | Completed |
| Canonical physical-table Long-Term Memory schema with live PostgreSQL data types | Completed |
| Legacy `sample_formatting` memory purged in favor of zero-shot column intent | Completed |
| Lazy query reformulation & vector rewriter isolation | Completed |
| Bulk schema modification engine (`drop_columns`, `combine_columns`, `add_columns`) | Completed |
| Multi-step write plan execution & deterministic exception-based error handling | Completed |
| Schema-typed literal validation in `db_editor.py` + `ExecuteRawSQLTool` delegation | Completed |
| Session-scoped `extract_artifacts_from_history` (no global mutable singletons) | Completed |
| Non-blocking fast semantic column profiler (37ms table data latency) | Completed |
| 3-Tier Agent Memory synchronization & PostgreSQL `UNIQUE` key integrity | Completed |
| Unified RF Certificate Normalization Engine across all ingestion pathways (`core/normalization.py`) | Completed |
| Sequence-state prefix checkpointing (`qwen35`), native `<think>` switch, & 4.6x chat latency speedup | Completed |
| Session-independent chat streaming (`AppLayout` global stream store + backend active job reattach + scoped HITL proposals) | Completed |
| Automated Certificate Expiry Alerting System (`expiry_alert_logs`, 30/60/90d configurable lead time, Force Rescan) | Completed |
| Google OAuth 2.0 + Gmail REST API sender & Stellantis-branded PDF renewal report attachment (`core/agent/report_pdf.py`) | Completed |
| `InsertRecordTool` (`insert_record`) with multi-table relational upsert & `CURRENT DATE` planner injection | Completed |
| Authentication, 3-Tier RBAC (`admin` unique, `editor`, `viewer`), invite-link onboarding, & per-user chat isolation | Completed |

---

## 15. Next Steps & Roadmap

| Priority | Feature |
|---|---|
| High | Certificate expiry alert dashboard (30/60/90 day heatmap) |
| Medium | Certificate renewal workflow tracking |
| Medium | Multi-program portfolio analytics |
| Low | Power BI / Tableau data connector |
| Low | Multi-language UI (French, Spanish, Italian, German) |

---

## 16. Critical Files Quick Reference

| File | Description |
|---|---|
| [`server/main.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/main.py) | All API endpoints, batch save, AI Diffing standardizer, fast column profiler |
| [`server/config.py`](file:///home/mouadnadzi3/rf-compliance-platform/server/config.py) | Central config (engine selection, VRAM limits, paths) |
| [`core/normalization.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/normalization.py) | Unified RF certificate normalizer (country, supplier, authority, date, empty cell invariance) |
| [`core/extractor.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/extractor.py) | PDF ingestion pipeline + notification creation |
| [`core/llm/qwen3_8_27b.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/llm/qwen3_8_27b.py) | Active production LLM engine |
| [`core/ocr/glm_ocr.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/ocr/glm_ocr.py) | Active production OCR engine |
| [`core/rag/orchestrator.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/rag/orchestrator.py) | RAG Q&A entry point with raw query routing |
| [`core/agent/tools.py`](file:///home/mouadnadzi3/rf-compliance-platform/core/agent/tools.py) | Registered agent tool registry & bulk schema tools |
| [`storage/dynamic_schema.py`](file:///home/mouadnadzi3/rf-compliance-platform/storage/dynamic_schema.py) | Dynamic PostgreSQL DDL/DML & automatic memory sync |
| [`schemas/extraction.py`](file:///home/mouadnadzi3/rf-compliance-platform/schemas/extraction.py) | All SQLAlchemy ORM models with unique constraints |
| [`storage/seed_lookups.py`](file:///home/mouadnadzi3/rf-compliance-platform/storage/seed_lookups.py) | Authority/supplier knowledge base seeder & DB migration |
| [`data/lookups/authorities.json`](file:///home/mouadnadzi3/rf-compliance-platform/data/lookups/authorities.json) | 183 global regulatory authority definitions |
| [`frontend/src/views/DatabasesView.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/views/DatabasesView.tsx) | Main compliance table UI + import workflow |
| [`frontend/src/components/Navbar/Navbar.tsx`](file:///home/mouadnadzi3/rf-compliance-platform/frontend/src/components/Navbar/Navbar.tsx) | Notification bell + unread badge |
| [`docker-compose.yaml`](file:///home/mouadnadzi3/rf-compliance-platform/docker-compose.yaml) | Container stack definition |
| [`entrypoint.sh`](file:///home/mouadnadzi3/rf-compliance-platform/entrypoint.sh) | Container boot sequence |

---

*Last updated: 2026-11-18 | Branch: `main`*
