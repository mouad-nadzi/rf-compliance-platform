# Agent Protocols & System Directives

## 0. Session Start — Mandatory Context Loading
- **READ `handoff_report.md` FIRST:** At the start of every new session, the agent MUST read `/home/mouadnadzi3/rf-compliance-platform/handoff_report.md` before doing any work. It is the single source of truth for the current architecture, live production status, model registry, benchmark results, and all engineering decisions.
- **Current repo layout:** FastAPI app lives in `server/` (`server.main:app`, `server.config`), AI engines in `core/` (formerly `engines/`), schemas in `schemas/`, DB layer in `storage/`, React 18 TypeScript SPA frontend in `frontend/` (served statically by FastAPI on `:8000`; legacy Streamlit UI is decommissioned). Runtime data is consolidated under `data/`. See section 3 of the handoff report for the full tree.

## 0.1 Systemic Engineering & Problem-Solving Doctrine (Mandatory)
Every problem, bug, or feature request in this codebase MUST be approached from a high-level architectural perspective using systemic root-cause solutions. Narrow-scope workarounds and point fixes are strictly forbidden:
- **No Narrow Regex Heuristics or Point-Fix DSLs:** Never patch symptoms by adding narrow regexes (such as hardcoded greeting/anaphora word lists, regex type-guessing on untyped database values, or regex index extraction from user prompts) or custom JSON operator DSLs tailored to a single test prompt. Identify why the system failed structurally and fix the underlying architecture.
- **Long-Term Memory (`agent_memories`) is the Single Source of Truth for Prompts:** Shared agent identity, tone, platform capabilities, registered tools, and canonical table schemas (`table_schema:<physical_table_name>` enriched with live PostgreSQL column names and data types) reside exclusively in Long-Term Memory (`core/agent/memory.py::format_memories_for_prompt`) and are injected into every prompt via the `config_*` functions in `core/prompts.py`.
- **Keep Sub-Prompts Clean and Task-Specific:** Each prompt template in `core/prompts.py` must contain only its specific task role, input/output contract, and JSON schema. Never repeat global identity, capability lists, or narrow single-case guardrails across individual sub-prompts.
- **Schema-Driven & Type-Safe Database Operations:** Always resolve table names to physical PostgreSQL tables via `storage.dynamic_schema.resolve_physical_table_name`. Validate literal mutations against live `information_schema.columns` data types (`core/agent/db_editor.py::_validate_typed_literal`) without mutating values in non-date columns. Delegate complex or conditional SQL expressions to `ExecuteRawSQLTool` against the live typed schema.
- **Session-Scoped State & Deterministic Error Handling:** Never use process-global mutable singletons to share state across chat turns; derive session context deterministically from conversation history (`extract_artifacts_from_history`). Propagate tool failures via explicit exceptions rather than substring-matching error words in human-readable strings.

## 1. Primary Production Directives — GCP NVIDIA L4 Instance & Docker Architecture (Active Production Host)

### A. Host & Container Infrastructure
- **Active Production Host:** GCP NVIDIA L4 GPU Instance (24 GB VRAM, Driver `580.173.02`, CUDA `13.0`).
- **Working Directory:** `/home/mouadnadzi3/rf-compliance-platform` (bind-mounted directly to `/app` inside the `rf_app` container).
- **Container Stack (`docker-compose.yaml`):**
  - `rf_app` (`rf-compliance-platform-app`): Runs FastAPI backend and serves the compiled React SPA (`frontend/dist`) on `:8000`.
  - `rf_postgres_db` (`pgvector/pgvector:pg16`): Persistent PostgreSQL 16 database with pgvector extension (`:5432`).
- **Production Execution:** All operations, services, and API calls run natively inside the Docker container ecosystem. Colab-specific ingress workarounds (`cloudflared`/`localtunnel`) and ephemeral sandbox workarounds are completely superseded in Production.

### B. GPU VRAM & Hardware Acceleration
- **Active Hardware:** Single GCP NVIDIA L4 GPU (24 GB VRAM footprint).
- **VRAM Coexistence:** Both GLM-OCR (~2.5 GB VRAM) and Qwen3.8-27B GGUF (~11 GB VRAM) co-exist in GPU VRAM, leaving ~10.5 GB VRAM headroom.
- **Model Quantization Rules:** Model quantization format/precision (e.g. `UD-IQ3_XXS` / `UD-IQ2_M`) must remain locked during debugging. Do not upgrade to FP16 or unquantized weights without evaluating VRAM impact.

### C. Git Directives
- **No Automatic Host Sync:** There is no automated sync to the remote GCP server anymore. Files are edited directly in this workspace, which is bind-mounted to `/app` inside the `rf_app` container. Manual `scp`/SSH syncs are no longer performed. If containerized app files (`server/main.py`, `server/config.py`, `core/*`, etc.) are modified, restart the service container (`sudo docker restart rf_app`) for the changes to take effect. If `frontend/src/*` files are modified, run `cd frontend && npm run build` before restarting `rf_app`.
- **Git Commit & Push Threshold:** Do NOT commit or push for minor UI tweaks, styling adjustments, formatting fixes, or routine file syncs. Git commits and `git push origin main` are strictly reserved for **major feature implementations, core engine integrations, milestone completions, or when the user explicitly requests a commit/push**.

### D. Code Style Directives
- **No Emojis:** Never write emojis (or decorative Unicode symbols such as arrows/dingbats) in the codebase — not in code, log messages, comments, or documentation.

---

## 2. Legacy Google Colab Sandbox Directives (Historical / Development Reference Only)

### A. Guardrails & Workspace Constraints
- **Strict File Isolation:** Never execute `drive.mount()` or mount the user's global Google Drive. 
- **Working Directory:** All operations inside Google Colab sandbox happen inside `/content/Project`.
- **Targeted Cloud Cache Routing:** Routes model downloads to `/content/model_cache/`.

### B. Strict Notebook & Code Execution Rules
- **Single Notebook Enforcement:** Prototyping inside Colab sandbox uses `main.ipynb`.
- **Hardware Acceleration:** T4 GPU sandbox ceiling (13 GB VRAM).
- **Cell Manipulation:** Use `colab-mcp` tools within `main.ipynb`.

### C. Unified Cloud Environment Initialization (Trigger: "connect")
1. **Immediate Execution Loop:** When the user types the trigger word "connect", the agent must immediately execute the setup pipeline without halting or prompting with multi-choice questionnaires.
2. **Browser Connection:** Automatically execute the `open_colab_browser_connection` proxy sequence to lock onto the running T4 GPU instance.
3. **Hybrid File Sync Rule (Zip vs. Base64/Direct Update):** For full or initial workspace initialization, use `project_sync.zip`. For small-to-medium single-file or incremental code changes (e.g. updating a script or fixing a bug in `main.py`), base64 decoding or direct file-write execution cells in `main.ipynb` are permitted and encouraged to update `/content/Project/` directly without requiring full zip re-uploads.
4. **Environment Paths Setup:** Force register `os.environ["HF_HOME"] = "/content/model_cache/"` inside the cloud container to isolate all upcoming model cache fetches to the fast local NVMe disk.
5. **No Upload Cell Pollution:** Immediately wipe and purge all temporary zip extraction cells from the visible `main.ipynb` canvas as soon as the files land safely.



### D. Optimized Execution & Lazy Synchronization Workflow
- **No Automatic Background Sync:** The agent is STRICTLY FORBIDDEN from running continuous, automated background file monitoring or sync checks. Local IDE code changes must never trigger an immediate background push.
- **Just-In-Time (JIT) Pre-Execution Sync:** The agent must execute a single-batch local-to-cloud workspace sync ONLY right before it triggers a notebook cell execution. This ensures the remote `/content/Project` environment is perfectly up-to-date with the local codebase at the exact moment of running, minimizing token consumption.
- **Reactive Cloud-to-Local Pull:** Automatically detect and download generated file outputs (such as trained weights `.pkl`, metrics logs, or `.csv` files) or notebook cell changes made during manual browser debugging sessions ONLY after a cell execution completely finishes.
- **Cell Preservation:** The agent MUST NEVER delete, wipe, or overwrite existing user cells in `main.ipynb`. All notebook updates must cleanly preserve all pre-existing user cells and append/update target cells without loss of user code.
- **Exclude Setup Cells from Sync:** Temporary helper cells used exclusively for workspace file uploads or internal sandbox initialization MUST NOT be synchronized back into `main.ipynb`. Only genuine development, testing, and application execution cells should be synchronized.



## 2. Model Integration Protocol
When asked to integrate or implement any new model (LLM, OCR, Embedding, etc.) into the registry architecture of this project, you MUST refer to and strictly follow the workflow document located at:
`.agents/MODEL_INTEGRATION_WORKFLOW.md`.

Do NOT attempt to write integration code until you have executed Phase 1 and Phase 2 of that workflow.

**Two recurring failure modes to respect when touching any LLM engine** (full detail in the workflow, Phase 2 + Phase 5):
- **`llama-cpp-python` pinning (Colab/T4-era, superseded in production):** The hard pin to the prebuilt `v0.3.34-cu122` GitHub release wheel was a **Google Colab / Tesla T4 sandbox constraint**, not a production requirement. On the GCP NVIDIA L4 production host, `llama-cpp-python` may be upgraded to a current CUDA build, or — for Multi-Token Prediction (MTP) / speculative decoding, which llama.cpp does **not** expose to bindings — the LLM may be served via a recent `llama-server` sidecar with `--spec-type draft-mtp` (main model + the repo's separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` draft head). Keep any `type_k`/`type_v` as integer GGML enums and never downgrade to a build whose bundled llama.cpp lacks the GGUF architecture in use (e.g. `gemma4`, `qwen3.8`).
- **`type_k`/`type_v` must be integer GGML enums, not strings:** `llama-cpp-python` 0.3.34 requires `type_k=llama_cpp.GGML_TYPE_Q8_0` (int), NOT `type_k="q8_0"`. The string raises `TypeError` and silently triggers the no-flash-attn fallback constructor, which pads the V cache to 2048 — blowing the VRAM budget and failing `llama_context` creation (app cannot load OCR + LLM at all). See `core/llm/gemma4_26b.py`.
- **Colab Sandbox VRAM Budget vs Production (OCR + LLM):** All single-GPU 16 GB Tesla T4 memory ceilings, CPU embedding placement (`EMBEDDING_DEVICE="cpu"` in `config.py`), and context limits (`DEFAULT_CONTEXT_WINDOW=8192`) are **strictly sandbox development guardrails for Google Colab testing**. They have no bearing on enterprise production deployments (which utilize multi-GPU enterprise infrastructure, decoupled OCR microservices, or unconstrained VRAM).
- **Transient peaks kill, idle-fit is a trap (Colab Sandbox Only, 2026-08-22):** In the single-T4 Colab sandbox, GLM-OCR (4.4 GB) + Qwen3.8-27B (6.9 GB) idle-fit at **13.43 GB VRAM**, leaving **~1.48 GB free**. However, during GLM-OCR vision inference on 200 DPI PDF pages, dynamic PyTorch tensor activations spike by **~1.47 GB**, pushing total allocation past the 15.36 GB Tesla T4 capacity and triggering `CUDA out of memory` (`Tried to allocate 1.47 GiB`). Therefore, Colab sandbox testing uses the two-phase worker `_run_batch` in `main.py` with GPU cache flushes between OCR and LLM extraction to guarantee headroom on single-T4 GPUs.
- **Transient peaks kill, idle-fit is a trap:** GLM-OCR's dynamic KV cache spiked free VRAM to ~350 MiB mid-generation and crashed Gemma with `ggml_abort` even though models "fit" at idle. `config.OCR_MAX_NEW_TOKENS=8192` but the engine caps generation at `min(OCR_MAX_NEW_TOKENS, 2048)` (see `core/ocr/glm_ocr.py`) — ALWAYS keep that cap at 2048 in the single-T4 sandbox, keep the `MIN_FREE_VRAM_MB=1024` headroom guard (`core/utils/vram.py::ensure_headroom`, raises graceful `MemoryError`  HTTP 507 via `main.py`), flush the cache between OCR and extraction, and truncate extraction input via `MAX_EXTRACTION_PROMPT_CHARS=20000`. Never delete or bypass these guardrails when operating inside the single-T4 Colab sandbox.
- **Empty `{}` LLM output:** mode tags (`/no_think` / `/think`) must be injected INSIDE the user block, never after `<|im_start|>assistant\n`; reasoning traces are scrubbed unconditionally before JSON parsing (see Phase 5).

## 3. Decoupled Architecture & Production Cleanup Protocol
For **ALL** tasks you perform in this codebase:
- **Clean Isolation:** Ensure all Colab-specific environment code, notebook workarounds, comments/markdown, and sandbox development tools are cleanly isolated (e.g., inside `colab/` or `sandbox/`) and structured so they can be easily removed without affecting production modules (`core/`, `schemas/`, `server/main.py`, `server/config.py`).
- **Production Removal Documentation:** Whenever you add, update, or modify any Colab-specific environment logic, custom notebook code, or development sandbox dependencies, you **MUST** update `handoff_report.md` with explicit, step-by-step instructions detailing how to safely remove or bypass those additions during the production deployment phase.
- **Colab Decision vs. Production Mapping:** You MUST document in `handoff_report.md` all architectural and framework decisions made specifically due to Colab constraints (e.g., `localtunnel`/`cloudflared` instead of standard reverse proxies, bash-compiled PostgreSQL vs. Docker containers, local disk staging vs. direct network mount reads), and explain explicitly how those decisions should be transformed or swapped in the production environment.

## 4. System Directive for Production Environment: GCP NVIDIA L4 Host + Docker Infrastructure
You are executing, testing, and developing code for a production GCP NVIDIA L4 GPU server (`gcp-gpu-server`). You MUST strictly adhere to the following infrastructure, hardware, dependency, and deployment constraints:

### A. Production Hardware & Container Runtime
- **Hardware:** GCP NVIDIA L4 GPU instance (24 GB VRAM, Driver `580.173.02`, CUDA `13.0`).
- **Container Infrastructure:** Deployed natively via `docker-compose.yaml`.
  - `rf_app`: FastAPI backend and compiled React SPA (`:8000`).
  - `rf_postgres_db`: PostgreSQL 16 + pgvector (`:5432`).
- **Path Isolation:** All container operations map directly to `/home/mouadnadzi3/rf-compliance-platform` mounted into `/app`.

### B. Security & Secret Management
- **Zero Credential Exposure:** Never hardcode or display API keys, tokens, or plaintext credentials in code, configuration files, or logs.
- **Environment Ingestion:** Ingest configuration and environment secrets securely from environment files or OS environment variables injected into the Docker container context.

### C. Legacy Sandbox Ingress Workarounds (Superseded in Production)
- In the production GCP environment, port `8000` (FastAPI + React SPA) is served directly. Reverse tunneling workarounds (`cloudflared`/`localtunnel`) and `.streamlit/config.toml` CORS overrides required for ephemeral Colab runtimes are not required in production.
