# Agent Protocols & System Directives

## 1. Antigravity Agent Directives - Google Colab Synchronization

### A. Guardrails & Workspace Constraints
- **Strict File Isolation:** Never execute `drive.mount()` or mount the user's global Google Drive. 
- **Working Directory:** All operations inside Google Colab must strictly happen inside the `/content/Project` directory.
- **Local Sandbox Only:** Do not read or write files outside the local project repository folder.
- **Targeted Cloud Cache Routing & Proactive OOM Prevention:** Whenever a code cell triggers a model download, file swap, or execution, ensure the system handles the data directly inside the cloud container's fast disk directory at `/content/model_cache/`. Crucially, BEFORE initializing the model loader or transferring weights, you MUST explicitly execute an aggressive GPU VRAM memory flush cell to clear pre-existing allocations and avoid OOM crashes.

### B. Strict Notebook & Code Execution Rules
- **No Local Python Execution:** Never execute `.py` scripts or test scripts on the user's local terminal machine.
- **Single Notebook Enforcement:** All code running, debugging, prototyping, and testing must happen **exclusively** inside the existing `main.ipynb` notebook file.
- **Hardware Acceleration & Rigid Size Ceilings:** You MUST ensure the Google Colab session is configured with a **T4 GPU** before running code. Crucially, when troubleshooting, debugging, or fixing model initialization errors, the agent is **STRICTLY FORBIDDEN** from changing or upgrading the model's quantization format or precision level (e.g., upgrading from 4-bit/INT4 to FP16 or unquantized weights). The active model size plus context must NEVER exceed a strict threshold of **13 GB of VRAM** to guarantee a buffer against uncatchable OOM kernel crashes. If a quantization format fails, you must debug the loading code parameters, dependencies, or configuration layers—NEVER upgrade to a heavier precision model tier.
- **No New Notebooks:** Do not create any new `.ipynb` files or temporary scratchpad notebooks in the workspace or in Colab.
- **Cell Manipulation:** Always use the `colab-mcp` tools to append, modify, or execute cells within the unified `main.ipynb` canvas.

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
- **CPU-only `llama-cpp-python`:** always install the **prebuilt v0.3.34-cu122 GitHub release wheel** (`llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl`) — do NOT use `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122` (no cp312 binary exists for 0.3.34; pip falls back to a ~15 min source compile). NEVER `pip install -U llama-cpp-python` — PyPI's higher CPU-only build (e.g. `0.3.35+`) silently wins, and after any C-extension wheel change you must restart the Colab kernel to flush the stale `libllama.so`. Do NOT downgrade to `0.3.19`: its bundled llama.cpp does not know the `gemma4` GGUF architecture used by the default `gemma4-26b-gguf` engine (`unknown model architecture: 'gemma4'`).
- **`type_k`/`type_v` must be integer GGML enums, not strings:** `llama-cpp-python` 0.3.34 requires `type_k=llama_cpp.GGML_TYPE_Q8_0` (int), NOT `type_k="q8_0"`. The string raises `TypeError` and silently triggers the no-flash-attn fallback constructor, which pads the V cache to 2048 — blowing the VRAM budget and failing `llama_context` creation (app cannot load OCR + LLM at all). See `engines/llm/gemma4_26b.py`.
- **Colab Sandbox VRAM Budget vs Production (OCR + LLM):** All single-GPU 16 GB Tesla T4 memory ceilings, CPU embedding placement (`EMBEDDING_DEVICE="cpu"` in `config.py`), and context limits (`DEFAULT_CONTEXT_WINDOW=8192`) are **strictly sandbox development guardrails for Google Colab testing**. They have no bearing on enterprise production deployments (which utilize multi-GPU enterprise infrastructure, decoupled OCR microservices, or unconstrained VRAM).
- **Transient peaks kill, idle-fit is a trap (Colab Sandbox Only, 2026-08-22):** In the single-T4 Colab sandbox, GLM-OCR (4.4 GB) + Qwen3.8-27B (6.9 GB) idle-fit at **13.43 GB VRAM**, leaving **~1.48 GB free**. However, during GLM-OCR vision inference on 200 DPI PDF pages, dynamic PyTorch tensor activations spike by **~1.47 GB**, pushing total allocation past the 15.36 GB Tesla T4 capacity and triggering `CUDA out of memory` (`Tried to allocate 1.47 GiB`). Therefore, Colab sandbox testing uses the two-phase worker `_run_batch` in `main.py` with GPU cache flushes between OCR and LLM extraction to guarantee headroom on single-T4 GPUs.
- **Transient peaks kill, idle-fit is a trap:** GLM-OCR's dynamic KV cache spiked free VRAM to ~350 MiB mid-generation and crashed Gemma with `ggml_abort` even though models "fit" at idle. `config.OCR_MAX_NEW_TOKENS=8192` but the engine caps generation at `min(OCR_MAX_NEW_TOKENS, 2048)` (see `engines/ocr/glm_ocr.py`) — ALWAYS keep that cap at 2048 in the single-T4 sandbox, keep the `MIN_FREE_VRAM_MB=1024` headroom guard (`engines/utils/vram.py::ensure_headroom`, raises graceful `MemoryError` → HTTP 507 via `main.py`), flush the cache between OCR and extraction, and truncate extraction input via `MAX_EXTRACTION_PROMPT_CHARS=20000`. Never delete or bypass these guardrails when operating inside the single-T4 Colab sandbox.
- **Empty `{}` LLM output:** mode tags (`/no_think` / `/think`) must be injected INSIDE the user block, never after `<|im_start|>assistant\n`; reasoning traces are scrubbed unconditionally before JSON parsing (see Phase 5).

## 3. Decoupled Architecture & Production Cleanup Protocol
For **ALL** tasks you perform in this codebase:
- **Clean Isolation:** Ensure all Colab-specific environment code, notebook workarounds, comments/markdown, and sandbox development tools are cleanly isolated (e.g., inside `colab/` or `sandbox/`) and structured so they can be easily removed without affecting production modules (`engines/`, `schemas/`, `main.py`, `config.py`).
- **Production Removal Documentation:** Whenever you add, update, or modify any Colab-specific environment logic, custom notebook code, or development sandbox dependencies, you **MUST** update `handoff_report.md` with explicit, step-by-step instructions detailing how to safely remove or bypass those additions during the production deployment phase.
- **Colab Decision vs. Production Mapping:** You MUST document in `handoff_report.md` all architectural and framework decisions made specifically due to Colab constraints (e.g., `localtunnel`/`cloudflared` instead of standard reverse proxies, bash-compiled PostgreSQL vs. Docker containers, local disk staging vs. direct network mount reads), and explain explicitly how those decisions should be transformed or swapped in the production environment.

## 4. System Directive for Environment: Google Colab + Streamlit + API Deployment
You are executing, testing, and developing code for an ephemeral Google Colab managed runtime. You MUST strictly adhere to the following infrastructure, hardware, dependency, network, and security constraints:

### A. Hardware, Memory & Package Collision Limits
- **Resource Constraints:** RAM is strictly capped (~12.7 GB) and GPU VRAM is limited (Tesla T4 16 GB). Intensive memory operations can trigger uncatchable OOM kernel crashes. Keep batching small and manage memory footprint aggressively.
- **Package Collisions & Version Drift:** Colab pre-installs older package versions that cause silent dependency collisions (e.g., outdated PyPI wheels). Always check for conflicts and force upgrade (`pip install -U ...`) when necessary — **EXCEPT for `llama-cpp-python`**, which must NEVER be upgraded blindly: pin `llama-cpp-python==0.3.34` together with `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122` (unpinned, PyPI's higher CPU-only build silently wins and runs on CPU; downgrading below 0.3.34 breaks the `gemma4` GGUF architecture of the default `gemma4-26b-gguf` engine). After any C-extension wheel reinstall, restart the Colab kernel to flush the stale `libllama.so` shared-library handle. Document all version constraints.
- **Docker Non-Usability in Colab:** Docker containers cannot run natively inside standard Google Colab runtimes. Infrastructure components (such as PostgreSQL 16 + pgvector) must be compiled/installed directly via local bash scripts (`colab/setup.sh`) during development/testing, while maintaining clean production configuration (like `docker-compose.yaml`).
- **Targeted Cloud Container Caching:** Cloud Direct configuration routes all model downloads directly to `/content/model_cache/` on the fast container disk, bypassing network mounts completely.

### B. Security & Secret Management
- **Zero Credential Exposure:** Never hardcode or display API keys, tokens, or plaintext credentials in code, notebook cells, outputs, or git commits.
- **Secure Secret Ingestion:** Retrieve platform secrets securely via Google Colab's encrypted engine:
  ```python
  from google.colab import userdata
  import os
  os.environ["YOUR_API_KEY"] = userdata.get('YOUR_API_KEY')
  ```
- **Streamlit Ingestion:** Inject these environment variables natively into the active OS environment block before initializing or calling sub-processes.

### C. Streamlit Networking, CORS & Ingress Workarounds
- **Closed Ingress Ports:** Public ingress to local ports (e.g., Streamlit 8501, FastAPI 8000) is structurally blocked by Google's firewall infrastructure.
- **Headless Enforcement:** Always initialize Streamlit servers headlessly:
  ```bash
  streamlit run ui/app.py --server.headless true --server.port 8501
  ```
- **Reverse Tunneling:** Expose interfaces publicly using routing providers like `cloudflared` or `localtunnel` to create an accessible bridge to port 8501 / FastAPI endpoints, bypassing HTML block screens and firewall blocks.
- **WebSocket & CORS Overrides:** Colab regularly drops standard WebSockets and blocks cross-origin traffic, resulting in stuck "Please Wait" screens or 403 Forbidden errors. Automatically generate a `.streamlit/config.toml` file before boot-up containing:
  ```toml
  [server]
  headless = true
  enableCORS = false
  enableXsrfProtection = false
  ```
