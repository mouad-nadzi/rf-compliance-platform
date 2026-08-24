# 🚀 Model Integration Workflow for Coding Agents

**CRITICAL RULE:** Whenever you are tasked with integrating a new model (LLM, OCR, Embedding, etc.) into the registry architecture, **DO NOT** assume the current environment, dependencies, prompt templates, or sampling parameters will work out of the box. 

You must execute the following workflow **before** writing any integration code.

---

## Phase 1: Upstream Documentation Verification

You must use your web search capabilities to visit the official Hugging Face model card AND the publisher's official documentation (e.g., Unsloth, Qwen, Meta).

**Exactly what you must look for:**

1. **Quantization Compatibility & Formats:** 
   - Identify the exact quantization format (e.g., standard `Q8_0` or `Q4_K_M` vs. proprietary `UD-Q4_K_XL`).
   - Does this specific format require a bleeding-edge version of the inference engine (e.g., `llama.cpp`, `vLLM`, `transformers`)?

2. **Chat Template & Prompt Format:** 
   - What is the exact prompt formatting requirement? (e.g., ChatML `<|im_start|>`, Llama-3 `<|begin_of_text|>`).
   - Does the model require specific system prompts to function correctly?

3. **Model Modes & "Thinking" Architectures:** 
   - Does the model have a built-in reasoning or "thinking mode" (like Qwen3 generating `<think>` tags)? **How is it toggled OFF?** This varies per model — verify from the model card / chat template, do not assume:
     * **Qwen3 (8B / 14B):** soft ` /no_think` / ` /think` tags injected INSIDE the user block; non-thinking sampling `temp=0.7, top_p=0.8`; thinking sampling `temp=0.6, top_p=0.95`.
     * **Qwen3.6 / Qwen-AgentWorld (35B):** `/no_think` is **IGNORED** (model thinks by default). Non-thinking is switched by emitting an already-closed empty `<think>\n</think>` block right after the assistant header (equiv. chat-template `enable_thinking=false`).
     * **Qwen2:** no mode tags — native `response_format={"type": "json_object"}` constraint, sampling `0.1/0.7`.
     * **Gemma 4:** no soft switch — `<start_of_turn>/<end_of_turn>` turn format, fixed sampling `0.7/0.8`.
   - **CRITICAL ChatML Tag Placement:** When receiving pre-formatted ChatML prompts (containing `<|im_start|>assistant\n`), **NEVER** append mode tags (` /no_think` or ` /think`) after the assistant turn header. Mode tags must be injected **inside the user block** (before `<|im_end|>\n<|im_start|>assistant`). Appending mode tags after the assistant header garbles generation and causes 0-token / empty `{}` output.
   - **Per-model ownership (refactor):** thinking/no-thinking mode switching is NOT centralised. Each engine implements its own native template + mode switch in the `build_prompt(system_prompt, user_prompt, disable_thinking)` hook (see Phase 5). Callers pass `(system_prompt, user_prompt)` and must NOT pre-build ChatML wrappers.

4. **Sampling Parameters (CRITICAL):**
   - Are there official recommended sampling parameters for your target task?
   - Example: Qwen3 in non-thinking mode requires `temperature=0.7`, `top_p=0.8`. In thinking mode, it requires `temperature=0.6`, `top_p=0.95`. **Verify before assuming low temperature (temp=0.1) is safe.**

---

## Phase 2: Environment & Dependency Validation

1. **Pre-compiled CUDA Wheel Indexing & Version Pinning (No CPU Fallback & No Source Builds):**
   - Source compilation (`CMAKE_ARGS="-DGGML_CUDA=on"`) is **strictly forbidden** (triggers 20+ minute raw builds).
   - When installing `llama-cpp-python`, **always pin the version** (e.g., `llama-cpp-python==0.3.19`) alongside `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122`. Without version pinning, `pip` picks higher version numbers (e.g. `0.3.34`) from PyPI, silently pulling down CPU-only builds.
   - Command standard:
     ```bash
     pip uninstall -y llama-cpp-python
     pip install llama-cpp-python==0.3.19 --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
     ```
   - **Hard CUDA GPU Assertion (`llama_supports_gpu_offload`):** In `llama-cpp-python` `v0.3.x`, check `llama_supports_gpu_offload()`. Check `llama_supports_gpu()` and `llama_supports_cuda()` as fallbacks:
     ```python
     gpu_ok = (
         getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)() or
         getattr(llama_cpp, "llama_supports_gpu", lambda: False)() or
         getattr(llama_cpp, "llama_supports_cuda", lambda: False)()
     )
     ```
     If `False`, immediately raise a `RuntimeError` instructing the operator to reinstall via the CUDA 12.2 wheel index.
   - **Kernel Flush After Any C-Extension Wheel Change:** Reinstalling or upgrading `llama-cpp-python` does NOT unload the already-`dlopen()`-ed `libllama.so` from process memory. Re-importing modules in the same kernel afterwards throws `undefined symbol: llama_params_fit` / similar link errors. After ANY low-level C-extension wheel change (reinstall, version bump, force-reinstall), **restart the Colab kernel** (cold restart) before importing the engine again.

2. **Filesystem, Hardware & VRAM Allocation:**
   - Always enforce full GPU offloading via `n_gpu_layers=-1`.
   - Enable KV cache quantization (`type_k="q8_0"`, `type_v="q8_0"`) and `flash_attn=True` to fit large context windows (up to 32,768 tokens) safely within 16 GB VRAM.
   - C-level libraries (like `llama.cpp`) fail to read large binary files (>4GB) through FUSE mounts (resulting in opaque "Failed to load model" errors).
   - If on a network mount, the model file must be copied to fast local storage (e.g., `/content/model_cache/`) before loading.

---

## Phase 3: Implementation Strategy (Decoupled Architecture)

1. **Configuration Centralization:**
   - Maintain clean, static constants in `config.py` (e.g. `DEFAULT_CONTEXT_WINDOW: int = 16384`, `DEFAULT_MAX_TOKENS: int = 8192`) without unnecessary `os.getenv` boilerplate for internal static constants.
   - Reserve `os.getenv` for environment-varying secrets (e.g., `DATABASE_URL`, `POSTGRES_*`) and cache paths (`CACHE_DIR`).

2. **Maintain Colab vs. Production Decoupling:** 
   - Keep environment-specific hacks (like downloading to Google Drive and copying to local disk) **out** of the production engine files (`core/`). 
   - Handle environment adaptations in `colab/setup.sh`, Jupyter notebook setups, or generic local-caching logic driven by externally configured variables (like `CACHE_DIR`).
   - **Mandatory Handoff Report Update:** Whenever Colab-specific code, scripts, or sandbox dependencies are added or modified, update `handoff_report.md` with explicit instructions on how to remove them for production deployment.

3. **Fail Gracefully (Singleton State):** 
   - In the engine initialization code (`core/llm/__init__.py`), ensure the singleton instance is only cached **AFTER** `engine.load()` successfully completes. This prevents a failed load from caching a broken instance in memory, which would otherwise require a kernel restart to clear.

---

## Phase 4: Strict Quantization & Precision Floor (Anti-OOM Directive)

1. **Never Change Quantization Format as a Debug Solution:**
   - When troubleshooting, debugging, or resolving model initialization or runtime errors, the agent is **STRICTLY FORBIDDEN** from changing, swapping, or upgrading the target model's quantization format or precision level (e.g., upgrading from `IQ2_M`/2-bit or `Q4_K_M`/4-bit to higher precision levels like 8-bit, FP16, or unquantized weights).
2. **Prevent OOM Kernel Crashes:**
   - Upgrading precision tiers inflates VRAM and RAM footprint beyond the strict 13 GB VRAM safety threshold, triggering uncatchable Out-Of-Memory (OOM) runtime kernel crashes on Tesla T4 GPUs (~12.7 GB RAM / 16 GB VRAM budget).
3. **Debug Root Cause Directly:**
   - If a requested model quantization format fails to load (due to C++ architecture incompatibility, parameter validation errors, or missing library bindings), the agent must debug the underlying binding code, dependencies, loading flags, or environment configuration—**NEVER** swap to a heavier precision tier as a shortcut solution.
4. **Allowed Last-Resort Exception (Same-Size / Same-Quantization Publisher Swap):**
   - As a **last resort** when all root-cause code and binding fixes have been exhausted, the agent **IS PERMITTED** to attempt swapping the model download source to an equivalent repository from a different publisher (e.g., `unsloth`, `bartowski`, `6block`, `google`), provided that:
     * The model file size (in GB) remains strictly equivalent.
     * The quantization precision level (e.g., `Q8_0`, `IQ2_M`, `Q4_K_M`) matches the target tier.
   - **Order of Priority:** Always debug parameters, C++ bindings, and dependencies first; use alternative publisher repos of identical size/quantization **ONLY** as a final fallback option.

---

## Phase 5: Thinking / No-Thinking Mode Handling (Per-Model, Streamlined Single-Hook Template Method)

**Applies to:** Every LLM engine that returns a JSON answer (`core/llm/`). Per-model reasoning-mode switching is managed inside each engine's `_generate_raw()` hook, coordinated by `BaseLLMEngine.generate_json()`.

**Single Source of Truth:** `BaseLLMEngine.generate_json(system_prompt, user_prompt, disable_thinking, max_tokens)` (`core/base.py`) is a **concrete template method**. It executes the per-model `_generate_raw` hook and automatically runs `extract_json`:

1. **`_generate_raw(system_prompt, user_prompt, disable_thinking, max_tokens)`** — single abstract hook implemented by each model to format its NATIVE prompt template, apply its own think/no-think switch, select sampling parameters, and execute raw completion.
2. **`extract_json(raw_content)`** — consolidated helper function in `core/base.py` (`extract_json` / `strip_reasoning_traces`) that unconditionally scrubs reasoning traces and parses strict JSON.

Engines MUST implement `_generate_raw()` and MUST NOT re-implement JSON extraction inline or override `generate_json()` itself. JSON extraction is shared and lives in `core/base.py`. Callers pass `(system_prompt, user_prompt)` — never pre-formatted ChatML.

### Known per-model switches (as deployed)
- **Qwen3 (8B / 14B):** ChatML; soft ` /no_think` / ` /think` tag injected INSIDE the user block (never after `<|im_start|>assistant\n`). Sampling = base default.
- **Qwen3.6 / Qwen-AgentWorld (35B):** ChatML; `/no_think` is **IGNORED** (thinks by default). Non-thinking = emit an already-closed empty `<think>\n</think>` block right after the assistant header (equiv. `enable_thinking=false`). Do NOT inject `/no_think`.
- **Gemma 4 26B:** `<start_of_turn>/<end_of_turn>` turns (incl. a `<start_of_turn>system` block). No soft switch → fixed sampling `0.7/0.8`.
- **Qwen2 7B:** ChatML + native `response_format={"type": "json_object"}` (with TypeError fallback that retries without it), sampling `0.1/0.7`.

### The Qwen3 empty-JSON bug (WHY scrubbing is unconditional)
The integration uses the **raw completion API** — NO `reasoning_content`/`content` split, so the entire `<think>...</think>` trace + answer arrive in ONE string. For short JSON-only tasks the model can bury the whole answer inside the thinking block, wrap the JSON inside the trace, or duplicate it across a code fence + a `</think>` block → empty `""`/`{}`. Therefore reasoning-trace scrubbing is **unconditional (mode-agnostic)**; gating it on `disable_thinking` is exactly what let stale traces leak and produce `{}`.

### `extract_json` parse ladder (current — raw-first)
1. **Scan the RAW content** for a balanced-brace JSON object that parses — done BEFORE scrubbing, so JSON trapped inside a think trace or duplicated across a fence + `</think>` block is still recovered. This also handles thinking blocks opened with `` thinking`` but *without* a closing `` response`` marker, which persist past the scrub.
2. Direct `json.loads(raw_content)`.
3. Scrub traces (`strip_reasoning_traces`): `\s*<thinking>.*?</thinking>` XML blocks, `\s*thinking.*?response\s*` text monologues, stray `</?think>` tags, and markdown fences — then re-run direct loads + balanced scan.
4. Fallback `"{}"` (accepted last-resort guard — must never be the common path).

### FREE-FORM generation, NOT grammar
In `llama-cpp-python` v0.3.x, `response_format={"type": "json_object"}` is **silently rejected** on Qwen3-family raw completion, and a GBNF `grammar=` does constrain JSON but makes sampling ~5x slower (~1.2s → ~5.1s/query on the 99-query benchmark; ~130 ms/token vs ~22 ms/token free-form). The verified approach is FREE-FORM generation + the unconditional scrub + `extract_json` ladder (gemma4-26b: 15/15 valid at ~1.1 s/query). Exception: Qwen2's `json_object` response_format IS supported on that model and is used with a TypeError fallback.

### Hard rules
- **Single-pass, no retries:** `generate_json` makes exactly **one** inference call. Do NOT introduce `max_attempts`/retry loops — retries were trialed and rejected for masking root-cause mode-handling defects.
- **Engine contract:** `generate_json()` returns a strict JSON **string**; engines delegate to `extract_json` and never `json.loads` themselves. All engines inherit the base template method (do not override `generate_json`).
- **Follow-up verification:** after integration, `py_compile` every changed engine and confirm no stray `json.loads` / `re.sub` reasoning-scrub logic exists outside `thinking.py`.

### Benchmark Evidence (Qwen3-35B — Qwen3.6-35B-A3B, IQ2_M, single load on T4, 99-query benchmark, non-thinking mode, max_tokens=4096)
- Pre-scrub (raw `response_format` path): **83.8% (83/99)**, ~529 s.
- With GBNF `grammar=json_object_grammar()`: **99.0% (98/99)**, 692 s, ~6.99 s/query, but grammar sampling cost ~5x latency. The single remaining mismatch is an intentionally-broken/truncated query (index 84, "...what are the roll-out of?"), so 98/98 valid queries classified correctly.
- **LATENCY REVERSION (current):** the GBNF grammar was removed from ALL engines in favor of free-form generation + trace scrubbing (see "FREE-FORM generation" above) because grammar-constrained sampling measured ~130 ms/token vs ~22 ms/token free-form (~5x slower). Verified on gemma4-26b: 15/15 valid classifications at ~1.1 s/query without grammar. Engines now call the raw completion with NO `grammar=` and rely on `extract_json`'s scrub→parse→extract ladder.
- **Model load note:** the ~12 GB GGUF can be memory-mapped/loaded only **once per Colab kernel**. Dropping the instance (`_engine_instance = None`) and re-loading in the same session fails with "Failed to load model from file" because the freed 12 GB region cannot be remapped alongside lingering driver/CUDA state. To re-run the benchmark, **restart the Colab runtime** (model stays disk-cached at `/content/model_cache/`, no re-download) and load once.
