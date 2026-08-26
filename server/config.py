"""
config.py — Central configuration & constants for the pipeline.

MODEL SELECTION:
    Change OCR_ENGINE or LLM_ENGINE to swap models across the entire project.
    Available keys are defined in core/registry.py.
"""

# ── Model Selection ──────────────────────────────────────────────────────────
# Which OCR engine to use (must match a key in core/registry.py)
# Options: "got-ocr2" (0.5B FP16, fast & reliable), "deepseek-ocr-2" (3B 4-bit), "glm-ocr" (0.9B, requires transformers>=5.0)
OCR_ENGINE: str = "glm-ocr"

# Which LLM engine to use (must match a key in core/registry.py)
LLM_ENGINE: str = "qwen3.8-27b-gguf"   # Options: "qwen3.8-27b-gguf", "qwen3.6-35b-gguf", "qwen3-35b", "gemma4-26b-gguf", "qwen3-8b", "qwen2-7b-gguf", "qwen3-14b-gguf", "qwen-agentworld-35b"

# Unified Embedding Model Selection
EMBEDDING_MODEL: str = "BAAI/bge-m3"
EMBEDDING_DIM: int = 1024


# ── Database Configuration ────────────────────────────────────────────────────
import os

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rf_certificates")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Shared Config ────────────────────────────────────────────────────────────
# File types recognised as processable documents
SUPPORTED_EXTENSIONS: tuple = ('.png', '.jpg', '.jpeg', '.webp', '.pdf')

# Centralized context window limit (tokens) for LLM initialization.
DEFAULT_CONTEXT_WINDOW: int = 8192

# Maximum token budget for LLM completions
DEFAULT_MAX_TOKENS: int = 8192

# ── Chat Session Context Budgeting ────────────────────────────────────────────
# Chat sessions freeze (stop accepting new turns) once the estimated cumulative
# prompt budget crosses this threshold. The threshold is intentionally below the
# hard context window so system prompts, retrieval context, and completions still
# fit without rejecting the llama_context.
CHAT_CONTEXT_WINDOW: int = DEFAULT_CONTEXT_WINDOW
CHAT_CONTEXT_FULL_THRESHOLD: int = int(DEFAULT_CONTEXT_WINDOW * 0.85)

# Fixed token overhead attributed to system prompts + retrieved context per turn.
CHAT_PROMPT_OVERHEAD_TOKENS: int = 700

# Message returned to the user once a session is frozen.
CHAT_CONTEXT_FULL_MESSAGE: str = "Context window full, open a new session."

# ── GPU VRAM Guardrails (T4: 15,360 MiB total; keep combined peak < 15 GB) ─────
# Minimum free VRAM required before starting a heavy inference stage. Below this,
# a graceful MemoryError is raised instead of an uncatchable kernel OOM.
MIN_FREE_VRAM_MB: int = 1024

# OCR generation token cap.
OCR_MAX_NEW_TOKENS: int = 8192

# Longest OCR document (chars) fed to the extraction LLM; truncation prevents an
# 8K-context overflow that would reject the llama_context.
MAX_EXTRACTION_PROMPT_CHARS: int = 20000

# Embedding encode batch size (bounds transformer activations on shared VRAM).
EMBEDDING_BATCH_SIZE: int = 16

# Device for the embedding model. "cpu" is deliberate: the L4's 23 GB is fully
# used by the co-resident GLM-OCR + Qwen engines, and OCR page generation has a
# large transient VRAM spike. Keeping bge-m3 on CPU reserves that headroom and
# avoids CUDA OOM during ingestion (embeddings only cost ~2-4 s/batch on CPU).
EMBEDDING_DEVICE: str = "cpu"

# Base project directory (project root — this file lives in the `app/` package)
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Runtime Data Directories (consolidated under data/) ──────────────────────
# Caching directory for all heavy AI models (uses HF_HOME or CACHE_DIR env var if set)
CACHE_DIR: str = os.getenv("HF_HOME", os.getenv("CACHE_DIR", os.path.join(BASE_DIR, "data", "model_cache")))

# ── Batch Ingestion Artifacts (sequential OCR  extraction lifecycle) ──────
# OCR markdown cache (survives app restarts  enables resume of interrupted batches).
OCR_CACHE_DIR: str = os.getenv("OCR_CACHE_DIR", os.path.join(BASE_DIR, "data", "ocr_cache"))
# Staging directory for uploaded files during batch ingestion.
BATCH_UPLOAD_DIR: str = os.getenv("BATCH_UPLOAD_DIR", os.path.join(BASE_DIR, "data", "uploads"))
# Persistent raw-file storage: uploaded documents are copied here and served statically
# via FastAPI on port 8000 at /files/<relative_path>.
FILES_STORAGE_DIR: str = os.getenv("FILES_STORAGE_DIR", os.path.join(BASE_DIR, "data", "files"))

# ── Public Network / API Settings ─────────────────────────────────────────────
PUBLIC_HOST: str = os.getenv("PUBLIC_HOST", "34.158.150.51")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
PUBLIC_API_URL: str = os.getenv("PUBLIC_API_URL", f"http://{PUBLIC_HOST}:{API_PORT}")

