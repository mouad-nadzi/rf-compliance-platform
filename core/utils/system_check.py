"""
core/utils/system_check.py — System readiness verification & initialization gate.

Single source of truth for "is this deployment ready to ingest documents?".
Used by:
  - GET /api/v1/system/check         full report (Streamlit sidebar + debugging)
  - POST /api/v1/system/init         idempotent init event (dirs + DB tables)
  - batch/parse guards               hard-fail before any heavy GPU action if
                                     required pieces (Postgres, tables) are down.

Checks (each returns a dict with `ok` and `detail`):
  - database    : PostgreSQL reachable, pgvector extension present, tables exist.
  - dirs        : model cache, OCR cache, batch upload dirs exist & are writable.
  - llm_model   : configured GGUF weight present in the local model cache.
  - ocr_model   : configured OCR model present in the local HF cache.
  - config      : OCR_ENGINE / LLM_ENGINE resolve to real registry entries.
"""

import logging
import os
import shutil
from typing import Any, Dict

from server import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────────


def check_database() -> Dict[str, Any]:
    """Verify PostgreSQL connectivity, pgvector, and the ORM tables."""
    result = {"ok": False, "detail": "", "required": True}
    try:
        from storage import init_db
        if not init_db():
            result["detail"] = (
                f"PostgreSQL not ready at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}. "
                "Run the environment setup (colab/setup.sh or docker-compose up -d) before using the app."
            )
            return result

        from storage.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            tables = {
                r[0] for r in conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
                ).fetchall()
            }
        missing = sorted({"certificates", "certificate_chunks"} - tables)
        if missing:
            result["detail"] = f"Missing tables: {', '.join(missing)}"
            return result

        result["ok"] = True
        result["detail"] = (
            f"PostgreSQL connected ({config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}), "
            "pgvector enabled, tables present."
        )
    except Exception as exc:  # noqa: BLE001 - report any connectivity failure
        result["detail"] = f"Database check failed: {exc}"
    return result


def check_dirs() -> Dict[str, Any]:
    """Verify required runtime directories exist and are writable."""
    paths = {
        "model_cache": config.CACHE_DIR,
        "ocr_cache": config.OCR_CACHE_DIR,
        "batch_uploads": config.BATCH_UPLOAD_DIR,
    }
    missing, unwritable = [], []
    for name, path in paths.items():
        os.makedirs(path, exist_ok=True)
        if not os.path.isdir(path):
            missing.append(name)
        elif not os.access(path, os.W_OK):
            unwritable.append(name)
    result = {
        "ok": not missing and not unwritable,
        "detail": "",
        "required": True,
    }
    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if unwritable:
        parts.append(f"not writable: {', '.join(unwritable)}")
    result["detail"] = ("; ".join(parts) + ".") if parts else "All runtime directories present and writable."
    return result


def check_llm_model() -> Dict[str, Any]:
    """Verify the configured GGUF weight exists locally (size > 500 MB)."""
    result = {"ok": False, "detail": "", "required": False}

    # Resolve the concrete engine class without loading weights.
    try:
        from core.registry import get_llm_engine
        engine = get_llm_engine(config.LLM_ENGINE)
    except Exception as exc:  # noqa: BLE001
        result["detail"] = f"Unknown LLM_ENGINE '{config.LLM_ENGINE}': {exc}"
        return result

    filename = getattr(engine, "FILENAME", None)
    if not filename:
        result["ok"] = True
        result["detail"] = f"LLM engine '{config.LLM_ENGINE}' has no local-weight requirement."
        return result

    found, total_bytes = _find_cached_file(config.CACHE_DIR, filename)
    if found and total_bytes > 500 * 1024 * 1024:
        result["ok"] = True
        result["detail"] = f"GGUF cached: {filename} ({total_bytes / (1024**3):.2f} GB)."
    elif found:
        result["detail"] = (
            f"GGUF '{filename}' found but looks truncated ({total_bytes / (1024**2):.0f} MB < 500 MB). "
            "It will be re-downloaded on first use."
        )
    else:
        result["detail"] = (
            f"GGUF '{filename}' not in cache. It will be downloaded (~9.3 GB) on the first batch "
            "— the first run will be slower."
        )
    return result


def check_ocr_model() -> Dict[str, Any]:
    """Verify the configured OCR model is present in the local HF cache."""
    result = {"ok": False, "detail": "", "required": False}
    try:
        from core.registry import get_ocr_engine
        engine = get_ocr_engine(config.OCR_ENGINE)
    except Exception as exc:  # noqa: BLE001
        result["detail"] = f"Unknown OCR_ENGINE '{config.OCR_ENGINE}': {exc}"
        return result

    model_id = getattr(engine, "MODEL_ID", "") or getattr(engine, "MODEL_NAME", "")
    if not model_id:
        result["ok"] = True
        result["detail"] = f"OCR engine '{config.OCR_ENGINE}' has no local-weight requirement."
        return result

    marker = _hf_cache_folder(config.CACHE_DIR, model_id)
    if marker and any(os.listdir(marker)):
        result["ok"] = True
        result["detail"] = f"OCR model cached: {model_id}."
    else:
        result["detail"] = (
            f"OCR model '{model_id}' not in cache. It will be downloaded on the first batch."
        )
    return result


def check_system_tools() -> Dict[str, Any]:
    """Verify system utilities (poppler-utils) and document parsing dependencies."""
    problems = []
    if not shutil.which("pdftoppm"):
        problems.append("Missing system package 'poppler-utils' (pdftoppm binary not found in PATH)")
    try:
        import pdf2image  # noqa: F401
    except ImportError:
        problems.append("Missing Python package 'pdf2image'")

    return {
        "ok": not problems,
        "detail": ("; ".join(problems) + ". Run setup.sh to install.") if problems else "poppler-utils and pdf2image available.",
        "required": True,
    }


def check_config() -> Dict[str, Any]:
    """Verify configured engine keys resolve in the registries."""
    from core.registry import OCR_REGISTRY, LLM_REGISTRY
    problems = []
    if config.OCR_ENGINE not in OCR_REGISTRY:
        problems.append(f"OCR_ENGINE='{config.OCR_ENGINE}' not in registry ({sorted(OCR_REGISTRY)})")
    if config.LLM_ENGINE not in LLM_REGISTRY:
        problems.append(f"LLM_ENGINE='{config.LLM_ENGINE}' not in registry ({sorted(LLM_REGISTRY)})")
    return {
        "ok": not problems,
        "detail": ("; ".join(problems) + ".") if problems else "Engine keys resolve in registry.",
        "required": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation & helpers
# ──────────────────────────────────────────────────────────────────────────────


def _find_cached_file(cache_dir: str, filename: str) -> tuple[bool, int]:
    """Best-effort search for `filename` under `cache_dir` (HF snapshot layout)."""
    if not os.path.isdir(cache_dir):
        return False, 0
    for root, _dirs, files in os.walk(cache_dir):
        if filename in files:
            try:
                return True, os.path.getsize(os.path.join(root, filename))
            except OSError:
                return True, 0
    return False, 0


def _hf_cache_folder(cache_dir: str, model_id: str) -> str | None:
    """Return the HF hub snapshot folder for a model id ('' if absent)."""
    safe = model_id.replace("/", "--")
    base = os.path.join(cache_dir, f"models--{safe}")
    if not os.path.isdir(base):
        return None
    snapshots = os.path.join(base, "snapshots")
    if os.path.isdir(snapshots):
        try:
            latest = sorted(os.listdir(snapshots))
            if latest:
                return os.path.join(snapshots, latest[-1])
        except OSError:
            pass
    return base


def run_system_check() -> Dict[str, Any]:
    """Run every check and aggregate into a single readiness report."""
    checks = {
        "config": check_config(),
        "system_tools": check_system_tools(),
        "dirs": check_dirs(),
        "database": check_database(),
        "llm_model": check_llm_model(),
        "ocr_model": check_ocr_model(),
    }
    all_ok = all(c["ok"] for c in checks.values())
    return {
        "ready": all_ok,
        "summary": (
            " All systems ready." if all_ok
            else " Not fully ready — review the details below before processing documents."
        ),
        "checks": checks,
    }


def ensure_ready() -> Dict[str, Any]:
    """
    Gate used by batch/parse endpoints. Runs the check; raises HTTPException(503)
    when a REQUIRED check fails so the app never starts a GPU job on a broken
    environment. Model-weight checks are informational (lazy download), so they
    do not block.
    """
    from fastapi import HTTPException
    report = run_system_check()
    failed = {
        name: c["detail"]
        for name, c in report["checks"].items()
        if c.get("required") and not c["ok"]
    }
    if failed:
        details = "; ".join(f"{k}: {v}" for k, v in failed.items())
        raise HTTPException(
            status_code=503,
            detail=f"System not ready. Fix before retrying: {details}",
        )
    return report


def initialize_system() -> Dict[str, Any]:
    """
    The explicit init event: idempotently create runtime dirs and database
    tables, then return a full readiness report. Model weights remain lazy
    (downloaded on first batch) to keep startup instant.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OCR_CACHE_DIR, exist_ok=True)
    os.makedirs(config.BATCH_UPLOAD_DIR, exist_ok=True)

    from storage import init_db
    db_ok = init_db()

    report = run_system_check()
    report["init"] = {"db_tables": db_ok}
    return report
