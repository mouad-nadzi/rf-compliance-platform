"""
main.py — Automotive Certificate Ingestion API

Production usage:
    uvicorn server.main:app --host 0.0.0.0 --port 8000

How it works:
    1. Initializes a FastAPI application with configured OCR + LLM models loaded on startup.
    2. Exposes a POST /api/v1/parse endpoint to accept document uploads.
    3. Runs OCR to get layout-aware Markdown with page tags.
    4. Extracts structured metadata directly from the document.
    5. Persists metadata and 1024-d chunk embeddings into PostgreSQL (pgvector).
    6. Returns a JSON response with validated certificate metadata and DB record IDs.

Model Selection:
    The active OCR and LLM engines are controlled by config.py (OCR_ENGINE / LLM_ENGINE).
    See core/registry.py for available options.
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
import shutil

from storage.database import get_db, SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from server import config

# ──────────────────────────────────────────────────────────────────────────────
# Global State & Config
# ──────────────────────────────────────────────────────────────────────────────

_ocr_engine = None

def get_shared_ocr_engine():
    """Returns the globally loaded shared OCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        from server.config import CACHE_DIR, OCR_ENGINE
        from core.registry import get_ocr_engine
        _ocr_engine = get_ocr_engine(OCR_ENGINE)
        _ocr_engine.load(CACHE_DIR)
    return _ocr_engine

OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), "ocr_outputs")

_batch_thread: threading.Thread | None = None
_batch_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Chat Session Store (PostgreSQL-persisted history + context budgeting)
# ──────────────────────────────────────────────────────────────────────────────

_CHAT_SESSIONS: Dict[str, Dict[str, Any]] = {}
_CHAT_SESSIONS_LOCK = threading.Lock()


def _estimate_tokens(text: str) -> int:
    """Rough token estimator (~4 chars per token)."""
    return max(1, len(str(text or "")) // 4)


def _session_usage_tokens(session: Dict[str, Any], extra_text: str = "") -> int:
    """Cumulative estimated prompt budget for a session (history + overhead)."""
    base = sum(_estimate_tokens(m.get("content", "")) for m in session.get("messages", []))
    return base + config.CHAT_PROMPT_OVERHEAD_TOKENS + _estimate_tokens(extra_text)


def _load_chat_sessions_from_db() -> None:
    """Hydrates the in-memory session cache from PostgreSQL (survives restarts)."""
    from schemas.extraction import ChatMessage, ChatSession

    try:
        db = SessionLocal()
        try:
            rows = db.query(ChatSession).order_by(ChatSession.created_at).all()
            for row in rows:
                session = {
                    "id": row.id,
                    "title": row.title or "",
                    "messages": [],
                    "created_at": row.created_at.isoformat() if row.created_at else "",
                    "frozen": bool(row.frozen),
                }
                for msg in row.messages:
                    session["messages"].append(
                        {"role": msg.role, "content": msg.content}
                    )
                _CHAT_SESSIONS[session["id"]] = session
            if rows:
                logger.info(f" Loaded {len(rows)} chat session(s) from PostgreSQL.")
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f" Could not load chat sessions from PostgreSQL: {exc}")


def _create_chat_session(title: str = "") -> Dict[str, Any]:
    from schemas.extraction import ChatSession

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    session = {
        "id": session_id,
        "title": (title or "").strip()[:60],
        "messages": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen": False,
    }
    with _CHAT_SESSIONS_LOCK:
        _CHAT_SESSIONS[session_id] = session

    try:
        db = SessionLocal()
        try:
            db.add(ChatSession(id=session_id, title=session["title"]))
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f" Could not persist new chat session: {exc}")

    return session


def _get_chat_session(session_id: str) -> Dict[str, Any] | None:
    with _CHAT_SESSIONS_LOCK:
        cached = _CHAT_SESSIONS.get(session_id)
    if cached is not None:
        return cached

    from schemas.extraction import ChatSession

    try:
        db = SessionLocal()
        try:
            row = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if row is None:
                return None
            session = {
                "id": row.id,
                "title": row.title or "",
                "messages": [
                    {"role": m.role, "content": m.content} for m in row.messages
                ],
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "frozen": bool(row.frozen),
            }
            with _CHAT_SESSIONS_LOCK:
                _CHAT_SESSIONS[session_id] = session
            return session
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f" Could not load chat session {session_id}: {exc}")
        return None


def _get_or_create_chat_session(session_id: str | None) -> Dict[str, Any]:
    if session_id:
        existing = _get_chat_session(session_id)
        if existing:
            return existing
    return _create_chat_session()


def _persist_chat_turn(session_id: str, role: str, content: str, title: str | None = None, frozen: bool | None = None) -> None:
    """Persists a message turn (and optional title/frozen updates) to PostgreSQL."""
    from schemas.extraction import ChatMessage, ChatSession

    try:
        db = SessionLocal()
        try:
            row = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if row is None:
                db.close()
                return
            db.add(ChatMessage(session_id=session_id, role=role, content=content))
            if title is not None:
                row.title = title
            if frozen is not None:
                row.frozen = frozen
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f" Could not persist chat turn: {exc}")


def _persist_chat_session_frozen(session_id: str, frozen: bool) -> None:
    _persist_chat_turn(session_id, role="", content="", frozen=frozen)


# ──────────────────────────────────────────────────────────────────────────────
# API Lifespan (Startup / Shutdown)
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes when the API starts up. Creates runtime directories and initializes
    the database. OCR and LLM engines run co-resident in GPU VRAM.
    """
    from server.config import CACHE_DIR, OCR_CACHE_DIR, BATCH_UPLOAD_DIR, FILES_STORAGE_DIR

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    os.makedirs(BATCH_UPLOAD_DIR, exist_ok=True)
    os.makedirs(FILES_STORAGE_DIR, exist_ok=True)

    logger.info(" Initializing database connection & pgvector tables...")
    from storage import init_db
    init_db()

    # Restore persisted chat sessions from PostgreSQL (survives restarts).
    _load_chat_sessions_from_db()

    # Eagerly load both engines into VRAM so the first user action is fast.
    global _ocr_engine
    from server.config import OCR_ENGINE, LLM_ENGINE, CACHE_DIR
    from core.registry import get_ocr_engine
    from core.llm import load_llm_engine

    logger.info(f" Preloading OCR engine '{OCR_ENGINE}' into VRAM...")
    if _ocr_engine is None:
        _ocr_engine = get_ocr_engine(OCR_ENGINE)
        _ocr_engine.load(CACHE_DIR)
    # Share the OCR engine with the agent ingestor (step-loop write step / HITL ingest).
    from core.agent.ingestor import set_ocr_engine

    set_ocr_engine(_ocr_engine)

    logger.info(f" Preloading LLM engine '{LLM_ENGINE}' into VRAM...")
    load_llm_engine()

    logger.info("Preloading embedding model into memory...")
    from core.rag.embeddings import get_embedding_model
    get_embedding_model()

    logger.info("OCR and LLM engines resident in VRAM; embedding model in RAM.")

    # Start the autonomous background scheduler (APScheduler AsyncIOScheduler).
    # The ingestion job runs at a configurable interval; failures are contained
    # inside the job so they can never crash the API. Config (on/off, interval,
    # persistent URL) is loaded from data/agent/scheduler_config.json (editable
    # from the CONTROL page), falling back to env/config defaults.
    try:
        from core.agent.worker import start_scheduler, shutdown_scheduler

        start_scheduler()
    except Exception as exc:
        logger.error(f" Failed to start autonomous scheduler: {exc}")

    yield  # This tells FastAPI it's ready to accept requests

    logger.info(" Shutting down API...")
    try:
        shutdown_scheduler()
    except Exception as exc:
        logger.warning(f" Scheduler shutdown error: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App Initialization
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Automotive Certificate Ingestion API",
    description="Extracts structured data from automotive certificate PDFs containing one or more certificates.",
    version="3.0.0",
    lifespan=lifespan
)

# Serve raw uploaded certificate files statically at /files/<path>
from server.config import FILES_STORAGE_DIR as _FILES_STORAGE_DIR
os.makedirs(_FILES_STORAGE_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=_FILES_STORAGE_DIR), name="uploaded_files")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    """Health check endpoint to verify API connection status."""
    from server.config import OCR_ENGINE, LLM_ENGINE
    return {
        "status": "online",
        "ocr_engine": OCR_ENGINE,
        "llm_engine": LLM_ENGINE,
    }


@app.get("/api/v1/system/check")
async def system_check():
    """Full system readiness report (DB, dirs, model caches, config)."""
    from core.utils.system_check import run_system_check
    return run_system_check()


@app.post("/api/v1/system/init")
async def system_init():
    """
    The explicit initialization event: idempotently creates runtime dirs and
    database tables, then returns the full readiness report. Safe to call any
    time (no-op when already initialized).
    """
    from core.utils.system_check import initialize_system
    return initialize_system()


@app.post("/api/v1/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Accepts a single document file (PDF or image), runs it through the full pipeline:
    OCR  Boundary Detection  Split  Per-Certificate Extraction.

    Uses the sequential lifecycle: loads OCR only  OCRs  unloads OCR  loads
    LLM only  extracts. Guarantees a single GPU model resident at a time.
    """
    global _ocr_engine

    temp_file_path = None

    try:
        # 0. Readiness gate before any GPU work (DB, dirs, config must be OK).
        from core.utils.system_check import ensure_ready
        ensure_ready()

        # 1. Securely save the incoming upload to a temporary file AND persistent storage
        file_extension = os.path.splitext(file.filename)[1]
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_extension)
        content = await file.read()

        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)

        # Copy to permanent file storage for static serving
        from server.config import FILES_STORAGE_DIR, PUBLIC_API_URL
        safe_fname = os.path.basename(file.filename or "upload")
        persistent_path = os.path.join(FILES_STORAGE_DIR, "parse", safe_fname)
        os.makedirs(os.path.dirname(persistent_path), exist_ok=True)
        shutil.copy2(temp_file_path, persistent_path)
        file_public_url = f"{PUBLIC_API_URL}/files/parse/{safe_fname}"

        logger.info(f"  Processing uploaded file: {file.filename}")

        # Copy to permanent file storage for static serving
        from server.config import FILES_STORAGE_DIR, PUBLIC_API_URL
        safe_fname = os.path.basename(file.filename or "upload")
        persistent_path = os.path.join(FILES_STORAGE_DIR, "parse", safe_fname)
        os.makedirs(os.path.dirname(persistent_path), exist_ok=True)
        shutil.copy2(temp_file_path, persistent_path)
        file_public_url = f"{PUBLIC_API_URL}/files/parse/{safe_fname}"

        logger.info(f"  Processing uploaded file: {file.filename}")

        # Delegate to single canonical ingestion pipeline (Defect 4 fix)
        from core.agent.ingestor import ingest_document_file
        res = ingest_document_file(temp_file_path, source_url=file_public_url)

        return res

    except MemoryError as e:
        error_msg = f"GPU memory exhausted while processing document: {str(e)}"
        logger.error(f"  {error_msg}")
        raise HTTPException(status_code=507, detail=error_msg)

    except Exception as e:
        error_msg = f"Failed to process document: {str(e)}"
        logger.error(f"  {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ──────────────────────────────────────────────────────────────────────────────
# Sequential Two-Phase Batch Ingestion (OCR  Extraction, single model resident)
# ──────────────────────────────────────────────────────────────────────────────

def _file_hash(file_name: str) -> str:
    """Deterministic cache key for an uploaded file name."""
    return hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:20]


def _manifest_path(batch_id: str) -> str:
    from server.config import OCR_CACHE_DIR
    return os.path.join(OCR_CACHE_DIR, f"manifest_{batch_id}.json")


def _load_manifest(batch_id: str) -> dict:
    path = _manifest_path(batch_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"  Could not load manifest '{batch_id}': {e}")
    return {}


def _save_manifest(batch_id: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(_manifest_path(batch_id)), exist_ok=True)
    with open(_manifest_path(batch_id), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def _resume_batch(batch_id: str, upload_dir: str, file_names: list[str]) -> bool:
    """Scans OCR cache + manifest; returns True if a previous run exists to resume."""
    manifest = _load_manifest(batch_id)
    return bool(manifest and manifest.get("phase") in ("processing", "ocr", "extract", "done"))


def _run_batch(batch_id: str, upload_dir: str, file_names: list[str]) -> None:
    """
    Background worker for per-file batch ingestion:
      For each file: OCR (GLM-OCR)  extract (Qwen3.8-27B)  persist  next file.

    Both OCR and LLM engines are loaded once and kept resident in VRAM (they
    co-exist), so there is no need for a separate all-OCR / all-extract phase.

    Progress/resume is tracked in the manifest JSON (written after every file).
    """
    global _ocr_engine

    manifest = _load_manifest(batch_id)
    if not manifest:
        manifest = {
            "batch_id": batch_id,
            "phase": "processing",
            "total": len(file_names),
            "ocr_done": 0,
            "extract_done": 0,
            "failed": 0,
            "skipped": 0,
            "current_file": None,
            "files": {name: {"status": "pending"} for name in file_names},
        }

    from server.config import CACHE_DIR, OCR_CACHE_DIR, OCR_ENGINE
    from core.registry import get_ocr_engine
    from core.extractor import extract_certificate_data, save_certificate_to_db, find_existing_certificate
    from storage.database import SessionLocal

    db_check = SessionLocal()

    def _save():
        _save_manifest(batch_id, manifest)

    try:
        manifest["phase"] = "processing"
        _save()
        _ocr_engine = get_shared_ocr_engine()

        for name in file_names:
            entry = manifest["files"].get(name, {})
            if entry.get("status") == "extracted":
                manifest["skipped"] += 1
                continue

            # ── Anti-duplicate: skip files already ingested (case-insensitive file_name) ──
            if find_existing_certificate(db_check, file_name=name, certif_number=None, country=None) is not None:
                logger.info(f"  Skipping duplicate '{name}' (already in database).")
                entry["status"] = "skipped"
                entry["error"] = "Duplicate: file already exists in database."
                manifest["skipped"] += 1
                _save()
                continue

            manifest["current_file"] = name
            _save()

            md_path = os.path.join(OCR_CACHE_DIR, f"{_file_hash(name)}.md")

            # ── Step 1: OCR (cached markdown is reused when resuming) ────────
            if not entry.get("ocr_ok"):
                try:
                    src_path = os.path.join(upload_dir, name)
                    text = _ocr_engine.process_document(src_path, OUTPUT_FOLDER)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    # Copy raw file to permanent storage for static serving
                    from server.config import FILES_STORAGE_DIR, PUBLIC_API_URL
                    dest_storage = os.path.join(FILES_STORAGE_DIR, batch_id, name)
                    os.makedirs(os.path.dirname(dest_storage), exist_ok=True)
                    shutil.copy2(src_path, dest_storage)
                    entry["file_url"] = f"{PUBLIC_API_URL}/files/{batch_id}/{name}"
                    entry["ocr_ok"] = True
                except Exception as e:
                    entry["status"] = "failed"
                    entry["error"] = str(e)
                    manifest["failed"] += 1
                    logger.error(f"  OCR failed for '{name}': {e}")
                    _save()
                    continue

            # ── Step 2: Extraction + persistence (LLM co-resident) ──────────
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                cert = extract_certificate_data(extracted_text, name)
                file_url = entry.get("file_url") or f"/files/{batch_id}/{name}"
                cert.cert_link = file_url
                rec = save_certificate_to_db(cert, extracted_text, name)
                entry["status"] = "extracted"
                entry["certificate_id"] = rec.certificate_id
                manifest["ocr_done"] += 1
                manifest["extract_done"] += 1
                logger.info(f"  Extracted + persisted: {name} -> {rec.certificate_id}")
            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = str(e)
                manifest["failed"] += 1
                logger.error(f"  Extraction failed for '{name}': {e}")
            _save()

        manifest["phase"] = "done"
        manifest["current_file"] = None
        _save()
    except Exception as e:
        logger.error(f"  Batch '{batch_id}' crashed: {e}")
        manifest["phase"] = "error"
        manifest["error"] = str(e)
        _save()
    finally:
        db_check.close()


@app.post("/api/v1/batch/ingest")
async def batch_ingest(files: list[UploadFile] = File(...)):
    """
    Starts per-file batch ingestion.

    Each file is processed end-to-end (OCR  extraction  persist) before the
    next file begins. Both the OCR and LLM engines are loaded once and kept
    resident in VRAM, so no separate all-OCR / all-extract phases are needed.

    Returns immediately with a batch_id; poll GET /api/v1/batch/status/{batch_id}.
    """
    global _batch_thread

    with _batch_lock:
        if _batch_thread is not None and _batch_thread.is_alive():
            raise HTTPException(status_code=409, detail="A batch is already running. Poll its status first.")

        if not files:
            raise HTTPException(status_code=422, detail="No files uploaded.")

        # Readiness gate: never start a GPU job on a broken environment.
        from core.utils.system_check import ensure_ready
        ensure_ready()

        # Deterministic batch id from the file set  enables resume across app restarts.
        name_set = sorted(os.path.basename(f.filename or f"file_{i}") for i, f in enumerate(files))
        batch_id = hashlib.sha256("|".join(name_set).encode("utf-8")).hexdigest()[:12]

        from server.config import BATCH_UPLOAD_DIR
        upload_dir = os.path.join(BATCH_UPLOAD_DIR, batch_id)
        os.makedirs(upload_dir, exist_ok=True)

        for f in files:
            dest = os.path.join(upload_dir, os.path.basename(f.filename or "file"))
            with open(dest, "wb") as out:
                out.write(await f.read())

        logger.info(f" Starting batch '{batch_id}' with {len(files)} file(s).")
        # Save initial manifest synchronously so status endpoint finds it immediately
        initial_manifest = {
            "batch_id": batch_id,
            "phase": "processing",
            "total": len(name_set),
            "ocr_done": 0,
            "extract_done": 0,
            "failed": 0,
            "skipped": 0,
            "current_file": None,
            "files": {name: {"status": "pending"} for name in name_set},
        }
        _save_manifest(batch_id, initial_manifest)

        _batch_thread = threading.Thread(
            target=_run_batch,
            args=(batch_id, upload_dir, name_set),
            daemon=True,
        )
        _batch_thread.start()

        return {"batch_id": batch_id, "status": "started", "total": len(name_set)}


@app.get("/api/v1/batch/status/{batch_id}")
async def batch_status(batch_id: str):
    """Returns the live progress manifest for a batch (phase, counts, per-file status)."""
    manifest = _load_manifest(batch_id)
    if not manifest:
        return {
            "batch_id": batch_id,
            "phase": "starting",
            "running": bool(_batch_thread and _batch_thread.is_alive()),
            "total": 0,
            "ocr_done": 0,
            "extract_done": 0,
            "failed": 0,
            "skipped": 0,
            "current_file": "Initializing batch..."
        }
    manifest["running"] = bool(_batch_thread and _batch_thread.is_alive())
    return manifest


@app.get("/api/v1/batch/status")
async def batch_status_current():
    """Returns the most recent batch manifest (for the Streamlit polling loop)."""
    from server.config import OCR_CACHE_DIR
    manifests = sorted(
        (os.path.join(OCR_CACHE_DIR, f) for f in os.listdir(OCR_CACHE_DIR)
         if f.startswith("manifest_") and f.endswith(".json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not manifests:
        return {"phase": "idle", "running": False}
    with open(manifests[0], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["running"] = bool(_batch_thread and _batch_thread.is_alive())
    return manifest


@app.get("/api/v1/certificates")
async def list_certificates(
    batch_id: str = Query(None, description="Batch manifest id to filter certificates by file names."),
    db: Session = Depends(get_db),
):
    """
    Lists persisted certificates. When `batch_id` is provided, only certificates
    whose file name appears in that batch's manifest are returned, along with the
    concatenated OCR markdown of the batch's successful files (for RAG indexing).
    """
    from schemas.extraction import CertificateMetadata
    from server.config import OCR_CACHE_DIR
    from core.extractor import format_exp_date

    if batch_id:
        manifest = _load_manifest(batch_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Batch not found.")
        file_names = [
            name for name, entry in manifest.get("files", {}).items()
            if entry.get("status") == "extracted"
        ]
    else:
        file_names = None

    query = db.query(CertificateMetadata).order_by(CertificateMetadata.created_at.desc())
    if file_names is not None:
        query = query.filter(CertificateMetadata.file_name.in_(file_names))
    rows = query.all()

    certificates = []
    for r in rows:
        certificates.append({
            "certificate_id": r.certificate_id,
            "component": r.component,
            "supplier": r.supplier,
            "country": r.country,
            "certif_number": r.certif_number,
            "authority": r.authority,
            "issue_date": str(r.issue_date) if r.issue_date else None,
            "exp_date": format_exp_date(r.exp_date),
            "cert_link": r.cert_link,
            "file_name": r.file_name,
            "last_update": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else None,
        })

    raw_markdown = ""
    if batch_id and file_names:
        parts = []
        for name in file_names:
            md_path = os.path.join(OCR_CACHE_DIR, f"{_file_hash(name)}.md")
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    parts.append(f"\n\n=== {name} ===\n\n" + f.read())
        raw_markdown = "".join(parts)

    return {"certificates": certificates, "raw_markdown": raw_markdown}


@app.get("/api/v1/certificates/exists", response_model=bool)
async def check_certificate_exists(
    file_name: str = Query(..., description="Source document file name to check for duplicate ingestion."),
    db: Session = Depends(get_db),
):
    """
    Checks whether a certificate with the exact given source `file_name` has already
    been parsed and persisted to the `certificates` table. Enables duplicate
    prevention during bulk document ingestion.

    Returns:
        bool: True if at least one certificate record with this file name exists,
              False otherwise.
    """
    clean_file_name = (file_name or "").strip()
    if not clean_file_name:
        raise HTTPException(status_code=422, detail="file_name query parameter is required and cannot be blank.")

    from schemas.extraction import CertificateMetadata
    from sqlalchemy import func

    try:
        exists = (
            db.query(CertificateMetadata)
            .filter(func.lower(CertificateMetadata.file_name) == clean_file_name.lower())
            .first()
            is not None
        )
    except SQLAlchemyError as e:
        logger.error(f" Error checking certificate existence for '{clean_file_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Database error while checking file: {str(e)}")

    logger.info(f" Duplicate check for '{clean_file_name}': exists={exists}")
    return exists


@app.get("/api/v1/lookups/authorities")
async def list_authorities(db: Session = Depends(get_db)):
    """Lists authority lookup reference table entries."""
    from storage.models import AuthorityLookup
    rows = db.query(AuthorityLookup).order_by(AuthorityLookup.canonical_authority.asc()).all()
    authorities = []
    for r in rows:
        authorities.append({
            "id": r.id,
            "canonical_authority": r.canonical_authority,
            "abbreviation": r.abbreviation,
            "country": r.country,
            "standard_validity_years": r.standard_validity_years,
            "aliases": r.aliases if isinstance(r.aliases, list) else []
        })
    return {"authorities": authorities}


@app.post("/api/v1/lookups/authorities")
async def add_authority(payload: dict, db: Session = Depends(get_db)):
    """Adds a new authority lookup reference entry."""
    from storage.models import AuthorityLookup, normalize_validity_years
    try:
        record = AuthorityLookup(
            canonical_authority=payload.get("canonical_authority"),
            abbreviation=payload.get("abbreviation"),
            country=payload.get("country"),
            standard_validity_years=normalize_validity_years(payload.get("standard_validity_years")),
            aliases=payload.get("aliases") or [],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f" Added authority lookup id={record.id}")
        return {"status": "success", "id": record.id}
    except Exception as e:
        db.rollback()
        logger.error(f" Error adding authority: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/lookups/suppliers")
async def add_supplier(payload: dict, db: Session = Depends(get_db)):
    """Adds a new supplier lookup reference entry."""
    from storage.models import SupplierLookup
    try:
        record = SupplierLookup(
            canonical_supplier=payload.get("canonical_supplier"),
            aliases=payload.get("aliases") or [],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f" Added supplier lookup id={record.id}")
        return {"status": "success", "id": record.id}
    except Exception as e:
        db.rollback()
        logger.error(f" Error adding supplier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/lookups/authorities/import")
async def import_authorities_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Imports authority lookups from an uploaded CSV or Excel file."""
    from storage.models import AuthorityLookup, normalize_validity_years
    import pandas as pd
    filename_lower = (file.filename or "").lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv, .xlsx, or .xls file.")
    try:
        content = await file.read()
        file_bytes = io.BytesIO(content)
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(file_bytes, dtype=str)
        else:
            df = pd.read_excel(file_bytes, dtype=str)
        df = df.fillna("")
        imported = 0
        for _, row in df.iterrows():
            canonical = str(row.get("canonical_authority") or "").strip()
            if not canonical:
                continue
            record = AuthorityLookup(
                canonical_authority=canonical,
                abbreviation=str(row.get("abbreviation") or "").strip() or None,
                country=str(row.get("country") or "").strip(),
                standard_validity_years=normalize_validity_years(row.get("standard_validity_years")),
                aliases=[a.strip() for a in str(row.get("aliases") or "").split(",") if a.strip()],
            )
            db.add(record)
            imported += 1
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f"Imported {imported} authority lookup records.")
        return {"status": "success", "imported_count": imported}
    except Exception as e:
        db.rollback()
        logger.error(f"Authority import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/lookups/suppliers/import")
async def import_suppliers_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Imports supplier lookups from an uploaded CSV or Excel file."""
    from storage.models import SupplierLookup
    import pandas as pd
    filename_lower = (file.filename or "").lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv, .xlsx, or .xls file.")
    try:
        content = await file.read()
        file_bytes = io.BytesIO(content)
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(file_bytes, dtype=str)
        else:
            df = pd.read_excel(file_bytes, dtype=str)
        df = df.fillna("")
        imported = 0
        for _, row in df.iterrows():
            canonical = str(row.get("canonical_supplier") or "").strip()
            if not canonical:
                continue
            record = SupplierLookup(
                canonical_supplier=canonical,
                aliases=[a.strip() for a in str(row.get("aliases") or "").split(",") if a.strip()],
            )
            db.add(record)
            imported += 1
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f"Imported {imported} supplier lookup records.")
        return {"status": "success", "imported_count": imported}
    except Exception as e:
        db.rollback()
        logger.error(f"Supplier import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/lookups/suppliers")
async def list_suppliers(db: Session = Depends(get_db)):
    """Lists supplier lookup reference table entries."""
    from storage.models import SupplierLookup
    rows = db.query(SupplierLookup).order_by(SupplierLookup.canonical_supplier.asc()).all()
    suppliers = []
    for r in rows:
        suppliers.append({
            "id": r.id,
            "canonical_supplier": r.canonical_supplier,
            "aliases": r.aliases if isinstance(r.aliases, list) else []
        })
    return {"suppliers": suppliers}


@app.put("/api/v1/lookups/authorities/{auth_id}")
async def update_authority(auth_id: int, payload: dict, db: Session = Depends(get_db)):
    """Updates an authority lookup reference entry."""
    from storage.models import AuthorityLookup, normalize_validity_years
    try:
        record = db.query(AuthorityLookup).filter(AuthorityLookup.id == auth_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Authority not found")
        if "canonical_authority" in payload:
            record.canonical_authority = payload["canonical_authority"]
        if "abbreviation" in payload:
            record.abbreviation = payload["abbreviation"] or None
        if "country" in payload:
            record.country = payload["country"]
        if "standard_validity_years" in payload:
            record.standard_validity_years = normalize_validity_years(payload["standard_validity_years"])
        if "aliases" in payload:
            record.aliases = payload["aliases"] or []
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f"Updated authority lookup id={auth_id}")
        return {"status": "success", "id": auth_id}
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating authority {auth_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/lookups/suppliers/{supp_id}")
async def update_supplier(supp_id: int, payload: dict, db: Session = Depends(get_db)):
    """Updates a supplier lookup reference entry."""
    from storage.models import SupplierLookup
    try:
        record = db.query(SupplierLookup).filter(SupplierLookup.id == supp_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Supplier not found")
        if "canonical_supplier" in payload:
            record.canonical_supplier = payload["canonical_supplier"]
        if "aliases" in payload:
            record.aliases = payload["aliases"] or []
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f"Updated supplier lookup id={supp_id}")
        return {"status": "success", "id": supp_id}
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating supplier {supp_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/lookups/authorities/{auth_id}")
async def delete_authority(auth_id: int, db: Session = Depends(get_db)):
    """Deletes an authority lookup reference entry."""
    from storage.models import AuthorityLookup
    try:
        record = db.query(AuthorityLookup).filter(AuthorityLookup.id == auth_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Authority not found")
        db.delete(record)
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f" Deleted authority lookup id={auth_id}")
        return {"status": "success", "deleted_id": auth_id}
    except Exception as e:
        db.rollback()
        logger.error(f" Error deleting authority {auth_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/lookups/suppliers/{supp_id}")
async def delete_supplier(supp_id: int, db: Session = Depends(get_db)):
    """Deletes a supplier lookup reference entry."""
    from storage.models import SupplierLookup
    try:
        record = db.query(SupplierLookup).filter(SupplierLookup.id == supp_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Supplier not found")
        db.delete(record)
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f" Deleted supplier lookup id={supp_id}")
        return {"status": "success", "deleted_id": supp_id}
    except Exception as e:
        db.rollback()
        logger.error(f" Error deleting supplier {supp_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Database CRUD Operations for UI
# ──────────────────────────────────────────────────────────────────────────────

@app.put("/api/v1/certificates/{cert_id}")
async def update_certificate(cert_id: str, payload: dict, db: Session = Depends(get_db)):
    """Updates editable fields of an existing certificate."""
    from schemas.extraction import CertificateMetadata
    from datetime import date
    from core.extractor import NO_EXPIRY_DATE, is_no_expiry_marker
    try:
        record = db.query(CertificateMetadata).filter(CertificateMetadata.certificate_id == cert_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Certificate not found")
        for field in ("component", "supplier", "country", "certif_number", "authority"):
            if field in payload and payload[field] is not None:
                setattr(record, field, payload[field])
        for field in ("issue_date", "exp_date"):
            if field in payload:
                val = payload[field]
                if not val:
                    setattr(record, field, None)
                elif field == "exp_date" and is_no_expiry_marker(val):
                    setattr(record, field, NO_EXPIRY_DATE)
                else:
                    setattr(record, field, date.fromisoformat(str(val)))
        if "cert_link" in payload:
            record.cert_link = payload["cert_link"] or None
        db.commit()
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        logger.info(f"Updated certificate: {cert_id}")
        return {"status": "success", "certificate_id": cert_id}
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating certificate {cert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/certificates/{cert_id}")
async def delete_certificate(cert_id: str, db: Session = Depends(get_db)):
    """Deletes a certificate and its associated chunks from PostgreSQL."""
    from schemas.extraction import CertificateMetadata
    try:
        record = db.query(CertificateMetadata).filter(CertificateMetadata.certificate_id == cert_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        db.delete(record)
        db.commit()
        
        # Trigger automated backup
        from storage.backup import export_database_to_sql
        export_database_to_sql()
        
        logger.info(f" Deleted certificate: {cert_id}")
        return {"status": "success", "deleted_id": cert_id}
    except Exception as e:
        db.rollback()
        logger.error(f" Error deleting certificate {cert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import Response, StreamingResponse
import csv
import io
from schemas.extraction import CertificateExtractionSchema, CertificateMetadata

@app.post("/api/v1/certificates/manual")
async def add_certificate_manual(cert_data: CertificateExtractionSchema, db: Session = Depends(get_db)):
    """Manually insert a certificate record without OCR."""
    from core.extractor import save_certificate_to_db
    try:
        record = save_certificate_to_db(
            cert_data,
            raw_markdown="Manual Entry",
            file_name="Manual Entry",
            db=db,
            dedup_file_name=False,
        )
        return {"status": "success", "certificate_id": record.certificate_id}
    except Exception as e:
        logger.error(f" Error adding manual certificate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/certificates/deduplicate")
async def deduplicate_certificates(db: Session = Depends(get_db)):
    """
    Merges legacy duplicate certificate records so each identity maps to a single row.

    Identity is `certif_number` + `country` (case-insensitive) when both are known,
    falling back to `file_name` (case-insensitive, excluding synthetic "Manual Entry"
    names) only when the certificate number is unavailable. For each duplicate group
    the most recently-created record is kept; older duplicates (and their vector
    chunks) are deleted via the ORM cascade.

    Returns:
        {"status": "success", "removed": <int>}
    """
    from collections import defaultdict
    from schemas.extraction import CertificateMetadata

    def _identity_key(r):
        if r.certif_number and r.country:
            cn = str(r.certif_number).strip().lower()
            co = str(r.country).strip().lower()
            if cn and co:
                return ("cert", cn, co)
        if r.file_name and str(r.file_name).strip().lower() != "manual entry":
            return ("file", str(r.file_name).strip().lower())
        return None

    rows = db.query(CertificateMetadata).order_by(CertificateMetadata.created_at.asc()).all()

    groups: dict = defaultdict(list)
    for r in rows:
        key = _identity_key(r)
        if key is not None:
            groups[key].append(r)

    removed_ids: set = set()
    for group in groups.values():
        if len(group) > 1:
            # rows are sorted ascending by created_at -> keep the newest (last).
            for dup in group[:-1]:
                if dup.certificate_id not in removed_ids:
                    removed_ids.add(dup.certificate_id)
                    db.delete(dup)

    db.commit()
    logger.info(f" Deduplication removed {len(removed_ids)} duplicate certificate record(s).")
    return {"status": "success", "removed": len(removed_ids)}


@app.get("/api/v1/certificates/export/csv")
async def export_certificates_csv(db: Session = Depends(get_db)):
    """Exports all certificate database records to a downloadable CSV file."""
    from schemas.extraction import CertificateMetadata
    from core.extractor import format_exp_date
    try:
        records = db.query(CertificateMetadata).order_by(CertificateMetadata.created_at.desc()).all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "certificate_id", "file_name", "component", "supplier",
            "country", "certif_number", "authority", "issue_date", "exp_date", "cert_link", "created_at"
        ])
        
        for r in records:
            writer.writerow([
                r.certificate_id or "",
                r.file_name or "",
                r.component or "",
                r.supplier or "",
                r.country or "",
                r.certif_number or "",
                r.authority or "",
                r.issue_date.strftime("%Y-%m-%d") if r.issue_date else "",
                format_exp_date(r.exp_date) or "",
                r.cert_link or "",
                r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
            ])
            
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=certificates_export.csv"}
        )
    except Exception as e:
        logger.error(f" Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=f"CSV Export failed: {str(e)}")


@app.get("/api/v1/certificates/export/excel")
async def export_certificates_excel(db: Session = Depends(get_db)):
    """Exports all certificate database records to a downloadable Excel (.xlsx) spreadsheet."""
    from schemas.extraction import CertificateMetadata
    from core.extractor import format_exp_date
    import pandas as pd

    try:
        records = db.query(CertificateMetadata).order_by(CertificateMetadata.created_at.desc()).all()
        
        data_list = []
        for r in records:
            data_list.append({
                "certificate_id": r.certificate_id or "",
                "file_name": r.file_name or "",
                "component": r.component or "",
                "supplier": r.supplier or "",
                "country": r.country or "",
                "certif_number": r.certif_number or "",
                "authority": r.authority or "",
                "issue_date": r.issue_date.strftime("%Y-%m-%d") if r.issue_date else "",
                "exp_date": format_exp_date(r.exp_date) or "",
                "cert_link": r.cert_link or "",
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
            })
            
        df = pd.DataFrame(data_list)
        excel_output = io.BytesIO()
        with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Certificates")
            
        excel_bytes = excel_output.getvalue()
        excel_output.close()

        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=certificates_export.xlsx"}
        )
    except Exception as e:
        logger.error(f" Error exporting Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Excel Export failed: {str(e)}")


@app.post("/api/v1/certificates/import/csv")
@app.post("/api/v1/certificates/import/excel")
@app.post("/api/v1/certificates/import/file")
async def import_certificates_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Imports certificate records from an uploaded CSV or Excel (.xlsx, .xls) file.
    Uses the LLM to automatically detect column mappings from any file structure."""
    from core.extractor import save_certificate_to_db, _parse_iso_date
    from schemas.extraction import CertificateExtractionSchema
    from core.llm import generate_json
    import pandas as pd
    import json as _json

    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv, .xlsx, or .xls file.")

    try:
        content = await file.read()
        file_bytes = io.BytesIO(content)

        if filename_lower.endswith(".csv"):
            df = pd.read_csv(file_bytes, dtype=str)
        else:
            df = pd.read_excel(file_bytes, dtype=str)

        df = df.fillna("")
        columns = list(df.columns)
        total_rows = len(df)

        # ── Step 1: LLM detects column mapping once for the whole file ──
        sample_row = df.iloc[0].to_dict() if total_rows > 0 else {}
        sample_str = "\n".join(f"  {k}: {v}" for k, v in sample_row.items())

        mapping_prompt = f"""You are a data schema mapper. You will receive a list of CSV/Excel column names and a sample row.
Map each column to one of these target schema fields (or null if no match):
- component
- supplier
- country
- certif_number
- authority
- issue_date
- exp_date
- cert_link
- file_name

Column names: {columns}

Sample row:
{sample_str}

Return ONLY a JSON object like:
{{
  "component": "<column name or null>",
  "supplier": "<column name or null>",
  "country": "<column name or null>",
  "certif_number": "<column name or null>",
  "authority": "<column name or null>",
  "issue_date": "<column name or null>",
  "exp_date": "<column name or null>",
  "cert_link": "<column name or null>",
  "file_name": "<column name or null>"
}}"""

        logger.info(f" Asking LLM to map columns for '{file.filename}': {columns}")
        raw_mapping = generate_json(
            system_prompt="You are a precise data schema mapper. Return only valid JSON.",
            user_prompt=mapping_prompt,
            disable_thinking=True
        )

        try:
            col_map = _json.loads(raw_mapping)
        except Exception:
            import re as _re
            m = _re.search(r"\{.*\}", raw_mapping, _re.DOTALL)
            col_map = _json.loads(m.group(0)) if m else {}

        logger.info(f" LLM column mapping: {col_map}")

        # Helper to extract value by mapped column name
        def get_val(row_dict, field):
            col = col_map.get(field)
            if col and col in row_dict:
                v = str(row_dict[col]).strip()
                return v if v not in ("", "nan", "None") else None
            return None

        # ── Step 2: Process all rows using detected mapping ──
        imported_count = 0
        for _, row in df.iterrows():
            row_dict = row.to_dict()

            comp = get_val(row_dict, "component") or "Not Found"
            supp = get_val(row_dict, "supplier") or "Not Found"
            coun = get_val(row_dict, "country") or "Not Found"
            cert_num = get_val(row_dict, "certif_number") or "Not Found"
            auth = get_val(row_dict, "authority") or "Not Found"
            iss_dt = get_val(row_dict, "issue_date")
            exp_dt = get_val(row_dict, "exp_date")
            fname = get_val(row_dict, "file_name") or f"import_{file.filename}"
            cert_lnk = get_val(row_dict, "cert_link")

            cert_schema = CertificateExtractionSchema(
                component=comp,
                supplier=supp,
                country=coun,
                certif_number=cert_num,
                authority=auth,
                issue_date=iss_dt,
                exp_date=exp_dt,
                cert_link=cert_lnk
            )

            raw_mk = (
                f"# Imported Record\n"
                f"Component: {comp}\nSupplier: {supp}\nCountry: {coun}\n"
                f"Certif Number: {cert_num}\nAuthority: {auth}\n"
                f"Issue Date: {iss_dt}\nExp Date: {exp_dt}\nCert Link: {cert_lnk}"
            )

            save_certificate_to_db(
                cert_schema,
                raw_markdown=raw_mk,
                file_name=fname,
                db=db,
                dedup_file_name=False,
            )
            imported_count += 1

        logger.info(f" File Import successful: {imported_count}/{total_rows} records processed from '{file.filename}'")
        return {
            "status": "success",
            "imported_count": imported_count,
            "total_rows": total_rows,
            "filename": file.filename,
            "column_mapping": col_map
        }
    except Exception as e:
        db.rollback()
        logger.error(f" Error importing file '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"File Import failed: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive RAG Q&A Chat Endpoint
# ──────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language compliance or certificate question.")
    session_id: str | None = Field(
        default=None,
        description="Existing chat session id. If omitted or unknown, a new session is created.",
    )


@app.get("/api/v1/chat/sessions")
def list_chat_sessions():
    """Lists all chat sessions (newest first) with title, message count, and freeze status."""
    from schemas.extraction import ChatMessage, ChatSession

    try:
        db = SessionLocal()
        try:
            rows = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
            result = []
            for row in rows:
                count = db.query(ChatMessage).filter(ChatMessage.session_id == row.id).count()
                result.append({
                    "id": row.id,
                    "title": row.title or "New chat",
                    "message_count": count,
                    "frozen": bool(row.frozen),
                    "created_at": row.created_at.isoformat() if row.created_at else "",
                })
            return result
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f" Could not list chat sessions: {exc}")
        return []


@app.post("/api/v1/chat/sessions")
def create_chat_session():
    """Creates a new empty chat session."""
    return _create_chat_session()


@app.delete("/api/v1/chat/sessions/{session_id}")
def delete_chat_session(session_id: str):
    """Deletes a chat session and its history (from PostgreSQL and cache)."""
    from schemas.extraction import ChatSession

    with _CHAT_SESSIONS_LOCK:
        cached = _CHAT_SESSIONS.pop(session_id, None)
    try:
        db = SessionLocal()
        try:
            row = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if row is None and cached is None:
                raise HTTPException(status_code=404, detail="Session not found.")
            if row is not None:
                db.delete(row)
                db.commit()
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f" Could not delete chat session {session_id}: {exc}")
        if cached is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": session_id}


@app.get("/api/v1/chat/sessions/{session_id}/messages")
def get_chat_session_messages(session_id: str):
    """Returns the full message history, context usage, and freeze status for a session."""
    session = _get_chat_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "id": session["id"],
        "title": session.get("title") or "New chat",
        "frozen": session.get("frozen", False),
        "usage_tokens": _session_usage_tokens(session),
        "context_threshold": config.CHAT_CONTEXT_FULL_THRESHOLD,
        "messages": session.get("messages", []),
    }


@app.post("/api/v1/chat")
async def chat_compliance_query(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Unified compliance Q&A chat endpoint with persistent multi-session history.

    Routes queries through the central RAG orchestrator:
      1. Intent classification (METADATA_QUERY, UNSTRUCTURED_RAG, HYBRID_QUERY).
      2. Optimal execution path (Text-to-SQL, Hybrid Vector RRF, or Dual-Path).
      3. Returns citation-backed answer, router decision, sources, and latency.

    Context-window guard: a session freezes once its cumulative estimated prompt
    budget crosses `config.CHAT_CONTEXT_FULL_THRESHOLD`; further turns return the
    freeze message until the user opens a new session.
    """
    clean_query = (request.query or "").strip()
    if not clean_query:
        raise HTTPException(status_code=422, detail="query field cannot be blank.")

    session = _get_or_create_chat_session(request.session_id)

    # Frozen-session guard: no new turns accepted.
    if session.get("frozen", False):
        return {
            "answer": config.CHAT_CONTEXT_FULL_MESSAGE,
            "intent": "SESSION_FROZEN",
            "reasoning": "Context window budget exhausted; open a new session.",
            "sources": [],
            "latency_ms": 0.0,
            "session_id": session["id"],
            "frozen": True,
            "usage_tokens": _session_usage_tokens(session),
            "context_threshold": config.CHAT_CONTEXT_FULL_THRESHOLD,
        }

    # Context-window budget guard: freeze the session if the new turn would overflow.
    projected_usage = _session_usage_tokens(session, clean_query)
    if projected_usage > config.CHAT_CONTEXT_FULL_THRESHOLD:
        with _CHAT_SESSIONS_LOCK:
            session["frozen"] = True
        _persist_chat_session_frozen(session["id"], True)
        return {
            "answer": config.CHAT_CONTEXT_FULL_MESSAGE,
            "intent": "SESSION_FROZEN",
            "reasoning": "Context window budget exhausted; open a new session.",
            "sources": [],
            "latency_ms": 0.0,
            "session_id": session["id"],
            "frozen": True,
            "usage_tokens": _session_usage_tokens(session),
            "context_threshold": config.CHAT_CONTEXT_FULL_THRESHOLD,
        }

    try:
        from core.rag import answer_compliance_query
        response = answer_compliance_query(
            user_query=clean_query,
            db_session=db,
            history=list(session.get("messages", [])),
        )

        # Persist the turn into the session history.
        with _CHAT_SESSIONS_LOCK:
            session.setdefault("messages", []).append(
                {"role": "user", "content": clean_query}
            )
            session["messages"].append(
                {"role": "assistant", "content": response.get("answer", "")}
            )
            if not session.get("title"):
                session["title"] = clean_query[:60]

        # Mirror the turn to PostgreSQL so it survives backend restarts.
        is_first_turn = len(session["messages"]) == 2
        _persist_chat_turn(
            session["id"],
            "user",
            clean_query,
            title=session["title"] if is_first_turn else None,
        )
        _persist_chat_turn(session["id"], "assistant", response.get("answer", ""))

        response["session_id"] = session["id"]
        response["frozen"] = False
        response["usage_tokens"] = _session_usage_tokens(session)
        response["context_threshold"] = config.CHAT_CONTEXT_FULL_THRESHOLD
        return response
    except Exception as exc:
        logger.error(f" Error during /api/v1/chat processing: {exc}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(exc)}")


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Chat (Job-based SSE) — fixes UI freeze by decoupling inference from
# the Streamlit script run. The pipeline runs in a background thread; the UI
# consumes tokens via a tailing SSE endpoint while staying interactive.
# ──────────────────────────────────────────────────────────────────────────────

_CHAT_STREAM_JOBS: Dict[str, Dict[str, Any]] = {}
_CHAT_STREAM_JOBS_LOCK = threading.Lock()

#: In-memory store of manual autonomous-discovery runs (dispatched from the
#: CONTROL page). Completed runs are purged periodically to bound memory.
_AUTONOMOUS_RUNS: Dict[str, Dict[str, Any]] = {}
_AUTONOMOUS_RUNS_LOCK = threading.Lock()

#: Serializes LLM inference across streaming jobs. The llama-cpp engine is
#: single-residency (one process, one context) and not thread-safe, so only one
#: RAG job may generate at a time; additional jobs queue until the current one
#: finishes.
_CHAT_INFERENCE_LOCK = threading.Lock()


def _sse_event(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _persist_streamed_turn(session_id: str, user_query: str, done_event: Dict[str, Any]) -> None:
    """Records the streamed turn into the in-memory session and PostgreSQL."""
    answer = str(done_event.get("answer", ""))
    with _CHAT_SESSIONS_LOCK:
        session = _CHAT_SESSIONS.get(session_id)
        if session is not None:
            session.setdefault("messages", []).append({"role": "user", "content": user_query})
            session["messages"].append({"role": "assistant", "content": answer})
            if not session.get("title"):
                session["title"] = (user_query or "")[:60]
        is_first_turn = bool(session and len(session.get("messages", [])) == 2)
        title = session["title"] if is_first_turn else None
    _persist_chat_turn(session_id, "user", user_query, title=title)
    _persist_chat_turn(session_id, "assistant", answer)


def _run_chat_stream_job(
    job: Dict[str, Any],
    user_query: str,
    session_id: str,
    history: List[Dict[str, str]],
) -> None:
    """Background worker: runs the streaming RAG pipeline and persists the turn.

    Token/status events are appended immediately; the terminal `done` event is
    only emitted AFTER the turn has been persisted, so a client that receives
    `done` can immediately re-fetch the session history.
    """
    from core.rag import answer_compliance_query_stream

    done_event = None
    try:
        with _CHAT_INFERENCE_LOCK:
            for evt in answer_compliance_query_stream(user_query=user_query, history=history):
                if evt["type"] == "done":
                    evt["session_id"] = session_id
                    done_event = evt
                    break
                with job["cond"]:
                    job["events"].append(evt)
                    job["cond"].notify_all()
    except Exception as exc:
        logger.error(f" Chat stream job failed: {exc}")
        done_event = {
            "type": "done",
            "session_id": session_id,
            "answer": f"Query processing failed: {str(exc)}",
            "intent": "UNSTRUCTURED_RAG",
            "reasoning": "Streaming job exception.",
            "sources": [],
            "latency_ms": 0.0,
        }

    # Persist the turn BEFORE signalling completion.
    if done_event is not None:
        try:
            _persist_streamed_turn(session_id, user_query, done_event)
        except Exception as exc:
            logger.warning(f" Could not persist streamed chat turn: {exc}")

    with job["cond"]:
        if done_event is not None:
            job["events"].append(done_event)
        job["done"] = True
        job["cond"].notify_all()


def _purge_stale_chat_jobs() -> None:
    """Removes completed chat jobs to bound memory (called on new job creation)."""
    from datetime import datetime as _dt
    now = _dt.now(timezone.utc).timestamp()
    with _CHAT_STREAM_JOBS_LOCK:
        stale = [
            job_id
            for job_id, job in _CHAT_STREAM_JOBS.items()
            if job["done"] and (now - job["created_at"]) > 300
        ]
        for job_id in stale:
            _CHAT_STREAM_JOBS.pop(job_id, None)


@app.post("/api/v1/chat/stream")
def start_chat_stream(request: ChatRequest):
    """
    Starts a streaming chat job. Returns immediately with a job_id; the answer is
    consumed progressively via GET /api/v1/chat/stream/{job_id} (SSE).
    """
    clean_query = (request.query or "").strip()
    if not clean_query:
        raise HTTPException(status_code=422, detail="query field cannot be blank.")

    session = _get_or_create_chat_session(request.session_id)
    base_response = {
        "job_id": None,
        "session_id": session["id"],
        "frozen": False,
        "usage_tokens": _session_usage_tokens(session),
        "context_threshold": config.CHAT_CONTEXT_FULL_THRESHOLD,
    }

    if session.get("frozen", False):
        base_response.update({
            "frozen": True,
            "intent": "SESSION_FROZEN",
            "answer": config.CHAT_CONTEXT_FULL_MESSAGE,
        })
        return base_response

    projected_usage = _session_usage_tokens(session, clean_query)
    if projected_usage > config.CHAT_CONTEXT_FULL_THRESHOLD:
        with _CHAT_SESSIONS_LOCK:
            session["frozen"] = True
        _persist_chat_session_frozen(session["id"], True)
        base_response.update({
            "frozen": True,
            "intent": "SESSION_FROZEN",
            "answer": config.CHAT_CONTEXT_FULL_MESSAGE,
        })
        return base_response

    _purge_stale_chat_jobs()

    job_id = f"chat_{uuid.uuid4().hex[:12]}"
    job = {
        "events": [],
        "done": False,
        "cond": threading.Condition(),
        "created_at": datetime.now(timezone.utc).timestamp(),
    }
    with _CHAT_STREAM_JOBS_LOCK:
        _CHAT_STREAM_JOBS[job_id] = job

    worker = threading.Thread(
        target=_run_chat_stream_job,
        args=(job, clean_query, session["id"], list(session.get("messages", []))),
        daemon=True,
    )
    worker.start()

    base_response["job_id"] = job_id
    return base_response


@app.get("/api/v1/chat/stream/{job_id}")
def stream_chat_job(job_id: str):
    """
    Tails a streaming chat job and yields its events as Server-Sent Events.
    """

    def event_iterator():
        with _CHAT_STREAM_JOBS_LOCK:
            job = _CHAT_STREAM_JOBS.get(job_id)
        if job is None:
            yield _sse_event({"type": "error", "message": "Unknown or expired chat job."})
            return
        sent = 0
        while True:
            with job["cond"]:
                while sent >= len(job["events"]) and not job["done"]:
                    job["cond"].wait(timeout=1.0)
                new_events = job["events"][sent:]
                sent = len(job["events"])
                done = job["done"]
            for evt in new_events:
                yield _sse_event(evt)
            if done:
                break

    return StreamingResponse(event_iterator(), media_type="text/event-stream")


@app.get("/api/v1/sources")
def list_sources(db: Session = Depends(get_db)):
    """Lists all scraper source URLs (newest first)."""
    from schemas.extraction import Source

    rows = db.query(Source).order_by(Source.created_at.desc()).all()
    return {
        "sources": [
            {
                "id": r.id,
                "url": r.url,
                "description": r.description,
                "active": bool(r.active),
                "cookie_header": getattr(r, "cookie_header", None),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


@app.post("/api/v1/sources")
def add_source(payload: dict, db: Session = Depends(get_db)):
    """Adds a new scraper source URL (case-insensitive uniqueness)."""
    from schemas.extraction import Source

    url = str((payload or {}).get("url") or "").strip()
    description = str((payload or {}).get("description") or "").strip() or None
    active = bool((payload or {}).get("active", True))
    cookie_header = str((payload or {}).get("cookie_header") or "").strip() or None

    if not url:
        raise HTTPException(status_code=422, detail="'url' is required.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=422, detail="'url' must start with http:// or https://.")

    existing = db.query(Source).filter(func.lower(Source.url) == url.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Source URL already exists (id={existing.id}).")

    row = Source(url=url, description=description, active=active, cookie_header=cookie_header)
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info(f" Added scraper source id={row.id}: {url}")
    return {
        "id": row.id,
        "url": row.url,
        "description": row.description,
        "active": bool(row.active),
        "cookie_header": row.cookie_header,
    }


@app.put("/api/v1/sources/{source_id}")
def update_source(source_id: int, payload: dict, db: Session = Depends(get_db)):
    """Updates a scraper source (url, description, active, cookie_header)."""
    from schemas.extraction import Source

    row = db.query(Source).filter(Source.id == source_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found.")

    url = payload.get("url")
    if url is not None:
        url = str(url).strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="'url' must start with http:// or https://.")
        clash = db.query(Source).filter(
            Source.id != source_id, func.lower(Source.url) == url.lower()
        ).first()
        if clash:
            raise HTTPException(status_code=409, detail=f"Another source already uses URL {url}.")
        row.url = url
    if "description" in payload:
        row.description = str(payload.get("description") or "").strip() or None
    if "active" in payload:
        row.active = bool(payload.get("active"))
    if "cookie_header" in payload:
        row.cookie_header = str(payload.get("cookie_header") or "").strip() or None

    db.commit()
    logger.info(f" Updated scraper source id={source_id}.")
    return {
        "id": row.id,
        "url": row.url,
        "description": row.description,
        "active": bool(row.active),
        "cookie_header": row.cookie_header,
    }


@app.delete("/api/v1/sources/{source_id}")
def delete_source(source_id: int, db: Session = Depends(get_db)):
    """Deletes a scraper source."""
    from schemas.extraction import Source

    row = db.query(Source).filter(Source.id == source_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found.")
    db.delete(row)
    db.commit()
    logger.info(f" Deleted scraper source id={source_id}.")
    return {"status": "success", "deleted_id": source_id}


def _ingest_document_file(file_path: str, source_url: str) -> dict:
    """Delegates document ingestion to core.agent.ingestor."""
    from core.agent.ingestor import ingest_document_file
    return ingest_document_file(file_path, source_url)


# ──────────────────────────────────────────────────────────────────────────────
# Agentic HITL Proposal Routes (DB_EDIT / SEND_EMAIL approval pipeline)
# ──────────────────────────────────────────────────────────────────────────────
# Proposals are staged PENDING actions that MUST be human-approved before any
# write or dispatch executes. Nothing here mutates state without approval.


@app.get("/api/v1/agent/proposals")
def list_agent_proposals():
    """Returns all PENDING action proposals for UI rendering."""
    from core.agent.proposals import proposal_manager

    return {"proposals": proposal_manager.list_pending_proposals()}


@app.post("/api/v1/agent/proposals/{proposal_id}/approve")
def approve_agent_proposal(proposal_id: str, db: Session = Depends(get_db)):
    """
    Executes an approved proposal inside a strict transaction.

    - DB_EDIT: builds the parameter-bound mutation and runs it in a single
      PostgreSQL transaction (rollback on any error).
    - SEND_EMAIL: reads the staged draft from data/agent/drafts/ and marks it
      dispatched (no live SMTP; dispatch is a HITL-marked action).
    """
    from core.agent.proposals import proposal_manager
    from core.agent.db_editor import build_mutation_sql, execute_mutation

    proposal = proposal_manager.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found.")
    if proposal.get("status") != "PENDING":
        raise HTTPException(status_code=409, detail=f"Proposal is already {proposal.get('status')}.")

    try:
        if proposal["type"] == "DB_EDIT":
            payload = proposal.get("payload", {})
            mutation = build_mutation_sql(
                op=payload.get("op", ""),
                table=payload.get("table", ""),
                values=payload.get("values", {}),
                row_filter=payload.get("row_filter"),
            )
            rowcount = execute_mutation(db, mutation)
            detail = {
                "op": mutation["op"],
                "table": mutation["table"],
                "rowcount": rowcount,
                "preview": mutation["preview"],
            }
        elif proposal["type"] == "SEND_EMAIL":
            draft_id = proposal.get("payload", {}).get("draft_id")
            if not draft_id:
                raise HTTPException(status_code=422, detail="SEND_EMAIL proposal missing 'draft_id' in payload.")
            from core.agent.tools import DRAFTS_DIR

            draft_path = os.path.join(DRAFTS_DIR, f"{draft_id}.json")
            if not os.path.exists(draft_path):
                raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found in drafts directory.")
            with open(draft_path, encoding="utf-8") as fh:
                draft = json.load(fh)
            draft["status"] = "dispatched"
            with open(draft_path, "w", encoding="utf-8") as fh:
                json.dump(draft, fh, indent=2, ensure_ascii=False)
            detail = {
                "draft_id": draft_id,
                "recipient": draft.get("recipient"),
                "message": "Draft marked as dispatched.",
            }
        elif proposal["type"] == "INGEST_DOCUMENT":
            file_path = proposal.get("payload", {}).get("file_path")
            source_url = proposal.get("payload", {}).get("source_url")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Document file not found: {file_path}")
            detail = _ingest_document_file(file_path, source_url)
        elif proposal["type"] == "AGENT_ACTION":
            # Approving the write-step proposal dispatches the write executor
            # (OCR -> ingest into the database) in the background; read steps
            # already ran freely before the approval was requested.
            from core.agent.agent_loop import dispatch_resolve_write_step

            detail = dispatch_resolve_write_step(proposal)
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported proposal type: {proposal['type']}")

        updated = proposal_manager.update_status(proposal_id, "APPROVED")
        return {
            "status": "success",
            "proposal_id": proposal_id,
            "proposal": updated,
            "execution": detail,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f" Proposal approval failed for {proposal_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Proposal execution failed: {str(exc)}")


@app.post("/api/v1/agent/proposals/{proposal_id}/reject")
def reject_agent_proposal(proposal_id: str):
    """Marks a proposal as REJECTED with no execution."""
    from core.agent.proposals import proposal_manager

    try:
        updated = proposal_manager.update_status(proposal_id, "REJECTED")
        return {"status": "success", "proposal_id": proposal_id, "proposal": updated}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/v1/agent/drafts/{draft_id}")
def get_agent_draft(draft_id: str):
    """Returns a staged email draft by id (for the CONTROL dashboard preview)."""
    from core.agent.tools import DRAFTS_DIR

    draft_path = os.path.join(DRAFTS_DIR, f"{draft_id}.json")
    if not os.path.exists(draft_path):
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found in drafts directory.")
    with open(draft_path, encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/api/v1/agent/autonomous/config")
async def get_autonomous_config():
    """Returns the scheduled-ingestion configuration (on/off, interval, persistent URL)."""
    from core.agent.worker import get_scheduler_status

    return get_scheduler_status()


@app.post("/api/v1/agent/autonomous/config")
async def update_autonomous_config(payload: dict):
    """
    Updates the scheduled-ingestion configuration and applies it live.

    Body (all optional): {"enabled": bool, "require_approval": bool, "interval_seconds": int >= 60,
    "target_url": str}. Persisted to data/agent/scheduler_config.json so it
    survives restarts. Also affects the manual-run URL fallback.
    """
    from core.agent.worker import update_scheduler_config

    enabled = payload.get("enabled")
    require_approval = payload.get("require_approval")
    interval_seconds = payload.get("interval_seconds")
    target_url = payload.get("target_url")

    if enabled is not None and not isinstance(enabled, bool):
        raise HTTPException(status_code=422, detail="'enabled' must be a boolean.")
    if require_approval is not None and not isinstance(require_approval, bool):
        raise HTTPException(status_code=422, detail="'require_approval' must be a boolean.")
    if interval_seconds is not None:
        try:
            interval_seconds = int(interval_seconds)
            if interval_seconds < 60:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="'interval_seconds' must be an integer >= 60.")
    if target_url is not None and not isinstance(target_url, str):
        raise HTTPException(status_code=422, detail="'target_url' must be a string.")

    try:
        return update_scheduler_config(enabled, require_approval, interval_seconds, target_url)
    except Exception as exc:
        logger.error(f" Failed to update autonomous scheduler config: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/agent/autonomous/run")
def run_autonomous_ingestion(payload: dict | None = None):
    """
    Manually dispatches the autonomous PDF discovery + download job.

    Accepts an optional body: {"target_url": "https://portal/listing",
    "fetcher": "html"|"playwright", "cookie_header": "<session cookie>"}.
    When target_url is omitted, the job checks ALL active sources in the
    `sources` table. fetcher selects the fetch backend. cookie_header is a
    TRANSIENT session cookie applied in-memory only: it is never stored, logged, or
    returned by the API (the run record only carries a has_cookie flag). The job
    (discovery -> verify -> dedup -> download, per source) runs in a daemon thread
    so the request returns immediately with a run_id; progress is polled via
    GET /api/v1/agent/autonomous/runs/{run_id}.
    """
    import threading as _threading
    from core.agent.worker import autonomous_ingestion_job
    from server.config import AUTONOMOUS_INGESTION_TARGET_URL, SCRAPER_FETCHER, SCRAPER_COOKIE_HEADER

    body = payload or {}
    target_url = str(body.get("target_url") or AUTONOMOUS_INGESTION_TARGET_URL or "").strip() or None
    fetcher = str(body.get("fetcher") or SCRAPER_FETCHER or "auto").strip().lower()
    cookie_header = str(body.get("cookie_header") or SCRAPER_COOKIE_HEADER or "").strip() or None

    run_id = uuid.uuid4().hex[:12]
    run = {
        "run_id": run_id,
        "status": "RUNNING",
        "target_url": target_url,
        "fetcher": fetcher,
        "has_cookie": bool(cookie_header),  # presence only; the value is never stored
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "done": False,
    }
    with _AUTONOMOUS_RUNS_LOCK:
        _AUTONOMOUS_RUNS[run_id] = run
        _purge_stale_autonomous_runs()

    def _runner():
        try:
            import asyncio as _asyncio
            summary = _asyncio.run(
                autonomous_ingestion_job(target_url, cookie_header=cookie_header, fetcher_type=fetcher)
            )
            with _AUTONOMOUS_RUNS_LOCK:
                run["status"] = "COMPLETED" if summary is not None else "SKIPPED"
                run["result"] = summary
                run["done"] = True
        except Exception as exc:
            logger.error(f" Manual autonomous ingestion run {run_id} failed: {exc}")
            with _AUTONOMOUS_RUNS_LOCK:
                run["status"] = "ERROR"
                run["error"] = str(exc)
                run["done"] = True

    _threading.Thread(target=_runner, daemon=True).start()
    logger.info(f" Manual autonomous ingestion dispatched (run_id={run_id}, fetcher={fetcher}, has_cookie={bool(cookie_header)})")
    return {"run_id": run_id, "status": "dispatched", "target_url": target_url, "fetcher": fetcher, "has_cookie": bool(cookie_header)}


@app.get("/api/v1/agent/autonomous/runs/{run_id}")
def get_autonomous_run(run_id: str):
    """Returns the status + result of a manual autonomous discovery run."""
    with _AUTONOMOUS_RUNS_LOCK:
        run = _AUTONOMOUS_RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Autonomous run {run_id} not found.")
    return run


def _ingest_document_file(file_path: str, source_url: str) -> dict:
    """Delegates to the shared core/agent/ingestor (OCR -> extraction -> persist)."""
    from core.agent.ingestor import ingest_document_file

    return ingest_document_file(file_path, source_url)


def _purge_stale_autonomous_runs(max_age_seconds: int = 1800) -> None:
    """Drop completed/errored runs older than max_age_seconds to bound memory."""
    from datetime import datetime as _dt

    now = _dt.now(timezone.utc)
    stale = [
        rid for rid, r in _AUTONOMOUS_RUNS.items()
        if r.get("done") and (now - _dt.fromisoformat(r["created_at"])).total_seconds() > max_age_seconds
    ]
    for rid in stale:
        _AUTONOMOUS_RUNS.pop(rid, None)


# ──────────────────────────────────────────────────────────────────────────────
# WORKFLOWS API ENDPOINTS (DAG Engine Integration)
# ──────────────────────────────────────────────────────────────────────────────
_WORKFLOWS_STORE: Dict[str, Dict[str, Any]] = {}
_WORKFLOW_RUNS: Dict[str, Dict[str, Any]] = {}


@app.get("/api/v1/workflows")
def list_workflows():
    """Lists all configured DAG workflows."""
    return {"workflows": list(_WORKFLOWS_STORE.values())}


@app.post("/api/v1/workflows")
def create_or_update_workflow(payload: dict):
    """Registers or updates a DAG workflow."""
    from core.workflow.dag_engine import DAGWorkflow
    wf = DAGWorkflow.from_dict(payload)
    _WORKFLOWS_STORE[wf.workflow_id] = wf.to_dict()
    logger.info(f" Registered DAG workflow '{wf.workflow_id}': {wf.title}")
    return wf.to_dict()


@app.post("/api/v1/workflows/{workflow_id}/execute")
def execute_workflow_endpoint(workflow_id: str):
    """Triggers execution of a DAG workflow."""
    from core.workflow.dag_engine import DAGWorkflow, execute_dag_workflow

    wf_dict = _WORKFLOWS_STORE.get(workflow_id)
    if not wf_dict:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    wf = DAGWorkflow.from_dict(wf_dict)
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    run_record = {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "status": "RUNNING",
        "result": None,
    }
    _WORKFLOW_RUNS[run_id] = run_record

    def _worker():
        res = execute_dag_workflow(wf)
        _WORKFLOW_RUNS[run_id]["status"] = res.status.upper()
        _WORKFLOW_RUNS[run_id]["result"] = res.to_dict()

    _threading.Thread(target=_worker, daemon=True).start()
    return {"run_id": run_id, "workflow_id": workflow_id, "status": "DISPATCHED"}


@app.get("/api/v1/workflows/runs/{run_id}")
def get_workflow_run_status(run_id: str):
    """Fetches execution status and outputs of a workflow run."""
    run = _WORKFLOW_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Workflow run '{run_id}' not found.")
    return run


# ──────────────────────────────────────────────────────────────────────────────
# LONG-TERM MEMORY API ENDPOINTS (Agent Memory Engine Integration)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/memories")
def list_memories(category: str | None = None):
    """Lists all active agent long-term memories."""
    from core.agent.memory import get_active_memories
    return {"memories": get_active_memories(category=category)}


@app.post("/api/v1/memories")
def save_memory_endpoint(payload: dict):
    """Saves a new persistent long-term memory fact."""
    from core.agent.memory import save_agent_memory
    key = str((payload or {}).get("memory_key") or "preference")
    fact = str((payload or {}).get("fact_text") or "").strip()
    session_id = (payload or {}).get("source_session_id")
    if not fact:
        raise HTTPException(status_code=422, detail="'fact_text' is required.")
    return save_agent_memory(key, fact, source_session_id=session_id)


@app.delete("/api/v1/memories/{memory_id}")
def delete_memory_endpoint(memory_id: int):
    """Deletes a long-term memory fact by ID."""
    from core.agent.memory import delete_agent_memory
    success = delete_agent_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")
    return {"status": "deleted", "id": memory_id}


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC SCHEMA & CUSTOM TABLE API ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/schema/tables")
def list_schema_tables():
    """Lists all database tables and column schema metadata."""
    from storage import dynamic_schema
    tables = dynamic_schema.list_all_user_tables()
    for t in tables:
        t["columns"] = dynamic_schema.get_table_columns(t["table_name"])
    return {"tables": tables}


@app.post("/api/v1/schema/tables")
def create_schema_table(payload: dict):
    """Creates a new custom database table."""
    from storage import dynamic_schema
    table_name = str((payload or {}).get("table_name") or "").strip()
    columns = (payload or {}).get("columns") or []
    if not table_name:
        raise HTTPException(status_code=422, detail="'table_name' is required.")
    try:
        return dynamic_schema.create_custom_table(table_name, columns)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/v1/schema/tables/{table_name}")
def drop_schema_table(table_name: str):
    """Drops a custom dynamic database table."""
    from storage import dynamic_schema
    try:
        return dynamic_schema.drop_custom_table(table_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/schema/tables/{table_name}/columns")
def add_schema_column(table_name: str, payload: dict):
    """Adds a new column to an existing table."""
    from storage import dynamic_schema
    col_name = str((payload or {}).get("column_name") or "").strip()
    col_type = str((payload or {}).get("column_type") or "string").strip()
    if not col_name:
        raise HTTPException(status_code=422, detail="'column_name' is required.")
    try:
        return dynamic_schema.add_column_to_table(table_name, col_name, col_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/v1/schema/tables/{table_name}/columns/{column_name}/rename")
def rename_schema_column(table_name: str, column_name: str, payload: dict):
    """Renames an existing column in a dynamic custom table."""
    from storage import dynamic_schema
    new_name = str((payload or {}).get("new_name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=422, detail="'new_name' is required.")
    try:
        return dynamic_schema.rename_column_in_table(table_name, column_name, new_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/v1/schema/tables/{table_name}/columns/{column_name}")
def drop_schema_column(table_name: str, column_name: str):
    """Drops a column from an existing table."""
    from storage import dynamic_schema
    try:
        return dynamic_schema.drop_column_from_table(table_name, column_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/schema/tables/{table_name}/data")
def fetch_schema_table_data(table_name: str, limit: int = 500):
    """Fetches all rows from a dynamic custom table."""
    from storage import dynamic_schema
    try:
        columns = dynamic_schema.get_table_columns(table_name)
        records = dynamic_schema.fetch_dynamic_records(table_name, limit=limit)
        return {"table_name": table_name, "columns": columns, "records": records}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/schema/tables/{table_name}/data")
def insert_schema_table_data(table_name: str, payload: dict):
    """Inserts a row into a dynamic custom table."""
    from storage import dynamic_schema
    try:
        return dynamic_schema.insert_dynamic_record(table_name, payload or {})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/v1/schema/tables/{table_name}/data/{record_id}")
def delete_schema_table_data(table_name: str, record_id: int):
    """Deletes a row by ID from a dynamic custom table."""
    from storage import dynamic_schema
    try:
        success = dynamic_schema.delete_dynamic_record(table_name, record_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")
        return {"status": "deleted", "id": record_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/v1/schema/tables/{table_name}/data/{record_id}")
def update_schema_table_data(table_name: str, record_id: int, payload: dict):
    """Updates a record in a custom dynamic table."""
    from storage import dynamic_schema
    try:
        success = dynamic_schema.update_dynamic_record(table_name, record_id, payload or {})
        if not success:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")
        return {"status": "success", "id": record_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/schema/tables/{table_name}/import")
async def import_schema_table_file(table_name: str, file: UploadFile = File(...)):
    """Imports CSV or Excel file records into a custom dynamic table."""
    import io
    import pandas as pd
    from storage import dynamic_schema
    filename_lower = (file.filename or "").lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv, .xlsx, or .xls file.")
    try:
        content = await file.read()
        file_bytes = io.BytesIO(content)
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(file_bytes, dtype=str)
        else:
            df = pd.read_excel(file_bytes, dtype=str)
        df = df.fillna("")
        records = df.to_dict(orient="records")
        imported_count = dynamic_schema.bulk_insert_dynamic_records(table_name, records)
        return {"status": "success", "imported_count": imported_count}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point (For running locally)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(" Starting FastAPI server on port 8000...")
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000)