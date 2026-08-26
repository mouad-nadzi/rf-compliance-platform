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
import shutil

from storage.database import get_db, SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from server import config

# ──────────────────────────────────────────────────────────────────────────────
# Global State & Config
# ──────────────────────────────────────────────────────────────────────────────

_ocr_engine = None

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

    logger.info(f" Preloading LLM engine '{LLM_ENGINE}' into VRAM...")
    load_llm_engine()

    logger.info("Preloading embedding model into memory...")
    from core.rag.embeddings import get_embedding_model
    get_embedding_model()

    logger.info("OCR and LLM engines resident in VRAM; embedding model in RAM.")

    yield  # This tells FastAPI it's ready to accept requests

    logger.info(" Shutting down API...")


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

        # 2. Ensure OCR Engine is initialized and resident
        from server.config import CACHE_DIR, OCR_ENGINE
        from core.registry import get_ocr_engine
        if _ocr_engine is None:
            _ocr_engine = get_ocr_engine(OCR_ENGINE)
            _ocr_engine.load(CACHE_DIR)

        # 3. Run OCR to get layout-aware Markdown with <Page X> tags
        extracted_text = _ocr_engine.process_document(temp_file_path, OUTPUT_FOLDER)

        # 4. Extract structured metadata directly from raw OCR output (LLM engine co-exists in VRAM)
        from core.extractor import extract_certificate_data
        certificates = [extract_certificate_data(extracted_text, file.filename)]

        # 6. Persist extracted certificate metadata and 1024-d embedded chunks into PostgreSQL
        from core.extractor import save_certificate_to_db
        db_certificate_ids = []
        for cert in certificates:
            try:
                cert.cert_link = file_public_url
                metadata_record = save_certificate_to_db(cert, extracted_text, file.filename)
                db_certificate_ids.append(metadata_record.certificate_id)
            except Exception as db_err:
                logger.warning(f"  Could not persist certificate to PostgreSQL: {db_err}")

        # 7. Return a clean JSON response with certificate objects, DB IDs, and raw markdown
        return {
            "status": "success",
            "filename": file.filename,
            "certificates_found": len(certificates),
            "certificates": [cert.model_dump() for cert in certificates],
            "database_records": db_certificate_ids,
            "raw_markdown": extracted_text
        }

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
    from core.extractor import extract_certificate_data, save_certificate_to_db

    def _save():
        _save_manifest(batch_id, manifest)

    try:
        manifest["phase"] = "processing"
        _save()
        if _ocr_engine is None:
            _ocr_engine = get_ocr_engine(OCR_ENGINE)
            _ocr_engine.load(CACHE_DIR)

        for name in file_names:
            entry = manifest["files"].get(name, {})
            if entry.get("status") == "extracted":
                manifest["skipped"] += 1
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
            "exp_date": str(r.exp_date) if r.exp_date else None,
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

    try:
        exists = (
            db.query(CertificateMetadata)
            .filter(CertificateMetadata.file_name == clean_file_name)
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
    from storage.models import AuthorityLookup
    try:
        record = AuthorityLookup(
            canonical_authority=payload.get("canonical_authority"),
            abbreviation=payload.get("abbreviation"),
            country=payload.get("country"),
            standard_validity_years=payload.get("standard_validity_years"),
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
    from storage.models import AuthorityLookup
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
                standard_validity_years=int(float(row["standard_validity_years"])) if str(row.get("standard_validity_years") or "").strip() else None,
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
    from storage.models import AuthorityLookup
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
            record.standard_validity_years = payload["standard_validity_years"]
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
                setattr(record, field, date.fromisoformat(str(val)) if val else None)
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

from fastapi.responses import Response
import csv
import io
from schemas.extraction import CertificateExtractionSchema, CertificateMetadata

@app.post("/api/v1/certificates/manual")
async def add_certificate_manual(cert_data: CertificateExtractionSchema, db: Session = Depends(get_db)):
    """Manually insert a certificate record without OCR."""
    from core.extractor import save_certificate_to_db
    try:
        record = save_certificate_to_db(cert_data, raw_markdown="Manual Entry", file_name="Manual Entry", db=db)
        return {"status": "success", "certificate_id": record.certificate_id}
    except Exception as e:
        logger.error(f" Error adding manual certificate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/certificates/export/csv")
async def export_certificates_csv(db: Session = Depends(get_db)):
    """Exports all certificate database records to a downloadable CSV file."""
    from schemas.extraction import CertificateMetadata
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
                r.exp_date.strftime("%Y-%m-%d") if r.exp_date else "",
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
                "exp_date": r.exp_date.strftime("%Y-%m-%d") if r.exp_date else "",
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

            save_certificate_to_db(cert_schema, raw_markdown=raw_mk, file_name=fname, db=db)
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
# CLI Entry Point (For running locally)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(" Starting FastAPI server on port 8000...")
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000)