"""
core/agent/ingestor.py — Shared document ingestion pipeline.

Runs the full OCR -> extraction -> persistence pipeline on a local document file.
Used by both the HITL approval route (INGEST_DOCUMENT proposals) and the agentic
step loop's `ingest_to_database` write step. Mirrors the single /api/v1/parse
flow but operates on an already-downloaded file path.
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Module-level OCR engine shared with server.main (set by the FastAPI lifespan).
_ocr_engine = None

#: Temporary folder for OCR intermediate images (mirrors server.main.OUTPUT_FOLDER).
OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), "ocr_outputs")


def set_ocr_engine(engine) -> None:
    """Register the shared OCR engine instance (called from server startup)."""
    global _ocr_engine
    _ocr_engine = engine


def ingest_document_file(
    file_path: str,
    source_url: str = "",
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """
    Run OCR -> extraction -> persistence on a local document file.

    Args:
        file_path: absolute path to the downloaded document.
        source_url: the source URL to store as cert_link (fallback: public file URL).
        status_callback: optional callback(stage, message) for live UI status reporting.

    Returns:
        dict with filename, source_url, certificates_found, certificates, database_records, raw_markdown.
    """
    global _ocr_engine
    from core.utils.system_check import ensure_ready
    from server.config import CACHE_DIR, OCR_ENGINE, FILES_STORAGE_DIR, PUBLIC_API_URL
    from core.registry import get_ocr_engine
    from core.extractor import extract_certificate_data, save_certificate_to_db

    base_name = os.path.basename(file_path)
    if status_callback:
        status_callback("ocr", f"Running GLM-OCR vision model on '{base_name}'...")

    ensure_ready()
    if _ocr_engine is None:
        try:
            from server.main import get_shared_ocr_engine
            _ocr_engine = get_shared_ocr_engine()
        except ImportError:
            _ocr_engine = get_ocr_engine(OCR_ENGINE)
            _ocr_engine.load(CACHE_DIR)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    extracted_text = _ocr_engine.process_document(file_path, OUTPUT_FOLDER)

    if status_callback:
        status_callback("extraction", f"Extracting compliance metadata & validity fields for '{base_name}'...")
    certificates = [extract_certificate_data(extracted_text, base_name)]

    safe_fname = os.path.basename(base_name or "document")
    persistent_path = os.path.join(FILES_STORAGE_DIR, "agent_ingest", safe_fname)
    os.makedirs(os.path.dirname(persistent_path), exist_ok=True)
    shutil.copy2(file_path, persistent_path)
    file_public_url = f"{PUBLIC_API_URL}/files/agent_ingest/{safe_fname}"

    clean_source_url = str(source_url or "").strip()
    if not clean_source_url.startswith(("http://", "https://")):
        clean_source_url = file_public_url

    if status_callback:
        status_callback("persistence", f"Computing vector embeddings & persisting '{base_name}' to PostgreSQL...")

    db_ids = []
    updated_ids = []
    new_ids = []
    for cert in certificates:
        cert.cert_link = clean_source_url
        try:
            record = save_certificate_to_db(cert, extracted_text, base_name)
            db_ids.append(record.certificate_id)
            if getattr(record, "_is_updated", False):
                updated_ids.append(record.certificate_id)
            else:
                new_ids.append(record.certificate_id)
        except Exception as db_err:
            logger.warning(f"  Could not persist ingested certificate to PostgreSQL: {db_err}")

    logger.info(f" Ingested document '{base_name}': {len(db_ids)} certificate record(s) ({len(updated_ids)} updated, {len(new_ids)} new).")
    return {
        "status": "success",
        "filename": base_name,
        "source_url": source_url,
        "certificates_found": len(certificates),
        "certificates": [cert.model_dump() for cert in certificates],
        "database_records": db_ids,
        "updated_records": updated_ids,
        "new_records": new_ids,
        "raw_markdown": extracted_text,
    }