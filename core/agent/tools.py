"""
core/agent/tools.py — Model-Agnostic Tool Registry

Defines the strict abstract contract every agentic tool must satisfy and hosts
the concrete tool registry. Tools are deliberately model-agnostic: they are pure
Python objects with no LLM/OCR coupling, so they can be driven by any supervisor.

Local-First: tool execution uses only the local filesystem and Python's standard
library (plus `requests` for HTTP downloads). No third-party APIs are referenced.

Safety: side-effectful tools (downloads, email drafting) are HITL-gated at the
supervisor/UI layer. Tools themselves never mutate the database.
"""

import csv
import io
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: Root directory where agent tools persist their local artifacts.
AGENT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "agent")
#: Directory where downloaded web resources are stored.
DOWNLOAD_DIR = os.path.join(AGENT_DATA_DIR, "downloads")
#: Directory where drafted email payloads are stored.
DRAFTS_DIR = os.path.join(AGENT_DATA_DIR, "drafts")

#: Default download timeout (seconds).
DOWNLOAD_TIMEOUT = 30


class BaseTool(ABC):
    """
    Abstract contract for all agentic tools.

    Every tool must implement:
      1. get_schema() -> returns a JSON schema describing the tool's expected arguments.
      2. execute(**kwargs) -> runs the tool with validated keyword arguments.

    Subclasses should also set a stable class attribute `name` used as the registry key.
    """

    name: str = "base"
    requires_approval: bool = False

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return a JSON Schema describing the tool's expected arguments.

        Returns:
            Dict[str, Any]: A JSON-schema-style mapping (e.g. {"type": "object",
                "properties": {...}, "required": [...]}).
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with validated keyword arguments.

        Args:
            **kwargs: Arguments matching the schema returned by get_schema().

        Returns:
            Any: The tool result (typed per implementation; callers should treat
                it as opaque and rely on the schema/documentation).
        """
        ...


def _ensure_dir(path: str) -> str:
    """Create a directory (and parents) if missing; returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def _derive_filename(url: str) -> str:
    """
    Derive a safe local filename from a URL's path component.

    Falls back to a unique timestamped name when the URL has no usable basename
    (e.g. URLs ending in "/" or query-only paths).
    """
    parsed = urlparse(url)
    base = os.path.basename(parsed.path).strip()
    if not base or base in (".", "/") or ".." in base:
        base = f"download_{int(time.time() * 1000)}"
    return base


class WebDownloaderTool(BaseTool):
    """
    Downloads a remote resource over HTTP(S) and stores it locally.

    Uses Python's `requests` library. The file is saved under a local temporary
    agent directory and the absolute path is returned. Raises on any failed
    request so callers can surface the error instead of persisting bad data.
    Supports an optional transient session-cookie header for authenticated
    portals (held in memory only, never stored or logged).
    """

    name: str = "web_downloader"
    requires_approval: bool = False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Fully-qualified HTTP(S) URL of the resource to download.",
                },
                "cookie_header": {
                    "type": "string",
                    "description": "Optional transient session cookie header value (e.g. 'session=abc'). "
                                   "Used in-memory only; never stored or logged.",
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> Any:
        url = str(kwargs.get("url") or "").strip()
        if not url:
            raise ValueError("WebDownloaderTool requires a non-empty 'url' argument.")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"WebDownloaderTool received an invalid URL: '{url}'")

        import requests

        headers = None
        cookie_header = kwargs.get("cookie_header")
        if cookie_header:
            headers = {"Cookie": str(cookie_header)}

        download_dir = _ensure_dir(DOWNLOAD_DIR)
        file_path = os.path.join(download_dir, _derive_filename(url))

        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as resp:
                resp.raise_for_status()
                with open(file_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as exc:
            # Clean up any partial download so failed requests leave no artifacts.
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            raise RuntimeError(f"WebDownloaderTool failed to download '{url}': {exc}") from exc

        logger.info(f" WebDownloaderTool saved '{url}' -> {file_path}")
        return {"path": os.path.abspath(file_path), "url": url}


class DataConverterTool(BaseTool):
    """
    Converts a list of row dictionaries into 'csv' or 'json' string output.

    Strictly validates the requested format ('csv' | 'json'). CSV output is
    produced with Python's built-in `csv` module via an in-memory string buffer;
    JSON output is a pretty-printed string block. Returns the formatted string.
    """

    name: str = "data_converter"
    requires_approval: bool = False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of dictionaries representing database rows to convert.",
                },
                "format": {
                    "type": "string",
                    "enum": ["csv", "json"],
                    "description": "Target output format. Strictly 'csv' or 'json'.",
                },
            },
            "required": ["data", "format"],
        }

    def execute(self, **kwargs: Any) -> Any:
        data = kwargs.get("data")
        fmt = str(kwargs.get("format") or "").strip().lower()

        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError("DataConverterTool requires 'data' to be a list of dictionaries.")

        if fmt not in ("csv", "json"):
            raise ValueError(f"DataConverterTool received invalid format '{fmt}'; must be 'csv' or 'json'.")

        if fmt == "csv":
            if not data:
                return ""
            fieldnames: List[str] = []
            for row in data:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
            return buffer.getvalue()

        return json.dumps(data, indent=2, ensure_ascii=False)


class EmailDraftingTool(BaseTool):
    """
    Drafts an email payload WITHOUT sending it (HITL guardrail).

    Compiles recipient/subject/body_content into a structured JSON payload, saves
    it under a local drafts/ directory, and returns a success message with the
    draft ID. Live SMTP dispatch is intentionally deferred; a human must approve
    and send the draft.
    """

    name: str = "email_drafting"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Email address of the intended recipient.",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line of the drafted email.",
                },
                "body_content": {
                    "type": "string",
                    "description": "Body content of the drafted email.",
                },
            },
            "required": ["recipient", "subject", "body_content"],
        }

    def execute(self, **kwargs: Any) -> Any:
        recipient = str(kwargs.get("recipient") or "").strip()
        subject = str(kwargs.get("subject") or "").strip()
        body_content = str(kwargs.get("body_content") or "").strip()

        if not recipient:
            raise ValueError("EmailDraftingTool requires a non-empty 'recipient'.")
        if not subject:
            raise ValueError("EmailDraftingTool requires a non-empty 'subject'.")

        draft_id = uuid.uuid4().hex[:12]
        payload = {
            "draft_id": draft_id,
            "recipient": recipient,
            "subject": subject,
            "body_content": body_content,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "draft",
        }

        drafts_dir = _ensure_dir(DRAFTS_DIR)
        draft_path = os.path.join(drafts_dir, f"{draft_id}.json")
        with open(draft_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info(f" EmailDraftingTool saved draft {draft_id} -> {draft_path}")
        return {
            "draft_id": draft_id,
            "message": f"Email draft saved (not sent - awaiting HITL approval).",
            "path": os.path.abspath(draft_path),
        }


class RememberFactTool(BaseTool):
    """
    Saves a persistent cross-session long-term memory fact to PostgreSQL.
    """

    name: str = "remember_fact"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_key": {
                    "type": "string",
                    "description": "Category key: 'preference', 'rule', 'contact', 'portal_note'.",
                },
                "fact_text": {
                    "type": "string",
                    "description": "The persistent long-term fact or user directive to remember across sessions.",
                },
            },
            "required": ["fact_text"],
        }

    def execute(self, **kwargs: Any) -> Any:
        from core.agent.memory import save_agent_memory
        memory_key = str(kwargs.get("memory_key") or "preference")
        fact_text = str(kwargs.get("fact_text") or "")
        return save_agent_memory(memory_key, fact_text)


class ManageSchemaTool(BaseTool):
    """
    Dynamically creates custom database tables, adds columns, drops columns,
    or inspects table schemas in PostgreSQL.
    """

    name: str = "manage_schema"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action type: 'create_table', 'add_column', 'drop_column', 'list_tables', 'get_columns'.",
                },
                "table_name": {
                    "type": "string",
                    "description": "Target database table name (e.g. 'vendor_audits').",
                },
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Column name (e.g. 'score')"},
                            "type": {"type": "string", "description": "SQL data type: 'string', 'text', 'int', 'float', 'date', 'timestamp', 'boolean'"},
                        },
                        "required": ["name"],
                    },
                    "description": "List of column definitions for create_table action.",
                },
                "column_name": {
                    "type": "string",
                    "description": "Target column name for add_column or drop_column actions.",
                },
                "column_type": {
                    "type": "string",
                    "description": "SQL data type for add_column action.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> Any:
        from storage import dynamic_schema
        action = str(kwargs.get("action") or "").strip().lower()
        table_name = str(kwargs.get("table_name") or "")
        columns = kwargs.get("columns") or []
        column_name = str(kwargs.get("column_name") or "")
        column_type = str(kwargs.get("column_type") or "string")

        if action == "create_table":
            if not table_name:
                raise ValueError("table_name is required for create_table action.")
            return dynamic_schema.create_custom_table(table_name, columns)
        elif action == "add_column":
            if not table_name or not column_name:
                raise ValueError("table_name and column_name are required for add_column action.")
            return dynamic_schema.add_column_to_table(table_name, column_name, column_type)
        elif action == "drop_column":
            if not table_name or not column_name:
                raise ValueError("table_name and column_name are required for drop_column action.")
            return dynamic_schema.drop_column_from_table(table_name, column_name)
        elif action == "rename_column":
            new_name = str(kwargs.get("new_name") or kwargs.get("new_column_name") or "")
            if not table_name or not column_name or not new_name:
                raise ValueError("table_name, column_name, and new_name are required for rename_column action.")
            return dynamic_schema.rename_column_in_table(table_name, column_name, new_name)
        elif action in ("drop_table", "delete_table"):
            if not table_name:
                raise ValueError("table_name is required for drop_table action.")
            return dynamic_schema.drop_custom_table(table_name)
        elif action == "list_tables":
            return dynamic_schema.list_all_user_tables()
        elif action == "get_columns":
            if not table_name:
                raise ValueError("table_name is required for get_columns action.")
            return dynamic_schema.get_table_columns(table_name)
        else:
            raise ValueError(f"Unknown action: '{action}'. Supported actions: create_table, drop_table, add_column, drop_column, list_tables, get_columns.")


class CheckUrlTool(BaseTool):
    name: str = "check_url"
    requires_approval: bool = False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}},
                "url": {"type": "string"}
            }
        }

    def execute(self, **kwargs: Any) -> Any:
        from core.agent.scraper import discover_pdf_urls
        from server.config import (
            SCRAPER_MAX_PAGES, SCRAPER_MAX_DEPTH, SCRAPER_TIMEOUT_SECONDS,
            SCRAPER_USER_AGENT, SCRAPER_POLITE_DELAY_SECONDS,
            SCRAPER_BLOCKED_KEYWORDS, SCRAPER_ALLOWED_KEYWORDS, SCRAPER_USE_LLM_FILTER,
        )

        urls = kwargs.get("urls") or ([kwargs.get("url")] if kwargs.get("url") else [])
        if not urls:
            return "I need a URL to check. Please provide the link."

        artifacts = kwargs.get("artifacts")
        all_verified: List[str] = []
        reports: List[str] = []

        for target_url in urls:
            def _run(u=target_url):
                return discover_pdf_urls(
                    u,
                    max_pages=SCRAPER_MAX_PAGES,
                    max_depth=SCRAPER_MAX_DEPTH,
                    timeout=SCRAPER_TIMEOUT_SECONDS,
                    user_agent=SCRAPER_USER_AGENT,
                    polite_delay=SCRAPER_POLITE_DELAY_SECONDS,
                    allowed_keywords=SCRAPER_ALLOWED_KEYWORDS,
                    blocked_keywords=SCRAPER_BLOCKED_KEYWORDS,
                    manifest_path=None,
                    use_llm=SCRAPER_USE_LLM_FILTER,
                    fetcher_type="auto",
                )

            try:
                from concurrent.futures import ThreadPoolExecutor
                from urllib.parse import unquote, urlparse
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(_run).result(timeout=300)
                verified = result.verified_urls
                all_verified.extend([v for v in verified if v not in all_verified])
                if verified:
                    names = [unquote(os.path.basename(urlparse(u).path)) or u for u in verified]
                    listed = "\n".join(f"- {n}" for n in names[:10])
                    more = f"\n... and {len(names) - 10} more" if len(names) > 10 else ""
                    reports.append(f"I checked {target_url} and found {len(verified)} certificate document(s):\n{listed}{more}")
                else:
                    reports.append(f"I checked {target_url} and found no downloadable certificate documents on it.")
            except Exception as exc:
                logger.error(f" URL check failed for {target_url}: {exc}")
                reports.append(f"I couldn't check {target_url} right now ({exc}). Please try again.")

        if artifacts:
            artifacts.source_url = ", ".join(urls)
            artifacts.verified_urls = all_verified
            artifacts.checked_report = "\n\n".join(reports)
            
        return "\n\n".join(reports)


class DownloadDocumentsTool(BaseTool):
    name: str = "download_documents"
    requires_approval: bool = False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}},
                "url": {"type": "string"},
                "target_file": {"type": "string"}
            }
        }

    def execute(self, **kwargs: Any) -> Any:
        from urllib.parse import unquote, urlparse
        
        artifacts = kwargs.get("artifacts")
        urls = (artifacts.verified_urls if artifacts else []) or []
        
        if not urls:
            source_urls = kwargs.get("urls") or ([kwargs.get("url")] if kwargs.get("url") else [])
            if not source_urls and artifacts and artifacts.source_url:
                source_urls = [u.strip() for u in artifacts.source_url.split(",") if u.strip()]
            if source_urls:
                from core.agent.scraper import discover_pdf_urls
                from server.config import (
                    SCRAPER_MAX_PAGES, SCRAPER_MAX_DEPTH, SCRAPER_TIMEOUT_SECONDS,
                    SCRAPER_USER_AGENT, SCRAPER_POLITE_DELAY_SECONDS,
                    SCRAPER_BLOCKED_KEYWORDS, SCRAPER_ALLOWED_KEYWORDS, SCRAPER_USE_LLM_FILTER,
                )
                for s_url in source_urls:
                    try:
                        res = discover_pdf_urls(
                            s_url,
                            max_pages=SCRAPER_MAX_PAGES,
                            max_depth=SCRAPER_MAX_DEPTH,
                            timeout=SCRAPER_TIMEOUT_SECONDS,
                            user_agent=SCRAPER_USER_AGENT,
                            polite_delay=SCRAPER_POLITE_DELAY_SECONDS,
                            allowed_keywords=SCRAPER_ALLOWED_KEYWORDS,
                            blocked_keywords=SCRAPER_BLOCKED_KEYWORDS,
                            use_llm=SCRAPER_USE_LLM_FILTER,
                            fetcher_type="auto",
                        )
                        urls.extend(res.verified_urls)
                    except Exception as exc:
                        logger.warning(f" Auto-discovery failed for {s_url}: {exc}")
                if artifacts:
                    artifacts.verified_urls = urls

        if not urls:
            return "There were no certificate documents found to download."

        target_file = (kwargs.get("target_file") or "").strip()
        if target_file:
            clean_target = unquote(target_file).lower()
            matching_urls = [
                u for u in urls
                if clean_target in unquote(os.path.basename(urlparse(u).path)).lower()
                or unquote(os.path.basename(urlparse(u).path)).lower() in clean_target
            ]
            if matching_urls:
                urls = matching_urls

        tool = WebDownloaderTool()
        downloaded: List[str] = []
        for url in urls:
            try:
                res = tool.execute(url=url, cookie_header=kwargs.get("cookie_header"))
                path = str(res.get("path", ""))
                if path:
                    downloaded.append(path)
            except Exception as exc:
                logger.warning(f" Download failed for {url}: {exc}")

        if artifacts:
            artifacts.file_paths = downloaded
            
        target_str = f" for '{target_file}'" if target_file else ""
        return f"I've downloaded {len(downloaded)} certificate PDF(s){target_str} to local storage."


class IngestToDatabaseTool(BaseTool):
    name: str = "ingest_to_database"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "source_url": {"type": "string"},
                "verified_urls": {"type": "array", "items": {"type": "string"}},
                "target_file": {"type": "string"}
            }
        }

    def execute(self, **kwargs: Any) -> Any:
        from core.agent.ingestor import ingest_document_file
        from urllib.parse import unquote, urlparse

        artifacts = kwargs.get("artifacts")
        status_callback = kwargs.get("status_callback")
        
        file_paths = kwargs.get("file_paths") or (artifacts.file_paths if artifacts else []) or []
        source_url = kwargs.get("source_url") or (artifacts.source_url if artifacts else "") or ""
        verified_urls = kwargs.get("verified_urls") or (artifacts.verified_urls if artifacts else []) or []
        target_file = (kwargs.get("target_file") or "").strip()

        if not file_paths:
            # Fallback to trigger download
            dl_tool = DownloadDocumentsTool()
            dl_tool.execute(**kwargs)
            file_paths = artifacts.file_paths if artifacts else []
            verified_urls = artifacts.verified_urls if artifacts else verified_urls

        if target_file and file_paths:
            clean_target = unquote(target_file).lower()
            matching = [
                p for p in file_paths
                if clean_target in unquote(os.path.basename(p)).lower()
                or unquote(os.path.basename(p)).lower() in clean_target
            ]
            if matching:
                file_paths = matching

        if (not file_paths or (target_file and not matching)) and os.path.exists(DOWNLOAD_DIR):
            local_files = [
                os.path.join(DOWNLOAD_DIR, f)
                for f in os.listdir(DOWNLOAD_DIR)
                if f.lower().endswith(".pdf")
            ]
            if target_file:
                clean_target = unquote(target_file).lower()
                local_matching = [
                    p for p in local_files
                    if clean_target in unquote(os.path.basename(p)).lower()
                    or unquote(os.path.basename(p)).lower() in clean_target
                ]
                if local_matching:
                    file_paths = local_matching
            elif local_files:
                file_paths = local_files

        ingest_mode = kwargs.get("ingest_mode") or ""
        if ingest_mode == "single" and not target_file and len(file_paths) > 1:
            file_paths = file_paths[:1]

        if not file_paths:
            return "Nothing to ingest - no downloaded certificate files."

        new_record_ids: List[str] = []
        updated_record_ids: List[str] = []
        for path in file_paths:
            fname = unquote(os.path.basename(path)).lower()
            file_source_url = next(
                (u for u in verified_urls if unquote(os.path.basename(urlparse(u).path)).lower() == fname),
                ""
            )
            if not file_source_url and source_url and source_url.startswith(("http://", "https://")):
                file_source_url = source_url

            try:
                detail = ingest_document_file(path, file_source_url, status_callback=status_callback)
                new_record_ids.extend(detail.get("new_records", []))
                updated_record_ids.extend(detail.get("updated_records", []))
            except Exception as exc:
                logger.error(f" Ingest failed for {path}: {exc}")

        target_str = f" for '{target_file}'" if target_file else ""
        if updated_record_ids and not new_record_ids:
            suffix = f" ({', '.join(updated_record_ids[:5])})"
            return f"Updated existing certificate record(s){target_str} in the database{suffix}."
        elif new_record_ids and updated_record_ids:
            return f"Added {len(new_record_ids)} new certificate(s) and updated {len(updated_record_ids)} existing record(s){target_str} in the database."
        else:
            suffix = f" ({', '.join(new_record_ids[:5])})" if new_record_ids else ""
            return f"Added {len(new_record_ids)} new certificate(s){target_str} to the database{suffix}."


class DeleteRecordTool(BaseTool):
    name: str = "delete_record"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "target_id": {"type": "string"},
                "target_file": {"type": "string"},
                "certificate_id": {"type": "string"},
                "filters": {"type": "object"},
                "delete_all": {"type": "boolean"},
                "row_filter": {"type": "object"}
            }
        }

    def execute(self, **kwargs: Any) -> Any:
        from core.agent.db_editor import build_mutation_sql, execute_mutation
        from storage.database import get_db_session

        table_name = kwargs.get("table_name")
        target_id = (kwargs.get("target_id") or kwargs.get("certificate_id") or kwargs.get("target_file") or "").strip()
        filters = kwargs.get("filters") or {}
        delete_all = bool(kwargs.get("delete_all"))
        row_filter = kwargs.get("row_filter")

        if not table_name and (target_id or filters or delete_all):
            # Default to certificates if no table specified but cert params provided
            table_name = "certificates"
            
        if not table_name:
            return "Cannot delete record: table_name is required."

        if row_filter:
            # Generic deletion via strict row filter
            try:
                mutation = build_mutation_sql("delete", table_name, values={}, row_filter=row_filter)
                with get_db_session() as db:
                    count = execute_mutation(db, mutation)
                    return f"Successfully deleted {count} record(s) from table '{table_name}'."
            except Exception as exc:
                return f"Failed to delete record from '{table_name}': {exc}"

        # Fuzzy / Complex Deletion
        try:
            if target_id:
                mutation = build_mutation_sql("delete", table_name, values={}, fuzzy_match_query=target_id)
            elif filters and isinstance(filters, dict):
                mutation = build_mutation_sql("delete", table_name, values={}, row_filter=filters)
            elif delete_all:
                mutation = build_mutation_sql("delete", table_name, values={}, allow_full_table=True)
            else:
                return f"No target_id, filters, or delete_all specified for table '{table_name}'."
                
            with get_db_session() as db:
                count = execute_mutation(db, mutation)
                if count > 0:
                    filter_str = f" matching '{target_id}'" if target_id else (f" matching {filters}" if filters else " (all records)")
                    return f"Successfully deleted {count} record(s) from '{table_name}'{filter_str}."
                else:
                    return f"No records found in '{table_name}' to delete."
        except Exception as exc:
            return f"Failed to delete record from '{table_name}': {exc}"


def get_tool_registry() -> Dict[str, BaseTool]:
    """
    Return the immutable tool registry mapping tool name -> tool instance.

    This is the single lookup point the agentic supervisor uses to resolve an
    AGENT_ACTION intent into a concrete tool.

    Returns:
        Dict[str, BaseTool]: name -> tool instance.
    """
    return {
        CheckUrlTool.name: CheckUrlTool(),
        DownloadDocumentsTool.name: DownloadDocumentsTool(),
        IngestToDatabaseTool.name: IngestToDatabaseTool(),
        DeleteRecordTool.name: DeleteRecordTool(),
        WebDownloaderTool.name: WebDownloaderTool(),
        DataConverterTool.name: DataConverterTool(),
        EmailDraftingTool.name: EmailDraftingTool(),
        RememberFactTool.name: RememberFactTool(),
        ManageSchemaTool.name: ManageSchemaTool(),
    }