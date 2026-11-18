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
    combines columns, renames columns, or inspects table schemas in PostgreSQL.
    Supports single and bulk operations.
    """

    name: str = "manage_schema"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Action type: 'create_table', 'add_column', 'add_columns', "
                        "'drop_column', 'drop_columns', 'combine_columns', 'rename_column', "
                        "'drop_table', 'list_tables', 'get_columns'."
                    ),
                },
                "table_name": {
                    "type": "string",
                    "description": "Target database table name (e.g. 'certificates_responsibles', 'vendor_audits').",
                },
                "columns": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string", "description": "Column name for drop_columns or source columns"},
                            {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Column name"},
                                    "type": {"type": "string", "description": "SQL data type: 'string', 'text', 'int', 'float', 'date', 'timestamp', 'boolean'"},
                                },
                                "required": ["name"],
                            },
                        ]
                    },
                    "description": "List of column definitions for create_table/add_columns, or list of column names to drop for drop_columns.",
                },
                "column_name": {
                    "type": "string",
                    "description": "Target column name (or comma-separated column names) for add_column or drop_column actions.",
                },
                "column_type": {
                    "type": "string",
                    "description": "SQL data type for add_column action.",
                },
                "source_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of source columns to concatenate for 'combine_columns' action (e.g. ['firstname', 'lastname']).",
                },
                "target_column": {
                    "type": "string",
                    "description": "Target column name to receive merged values for 'combine_columns' action (e.g. 'responsible_name').",
                },
                "separator": {
                    "type": "string",
                    "description": "Separator string for 'combine_columns' action (default: ' ').",
                },
                "drop_sources": {
                    "type": "boolean",
                    "description": "Whether to drop the source columns after combining them into target_column (default: false).",
                },
                "new_name": {
                    "type": "string",
                    "description": "New column name for rename_column action.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> Any:
        from storage import dynamic_schema
        raw_act = str(kwargs.get("action_type") or "").strip()
        if not raw_act or raw_act.lower() == "manage_schema":
            raw_act = str(kwargs.get("action") or "").strip()
        if raw_act.lower() == "manage_schema":
            raw_act = str(kwargs.get("subaction") or kwargs.get("schema_action") or "").strip()
        action = raw_act.lower()
        table_name = str(kwargs.get("table_name") or "")
        columns = kwargs.get("columns") or []
        column_name = str(kwargs.get("column_name") or "")
        column_type = str(kwargs.get("column_type") or "string")

        if action == "create_table":
            if not table_name:
                raise ValueError("table_name is required for create_table action.")
            return dynamic_schema.create_custom_table(table_name, columns)
        elif action in ("add_column", "add_columns"):
            if not table_name:
                raise ValueError("table_name is required for add_column(s) action.")
            if columns and isinstance(columns, (list, tuple)):
                return dynamic_schema.add_columns_to_table(table_name, list(columns))
            if not column_name:
                raise ValueError("column_name or columns list is required for add_column action.")
            return dynamic_schema.add_column_to_table(table_name, column_name, column_type)
        elif action in ("drop_column", "drop_columns"):
            if not table_name:
                raise ValueError("table_name is required for drop_column(s) action.")
            target_cols = columns if columns else column_name
            if not target_cols:
                raise ValueError("column_name or columns list is required for drop_column action.")
            return dynamic_schema.drop_columns_from_table(table_name, target_cols)
        elif action == "combine_columns":
            if not table_name:
                raise ValueError("table_name is required for combine_columns action.")
            target_col = str(kwargs.get("target_column") or column_name or kwargs.get("new_name") or "").strip()
            source_cols = kwargs.get("source_columns") or columns
            if not target_col or not source_cols:
                raise ValueError("target_column and source_columns are required for combine_columns action.")
            sep = str(kwargs.get("separator") or " ")
            drop_src = bool(kwargs.get("drop_sources", False))
            return dynamic_schema.combine_columns_in_table(
                table_name=table_name,
                target_column=target_col,
                source_columns=list(source_cols),
                separator=sep,
                drop_sources=drop_src,
            )
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
            raise ValueError(
                f"Unknown action: '{action}'. Supported actions: "
                "create_table, drop_table, add_column, add_columns, drop_column, drop_columns, "
                "combine_columns, rename_column, list_tables, get_columns."
            )


class CheckUrlTool(BaseTool):
    """Inspects a web URL or portal page and discovers downloadable certificate PDF links."""
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
                    listed_lines = []
                    for idx, u in enumerate(verified[:15], 1):
                        fname = unquote(os.path.basename(urlparse(u).path)) or u
                        listed_lines.append(f"{idx}. {fname} ({u})")
                    listed = "\n".join(listed_lines)
                    more = f"\n... and {len(verified) - 15} more" if len(verified) > 15 else ""
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
    """Downloads one or more discovered certificate PDFs (or a specific target_file) from a portal or URL to local storage."""
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
        from core.agent.scraper import discover_pdf_urls, verify_pdf
        
        artifacts = kwargs.get("artifacts")
        urls = list((artifacts.verified_urls if artifacts and artifacts.verified_urls else kwargs.get("verified_urls")) or [])
        target_file = (kwargs.get("target_file") or "").strip()

        # If target_file itself is a direct HTTP(S) URL, use it immediately
        if target_file.startswith(("http://", "https://")) and verify_pdf(target_file):
            if target_file not in urls:
                urls.append(target_file)

        source_urls = kwargs.get("urls") or ([kwargs.get("url")] if kwargs.get("url") else [])
        if not source_urls and artifacts and artifacts.source_url:
            source_urls = [u.strip() for u in artifacts.source_url.split(",") if u.strip()]

        # If any passed source_url is already a direct PDF URL, include it directly
        direct_pdf_sources = [u for u in source_urls if u.lower().split("?")[0].endswith(".pdf")]
        if direct_pdf_sources:
            for du in direct_pdf_sources:
                if du not in urls:
                    urls.append(du)

        portal_source_urls = [u for u in source_urls if u not in direct_pdf_sources]
        if not urls and portal_source_urls:
            from server.config import (
                SCRAPER_MAX_PAGES, SCRAPER_MAX_DEPTH, SCRAPER_TIMEOUT_SECONDS,
                SCRAPER_USER_AGENT, SCRAPER_POLITE_DELAY_SECONDS,
                SCRAPER_BLOCKED_KEYWORDS, SCRAPER_ALLOWED_KEYWORDS, SCRAPER_USE_LLM_FILTER,
            )
            for s_url in portal_source_urls:
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
                    for vu in res.verified_urls:
                        if vu not in urls:
                            urls.append(vu)
                except Exception as exc:
                    logger.warning(f" Auto-discovery failed for {s_url}: {exc}")
            if artifacts:
                artifacts.verified_urls = list(urls)

        if not urls:
            return "There were no certificate documents found to download."

        if target_file:
            clean_target = unquote(target_file).strip().lower()
            matching_urls = [
                u for u in urls
                if clean_target in unquote(os.path.basename(urlparse(u).path)).lower()
                or unquote(os.path.basename(urlparse(u).path)).lower() in clean_target
                or clean_target.replace(" ", "_") in unquote(os.path.basename(urlparse(u).path)).lower().replace(" ", "_")
            ]
            if matching_urls:
                urls = matching_urls

        tool = WebDownloaderTool()
        downloaded: List[str] = []
        for url in urls:
            try:
                res = tool.execute(url=url, cookie_header=kwargs.get("cookie_header"))
                path = str(res.get("path", "")) if isinstance(res, dict) else ""
                if path:
                    downloaded.append(path)
            except Exception as exc:
                logger.warning(f" Download failed for {url}: {exc}")

        if artifacts:
            existing_paths = list(artifacts.file_paths or [])
            for p in downloaded:
                if p not in existing_paths:
                    existing_paths.append(p)
            artifacts.file_paths = existing_paths
            if not artifacts.verified_urls:
                artifacts.verified_urls = list(urls)
            
        target_str = f" for '{target_file}'" if target_file else ""
        if downloaded:
            names = [os.path.basename(p) for p in downloaded]
            return f"Downloaded {len(downloaded)} certificate PDF(s){target_str} to local storage: {', '.join(names)}."
        return f"Could not download certificate PDF(s){target_str}."


class IngestToDatabaseTool(BaseTool):
    """Downloads (if not yet downloaded), runs GLM-OCR + LLM metadata extraction, and ingests certificate PDF(s) into PostgreSQL."""
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
        from core.agent.agent_loop import RunArtifacts
        from urllib.parse import unquote, urlparse

        artifacts = kwargs.get("artifacts")
        status_callback = kwargs.get("status_callback")
        
        file_paths = list(kwargs.get("file_paths") or (artifacts.file_paths if artifacts and artifacts.file_paths else []) or [])
        source_url = kwargs.get("source_url") or (artifacts.source_url if artifacts and artifacts.source_url else "") or ""
        verified_urls = list(kwargs.get("verified_urls") or (artifacts.verified_urls if artifacts and artifacts.verified_urls else []) or [])
        target_file = (kwargs.get("target_file") or "").strip()

        if artifacts is None:
            artifacts = RunArtifacts(
                source_url=source_url,
                verified_urls=list(verified_urls),
                file_paths=list(file_paths),
            )

        def _match_paths(paths: List[str], tf: str) -> List[str]:
            if not tf:
                return paths
            clean_tf = unquote(tf).strip().lower()
            return [
                p for p in paths
                if os.path.exists(p) and (
                    clean_tf in unquote(os.path.basename(p)).lower()
                    or unquote(os.path.basename(p)).lower() in clean_tf
                    or clean_tf.replace(" ", "_") in unquote(os.path.basename(p)).lower().replace(" ", "_")
                )
            ]

        matching = _match_paths(file_paths, target_file) if target_file else [p for p in file_paths if os.path.exists(p)]

        if not matching and os.path.exists(DOWNLOAD_DIR):
            local_files = [
                os.path.join(DOWNLOAD_DIR, f)
                for f in os.listdir(DOWNLOAD_DIR)
                if f.lower().endswith(".pdf")
            ]
            matching = _match_paths(local_files, target_file) if target_file else local_files

        if not matching:
            # Trigger DownloadDocumentsTool with the session's RunArtifacts
            dl_kwargs = dict(kwargs)
            dl_kwargs["artifacts"] = artifacts
            if source_url and not dl_kwargs.get("url") and not dl_kwargs.get("urls"):
                dl_kwargs["url"] = source_url
            dl_tool = DownloadDocumentsTool()
            dl_tool.execute(**dl_kwargs)
            file_paths = list(artifacts.file_paths or [])
            verified_urls = list(artifacts.verified_urls or verified_urls)
            matching = _match_paths(file_paths, target_file) if target_file else [p for p in file_paths if os.path.exists(p)]
            if not matching and os.path.exists(DOWNLOAD_DIR):
                local_files = [
                    os.path.join(DOWNLOAD_DIR, f)
                    for f in os.listdir(DOWNLOAD_DIR)
                    if f.lower().endswith(".pdf")
                ]
                matching = _match_paths(local_files, target_file) if target_file else local_files

        file_paths = matching

        ingest_mode = kwargs.get("ingest_mode") or ""
        if ingest_mode == "single" and not target_file and len(file_paths) > 1:
            file_paths = file_paths[:1]

        if not file_paths:
            raise RuntimeError("No matching downloaded certificate files found to ingest.")

        new_record_ids: List[str] = []
        updated_record_ids: List[str] = []
        extracted_summaries: List[str] = []
        ingest_errors: List[str] = []
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
                for c in detail.get("certificates", []):
                    extracted_summaries.append(
                        f"Component: {c.get('component')}, Supplier: {c.get('supplier')}, "
                        f"Country: {c.get('country')}, Authority: {c.get('authority')}, "
                        f"Cert #: {c.get('certif_number')}, Issue Date: {c.get('issue_date')}, "
                        f"Exp Date: {c.get('exp_date')}, Link: {c.get('cert_link')}"
                    )
            except Exception as exc:
                logger.error(f" Ingest failed for {path}: {exc}")
                ingest_errors.append(f"{os.path.basename(path)}: {exc}")

        if not new_record_ids and not updated_record_ids and ingest_errors:
            raise RuntimeError(f"Failed to ingest certificate file(s): {'; '.join(ingest_errors)}")

        target_str = f" for '{target_file}'" if target_file else ""
        details_block = f" Extracted Metadata: {'; '.join(extracted_summaries)}" if extracted_summaries else ""
        if updated_record_ids and not new_record_ids:
            suffix = f" ({', '.join(updated_record_ids[:5])})"
            return f"Updated existing certificate record(s){target_str} in the database{suffix}.{details_block}"
        elif new_record_ids and updated_record_ids:
            return f"Added {len(new_record_ids)} new certificate(s) and updated {len(updated_record_ids)} existing record(s){target_str} in the database.{details_block}"
        else:
            suffix = f" ({', '.join(new_record_ids[:5])})" if new_record_ids else ""
            return f"Added {len(new_record_ids)} new certificate(s){target_str} to the database{suffix}.{details_block}"

class ExecuteRawSQLTool(BaseTool):
    """
    Executes raw SQL queries (DML/DDL) directly against the database.
    Useful for merging tables, performing complex aggregations, or advanced ETL logic.
    """
    name: str = "execute_raw_sql"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "The raw PostgreSQL query to execute."
                },
                "params": {
                    "type": "object",
                    "description": "Optional parameters to bind to the query (key-value mapping)."
                }
            },
            "required": ["sql_query"]
        }

    @staticmethod
    def generate_sql(objective: str) -> str:
        """Dynamically generate SQL just-in-time using the cached live database schema and an LLM."""
        from core.rag.sql_engine import build_schema_description
        from core.prompts import config_sql_generator_system_prompt

        try:
            schema_block = build_schema_description()
        except Exception as e:
            schema_block = f"(Schema unavailable: {e})"

        system_prompt = config_sql_generator_system_prompt(objective=objective, schema=schema_block)

        try:
            import json
            from core.llm import generate_json
            raw_response = generate_json(
                system_prompt=system_prompt,
                user_prompt=f"OBJECTIVE: {objective}\n\nReturn ONLY the raw JSON output matching the schema.",
                disable_thinking=True,
                max_tokens=1024,
                response_prefix='{"sql_query": "',
                hop_name="raw_sql_gen",
            )
            parsed = json.loads(raw_response)
            sql = str(parsed.get("sql_query", "")).strip()
            sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE).strip()
            sql = re.sub(r"\s*```$", "", sql).strip()
            return sql
        except Exception as e:
            return f"-- Failed to generate SQL: {e}"

    def execute(self, **kwargs: Any) -> Any:
        from storage.database import get_db_session
        from sqlalchemy import text
        
        sql_query = str(kwargs.get("sql_query") or "").strip()
        params = kwargs.get("params") or {}
        
        if not sql_query:
            description = str(kwargs.get("description") or kwargs.get("objective") or "Execute raw SQL query")
            sql_query = ExecuteRawSQLTool.generate_sql(description)
            if not sql_query or sql_query.startswith("-- Failed"):
                raise ValueError(f"sql_query is required and lazy generation failed: {sql_query}")
            
        with get_db_session() as db:
            try:
                result = db.execute(text(sql_query), params)
                rowcount = result.rowcount if result.rowcount is not None else -1
                
                if sql_query.lower().startswith("select") or result.returns_rows:
                    rows = result.fetchall()
                    data = [dict(row._mapping) for row in rows]
                    db.commit()
                    return {"status": "success", "rowcount": len(data), "data": data, "sql": sql_query}
                    
                db.commit()
                from storage.backup import export_database_to_sql
                export_database_to_sql()
                return f"Successfully executed SQL affecting {rowcount} record(s): {sql_query}"
            except Exception as exc:
                db.rollback()
                logger.error(f" ExecuteRawSQLTool failed: {exc}")
                raise


class DeleteRecordTool(BaseTool):
    """Deletes one or more records from a database table matching a target_id, filters, or delete_all."""
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
        from storage.dynamic_schema import resolve_physical_table_name

        table_name = resolve_physical_table_name(kwargs.get("table_name") or "certificates")
        target_id = (kwargs.get("target_id") or kwargs.get("certificate_id") or kwargs.get("target_file") or "").strip()
        filters = kwargs.get("filters") or kwargs.get("row_filter") or {}
        delete_all = bool(kwargs.get("delete_all"))
        description = str(kwargs.get("description") or "").strip()

        try:
            if target_id:
                mutation = build_mutation_sql("delete", table_name, values={}, fuzzy_match_query=target_id)
            elif filters and isinstance(filters, dict):
                mutation = build_mutation_sql("delete", table_name, values={}, row_filter=filters)
            elif delete_all:
                mutation = build_mutation_sql("delete", table_name, values={}, allow_full_table=True)
            else:
                raise ValueError(f"No literal target_id, filters, or delete_all specified for table '{table_name}'.")
                
            with get_db_session() as db:
                count = execute_mutation(db, mutation)
                from storage.backup import export_database_to_sql
                export_database_to_sql()
                filter_str = f" matching '{target_id}'" if target_id else (f" matching {filters}" if filters else " (all records)")
                return f"Successfully deleted {count} record(s) from '{table_name}'{filter_str}."
        except ValueError:
            objective = (
                f"Generate a single PostgreSQL DELETE statement on table '{table_name}'. "
                f"Task description: {description}. Filters/criteria: {filters or target_id}."
            )
            return ExecuteRawSQLTool().execute(objective=objective)


class UpdateRecordTool(BaseTool):
    """Update or transform records in a database table matching specific filters or search targets."""

    name: str = "update_record"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "Target table name (e.g. 'certificates')"},
                "values": {"type": "object", "description": "Field-value pairs to set (e.g. {'exp_date': '2999-01-01'})"},
                "filters": {"type": "object", "description": "Column filter conditions (e.g. {'exp_date': '1999-01-01'})"},
                "target_id": {"type": "string", "description": "Optional specific certificate_id or search string"},
                "allow_full_table": {"type": "boolean", "description": "Allow updating all records in table"},
            },
            "required": ["values"],
        }

    def execute(self, **kwargs) -> str:
        from core.agent.db_editor import build_mutation_sql, execute_mutation
        from storage.database import get_db_session
        from storage.dynamic_schema import resolve_physical_table_name

        table_name = resolve_physical_table_name(kwargs.get("table_name") or "certificates")
        values = kwargs.get("values") or kwargs.get("updates") or {}
        filters = kwargs.get("filters") or kwargs.get("row_filter") or {}
        target_id = kwargs.get("target_id")
        allow_full_table = bool(kwargs.get("allow_full_table", False))
        description = str(kwargs.get("description") or "").strip()

        if not values and not description:
            raise ValueError("Cannot update record: 'values' or 'description' is required.")

        try:
            mutation = build_mutation_sql(
                "update",
                table_name,
                values=values,
                row_filter=filters if filters else None,
                fuzzy_match_query=target_id if target_id else None,
                allow_full_table=allow_full_table or not (filters or target_id),
            )
            with get_db_session() as db:
                count = execute_mutation(db, mutation)
                if count == 0 and table_name == "components" and isinstance(filters, dict) and values:
                    comp_key = next(
                        (k for k in filters if k.strip().lower() == "componentname" and str(filters[k] or "").strip()),
                        None,
                    )
                    if comp_key is not None:
                        upsert_vals = {**filters, **values}
                        insert_mut = build_mutation_sql("insert", table_name, values=upsert_vals)
                        count = execute_mutation(db, insert_mut)
                from storage.backup import export_database_to_sql
                export_database_to_sql()
                return f"Successfully updated {count} record(s) in table '{table_name}' with {values}."
        except ValueError:
            objective = (
                f"Generate a single PostgreSQL UPDATE statement on table '{table_name}'. "
                f"Task description: {description}. Requested updates: {values}. Filter criteria: {filters or target_id}. "
                f"If replacing a year on a DATE column, preserve the month and day (e.g. MAKE_DATE or REPLACE(col::text, old, new)::date)."
            )
            return ExecuteRawSQLTool().execute(objective=objective)


class InsertRecordTool(BaseTool):
    """Inserts one or more new records into a database table (e.g. certificates, responsibles, components, sources)."""

    name: str = "insert_record"
    requires_approval: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Target table name (e.g. 'certificates', 'responsibles', 'components', 'sources')",
                },
                "values": {
                    "type": "object",
                    "description": "Column-value mapping for a single record to insert",
                },
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional list of column-value dicts when inserting multiple records in one step",
                },
            },
            "required": ["table_name"],
        }

    def execute(self, **kwargs: Any) -> Any:
        from sqlalchemy import text
        from core.agent.db_editor import build_mutation_sql, execute_mutation
        from storage.database import get_db_session
        from storage.dynamic_schema import resolve_physical_table_name

        table_name = resolve_physical_table_name(kwargs.get("table_name") or "certificates")
        description = str(kwargs.get("description") or "").strip()

        raw_records: List[Dict[str, Any]] = []
        if isinstance(kwargs.get("records"), list):
            raw_records = [r for r in kwargs["records"] if isinstance(r, dict) and r]
        elif isinstance(kwargs.get("values"), list):
            raw_records = [r for r in kwargs["values"] if isinstance(r, dict) and r]
        elif isinstance(kwargs.get("values"), dict) and kwargs["values"]:
            raw_records = [kwargs["values"]]
        elif isinstance(kwargs.get("record"), dict) and kwargs["record"]:
            raw_records = [kwargs["record"]]

        if not raw_records:
            if description:
                objective = (
                    f"Generate a PostgreSQL INSERT statement into table '{table_name}'. "
                    f"Task description: {description}."
                )
                return ExecuteRawSQLTool().execute(objective=objective)
            raise ValueError(f"Cannot insert into '{table_name}': 'values' or 'records' is required.")

        try:
            inserted_count = 0
            with get_db_session() as db:
                for row_dict in raw_records:
                    clean_row = dict(row_dict)
                    if table_name == "certificates":
                        try:
                            from core.normalization import normalize_certificate_dict
                            norm = normalize_certificate_dict(clean_row, db)
                            for k in list(clean_row.keys()):
                                kl = k.strip().lower()
                                if kl in norm and norm[kl] not in ("", None):
                                    clean_row[k] = norm[kl]
                            # Include auto-filled country/authority/supplier if derived
                            for auto_k in ("country", "authority", "supplier"):
                                if norm.get(auto_k) and not clean_row.get(auto_k):
                                    clean_row[auto_k] = norm[auto_k]
                        except Exception:
                            pass

                    # Upsert semantics on natural unique keys for responsibles (email) and components (componentname)
                    if table_name == "responsibles":
                        email_val = next(
                            (str(v).strip() for k, v in clean_row.items() if k.strip().lower() == "email" and str(v or "").strip()),
                            None,
                        )
                        if email_val:
                            existing = db.execute(
                                text("SELECT id FROM responsibles WHERE LOWER(TRIM(email)) = LOWER(TRIM(:email)) LIMIT 1"),
                                {"email": email_val},
                            ).fetchone()
                            if existing:
                                upd_mut = build_mutation_sql("update", table_name, values=clean_row, row_filter={"id": existing.id})
                                execute_mutation(db, upd_mut)
                                inserted_count += 1
                                continue

                    elif table_name == "components":
                        comp_val = next(
                            (str(v).strip() for k, v in clean_row.items() if k.strip().lower() == "componentname" and str(v or "").strip()),
                            None,
                        )
                        if comp_val:
                            existing = db.execute(
                                text("SELECT id FROM components WHERE LOWER(TRIM(componentname)) = LOWER(TRIM(:comp)) LIMIT 1"),
                                {"comp": comp_val},
                            ).fetchone()
                            if existing:
                                upd_mut = build_mutation_sql("update", table_name, values=clean_row, row_filter={"id": existing.id})
                                execute_mutation(db, upd_mut)
                                inserted_count += 1
                                continue

                    mutation = build_mutation_sql("insert", table_name, values=clean_row)
                    execute_mutation(db, mutation)
                    inserted_count += 1

                from storage.backup import export_database_to_sql
                export_database_to_sql()
                return f"Successfully inserted {inserted_count} record(s) into table '{table_name}': {raw_records}"
        except ValueError:
            objective = (
                f"Generate a PostgreSQL INSERT statement into table '{table_name}'. "
                f"Task description: {description}. Records to insert: {raw_records}."
            )
            return ExecuteRawSQLTool().execute(objective=objective)


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
        InsertRecordTool.name: InsertRecordTool(),
        DeleteRecordTool.name: DeleteRecordTool(),
        UpdateRecordTool.name: UpdateRecordTool(),
        DataConverterTool.name: DataConverterTool(),
        EmailDraftingTool.name: EmailDraftingTool(),
        RememberFactTool.name: RememberFactTool(),
        ManageSchemaTool.name: ManageSchemaTool(),
        ExecuteRawSQLTool.name: ExecuteRawSQLTool(),
    }