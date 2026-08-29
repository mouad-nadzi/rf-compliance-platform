"""
core/agent — Agentic Tool Registry, Discovery & HITL Proposal Pipeline Package.

Provides the model-agnostic tool scaffolding, the autonomous PDF discovery
engine, the Database Editor engine, and the Human-In-The-Loop proposal staging
manager consumed by the agentic supervisor (core/rag/router.py). All components
are model-agnostic pure Python with no LLM/OCR coupling.
"""

from core.agent.tools import (
    BaseTool,
    WebDownloaderTool,
    DataConverterTool,
    EmailDraftingTool,
    get_tool_registry,
)
from core.agent.db_editor import build_mutation_sql, execute_mutation, ALLOWED_TABLES
from core.agent.proposals import ProposalManager, proposal_manager
from core.agent.scraper import discover_pdf_urls, DiscoveryResult
from core.agent.ingestor import ingest_document_file, set_ocr_engine
from core.agent.agent_loop import plan_agent_action, resolve_write_step, dispatch_resolve_write_step

__all__ = [
    "BaseTool",
    "WebDownloaderTool",
    "DataConverterTool",
    "EmailDraftingTool",
    "get_tool_registry",
    "build_mutation_sql",
    "execute_mutation",
    "ALLOWED_TABLES",
    "ProposalManager",
    "proposal_manager",
    "discover_pdf_urls",
    "DiscoveryResult",
    "ingest_document_file",
    "set_ocr_engine",
    "plan_agent_action",
    "resolve_write_step",
    "dispatch_resolve_write_step",
]