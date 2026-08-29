"""
core/agent/agent_loop.py — Step-Based Agent Execution Loop

Decomposes an AGENT_ACTION query into a sequence of concrete steps, executes the
READ steps immediately (narrating each so the chat shows progress), and gates
only the WRITE steps (e.g. adding to the database) behind human-in-the-loop
approval. Future actions register here as new executors.

Read steps never ask permission; only mutating steps do.
"""

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

@dataclass
class PlanResult:
    steps: List['Step']
    is_direct_command: bool = False


@dataclass
class Step:
    action: str
    kind: str  # "read" | "write"
    description: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    """State produced by read steps, consumed by write steps."""
    source_url: str = ""
    verified_urls: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    checked_report: str = ""


def get_registered_actions_prompt_block() -> str:
    """Dynamically formats the registered step executors for the LLM planner prompt."""
    from core.agent.tools import get_tool_registry
    registry = get_tool_registry()
    lines = []
    for idx, (name, tool) in enumerate(registry.items(), 1):
        doc = (tool.__doc__ or "").strip().split("\\n")[0] if tool.__doc__ else name
        if ":" in doc:
            doc = doc.split(":", 1)[1].strip()
        kind = "write" if tool.requires_approval else "read"
        req_approval = "THIS IS A DATABASE WRITE and requires human approval." if kind == "write" else "NO side effect, NO permission needed."
        lines.append(f'{idx}. "{name}" (kind "{kind}"): {doc} {req_approval}')
    return "\\n".join(lines)



from core.utils.history import format_conversation_history as _format_history_block


def _llm_plan(
    query: str,
    urls: List[str],
    history: Optional[List[Dict[str, str]]] = None,
) -> PlanResult:
    """Ask the LLM to decompose the query into steps; returns PlanResult with empty steps on failure."""
    from core.prompts import config_planner_system_prompt
    from core.llm import generate_json

    history_text = _format_history_block(history)
    urls_formatted = ", ".join(urls) if urls else ""
    user_prompt = (
        (f"{history_text}\n\n" if history_text else "")
        + f"USER REQUEST: {query}\n\n"
        + (f"ACTIVE URL(S): {urls_formatted}\n\n" if urls_formatted else "")
        + "Return ONLY the raw JSON output matching the schema."
    )
    primary_url = urls[0] if urls else ""
    try:
        raw_response = generate_json(
            system_prompt=config_planner_system_prompt(),
            user_prompt=user_prompt,
            disable_thinking=True,
        )
        parsed = json.loads(raw_response)
        is_direct = bool(parsed.get("is_direct_command", False))
        steps: List[Step] = []
        for item in parsed.get("steps", []):
            action = str(item.get("action", "")).strip()
            from core.agent.tools import get_tool_registry
            registry = get_tool_registry()
            if action not in registry:
                continue
            tool = registry[action]
            kind = "write" if tool.requires_approval else "read"
            target_file = str(item.get("target_file", "")).strip()
            target_id = str(item.get("target_id", "") or item.get("certificate_id", "")).strip()
            payload = {"url": primary_url, "urls": urls}
            if target_file:
                payload["target_file"] = target_file
            if target_id:
                payload["target_id"] = target_id
            if "ingest_mode" in item and isinstance(item["ingest_mode"], str):
                payload["ingest_mode"] = item["ingest_mode"].strip()
            if "filters" in item and isinstance(item["filters"], dict):
                payload["filters"] = item["filters"]
            if "delete_all" in item:
                payload["delete_all"] = bool(item["delete_all"])
            steps.append(
                Step(
                    action=action,
                    kind=kind,
                    description=str(item.get("description", "")).strip(),
                    payload=payload,
                )
            )
        return PlanResult(steps=steps, is_direct_command=is_direct)
    except Exception as exc:
        logger.warning(f" Agent planner failed ({exc}); using deterministic fallback.")
        return PlanResult(steps=[], is_direct_command=False)


def plan_agent_action(
    query: str,
    urls: Union[str, List[str], None] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> PlanResult:
    """
    Decompose an AGENT_ACTION query into a step plan using the LLM planner.

    Supports single URL strings or lists of URLs. If no URLs are provided,
    extracts all URLs from current query text or conversation history.
    Returns PlanResult with empty steps if no registered tool matches the request.
    """
    url_list: List[str] = []
    if isinstance(urls, str) and urls.strip():
        url_list = [urls.strip()]
    elif isinstance(urls, list):
        url_list = [u.strip() for u in urls if u and isinstance(u, str)]

    # URL matcher for inline extraction
    _url_re = re.compile(r"https?://[^\s'\"<>]+")

    # If no URLs passed explicitly, extract all URLs from query text
    if not url_list:
        matches = _url_re.findall(query or "")
        for m in matches:
            u = m.rstrip(".,;").strip()
            if u and u not in url_list:
                url_list.append(u)

    # Inherit URLs from history if missing in current turn
    if not url_list and history:
        for turn in reversed(history):
            matches = _url_re.findall(turn.get("content", ""))
            for m in matches:
                u = m.rstrip(".,;").strip()
                if u and u not in url_list:
                    url_list.append(u)
            if url_list:
                logger.info(f" Inherited active URL(s) from history: {url_list}")
                break

    plan_result = _llm_plan(query, url_list, history)

    # Ensure URL(s) are populated in step payloads
    primary_url = url_list[0] if url_list else ""
    for s in plan_result.steps:
        if url_list and not s.payload.get("urls"):
            s.payload["urls"] = url_list
        if primary_url and not s.payload.get("url"):
            s.payload["url"] = primary_url

    return plan_result



def iter_read_steps(plan: List[Step], url: str, artifacts: RunArtifacts):
    """
    Execute the READ steps of a plan in order, yielding a narration string per
    step as it completes, and filling `artifacts` for the write step.
    """
    from core.agent.tools import get_tool_registry
    registry = get_tool_registry()
    for step in plan:
        if step.kind != "read":
            continue
        if not step.payload.get("url"):
            step.payload["url"] = url
        tool = registry.get(step.action)
        if tool is None:
            continue
        try:
            yield tool.execute(**step.payload, artifacts=artifacts)
        except Exception as exc:
            logger.error(f" Agent step '{step.action}' failed: {exc}")
            yield f"Step '{step.description or step.action}' failed ({exc})."



def stage_write_step(write_step: Step, artifacts: RunArtifacts) -> Dict[str, Any]:
    """Stage a proposal for the write step, carrying its artifacts and descriptive payload."""
    from core.agent.proposals import proposal_manager

    target_file = write_step.payload.get("target_file")
    target_id = write_step.payload.get("target_id") or write_step.payload.get("certificate_id")
    filters = write_step.payload.get("filters")
    delete_all = write_step.payload.get("delete_all")
    row_filter = write_step.payload.get("row_filter")

    desc = write_step.description or write_step.action

    if write_step.action == "delete_record":
        try:
            from core.agent.db_editor import build_mutation_sql
            table_name = write_step.payload.get("table_name") or "certificates"
            
            if target_id:
                mutation = build_mutation_sql("delete", table_name, values={}, fuzzy_match_query=target_id)
            elif filters and isinstance(filters, dict):
                mutation = build_mutation_sql("delete", table_name, values={}, row_filter=filters)
            elif delete_all:
                mutation = build_mutation_sql("delete", table_name, values={}, allow_full_table=True)
            elif row_filter:
                mutation = build_mutation_sql("delete", table_name, values={}, row_filter=row_filter)
            else:
                mutation = {"preview": desc}
                
            desc = mutation.get("preview", desc)
        except Exception as e_desc:
            logger.warning(f" Could not build detailed deletion preview: {e_desc}")

    elif target_file and target_file not in desc:
        desc = f"{desc} ({target_file})"

    payload = {
        "action": write_step.action,
        "description": desc,
        "source_url": artifacts.source_url,
        "file_paths": artifacts.file_paths,
        "verified_urls": artifacts.verified_urls,
        "target_file": target_file,
        "target_id": target_id,
        "filters": filters,
        "delete_all": delete_all,
        "row_filter": row_filter,
    }
    if "table_name" in write_step.payload:
        payload["table_name"] = write_step.payload["table_name"]

    write_step.description = desc

    return proposal_manager.create_proposal(
        "AGENT_ACTION",
        payload,
        sql_preview=desc or artifacts.source_url or None,
    )




def resolve_write_step(
    proposal: Dict[str, Any],
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Execute the write step represented by an APPROVED proposal."""
    from core.agent.tools import get_tool_registry
    registry = get_tool_registry()
    
    payload = proposal.get("payload", {})
    action = payload.get("action")
    tool = registry.get(action)
    if tool is None:
        return "I couldn't resolve this action."
        
    artifacts = RunArtifacts(
        source_url=payload.get("source_url", ""),
        verified_urls=payload.get("verified_urls", []),
        file_paths=payload.get("file_paths", []),
    )
    
    return tool.execute(**payload, artifacts=artifacts, status_callback=status_callback)



def dispatch_resolve_write_step(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve an approved write-step proposal in a background thread and return
    immediately (OCR ingestion of many files can take minutes).
    """
    def _run():
        try:
            resolve_write_step(proposal)
        except Exception as exc:
            logger.error(f" Write-step resolution failed for {proposal.get('proposal_id')}: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return {
        "dispatched": True,
        "message": "Write action dispatched in the background.",
    }