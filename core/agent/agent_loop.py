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
    """Session-scoped state produced by read steps or reconstructed from conversation history, consumed by write steps."""
    source_url: str = ""
    verified_urls: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    checked_report: str = ""


_INLINE_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")


def extract_artifacts_from_history(
    history: Optional[List[Dict[str, str]]] = None,
    explicit_urls: Optional[List[str]] = None,
) -> RunArtifacts:
    """
    Reconstruct session-scoped RunArtifacts (source portal URLs and discovered PDF URLs)
    directly from the current conversation's history and explicit query URLs.
    Eliminates process-global state so sessions never collide and state survives restarts.
    """
    explicit = [u.rstrip(".,;").strip() for u in (explicit_urls or []) if u and u.strip()]
    explicit_pdfs = [u for u in explicit if u.lower().split("?")[0].endswith(".pdf")]
    explicit_portals = [u for u in explicit if u not in explicit_pdfs]

    history_pdfs: List[str] = []
    history_portals: List[str] = []

    if history:
        for turn in reversed(history):
            content = str(turn.get("content", "") or "")
            matches = [m.rstrip(".,;").strip() for m in _INLINE_URL_RE.findall(content) if m.strip()]
            turn_pdfs = [u for u in matches if u.lower().split("?")[0].endswith(".pdf")]
            turn_portals = [u for u in matches if u not in turn_pdfs]
            if turn_pdfs and not history_pdfs:
                for u in turn_pdfs:
                    if u not in history_pdfs:
                        history_pdfs.append(u)
            for u in turn_portals:
                if u not in history_portals:
                    history_portals.append(u)

    # If the user explicitly supplied a new portal URL not seen in history, start fresh for that portal
    if explicit_portals and history_portals and explicit_portals[0] not in history_portals:
        return RunArtifacts(
            source_url=", ".join(explicit_portals),
            verified_urls=list(explicit_pdfs),
        )

    portal_urls = explicit_portals or history_portals[:1]
    verified_urls = list(explicit_pdfs)
    for u in history_pdfs:
        if u not in verified_urls:
            verified_urls.append(u)

    return RunArtifacts(
        source_url=", ".join(portal_urls),
        verified_urls=verified_urls,
    )


def get_registered_actions_prompt_block() -> str:
    """Dynamically formats the registered step executors for the LLM planner prompt."""
    from core.agent.tools import get_tool_registry
    registry = get_tool_registry()
    lines = []
    for idx, (name, tool) in enumerate(registry.items(), 1):
        doc = (tool.__doc__ or "").strip().splitlines()[0].strip() if tool.__doc__ else name
        if ":" in doc:
            doc = doc.split(":", 1)[1].strip()
        kind = "write" if tool.requires_approval else "read"
        req_approval = "THIS IS A DATABASE WRITE and requires human approval." if kind == "write" else "NO side effect, NO permission needed."
        lines.append(f'{idx}. "{name}" (kind "{kind}"): {doc} {req_approval}')
    return "\n".join(lines)



from core.utils.history import format_conversation_history as _format_history_block


# The prefix injected into the assistant turn of every planner call.
# llama.cpp's echo=False means the model's output starts AFTER these tokens,
# so the model is physically forced to continue the JSON object — it cannot
# produce a flat dict or any other format.
_PLAN_SCHEMA_PREFIX = '{"is_direct_command":'


def _llm_plan(
    query: str,
    urls: List[str],
    history: Optional[List[Dict[str, str]]] = None,
    error_feedback: str = "",
) -> PlanResult:
    """
    Ask the LLM to decompose the query into a structured plan.

    Uses response prefix injection: the assistant turn is pre-filled with
    '{"is_direct_command":' so the model is forced at the token level to
    continue with the correct schema. This eliminates the flat-dict failure
    mode without any query-specific heuristics or retries.

    When error_feedback is provided (ReAct retry loop), it is injected into
    the user prompt so the planner can correct its previous failed step.

    Returns PlanResult with empty steps on JSON parse failure.
    """
    from core.prompts import config_planner_system_prompt
    from core.llm import generate_json

    history_text = _format_history_block(history)
    urls_formatted = ", ".join(urls) if urls else ""
    error_block = (
        f"PREVIOUS EXECUTION ERROR (re-plan to fix this):\n{error_feedback}\n\n"
        if error_feedback
        else ""
    )
    user_prompt = (
        (f"{history_text}\n\n" if history_text else "")
        + error_block
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
            response_prefix=_PLAN_SCHEMA_PREFIX,
        )
        logger.info(f"PLANNER RAW RESPONSE:\n{raw_response}")
        parsed = json.loads(raw_response)

        is_direct = bool(parsed.get("is_direct_command", False))
        steps: List[Step] = []
        from core.agent.tools import get_tool_registry
        registry = get_tool_registry()

        for item in parsed.get("steps", []):
            action = str(item.get("action", "")).strip()
            if action not in registry:
                continue
            tool = registry[action]
            kind = "write" if tool.requires_approval else "read"

            payload: Dict[str, Any] = {}
            if primary_url:
                payload["url"] = primary_url
                payload["urls"] = urls

            for k, v in item.items():
                if k not in ("action", "kind", "description") and v is not None:
                    payload[k] = v

            if "certificate_id" in payload and "target_id" not in payload:
                payload["target_id"] = str(payload["certificate_id"])

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
        logger.warning(f" Agent planner failed ({exc}); returning empty plan.")
        return PlanResult(steps=[], is_direct_command=False)




def plan_agent_action(
    query: str,
    urls: Union[str, List[str], None] = None,
    history: Optional[List[Dict[str, str]]] = None,
    error_feedback: str = "",
) -> PlanResult:
    """
    Decompose an AGENT_ACTION query into a step plan using the LLM planner.

    Supports single URL strings or lists of URLs. If no URLs are provided,
    extracts all URLs from current query text.

    error_feedback is used by the ReAct retry loop in orchestrator.py: when
    a previous execution attempt failed, the error message is injected into
    the planner prompt so it can correct its plan.

    Returns PlanResult with empty steps if no registered tool matches the request.
    """
    url_list: List[str] = []
    if isinstance(urls, str) and urls.strip():
        url_list = [urls.strip()]
    elif isinstance(urls, list):
        url_list = [u.strip() for u in urls if u and isinstance(u, str)]

    # If no URLs passed explicitly, extract from query text or session history
    if not url_list:
        matches = _INLINE_URL_RE.findall(query or "")
        for m in matches:
            u = m.rstrip(".,;").strip()
            if u and u not in url_list:
                url_list.append(u)

    if not url_list and history:
        hist_artifacts = extract_artifacts_from_history(history)
        if hist_artifacts.source_url:
            for u in hist_artifacts.source_url.split(","):
                u = u.strip()
                if u and u not in url_list:
                    url_list.append(u)
        for u in hist_artifacts.verified_urls:
            if u not in url_list:
                url_list.append(u)

    plan_result = _llm_plan(query, url_list, history, error_feedback=error_feedback)

    # Ensure URL(s) are populated in step payloads if active URLs are present
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
    values = write_step.payload.get("values") or {}

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

    elif write_step.action == "update_record":
        try:
            from core.agent.db_editor import build_mutation_sql
            table_name = write_step.payload.get("table_name") or "certificates"
            mutation = build_mutation_sql(
                "update",
                table_name,
                values=values,
                row_filter=filters or row_filter,
                fuzzy_match_query=target_id,
                allow_full_table=not (filters or row_filter or target_id),
            )
            desc = mutation.get("preview", desc)
        except Exception as e_desc:
            logger.warning(f" Could not build detailed update preview: {e_desc}")

    elif write_step.action == "execute_raw_sql":
        sql_query = write_step.payload.get("sql_query")
        if not sql_query:
            try:
                from core.agent.tools import ExecuteRawSQLTool
                sql_query = ExecuteRawSQLTool.generate_sql(write_step.description)
                write_step.payload["sql_query"] = sql_query
                desc = sql_query
            except Exception as e:
                logger.warning(f" Could not generate lazy SQL: {e}")
                desc = f"Raw SQL execution (failed to generate preview: {e})"
        else:
            desc = sql_query

    elif target_file and target_file not in desc:
        desc = f"{desc} ({target_file})"

    payload = dict(write_step.payload)
    payload.update({
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
    })

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
        raise ValueError(f"Unknown write action '{action}'.")
        
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