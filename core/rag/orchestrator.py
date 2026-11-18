"""
core/rag/orchestrator.py — End-to-End Dual-Path RAG Orchestrator

Unifies intent classification (core.rag.router), Text-to-SQL (core.rag.sql_engine),
and Hybrid RRF vector retrieval (core.rag.hybrid_engine) into a single production entry point:
`answer_compliance_query(user_query, db_session)`.

Routing logic:
  - METADATA_QUERY   ──► Fast Text-to-SQL pipeline over PostgreSQL certificates table.
  - UNSTRUCTURED_RAG ──► Hybrid Dense/Sparse RRF + Adaptive Parent Expansion.
  - HYBRID_QUERY     ──► Dual-path execution combining SQL facts + Hybrid RRF context.
  - CASUAL_CONVERSATION ──► Short-circuit conversational reply (no DB, no retrieval).
  - AGENT_ACTION     ──► Short-circuit tool dispatch placeholder (HITL gated, deferred).
"""

import time
import logging
import json
import re
import threading
from typing import Any, Dict, List, Optional

from server import config
from core.rag.unified_agent import QueryIntent, unified_tool_select
from core.rag.sql_engine import (
    execute_metadata_query,
    execute_sql_structured_context,
    FALLBACK_MESSAGE as SQL_FALLBACK,
)
from core.rag.hybrid_engine import (
    retrieve_hybrid_context,
    execute_unstructured_query,
    build_unstructured_qa_prompt,
    FALLBACK_NOT_FOUND_MESSAGE,
)

logger = logging.getLogger(__name__)


from core.utils.history import format_conversation_history as _format_history


def _reformulate_query(user_query: str, history) -> str:
    """
    Rewrites the latest query into a standalone, self-contained query that carries
    forward entities from prior turns (resolves "the others", "these", "them", etc.).

    Falls back to the original query unchanged if no history exists or the LLM rewrite fails.
    """
    if not history:
        return user_query

    history_text = _format_history(history)
    rewritten = user_query

    try:
        from core.prompts import config_query_rewrite_system_prompt
        from core.llm import generate_json

        raw_response = generate_json(
            system_prompt=config_query_rewrite_system_prompt(),
            user_prompt=(
                f"{history_text}\n\n"
                f"USER'S LATEST QUERY: {user_query}\n\n"
                f"Return ONLY the raw JSON output matching the schema."
            ),
            disable_thinking=True,
            max_tokens=128,
            response_prefix='{"rewritten_query": "',
            hop_name="query_rewrite",
        )

        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            candidate = str(parsed.get("rewritten_query", "")).strip()
            if candidate:
                rewritten = candidate
    except Exception as exc:
        logger.warning(f" Query reformulation failed ({exc}); using original query.")

    if rewritten.strip() != user_query.strip():
        logger.info(f" Query rewritten: '{user_query[:50]}' -> '{rewritten[:80]}'")
    return rewritten


def _casual_conversation_reply(user_query: str, history_text: str = "") -> str:
    """
    Generates a friendly casual reply WITHOUT touching the database or retrieval.

    Uses a lightweight LLM call (disable_thinking) against a dedicated casual
    prompt; falls back to a canned greeting if generation fails. Stateless.
    """
    from core.prompts import config_casual_conversation_system_prompt
    from core.llm import generate_json

    try:
        user_prompt = (
            f"{history_text}\n\n" if history_text else ""
        ) + f"USER MESSAGE: {user_query}\n\nReturn ONLY the raw JSON output matching the schema."
        raw_response = generate_json(
            system_prompt=config_casual_conversation_system_prompt(),
            user_prompt=user_prompt,
            disable_thinking=True,
            max_tokens=384,
            response_prefix='{"answer": "',
            hop_name="casual_reply",
        )
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            answer = str(parsed.get("answer", "")).strip()
            if answer:
                return answer
    except Exception as exc:
        logger.warning(f" Casual conversation reply generation failed: {exc}")

    return (
        "Hello! I'm here to help with your certificate compliance questions. "
        "Ask me about suppliers, countries, expiry dates, or document contents."
    )


def _build_action_synthesis_user_prompt(
    user_query: str,
    action_name: str,
    execution_outcome: Any,
    read_narrations: Optional[List[str]] = None,
    history_text: str = "",
) -> str:
    """Builds a prefix-aligned user prompt for agent action synthesis."""
    if isinstance(execution_outcome, dict):
        outcome_str = json.dumps(execution_outcome, ensure_ascii=False, default=str)
    elif execution_outcome is not None:
        outcome_str = str(execution_outcome)
    else:
        outcome_str = "Action completed."

    read_context = ""
    if read_narrations:
        valid_narrations = [str(n).strip() for n in read_narrations if n and str(n).strip()]
        if valid_narrations:
            read_context = "\n".join(f"- {n}" for n in valid_narrations)

    prompt_parts = []
    if history_text:
        prompt_parts.append(history_text)
    prompt_parts.append(f"USER REQUEST: {user_query}")
    prompt_parts.append(f"ACTION EXECUTED: {action_name}")
    prompt_parts.append(f"EXECUTION OUTCOME: {outcome_str}")
    if read_context:
        prompt_parts.append(f"INSPECTION / CONTEXT FINDINGS:\n{read_context}")
    prompt_parts.append("\nReturn ONLY the raw JSON output matching the schema.")
    return "\n\n".join(prompt_parts)


def _fallback_action_outcome_str(execution_outcome: Any) -> str:
    if isinstance(execution_outcome, dict):
        status = execution_outcome.get("status", "completed")
        msg = execution_outcome.get("message", "Action executed.")
        rowc = execution_outcome.get("rowcount")
        count_str = f" ({rowc} rows affected)" if rowc is not None and rowc >= 0 else ""
        return f"{status.title()}: {msg}{count_str}"
    elif execution_outcome:
        return str(execution_outcome)
    return "Action executed successfully."


def _synthesize_agent_action_reply(
    user_query: str,
    action_name: str,
    execution_outcome: Any,
    read_narrations: Optional[List[str]] = None,
    history_text: str = "",
) -> str:
    """
    Synthesizes a natural, helpful, conversational response summarizing the
    execution outcome of any agent action (mutations, schema changes, ingestions, inspections, etc.).
    """
    from core.prompts import config_action_synthesis_system_prompt
    from core.llm import generate_json

    user_prompt = _build_action_synthesis_user_prompt(
        user_query=user_query,
        action_name=action_name,
        execution_outcome=execution_outcome,
        read_narrations=read_narrations,
        history_text=history_text,
    )

    try:
        raw_response = generate_json(
            system_prompt=config_action_synthesis_system_prompt(),
            user_prompt=user_prompt,
            disable_thinking=True,
            max_tokens=512,
            response_prefix='{"answer": "',
            hop_name="action_synthesis",
        )
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            ans = str(parsed.get("answer", "")).strip()
            if ans:
                return ans
    except Exception as exc:
        logger.warning(f" Agent action synthesis generation failed: {exc}")

    return _fallback_action_outcome_str(execution_outcome)


def _stream_synthesize_agent_action_reply(
    user_query: str,
    action_name: str,
    execution_outcome: Any,
    collector: Dict[str, Any],
    read_narrations: Optional[List[str]] = None,
    history_text: str = "",
):
    """
    Streams the agent action synthesis token-by-token in real time without artificial sleep delays.
    """
    from core.prompts import config_action_synthesis_system_prompt
    from core.rag.qa import stream_synthesize_answer

    user_prompt = _build_action_synthesis_user_prompt(
        user_query=user_query,
        action_name=action_name,
        execution_outcome=execution_outcome,
        read_narrations=read_narrations,
        history_text=history_text,
    )
    for evt in stream_synthesize_answer(
        config_action_synthesis_system_prompt(),
        user_prompt,
        user_query,
        disable_thinking=True,
        max_tokens=512,
        hop_name="action_synthesis",
    ):
        if evt["type"] in ("token", "thinking"):
            yield evt
        elif evt["type"] == "synthesis_done":
            collector["answer"] = evt.get("answer", "")

    if not collector.get("answer"):
        fb = _fallback_action_outcome_str(execution_outcome)
        collector["answer"] = fb
        yield {"type": "token", "text": fb}


_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")


def _extract_urls(query: str) -> List[str]:
    matches = _URL_RE.findall(query or "")
    urls: List[str] = []
    for m in matches:
        u = m.rstrip(".,;").strip()
        if u and u not in urls:
            urls.append(u)
    return urls


def _extract_url(query: str) -> str:
    urls = _extract_urls(query)
    return urls[0] if urls else ""


def _handle_chat_approval_stream(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
):
    """
    Generator: handles chat-based HITL confirmations for the most recent PENDING
    AGENT_ACTION proposal scoped to the active session and streams live status events.
    """
    import queue
    import threading
    from core.agent.proposals import proposal_manager
    from core.llm import get_turn_telemetry

    query = str(user_query or "").strip()
    if not query:
        return

    pending = [
        p for p in proposal_manager.list_pending_proposals(session_id=session_id)
        if p.get("type") == "AGENT_ACTION"
    ]
    if not pending:
        return
    proposal = pending[0]
    proposal_id = proposal["proposal_id"]
    desc = (proposal.get("payload") or {}).get("description") or proposal.get("sql_preview") or "Agent action"

    decision = "NEW_QUERY"
    try:
        from core.prompts import config_approval_classifier_system_prompt
        from core.llm import generate_json

        system_prompt = config_approval_classifier_system_prompt(
            proposal_description=desc,
            user_message=query,
        )
        raw_res = generate_json(
            system_prompt=system_prompt,
            user_prompt=f"USER MESSAGE: {query}\n\nReturn ONLY raw JSON.",
            disable_thinking=True,
            max_tokens=64,
            response_prefix='{"decision": "',
            hop_name="approval_classifier",
        )
        parsed = json.loads(raw_res)
        if isinstance(parsed, dict):
            decision = str(parsed.get("decision", "")).strip().upper()
    except Exception as exc:
        logger.warning(f" LLM approval classifier failed ({exc}); using keyword fallback.")
        q_lower = query.lower()
        if any(w in q_lower for w in ["yes", "go", "do it", "sure", "proceed", "approve", "ok", "okay", "yep", "yeah", "confirm"]):
            decision = "APPROVE"
        elif any(w in q_lower for w in ["no", "cancel", "stop", "don't", "dont", "skip", "reject", "decline", "never mind"]):
            decision = "REJECT"

    if decision == "REJECT":
        try:
            for p in pending:
                proposal_manager.update_status(p["proposal_id"], "REJECTED")
        except Exception as exc:
            logger.warning(f" Could not reject proposal {proposal_id}: {exc}")
            return
        logger.info(f" Chat HITL: rejected {len(pending)} AGENT_ACTION proposal(s) starting with {proposal_id}.")
        reply = (
            f"OK - I've rejected proposal `{proposal_id}`. No action was executed. "
            "Let me know if you'd like to try a different link or command."
        )
        yield {"type": "status", "stage": "agent_action", "message": "Proposal rejected."}
        yield {"type": "token", "text": reply}
        yield {
            "type": "done",
            "answer": reply,
            "intent": QueryIntent.AGENT_ACTION.value,
            "reasoning": "User rejected pending agent action proposal.",
            "sources": [],
            "latency_ms": 0.0,
            "timings": get_turn_telemetry(),
        }
        return

    if decision == "APPROVE":
        yield {"type": "status", "stage": "agent_action", "message": "Confirming approval..."}
        try:
            for p in pending:
                proposal_manager.update_status(p["proposal_id"], "APPROVED")
        except Exception as exc:
            logger.warning(f" Could not approve proposal {proposal_id}: {exc}")
            return

        logger.info(f" Chat HITL: approved {len(pending)} AGENT_ACTION proposal(s) starting with {proposal_id}.")
        from core.agent.agent_loop import resolve_write_step

        executed_outcomes = []
        action_names = []
        action_descs = []

        for p in pending:
            p_payload = p.get("payload") or {}
            action_names.append(p_payload.get("action") or "agent_action")
            action_descs.append(p_payload.get("description") or p.get("sql_preview") or desc)

            q: queue.Queue = queue.Queue()

            def _cb(stage_name: str, msg_text: str, _q=q):
                _q.put({"type": "status", "stage": stage_name, "message": msg_text})

            exec_outcome = [None]
            def _worker(_p=p, _cb_fn=_cb, _out=exec_outcome, _q=q):
                try:
                    _out[0] = resolve_write_step(_p, status_callback=_cb_fn)
                except Exception as e:
                    _out[0] = f"Execution failed: {e}"
                finally:
                    _q.put(None)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            while True:
                try:
                    item = q.get(timeout=0.05)
                    if item is None:
                        break
                    yield item
                except queue.Empty:
                    if not t.is_alive() and q.empty():
                        break

            t.join()
            executed_outcomes.append(exec_outcome[0])

        raw_outcome = executed_outcomes[0] if len(executed_outcomes) == 1 else "\n".join(str(o) for o in executed_outcomes)
        action_name = ", ".join(action_names)
        action_desc = "; ".join(action_descs)

        yield {"type": "status", "stage": "generation", "message": "Synthesizing result..."}
        collector: Dict[str, Any] = {}
        yield from _stream_synthesize_agent_action_reply(
            user_query=action_desc,
            action_name=action_name,
            execution_outcome=raw_outcome,
            collector=collector,
            history_text=_format_history(history),
        )
        reply = collector.get("answer", "")

        yield {
            "type": "done",
            "answer": reply,
            "intent": QueryIntent.AGENT_ACTION.value,
            "reasoning": "User approved pending agent action proposal.",
            "sources": [],
            "latency_ms": 0.0,
            "timings": get_turn_telemetry(),
        }
        return


def answer_compliance_query(
    user_query: str,
    db_session=None,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
    deep_thinking: bool = False,
) -> Dict[str, Any]:
    """
    Unified entry point for query processing across all routing paths.
    """
    from core.llm import start_turn_telemetry, get_turn_telemetry

    start_turn_telemetry()
    t_start = time.perf_counter()
    clean_query = str(user_query or "").strip()
    history_text = _format_history(history)
    disable_thinking = not deep_thinking

    if not clean_query:
        return {
            "answer": "Please submit a non-empty question.",
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": "Blank query submitted.",
            "sources": [],
            "latency_ms": 0.0,
            "timings": [],
        }

    resolved_query = clean_query

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal

        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        logger.info(f" Unified agent tool selection for: '{clean_query[:60]}...'")
        tool_decision = unified_tool_select(clean_query, history=history)
        intent = tool_decision.get("intent", QueryIntent.UNSTRUCTURED_RAG.value)
        reasoning = tool_decision.get("reasoning", "Unified agent default routing.")

        logger.info(f" Unified agent decision: tool={tool_decision.get('tool')} intent={intent} ({reasoning[:60]})'")

        answer_text = ""
        sources: List[Dict[str, Any]] = []

        if intent == QueryIntent.METADATA_QUERY.value:
            logger.info(" Executing METADATA_QUERY path (Text-to-SQL)...")
            answer_text = execute_metadata_query(
                clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
            )
            sources = [{"type": "database", "table": "certificates", "query_type": "relational_sql"}]

        elif intent == QueryIntent.HYBRID_QUERY.value:
            logger.info(" Executing HYBRID_QUERY path (Dual-Path SQL + Vector RRF)...")
            resolved_query = _reformulate_query(clean_query, history)
            try:
                sql_facts = execute_sql_structured_context(clean_query, db_session, history_text=history_text)
                hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
                context_text = hybrid_payload.get("context_text", "").strip()
                sources = hybrid_payload.get("sources", [])

                if context_text and sql_facts:
                    dual_prompt = (
                        f"{history_text}\n\n" if history_text else ""
                    ) + (
                        f"--- STRUCTURED METADATA FACTS (POSTGRESQL) ---\n"
                        f"{sql_facts}\n\n"
                        f"--- UNSTRUCTURED DOCUMENT CONTEXT ---\n"
                        f"{context_text}\n\n"
                        f"USER QUESTION: {clean_query}\n\n"
                        f"Synthesize a unified, comprehensive answer combining both the structured metadata facts "
                        f"and unstructured document details above. Preserve citation markers (e.g. <Page X>)."
                    )

                    from core.prompts import config_qa_system_prompt
                    from core.llm import generate_json

                    raw_response = generate_json(
                        system_prompt=config_qa_system_prompt(),
                        user_prompt=dual_prompt,
                        disable_thinking=disable_thinking,
                        max_tokens=1024,
                        response_prefix='{"answer": "' if disable_thinking else "",
                        hop_name="hybrid_synthesis",
                    )

                    try:
                        parsed = json.loads(raw_response)
                        if isinstance(parsed, dict) and "answer" in parsed:
                            answer_text = str(parsed.get("answer", "")).strip()
                    except Exception:
                        pass

                    if not answer_text:
                        answer_text = re.sub(r"```(?:json)?\s*|\s*```", "", raw_response).strip()

                elif context_text:
                    answer_text = execute_unstructured_query(
                        resolved_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                    )
                else:
                    answer_text = execute_metadata_query(
                        clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                    )
                    if not answer_text or answer_text == SQL_FALLBACK:
                        answer_text = FALLBACK_NOT_FOUND_MESSAGE

            except Exception as exc:
                logger.warning(f" Dual-path hybrid execution failed: {exc}. Falling back to unstructured query.")
                answer_text = execute_unstructured_query(
                    resolved_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                )

        elif intent == QueryIntent.CASUAL_CONVERSATION.value:
            logger.info(" Executing CASUAL_CONVERSATION path (no DB, no retrieval)...")
            answer_text = _casual_conversation_reply(clean_query, history_text)

        elif intent == QueryIntent.AGENT_ACTION.value:
            logger.info(" Executing AGENT_ACTION path (step-based execution)...")
            tokens = []
            for event in answer_compliance_query_stream(
                user_query,
                db_session=db_session,
                history=history,
                session_id=session_id,
                deep_thinking=deep_thinking,
                reset_telemetry=False,
            ):
                if event.get("type") == "token":
                    tokens.append(event.get("text", ""))
            answer_text = "".join(tokens)
            sources = [{"type": "agent_action", "url": _extract_url(clean_query)}]

        elif intent == QueryIntent.UNSTRUCTURED_RAG.value:
            logger.info(" Executing UNSTRUCTURED_RAG path (Hybrid RRF + Parent Expansion)...")
            resolved_query = _reformulate_query(clean_query, history)
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                answer_text = execute_unstructured_query(
                    resolved_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                )
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                answer_text = execute_metadata_query(
                    clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                )
                if not answer_text or answer_text == SQL_FALLBACK:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE

        else:
            logger.warning(f" Unhandled intent '{intent}'; falling back to UNSTRUCTURED_RAG path.")
            resolved_query = _reformulate_query(clean_query, history)
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                answer_text = execute_unstructured_query(
                    resolved_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                )
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                answer_text = execute_metadata_query(
                    clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                )
                if not answer_text or answer_text == SQL_FALLBACK:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return {
            "answer": answer_text if answer_text else FALLBACK_NOT_FOUND_MESSAGE,
            "intent": intent,
            "reasoning": reasoning,
            "sources": sources,
            "latency_ms": round(latency_ms, 2),
            "timings": get_turn_telemetry(),
        }

    except Exception as exc:
        logger.error(f" Exception in answer_compliance_query: {exc}. Executing emergency fallback.")
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        try:
            fallback_answer = execute_unstructured_query(
                resolved_query, db_session, history_text=history_text, disable_thinking=disable_thinking
            )
        except Exception:
            fallback_answer = FALLBACK_NOT_FOUND_MESSAGE

        return {
            "answer": fallback_answer,
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": f"Emergency fallback triggered due to exception: {str(exc)}",
            "sources": [],
            "latency_ms": round(latency_ms, 2),
            "timings": get_turn_telemetry(),
        }

    finally:
        if close_session_on_exit:
            db_session.close()


def _stream_and_collect(
    system_prompt: str,
    user_prompt: str,
    query: str,
    collector: Dict[str, Any],
    disable_thinking: bool = True,
    max_tokens: int = 1024,
    hop_name: str = "qa_synthesis",
):
    """
    Forwards streaming answer-token events from a synthesis and records the final
    (answer, citations) into `collector` via the synthesis_done event.
    """
    from core.rag.qa import stream_synthesize_answer

    for evt in stream_synthesize_answer(
        system_prompt,
        user_prompt,
        query,
        disable_thinking=disable_thinking,
        max_tokens=max_tokens,
        hop_name=hop_name,
    ):
        if evt["type"] in ("token", "thinking"):
            yield evt
        elif evt["type"] == "synthesis_done":
            collector["answer"] = evt.get("answer", "")
            collector["citations"] = evt.get("citations", [])


def answer_compliance_query_stream(
    user_query: str,
    db_session=None,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
    deep_thinking: bool = False,
    reset_telemetry: bool = True,
):
    """
    Streaming variant of answer_compliance_query() for the interactive chat UI.
    """
    from core.llm import start_turn_telemetry, get_turn_telemetry

    if reset_telemetry:
        start_turn_telemetry()

    t_start = time.perf_counter()
    clean_query = str(user_query or "").strip()
    history_text = _format_history(history)
    disable_thinking = not deep_thinking

    yield {"type": "status", "stage": "routing", "message": "Understanding your request..."}

    if not clean_query:
        yield {
            "type": "done",
            "answer": "Please submit a non-empty question.",
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": "Blank query submitted.",
            "sources": [],
            "latency_ms": 0.0,
            "timings": [],
        }
        return

    # Chat-based HITL confirmation ("yes" / "approve" / "no") for a pending
    # AGENT_ACTION proposal scoped to this session, handled before routing.
    has_approval_events = False
    for evt in _handle_chat_approval_stream(clean_query, history=history, session_id=session_id):
        has_approval_events = True
        yield evt
    if has_approval_events:
        return

    resolved_query = clean_query

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal

        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        # Step 1: Unified tool selection (single LLM hop replaces 2-stage router)
        tool_decision = unified_tool_select(clean_query, history=history)
        intent = tool_decision.get("intent", QueryIntent.UNSTRUCTURED_RAG.value)
        reasoning = tool_decision.get("reasoning", "Unified agent default routing.")
        yield {
            "type": "status",
            "stage": "routing",
            "message": f"Tool selected: {tool_decision.get('tool', intent)}",
            "intent": intent,
        }

        answer_text = ""
        sources: List[Dict[str, Any]] = []

        # Step 2: Route Execution
        if intent == QueryIntent.METADATA_QUERY.value:
            yield {"type": "status", "stage": "retrieval", "message": "Querying certificate database..."}
            sources = [{"type": "database", "table": "certificates", "query_type": "relational_sql"}]
            from core.rag.sql_engine import stream_metadata_answer
            answer_text = ""
            for chunk in stream_metadata_answer(
                clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
            ):
                answer_text += chunk
                yield {"type": "token", "text": chunk}
            if not answer_text:
                answer_text = SQL_FALLBACK
                yield {"type": "token", "text": answer_text}

        elif intent == QueryIntent.HYBRID_QUERY.value:
            yield {"type": "status", "stage": "retrieval", "message": "Combining database facts and document context..."}
            resolved_query = _reformulate_query(clean_query, history)
            try:
                sql_facts = execute_sql_structured_context(clean_query, db_session, history_text=history_text)
                hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
                context_text = hybrid_payload.get("context_text", "").strip()
                sources = hybrid_payload.get("sources", [])

                if context_text and sql_facts:
                    dual_prompt = (
                        f"{history_text}\n\n" if history_text else ""
                    ) + (
                        f"--- STRUCTURED METADATA FACTS (POSTGRESQL) ---\n"
                        f"{sql_facts}\n\n"
                        f"--- UNSTRUCTURED DOCUMENT CONTEXT ---\n"
                        f"{context_text}\n\n"
                        f"USER QUESTION: {clean_query}\n\n"
                        f"Synthesize a unified, comprehensive answer combining both the structured metadata facts "
                        f"and unstructured document details above. Preserve citation markers (e.g. <Page X>)."
                    )
                    yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                    collector: Dict[str, Any] = {}
                    yield from _stream_and_collect(
                        config_qa_system_prompt(),
                        dual_prompt,
                        clean_query,
                        collector,
                        disable_thinking=disable_thinking,
                        hop_name="hybrid_synthesis",
                    )
                    answer_text = collector.get("answer", "")
                elif context_text:
                    yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                    user_prompt = build_unstructured_qa_prompt(clean_query, context_text, history_text)
                    collector: Dict[str, Any] = {}
                    yield from _stream_and_collect(
                        config_qa_system_prompt(),
                        user_prompt,
                        clean_query,
                        collector,
                        disable_thinking=disable_thinking,
                    )
                    answer_text = collector.get("answer", "")
                else:
                    from core.rag.sql_engine import stream_metadata_answer
                    answer_text = ""
                    for chunk in stream_metadata_answer(
                        clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                    ):
                        answer_text += chunk
                        yield {"type": "token", "text": chunk}
                    if not answer_text or answer_text == SQL_FALLBACK:
                        answer_text = FALLBACK_NOT_FOUND_MESSAGE
                        yield {"type": "token", "text": answer_text}
            except Exception as exc:
                logger.warning(f" Dual-path hybrid streaming failed: {exc}. Falling back to unstructured query.")
                hybrid_fallback = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
                fallback_context = hybrid_fallback.get("context_text", "").strip()
                if fallback_context:
                    user_prompt = build_unstructured_qa_prompt(clean_query, fallback_context, history_text)
                    collector: Dict[str, Any] = {}
                    yield from _stream_and_collect(
                        config_qa_system_prompt(),
                        user_prompt,
                        clean_query,
                        collector,
                        disable_thinking=disable_thinking,
                    )
                    answer_text = collector.get("answer", "")
                else:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE
                    yield {"type": "token", "text": answer_text}

        elif intent == QueryIntent.CASUAL_CONVERSATION.value:
            yield {"type": "status", "stage": "casual", "message": "Generating reply..."}
            from core.prompts import config_casual_conversation_system_prompt
            casual_prompt = (
                f"{history_text}\n\n" if history_text else ""
            ) + f"USER MESSAGE: {clean_query}\n\nReturn ONLY the raw JSON output matching the schema."
            collector: Dict[str, Any] = {}
            yield from _stream_and_collect(
                config_casual_conversation_system_prompt(),
                casual_prompt,
                clean_query,
                collector,
                disable_thinking=True,
                max_tokens=384,
                hop_name="casual_reply",
            )
            answer_text = collector.get("answer", "")
            if not answer_text:
                answer_text = _casual_conversation_reply(clean_query, history_text)
                yield {"type": "token", "text": answer_text}

        elif intent == QueryIntent.AGENT_ACTION.value:
            yield {"type": "status", "stage": "agent_action", "message": "Planning agent action..."}
            from core.agent.agent_loop import (
                plan_agent_action, iter_read_steps, stage_write_step,
                resolve_write_step, extract_artifacts_from_history
            )
            explicit_urls = _extract_urls(clean_query)
            artifacts = extract_artifacts_from_history(history=history, explicit_urls=explicit_urls)

            urls = list(explicit_urls)
            if not urls and artifacts.source_url:
                urls = [u.strip() for u in artifacts.source_url.split(",") if u.strip()]
            if not urls and artifacts.verified_urls:
                urls = list(artifacts.verified_urls)

            plan_result = plan_agent_action(clean_query, urls=urls, history=history)
            primary_url = urls[0] if urls else ""

            _parts = []
            for step in plan_result.steps:
                if step.kind == "read":
                    desc = step.description or step.action
                    yield {"type": "status", "stage": "agent_action", "message": f"Executing: {desc}..."}
                    for narration in iter_read_steps([step], primary_url, artifacts):
                        _parts.append(narration)

            write_steps = [s for s in plan_result.steps if s.kind == "write"]
            if write_steps:
                is_direct = plan_result.is_direct_command

                if is_direct:
                    import queue
                    import threading

                    MAX_AGENT_RETRIES = 3
                    error_feedback = ""
                    raw_res: Any = None
                    executed_outcomes: List[Any] = []
                    answer_text = ""

                    for attempt in range(MAX_AGENT_RETRIES):
                        if attempt > 0:
                            yield {"type": "status", "stage": "agent_action", "message": f"Execution failed. Re-planning (attempt {attempt + 1}/{MAX_AGENT_RETRIES})..."}
                            plan_result = plan_agent_action(clean_query, urls=urls, history=history, error_feedback=error_feedback)
                            write_steps = [s for s in plan_result.steps if s.kind == "write"]
                            if not write_steps:
                                raw_res = "Failed to plan a valid recovery action."
                                break

                        executed_outcomes = []
                        attempt_failed = False

                        for step_idx, write_step in enumerate(write_steps, 1):
                            step_desc = write_step.description or write_step.action
                            yield {
                                "type": "status",
                                "stage": "agent_action",
                                "message": f"Executing step {step_idx}/{len(write_steps)}: {step_desc} (attempt {attempt + 1})...",
                            }

                            q: queue.Queue = queue.Queue()

                            def _cb(stage_name: str, msg_text: str, _q=q):
                                _q.put({"type": "status", "stage": stage_name, "message": msg_text})

                            proposal_payload = dict(write_step.payload)
                            proposal_payload.update({
                                "action": write_step.action,
                                "description": write_step.description,
                                "source_url": artifacts.source_url,
                                "file_paths": artifacts.file_paths,
                                "verified_urls": artifacts.verified_urls,
                                "session_id": session_id,
                            })
                            dummy_proposal = {"payload": proposal_payload}

                            exec_outcome = [None]
                            step_failed = [False]

                            def _worker(_prop=dummy_proposal, _cb_fn=_cb, _out=exec_outcome, _fail=step_failed, _q=q):
                                try:
                                    _out[0] = resolve_write_step(_prop, status_callback=_cb_fn)
                                except Exception as e:
                                    _fail[0] = True
                                    _out[0] = f"Execution failed: {e}"
                                finally:
                                    _q.put(None)

                            t = threading.Thread(target=_worker, daemon=True)
                            t.start()

                            while True:
                                try:
                                    item = q.get(timeout=0.05)
                                    if item is None:
                                        break
                                    yield item
                                except queue.Empty:
                                    if not t.is_alive() and q.empty():
                                        break

                            t.join()

                            if step_failed[0]:
                                attempt_failed = True
                                prior_str = f"Prior completed steps: {executed_outcomes}. " if executed_outcomes else ""
                                error_feedback = f"{prior_str}Step {step_idx} ({write_step.action}) failed: {exec_outcome[0]}"
                                raw_res = error_feedback
                                break
                            else:
                                executed_outcomes.append(exec_outcome[0])

                        if attempt_failed:
                            continue
                        else:
                            error_feedback = ""
                            if len(executed_outcomes) == 1:
                                raw_res = executed_outcomes[0]
                            else:
                                raw_res = [
                                    {"step": idx + 1, "action": ws.action, "outcome": out}
                                    for idx, (ws, out) in enumerate(zip(write_steps, executed_outcomes))
                                ]
                            break

                    action_summary = ", ".join(ws.action for ws in write_steps) if write_steps else "agent_action"
                    yield {"type": "status", "stage": "generation", "message": "Synthesizing response..."}
                    collector: Dict[str, Any] = {}
                    yield from _stream_synthesize_agent_action_reply(
                        user_query=clean_query,
                        action_name=action_summary,
                        execution_outcome=raw_res,
                        collector=collector,
                        read_narrations=_parts,
                        history_text=history_text,
                    )
                    answer_text = collector.get("answer", "")

                else:
                    # Implicit action decided by agent -> stage all write_steps for HITL approval!
                    staged_descs = []
                    for ws in write_steps:
                        try:
                            stage_write_step(ws, artifacts, session_id=session_id)
                        except Exception as exc:
                            logger.warning(f" Could not stage write step: {exc}")
                        target_f = ws.payload.get("target_file")
                        desc = ws.description or ws.action
                        if target_f and target_f not in desc:
                            desc = f"{desc} ({target_f})"
                        staged_descs.append(desc)

                    combined_desc = "; ".join(staged_descs)
                    _parts.append(
                        f"To do the final step ({combined_desc}), "
                        "I need your approval. Is that OK? Reply 'yes' to proceed or 'no' to cancel."
                    )
                    answer_text = "\n\n".join(_parts)
                    yield {"type": "token", "text": answer_text}

            elif not write_steps and _parts:
                # Read-only actions (inspections, web fetching, etc.)
                yield {"type": "status", "stage": "generation", "message": "Synthesizing response..."}
                collector: Dict[str, Any] = {}
                yield from _stream_synthesize_agent_action_reply(
                    user_query=clean_query,
                    action_name="inspect_sources",
                    execution_outcome="Inspection completed.",
                    collector=collector,
                    read_narrations=_parts,
                    history_text=history_text,
                )
                answer_text = collector.get("answer", "")
            else:
                answer_text = "I do not currently have a registered tool to execute that specific action. My available actions include inspecting URL sources, downloading & ingesting certificate files, updating or deleting database records, executing dynamic database operations, and querying compliance data."
                yield {"type": "token", "text": answer_text}

            sources = [{"type": "agent_action", "url": _extract_url(clean_query)}]

        elif intent == QueryIntent.UNSTRUCTURED_RAG.value:
            yield {"type": "status", "stage": "retrieval", "message": "Searching document context..."}
            resolved_query = _reformulate_query(clean_query, history)
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                user_prompt = build_unstructured_qa_prompt(clean_query, context_text, history_text)
                collector: Dict[str, Any] = {}
                yield from _stream_and_collect(
                    config_qa_system_prompt(),
                    user_prompt,
                    clean_query,
                    collector,
                    disable_thinking=disable_thinking,
                )
                answer_text = collector.get("answer", "")
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                from core.rag.sql_engine import stream_metadata_answer
                answer_text = ""
                for chunk in stream_metadata_answer(
                    clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                ):
                    answer_text += chunk
                    yield {"type": "token", "text": chunk}
                if not answer_text or answer_text == SQL_FALLBACK:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE
                    yield {"type": "token", "text": answer_text}

        else:
            logger.warning(f" Unhandled intent '{intent}'; falling back to UNSTRUCTURED_RAG path.")
            yield {"type": "status", "stage": "retrieval", "message": "Searching document context..."}
            resolved_query = _reformulate_query(clean_query, history)
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                user_prompt = build_unstructured_qa_prompt(clean_query, context_text, history_text)
                collector: Dict[str, Any] = {}
                yield from _stream_and_collect(
                    config_qa_system_prompt(),
                    user_prompt,
                    clean_query,
                    collector,
                    disable_thinking=disable_thinking,
                )
                answer_text = collector.get("answer", "")
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                from core.rag.sql_engine import stream_metadata_answer
                answer_text = ""
                for chunk in stream_metadata_answer(
                    clean_query, db_session, history_text=history_text, disable_thinking=disable_thinking
                ):
                    answer_text += chunk
                    yield {"type": "token", "text": chunk}
                if not answer_text or answer_text == SQL_FALLBACK:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE
                    yield {"type": "token", "text": answer_text}

        if not answer_text:
            answer_text = FALLBACK_NOT_FOUND_MESSAGE

        yield {
            "type": "done",
            "answer": answer_text,
            "intent": intent,
            "reasoning": reasoning,
            "sources": sources,
            "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "timings": get_turn_telemetry(),
        }

    except Exception as exc:
        logger.error(f" Exception in answer_compliance_query_stream: {exc}. Emitting fallback.")
        yield {
            "type": "done",
            "answer": FALLBACK_NOT_FOUND_MESSAGE,
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": f"Streaming failure triggered fallback: {str(exc)}",
            "sources": [],
            "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "timings": get_turn_telemetry(),
        }

    finally:
        if close_session_on_exit:
            db_session.close()


def config_qa_system_prompt() -> str:
    """Lazily returns the QA synthesis system prompt with active long-term memories injected."""
    from core.prompts import config_qa_system_prompt as _config_qa_system_prompt
    return _config_qa_system_prompt()
