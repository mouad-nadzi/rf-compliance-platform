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
from core.rag.router import classify_intent, QueryIntent
from core.rag.sql_engine import execute_metadata_query, FALLBACK_MESSAGE as SQL_FALLBACK
from core.rag.hybrid_engine import (
    retrieve_hybrid_context,
    execute_unstructured_query,
    build_unstructured_qa_prompt,
    FALLBACK_NOT_FOUND_MESSAGE,
)

logger = logging.getLogger(__name__)


def _format_history(history) -> str:
    """
    Formats a list of {role, content} turns into a compact conversation-history
    block for prompt injection. Empty input yields an empty string.
    """
    if not history:
        return ""
    lines = ["--- CONVERSATION HISTORY ---"]
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append("--- END CONVERSATION HISTORY ---")
    return "\n".join(lines)


_ANAPHORA_RE = re.compile(
    r"\b(the others|the other|these|those|them|this|that|it|they|"
    r"what about|how about|and the rest|the rest|also|ones)\b",
    re.IGNORECASE,
)


def _reformulate_query(user_query: str, history) -> str:
    """
    Rewrites the latest query into a standalone, self-contained query that carries
    forward entities from prior turns (resolves "the others", "these", "them", etc.).

    Falls back to the original query unchanged if no history exists or the LLM rewrite fails.
    A deterministic anaphora fallback reconstructs the query from the last user turn
    when the LLM returned the query unchanged despite clear references to prior context.
    """
    if not history:
        return user_query

    history_text = _format_history(history)
    rewritten = user_query
    try:
        from core.prompts import QUERY_REWRITE_SYSTEM_PROMPT
        from core.llm import generate_json

        raw_response = generate_json(
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
            user_prompt=(
                f"{history_text}\n\n"
                f"USER'S LATEST QUERY: {user_query}\n\n"
                f"Return ONLY the raw JSON output matching the schema."
            ),
            disable_thinking=True,
        )

        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            candidate = str(parsed.get("rewritten_query", "")).strip()
            if candidate:
                rewritten = candidate
    except Exception as exc:
        logger.warning(f" Query reformulation failed ({exc}); using original query.")

    # Deterministic fallback: if the LLM left an anaphoric query unchanged, rebuild
    # it from the last user turn so downstream routing/SQL still has the context.
    if rewritten.strip().lower() == user_query.strip().lower() and _ANAPHORA_RE.search(user_query):
        last_user = ""
        for turn in reversed(history or []):
            if turn.get("role") == "user":
                last_user = str(turn.get("content", "")).strip()
                if last_user:
                    break
        if last_user:
            fallback = f"{user_query.strip()} - referring to the certificates from the previous question: {last_user}"
            logger.info(f" Anaphora fallback rewrite: '{user_query[:50]}' -> '{fallback[:90]}'")
            return fallback

    if rewritten.strip() != user_query.strip():
        logger.info(f" Query rewritten: '{user_query[:50]}' -> '{rewritten[:80]}'")
    return rewritten


def _casual_conversation_reply(user_query: str, history_text: str = "") -> str:
    """
    Generates a friendly casual reply WITHOUT touching the database or retrieval.

    Uses a lightweight LLM call (disable_thinking) against a dedicated casual
    prompt; falls back to a canned greeting if generation fails. Stateless.
    """
    from core.prompts import CASUAL_CONVERSATION_SYSTEM_PROMPT
    from core.llm import generate_json

    try:
        user_prompt = (
            f"{history_text}\n\n" if history_text else ""
        ) + f"USER MESSAGE: {user_query}\n\nReturn ONLY the raw JSON output matching the schema."
        raw_response = generate_json(
            system_prompt=CASUAL_CONVERSATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            disable_thinking=True,
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


_URL_RE = re.compile(r"https?://[^\s'\"<>]+")


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
):
    """
    Generator: handles chat-based HITL confirmations for the most recent PENDING
    AGENT_ACTION proposal and streams live status events (Running OCR, Extracting
    metadata, Persisting to DB) while executing the approved action.
    """
    import queue
    import threading
    from core.agent.proposals import proposal_manager

    query = str(user_query or "").strip()
    if not query:
        return

    pending = [p for p in proposal_manager.list_pending_proposals() if p.get("type") == "AGENT_ACTION"]
    if not pending:
        return
    proposal = pending[0]
    proposal_id = proposal["proposal_id"]
    desc = (proposal.get("payload") or {}).get("description") or proposal.get("sql_preview") or "Agent action"

    decision = "NEW_QUERY"
    try:
        from core.prompts import APPROVAL_CLASSIFIER_SYSTEM_PROMPT
        from core.llm import generate_json

        system_prompt = APPROVAL_CLASSIFIER_SYSTEM_PROMPT.format(
            PROPOSAL_DESCRIPTION=desc,
            USER_MESSAGE=query,
        )
        raw_res = generate_json(
            system_prompt=system_prompt,
            user_prompt=f"USER MESSAGE: {query}\n\nReturn ONLY raw JSON.",
            disable_thinking=True,
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
            proposal_manager.update_status(proposal_id, "REJECTED")
        except Exception as exc:
            logger.warning(f" Could not reject proposal {proposal_id}: {exc}")
            return
        logger.info(f" Chat HITL: rejected AGENT_ACTION proposal {proposal_id}.")
        reply = (
            f"OK - I've rejected proposal `{proposal_id}`. No action was executed. "
            "Let me know if you'd like to try a different link or command."
        )
        yield {"type": "status", "stage": "agent_action", "message": "Proposal rejected."}
        words = reply.split(" ")
        for i, w in enumerate(words):
            yield {"type": "token", "text": w + (" " if i < len(words) - 1 else "")}
            time.sleep(0.012)
        yield {
            "type": "done",
            "answer": reply,
            "intent": QueryIntent.AGENT_ACTION.value,
            "reasoning": "User rejected pending agent action proposal.",
            "sources": [],
            "latency_ms": 0.0,
        }
        return

    if decision == "APPROVE":
        yield {"type": "status", "stage": "agent_action", "message": "Confirming approval..."}
        try:
            proposal_manager.update_status(proposal_id, "APPROVED")
            target = (proposal.get("payload") or {}).get("target_file") or ""
            if target:
                for other in proposal_manager.list_pending_proposals():
                    if other.get("type") == "AGENT_ACTION":
                        other_target = (other.get("payload") or {}).get("target_file") or ""
                        if other_target == target:
                            proposal_manager.update_status(other["proposal_id"], "APPROVED")
        except Exception as exc:
            logger.warning(f" Could not approve proposal {proposal_id}: {exc}")
            return

        logger.info(f" Chat HITL: approved AGENT_ACTION proposal {proposal_id}.")
        from core.agent.agent_loop import resolve_write_step

        target = (proposal.get("payload") or {}).get("target_file") or ""
        target_suffix = f" for '{target}'" if target else ""

        q: queue.Queue = queue.Queue()

        def _cb(stage_name: str, msg_text: str):
            q.put({"type": "status", "stage": stage_name, "message": msg_text})

        exec_outcome = [None]
        def _worker():
            try:
                exec_outcome[0] = resolve_write_step(proposal, status_callback=_cb)
            except Exception as e:
                exec_outcome[0] = f"Execution failed: {e}"
            finally:
                q.put(None)

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

        result_text = exec_outcome[0] or "Action executed."
        reply = f"Approved! I've executed the action{target_suffix}:\n\n{result_text}"

        yield {"type": "status", "stage": "generation", "message": "Streaming result..."}
        words = reply.split(" ")
        for i, w in enumerate(words):
            yield {"type": "token", "text": w + (" " if i < len(words) - 1 else "")}
            time.sleep(0.012)

        yield {
            "type": "done",
            "answer": reply,
            "intent": QueryIntent.AGENT_ACTION.value,
            "reasoning": "User approved pending agent action proposal.",
            "sources": [],
            "latency_ms": 0.0,
        }
        return


def _iter_agent_action(
    clean_query: str,
    resolved_query: str,
    history: Optional[List[Dict[str, str]]] = None,
):
    """
    Step-based execution of an AGENT_ACTION request. Yields a string per READ
    step as it completes (so the chat streams progress), then the approval
    question if a WRITE step remains. Read steps run without permission; only
    the write step is gated.
    """
    from core.agent.agent_loop import plan_agent_action, iter_read_steps, stage_write_step, RunArtifacts

    urls = _extract_urls(resolved_query) or _extract_urls(clean_query)
    target_query = resolved_query or clean_query
    plan_result = plan_agent_action(target_query, urls=urls, history=history)
    primary_url = urls[0] if urls else ""
    artifacts = RunArtifacts()

    yielded = False
    for narration in iter_read_steps(plan_result.steps, primary_url, artifacts):
        yielded = True
        yield narration


    write_steps = [s for s in plan_result.steps if s.kind == "write"]
    if write_steps:
        from core.agent.agent_loop import resolve_write_step, stage_write_step
        w_step = write_steps[0]

        if plan_result.is_direct_command:
            try:
                # Direct command: execute write action immediately without requiring 'yes' confirmation turn
                payload_dict = dict(w_step.payload)
                payload_dict.update({
                    "action": w_step.action,
                    "description": w_step.description,
                    "source_url": artifacts.source_url,
                    "file_paths": artifacts.file_paths,
                    "verified_urls": artifacts.verified_urls,
                })
                result_msg = resolve_write_step({
                    "payload": payload_dict,
                    "action": w_step.action,
                })
                yield result_msg
                yielded = True
            except Exception as exc:
                yield f"Action failed: {exc}"
                yielded = True
        else:
            # Discovery / implicit command: stage proposal and request approval
            try:
                stage_write_step(w_step, artifacts)
                desc = w_step.description or 'add to the database'
                yield (
                    f"To do the final step ({desc}), "
                    "I need your approval. Is that OK? Reply 'yes' to proceed or 'no' to cancel."
                )
                yielded = True
            except Exception as exc:
                logger.warning(f" Could not stage write step: {exc}")

    if not yielded:
        yield "I cannot execute what you are asking because I do not have a tool available for this action."



def answer_compliance_query(
    user_query: str,
    db_session=None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Unified entry point for query processing across all routing paths.

    Args:
        user_query (str): Natural language user question.
        db_session: SQLAlchemy session (or None to auto-manage a local session).
        history (Optional[List[Dict]]): Prior conversation turns ({role, content})
            injected into LLM prompts so follow-up questions keep context.

    Returns:
        Dict[str, Any] standardized payload:
            - answer (str): Synthesized natural language answer.
            - intent (str): "METADATA_QUERY", "UNSTRUCTURED_RAG", "HYBRID_QUERY",
              "CASUAL_CONVERSATION", or "AGENT_ACTION".
            - reasoning (str): Explanation for router classification decision.
            - sources (List[Dict]): List of source certificates and page references.
            - latency_ms (float): Total processing latency in milliseconds.
    """
    t_start = time.perf_counter()
    clean_query = str(user_query or "").strip()
    history_text = _format_history(history)

    if not clean_query:
        return {
            "answer": "Please submit a non-empty question.",
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": "Blank query submitted.",
            "sources": [],
            "latency_ms": 0.0,
        }

    # Chat-based HITL confirmation ("yes" / "approve" / "no") for a pending
    # AGENT_ACTION proposal, handled before routing.
    approval_reply = _handle_chat_approval(clean_query, history=history)
    if approval_reply:
        t_end = time.perf_counter()
        return {
            "answer": approval_reply,
            "intent": QueryIntent.AGENT_ACTION.value,
            "reasoning": "User confirmed a pending agent action via chat.",
            "sources": [],
            "latency_ms": round((t_end - t_start) * 1000.0, 2),
        }

    # Resolve anaphoric follow-ups ("the others", "these", "them") into a
    # standalone query carrying entities from the conversation history.
    resolved_query = _reformulate_query(clean_query, history)

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal

        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        # Step 1: Intent Classification via Router (history-aware, on the resolved query)
        logger.info(f" Classifying query intent for: '{clean_query[:60]}...'")
        router_decision = classify_intent(resolved_query, history=history)
        intent = router_decision.get("intent", QueryIntent.UNSTRUCTURED_RAG.value)
        reasoning = router_decision.get("reasoning", "Default routing classification.")

        logger.info(f" Router decision: intent={intent} ({reasoning})")

        answer_text = ""
        sources: List[Dict[str, Any]] = []

        # Step 2: Route Execution
        if intent == QueryIntent.METADATA_QUERY.value:
            # Path A: Text-to-SQL
            logger.info(" Executing METADATA_QUERY path (Text-to-SQL)...")
            answer_text = execute_metadata_query(resolved_query, db_session, history_text=history_text)
            sources = [{"type": "database", "table": "certificates", "query_type": "relational_sql"}]

        elif intent == QueryIntent.HYBRID_QUERY.value:
            # Path C: Dual-Path Hybrid (Structured SQL + Unstructured Vector RRF)
            logger.info(" Executing HYBRID_QUERY path (Dual-Path SQL + Vector RRF)...")
            try:
                # Structured facts from SQL engine
                sql_answer = execute_metadata_query(resolved_query, db_session, history_text=history_text)

                # Unstructured context from Hybrid RRF
                hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
                context_text = hybrid_payload.get("context_text", "").strip()
                sources = hybrid_payload.get("sources", [])

                if context_text and sql_answer and sql_answer != SQL_FALLBACK:
                    dual_prompt = (
                        f"{history_text}\n\n" if history_text else ""
                    ) + (
                        f"--- STRUCTURED METADATA FACTS (POSTGRESQL) ---\n"
                        f"{sql_answer}\n\n"
                        f"--- UNSTRUCTURED DOCUMENT CONTEXT ---\n"
                        f"{context_text}\n\n"
                        f"USER QUESTION: {resolved_query}\n\n"
                        f"Synthesize a unified, comprehensive answer combining both the structured metadata facts "
                        f"and unstructured document details above. Preserve citation markers (e.g. <Page X>)."
                    )

                    from core.prompts import QA_SYNTHESIS_SYSTEM_PROMPT
                    from core.llm import generate_json

                    raw_response = generate_json(
                        system_prompt=QA_SYNTHESIS_SYSTEM_PROMPT,
                        user_prompt=dual_prompt,
                        disable_thinking=False,
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
                    answer_text = execute_unstructured_query(resolved_query, db_session, history_text=history_text)
                else:
                    answer_text = sql_answer
                    if not answer_text or answer_text == SQL_FALLBACK:
                        answer_text = FALLBACK_NOT_FOUND_MESSAGE

            except Exception as exc:
                logger.warning(f" Dual-path hybrid execution failed: {exc}. Falling back to unstructured query.")
                answer_text = execute_unstructured_query(resolved_query, db_session, history_text=history_text)

        elif intent == QueryIntent.CASUAL_CONVERSATION.value:
            # Path D: Casual conversation — short-circuit. No DB, no retrieval.
            logger.info(" Executing CASUAL_CONVERSATION path (no DB, no retrieval)...")
            answer_text = _casual_conversation_reply(clean_query, history_text)

        elif intent == QueryIntent.AGENT_ACTION.value:
            # Step-based agent action: read steps run freely; the write step
            # (if any) is staged and waits for approval.
            logger.info(" Executing AGENT_ACTION path (step-based execution)...")
            _parts = list(_iter_agent_action(clean_query, resolved_query, history=history))
            answer_text = "\n\n".join(_parts)
            sources = [{"type": "agent_action", "url": _extract_url(resolved_query) or _extract_url(clean_query)}]

        elif intent == QueryIntent.UNSTRUCTURED_RAG.value:
            # Path B: Standard UNSTRUCTURED_RAG (Hybrid RRF + Parent Expansion)
            logger.info(" Executing UNSTRUCTURED_RAG path (Hybrid RRF + Parent Expansion)...")
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                answer_text = execute_unstructured_query(resolved_query, db_session, history_text=history_text)
            else:
                # Relevance gate / low-signal query: the vector store had no usable
                # context, so fall back to the structured metadata (SQL) path.
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                answer_text = execute_metadata_query(resolved_query, db_session, history_text=history_text)
                if not answer_text or answer_text == SQL_FALLBACK:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE

        else:
            # Safety net: unknown intent tokens should never reach here (router
            # validates against QueryIntent); treat as UNSTRUCTURED_RAG.
            logger.warning(f" Unhandled intent '{intent}'; falling back to UNSTRUCTURED_RAG path.")
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                answer_text = execute_unstructured_query(resolved_query, db_session, history_text=history_text)
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                answer_text = execute_metadata_query(resolved_query, db_session, history_text=history_text)
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
        }

    except Exception as exc:
        logger.error(f" Exception in answer_compliance_query: {exc}. Executing emergency fallback.")
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        try:
            fallback_answer = execute_unstructured_query(resolved_query, db_session, history_text=history_text)
        except Exception:
            fallback_answer = FALLBACK_NOT_FOUND_MESSAGE

        return {
            "answer": fallback_answer,
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": f"Emergency fallback triggered due to exception: {str(exc)}",
            "sources": [],
            "latency_ms": round(latency_ms, 2),
        }

    finally:
        if close_session_on_exit:
            db_session.close()


def _stream_and_collect(
    system_prompt: str,
    user_prompt: str,
    query: str,
    collector: Dict[str, Any],
    disable_thinking: bool = False,
):
    """
    Forwards streaming answer-token events from a synthesis and records the final
    (answer, citations) into `collector` via the synthesis_done event.
    """
    from core.rag.qa import stream_synthesize_answer

    for evt in stream_synthesize_answer(system_prompt, user_prompt, query, disable_thinking=disable_thinking):
        if evt["type"] in ("token", "thinking"):
            yield evt
        elif evt["type"] == "synthesis_done":
            collector["answer"] = evt.get("answer", "")
            collector["citations"] = evt.get("citations", [])


def answer_compliance_query_stream(
    user_query: str,
    db_session=None,
    history: Optional[List[Dict[str, str]]] = None,
):
    """
    Streaming variant of answer_compliance_query() for the interactive chat UI.

    Performs the same routing/retrieval pipeline but emits the final answer
    generation token-by-token. Yields event dicts:

      {"type": "status", "stage": ..., "message": ...}
      {"type": "token", "text": <incremental answer text>}
      {"type": "done", "answer": ..., "intent": ..., "reasoning": ...,
       "sources": [...], "latency_ms": ...}

    For metadata-only (SQL) paths the answer has no LLM synthesis step, so a
    single full token event is emitted instead.
    """
    t_start = time.perf_counter()
    clean_query = str(user_query or "").strip()
    history_text = _format_history(history)

    yield {"type": "status", "stage": "routing", "message": "Routing query..."}

    if not clean_query:
        yield {
            "type": "done",
            "answer": "Please submit a non-empty question.",
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": "Blank query submitted.",
            "sources": [],
            "latency_ms": 0.0,
        }
        return

    # Chat-based HITL confirmation ("yes" / "approve" / "no") for a pending
    # AGENT_ACTION proposal, handled before routing.
    has_approval_events = False
    for evt in _handle_chat_approval_stream(clean_query, history=history):
        has_approval_events = True
        yield evt
    if has_approval_events:
        return

    resolved_query = _reformulate_query(clean_query, history)

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal

        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        # Step 1: Intent Classification via Router (history-aware, on the resolved query)
        router_decision = classify_intent(resolved_query, history=history)
        intent = router_decision.get("intent", QueryIntent.UNSTRUCTURED_RAG.value)
        reasoning = router_decision.get("reasoning", "Default routing classification.")
        yield {
            "type": "status",
            "stage": "routing",
            "message": f"Intent: {intent} - {reasoning}",
        }

        answer_text = ""
        sources: List[Dict[str, Any]] = []

        # Step 2: Route Execution
        if intent == QueryIntent.METADATA_QUERY.value:
            yield {"type": "status", "stage": "retrieval", "message": "Querying certificate database..."}
            sources = [{"type": "database", "table": "certificates", "query_type": "relational_sql"}]
            from core.rag.sql_engine import stream_metadata_answer
            answer_text = ""
            for chunk in stream_metadata_answer(resolved_query, db_session, history_text=history_text):
                answer_text += chunk
                yield {"type": "token", "text": chunk}
            if not answer_text:
                answer_text = SQL_FALLBACK
                yield {"type": "token", "text": answer_text}

        elif intent == QueryIntent.HYBRID_QUERY.value:
            yield {"type": "status", "stage": "retrieval", "message": "Combining database facts and document context..."}
            try:
                sql_answer = execute_metadata_query(resolved_query, db_session, history_text=history_text)
                hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
                context_text = hybrid_payload.get("context_text", "").strip()
                sources = hybrid_payload.get("sources", [])

                if context_text and sql_answer and sql_answer != SQL_FALLBACK:
                    dual_prompt = (
                        f"{history_text}\n\n" if history_text else ""
                    ) + (
                        f"--- STRUCTURED METADATA FACTS (POSTGRESQL) ---\n"
                        f"{sql_answer}\n\n"
                        f"--- UNSTRUCTURED DOCUMENT CONTEXT ---\n"
                        f"{context_text}\n\n"
                        f"USER QUESTION: {resolved_query}\n\n"
                        f"Synthesize a unified, comprehensive answer combining both the structured metadata facts "
                        f"and unstructured document details above. Preserve citation markers (e.g. <Page X>)."
                    )
                    yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                    collector: Dict[str, Any] = {}
                    yield from _stream_and_collect(
                        config_qa_system_prompt(), dual_prompt, resolved_query, collector
                    )
                    answer_text = collector.get("answer", "")
                elif context_text:
                    yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                    user_prompt = build_unstructured_qa_prompt(resolved_query, context_text, history_text)
                    collector: Dict[str, Any] = {}
                    yield from _stream_and_collect(
                        config_qa_system_prompt(), user_prompt, resolved_query, collector
                    )
                    answer_text = collector.get("answer", "")
                else:
                    answer_text = sql_answer
                    if not answer_text or answer_text == SQL_FALLBACK:
                        answer_text = FALLBACK_NOT_FOUND_MESSAGE
                    yield {"type": "token", "text": answer_text}
            except Exception as exc:
                logger.warning(f" Dual-path hybrid streaming failed: {exc}. Falling back to unstructured query.")
                hybrid_fallback = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
                fallback_context = hybrid_fallback.get("context_text", "").strip()
                if fallback_context:
                    user_prompt = build_unstructured_qa_prompt(resolved_query, fallback_context, history_text)
                    collector: Dict[str, Any] = {}
                    yield from _stream_and_collect(
                        config_qa_system_prompt(), user_prompt, resolved_query, collector
                    )
                    answer_text = collector.get("answer", "")
                else:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE
                    yield {"type": "token", "text": answer_text}

        elif intent == QueryIntent.CASUAL_CONVERSATION.value:
            yield {"type": "status", "stage": "casual", "message": "Responding conversationally..."}
            from core.prompts import CASUAL_CONVERSATION_SYSTEM_PROMPT
            casual_prompt = (
                f"{history_text}\n\n" if history_text else ""
            ) + f"USER MESSAGE: {clean_query}\n\nReturn ONLY the raw JSON output matching the schema."
            collector: Dict[str, Any] = {}
            yield from _stream_and_collect(
                CASUAL_CONVERSATION_SYSTEM_PROMPT,
                casual_prompt,
                clean_query,
                collector,
                disable_thinking=True,
            )
            answer_text = collector.get("answer", "")
            if not answer_text:
                answer_text = _casual_conversation_reply(clean_query, history_text)
                yield {"type": "token", "text": answer_text}

        elif intent == QueryIntent.AGENT_ACTION.value:
            yield {"type": "status", "stage": "agent_action", "message": "Planning agent action..."}
            from core.agent.agent_loop import (
                plan_agent_action, iter_read_steps, stage_write_step,
                resolve_write_step, RunArtifacts
            )
            urls = _extract_urls(resolved_query) or _extract_urls(clean_query)
            target_query = resolved_query or clean_query
            plan_result = plan_agent_action(target_query, urls=urls, history=history)
            primary_url = urls[0] if urls else ""
            artifacts = RunArtifacts()

            _parts = []
            for step in plan_result.steps:
                if step.kind == "read":
                    desc = step.description or step.action
                    yield {"type": "status", "stage": "agent_action", "message": f"Executing: {desc}..."}
                    for narration in iter_read_steps([step], primary_url, artifacts):
                        _parts.append(narration)

            write_steps = [s for s in plan_result.steps if s.kind == "write"]
            if write_steps:
                write_step = write_steps[0]
                is_direct = plan_result.is_direct_command

                if is_direct:
                    # Explicit direct user command -> execute immediately without asking for approval!
                    yield {"type": "status", "stage": "agent_action", "message": "Executing database action..."}
                    
                    import queue
                    import threading
                    q: queue.Queue = queue.Queue()

                    def _cb(stage_name: str, msg_text: str):
                        q.put({"type": "status", "stage": stage_name, "message": msg_text})

                    target_f = write_step.payload.get("target_file")
                    proposal_payload = dict(write_step.payload)
                    proposal_payload.update({
                        "action": write_step.action,
                        "description": write_step.description,
                        "source_url": artifacts.source_url,
                        "file_paths": artifacts.file_paths,
                        "verified_urls": artifacts.verified_urls,
                    })
                    dummy_proposal = {"payload": proposal_payload}

                    exec_outcome = [None]
                    def _worker():
                        try:
                            exec_outcome[0] = resolve_write_step(dummy_proposal, status_callback=_cb)
                        except Exception as e:
                            exec_outcome[0] = f"Execution failed: {e}"
                        finally:
                            q.put(None)

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
                    res_msg = exec_outcome[0] or "Action executed."
                    _parts.append(f"Executed database action: {res_msg}")

                else:
                    # Implicit action decided by agent -> stage proposal for HITL approval!
                    try:
                        stage_write_step(write_step, artifacts)
                    except Exception as exc:
                        logger.warning(f" Could not stage write step: {exc}")

                    target_f = write_step.payload.get("target_file")
                    desc = write_step.description or 'add to the database'
                    if target_f and target_f not in desc:
                        desc = f"{desc} ({target_f})"
                    _parts.append(
                        f"To do the final step ({desc}), "
                        "I need your approval. Is that OK? Reply 'yes' to proceed or 'no' to cancel."
                    )

            if not _parts:
                _parts.append("I couldn't plan that action - please rephrase.")

            answer_text = "\n\n".join(_parts)

            # Stream the exact detailed answer text token-by-token (word-by-word) progressively
            yield {"type": "status", "stage": "generation", "message": "Streaming results..."}
            words = answer_text.split(" ")
            for i, w in enumerate(words):
                w_text = w + (" " if i < len(words) - 1 else "")
                yield {"type": "token", "text": w_text}
                time.sleep(0.012)

            sources = [{"type": "agent_action", "url": _extract_url(resolved_query) or _extract_url(clean_query)}]


        elif intent == QueryIntent.UNSTRUCTURED_RAG.value:
            yield {"type": "status", "stage": "retrieval", "message": "Searching document context..."}
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                user_prompt = build_unstructured_qa_prompt(resolved_query, context_text, history_text)
                collector: Dict[str, Any] = {}
                yield from _stream_and_collect(
                    config_qa_system_prompt(), user_prompt, resolved_query, collector
                )
                answer_text = collector.get("answer", "")
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                answer_text = execute_metadata_query(resolved_query, db_session, history_text=history_text)
                if not answer_text or answer_text == SQL_FALLBACK:
                    answer_text = FALLBACK_NOT_FOUND_MESSAGE
                yield {"type": "token", "text": answer_text}

        else:
            # Safety net: unknown intent tokens should never reach here; treat as UNSTRUCTURED_RAG.
            logger.warning(f" Unhandled intent '{intent}'; falling back to UNSTRUCTURED_RAG path.")
            yield {"type": "status", "stage": "retrieval", "message": "Searching document context..."}
            hybrid_payload = retrieve_hybrid_context(resolved_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            context_text = hybrid_payload.get("context_text", "").strip()

            if context_text:
                yield {"type": "status", "stage": "generation", "message": "Generating answer..."}
                user_prompt = build_unstructured_qa_prompt(resolved_query, context_text, history_text)
                collector: Dict[str, Any] = {}
                yield from _stream_and_collect(
                    config_qa_system_prompt(), user_prompt, resolved_query, collector
                )
                answer_text = collector.get("answer", "")
            else:
                logger.info(" RAG retrieval yielded no usable context; falling back to metadata (SQL).")
                answer_text = execute_metadata_query(resolved_query, db_session, history_text=history_text)
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
        }

    finally:
        if close_session_on_exit:
            db_session.close()


def config_qa_system_prompt() -> str:
    """Lazily returns the QA synthesis system prompt (avoids import cycle at module load)."""
    from core.prompts import QA_SYNTHESIS_SYSTEM_PROMPT
    return QA_SYNTHESIS_SYSTEM_PROMPT
