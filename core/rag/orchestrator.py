"""
core/rag/orchestrator.py — End-to-End Dual-Path RAG Orchestrator

Unifies intent classification (core.rag.router), Text-to-SQL (core.rag.sql_engine),
and Hybrid RRF vector retrieval (core.rag.hybrid_engine) into a single production entry point:
`answer_compliance_query(user_query, db_session)`.

Routing logic:
  - METADATA_QUERY   ──► Fast Text-to-SQL pipeline over PostgreSQL certificates table.
  - UNSTRUCTURED_RAG ──► Hybrid Dense/Sparse RRF + Adaptive Parent Expansion.
  - HYBRID_QUERY     ──► Dual-path execution combining SQL facts + Hybrid RRF context.
"""

import time
import logging
import json
import re
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


def answer_compliance_query(
    user_query: str,
    db_session=None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Unified entry point for query processing across all three routing paths.

    Args:
        user_query (str): Natural language user question.
        db_session: SQLAlchemy session (or None to auto-manage a local session).
        history (Optional[List[Dict]]): Prior conversation turns ({role, content})
            injected into LLM prompts so follow-up questions keep context.

    Returns:
        Dict[str, Any] standardized payload:
            - answer (str): Synthesized natural language answer.
            - intent (str): "METADATA_QUERY", "UNSTRUCTURED_RAG", or "HYBRID_QUERY".
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

        else:
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


def _stream_and_collect(system_prompt: str, user_prompt: str, query: str, collector: Dict[str, Any]):
    """
    Forwards streaming answer-token events from a synthesis and records the final
    (answer, citations) into `collector` via the synthesis_done event.
    """
    from core.rag.qa import stream_synthesize_answer

    for evt in stream_synthesize_answer(system_prompt, user_prompt, query):
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

        else:
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
