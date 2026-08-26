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

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal

        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        # Step 1: Intent Classification via Router
        logger.info(f" Classifying query intent for: '{clean_query[:60]}...'")
        router_decision = classify_intent(clean_query)
        intent = router_decision.get("intent", QueryIntent.UNSTRUCTURED_RAG.value)
        reasoning = router_decision.get("reasoning", "Default routing classification.")

        logger.info(f" Router decision: intent={intent} ({reasoning})")

        answer_text = ""
        sources: List[Dict[str, Any]] = []

        # Step 2: Route Execution
        if intent == QueryIntent.METADATA_QUERY.value:
            # Path A: Text-to-SQL
            logger.info(" Executing METADATA_QUERY path (Text-to-SQL)...")
            answer_text = execute_metadata_query(clean_query, db_session, history_text=history_text)
            sources = [{"type": "database", "table": "certificates", "query_type": "relational_sql"}]

        elif intent == QueryIntent.HYBRID_QUERY.value:
            # Path C: Dual-Path Hybrid (Structured SQL + Unstructured Vector RRF)
            logger.info(" Executing HYBRID_QUERY path (Dual-Path SQL + Vector RRF)...")
            try:
                # Structured facts from SQL engine
                sql_answer = execute_metadata_query(clean_query, db_session, history_text=history_text)

                # Unstructured context from Hybrid RRF
                hybrid_payload = retrieve_hybrid_context(clean_query, db_session, top_k=5)
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
                        f"USER QUESTION: {clean_query}\n\n"
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
                    answer_text = execute_unstructured_query(clean_query, db_session, history_text=history_text)
                else:
                    answer_text = sql_answer

            except Exception as exc:
                logger.warning(f" Dual-path hybrid execution failed: {exc}. Falling back to unstructured query.")
                answer_text = execute_unstructured_query(clean_query, db_session, history_text=history_text)

        else:
            # Path B: Standard UNSTRUCTURED_RAG (Hybrid RRF + Parent Expansion)
            logger.info(" Executing UNSTRUCTURED_RAG path (Hybrid RRF + Parent Expansion)...")
            hybrid_payload = retrieve_hybrid_context(clean_query, db_session, top_k=5)
            sources = hybrid_payload.get("sources", [])
            answer_text = execute_unstructured_query(clean_query, db_session, history_text=history_text)

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
            fallback_answer = execute_unstructured_query(clean_query, db_session, history_text=history_text)
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


if __name__ == "__main__":
    print("Testing Orchestrator imports and function signature...")
    print(" orchestrator.py module compiled successfully.")
