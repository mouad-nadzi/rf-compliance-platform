"""
core/rag/unified_agent.py - Unified Tool-Calling Agent

Replaces the 2-stage Intent Router -> Specialized Engine pipeline with a single
LLM call that selects the appropriate tool and returns a structured decision.

TOOL DECISIONS:
  - search_database_sql       -> Text-to-SQL engine (METADATA_QUERY path)
  - search_document_chunks    -> Hybrid RRF vector retrieval (UNSTRUCTURED_RAG path)
  - search_hybrid             -> Dual-path SQL + vector synthesis (HYBRID_QUERY path)
  - respond_conversationally  -> Casual conversation reply (CASUAL_CONVERSATION path)
  - execute_agent_action      -> Agent planner + tool execution (AGENT_ACTION path)

The existing downstream engines (sql_engine, hybrid_engine, agent_loop, tools) are
100% preserved. Only the routing decision layer changes.
"""

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Unified routing intent labels used by the orchestrator dispatch tree and API responses."""
    METADATA_QUERY      = "METADATA_QUERY"
    UNSTRUCTURED_RAG    = "UNSTRUCTURED_RAG"
    HYBRID_QUERY        = "HYBRID_QUERY"
    CASUAL_CONVERSATION = "CASUAL_CONVERSATION"
    AGENT_ACTION        = "AGENT_ACTION"


# Maps unified tool names to QueryIntent values
TOOL_TO_INTENT = {
    "search_database_sql":      QueryIntent.METADATA_QUERY.value,
    "search_document_chunks":   QueryIntent.UNSTRUCTURED_RAG.value,
    "search_hybrid":            QueryIntent.HYBRID_QUERY.value,
    "respond_conversationally": QueryIntent.CASUAL_CONVERSATION.value,
    "execute_agent_action":     QueryIntent.AGENT_ACTION.value,
}

VALID_TOOLS = set(TOOL_TO_INTENT.keys())


from core.utils.history import format_conversation_history as _format_history


def unified_tool_select(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Single LLM call that selects the appropriate tool for the given query.

    Returns a dict with:
        - "tool": one of the VALID_TOOLS strings
        - "reasoning": one-sentence explanation
        - "intent": the equivalent legacy QueryIntent value (for API compat)

    Falls back to "search_document_chunks" (UNSTRUCTURED_RAG) if the LLM
    fails or returns an unrecognized tool name.
    """
    from core.llm import generate_json
    from core.prompts import config_unified_agent_system_prompt

    history_block = _format_history(history)
    user_prompt = (
        f"{history_block}\n\n" if history_block else ""
    ) + f"USER MESSAGE: {query}\n\nReturn ONLY the raw JSON output matching the schema."

    fallback = {
        "tool": "search_document_chunks",
        "reasoning": "Fallback: defaulting to vector document search.",
        "intent": "UNSTRUCTURED_RAG",
    }

    try:
        raw = generate_json(
            system_prompt=config_unified_agent_system_prompt(),
            user_prompt=user_prompt,
            disable_thinking=True,
        )
        parsed = json.loads(raw)
        tool = str(parsed.get("tool", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()

        if tool not in VALID_TOOLS:
            logger.warning(
                f" Unified agent returned unknown tool '{tool}'; falling back to vector search."
            )
            return fallback

        intent = TOOL_TO_INTENT[tool]
        logger.info(f" Unified agent selected tool='{tool}' intent='{intent}' ({reasoning[:80]})")
        return {"tool": tool, "reasoning": reasoning, "intent": intent}

    except Exception as exc:
        logger.warning(f" Unified agent tool selection failed ({exc}); falling back to vector search.")
        return fallback
