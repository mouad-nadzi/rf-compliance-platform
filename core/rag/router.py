"""
core/rag/router.py — Intelligent Dual-Path Query Intent Classifier

Classifies natural language user queries into three distinct routing intents:
  1. METADATA_QUERY: Structured SQL filtering/aggregations over certificate attributes.
  2. UNSTRUCTURED_RAG: Semantic dense vector chunk retrieval across document text.
  3. HYBRID_QUERY: Combined relational metadata filtering + semantic vector search.

Uses the local LLM facade (generate_json) and includes robust fallback exception handling.
"""

import json
import logging
from enum import Enum
from typing import Any, Dict

from core.prompts import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """
    Enum representing the supported routing intent classifications.
    """
    METADATA_QUERY = "METADATA_QUERY"
    UNSTRUCTURED_RAG = "UNSTRUCTURED_RAG"
    HYBRID_QUERY = "HYBRID_QUERY"


try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:
    class RouterDecision(BaseModel):
        """
        Pydantic model representing the output of the query router.
        """
        intent: QueryIntent = Field(
            ...,
            description="The classified routing intent (METADATA_QUERY, UNSTRUCTURED_RAG, or HYBRID_QUERY)."
        )
        reasoning: str = Field(
            ...,
            description="Brief 1-sentence explanation justifying the classification decision."
        )
else:
    class RouterDecision:
        def __init__(self, intent: str, reasoning: str):
            self.intent = intent
            self.reasoning = reasoning

        def model_dump(self) -> Dict[str, Any]:
            return {"intent": str(self.intent), "reasoning": self.reasoning}


def _format_history(history) -> str:
    """Formats conversation turns into a compact history block for the router."""
    if not history:
        return ""
    lines = ["--- PRIOR CONVERSATION HISTORY ---"]
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append("--- END CONVERSATION HISTORY ---")
    return "\n".join(lines)


def classify_intent(user_query: str, history=None) -> Dict[str, str]:
    """
    Evaluates a user query using the local LLM facade and classifies it strictly
    into one of three routing intents: METADATA_QUERY, UNSTRUCTURED_RAG, or HYBRID_QUERY.

    Args:
        user_query (str): The natural language query submitted by the user.
        history (Optional[List[Dict]]): Prior conversation turns ({role, content})
            used to resolve anaphoric follow-up queries (e.g., "the others", "these").

    Returns:
        Dict[str, str]: A dictionary containing:
            - "intent": Classified intent string ("METADATA_QUERY", "UNSTRUCTURED_RAG", or "HYBRID_QUERY")
            - "reasoning": Brief explanation for the decision.
    """
    # 1. Defensive check for empty or whitespace query
    if not user_query or not str(user_query).strip():
        logger.warning("Empty or blank query provided to classify_intent. Returning fallback UNSTRUCTURED_RAG.")
        return {
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": "Blank or empty user query provided; defaulting to UNSTRUCTURED_RAG."
        }

    clean_query = str(user_query).strip()
    history_block = _format_history(history)

    # 2. Call local LLM engine facade with DEFAULT_MAX_TOKENS & disable_thinking=True
    #    The engine formats (system_prompt, user_prompt) into its own native template.
    try:
        from core.llm import generate_json
        user_prompt = (
            f"{history_block}\n\n" if history_block else ""
        ) + f"USER QUERY: {clean_query}\n\nReturn ONLY the raw JSON output matching the schema."
        raw_json_response = generate_json(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            disable_thinking=True,
        )

        # Sanitize potential markdown code block wrappers or extra text
        cleaned_str = raw_json_response.strip()
        if cleaned_str.startswith("```"):
            lines = cleaned_str.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_str = "\n".join(lines).strip()

        # 4. Parse JSON with regex fallback
        try:
            parsed_data = json.loads(cleaned_str)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", cleaned_str, re.DOTALL)
            if match:
                parsed_data = json.loads(match.group(0))
            else:
                raise
        raw_intent = str(parsed_data.get("intent", "")).upper().strip()
        reasoning = str(parsed_data.get("reasoning", "Classification generated by local LLM router.")).strip()

        # 5. Validate intent token against QueryIntent Enum
        valid_intents = {e.value for e in QueryIntent}
        if raw_intent in valid_intents:
            logger.info(f" Query intent classified as '{raw_intent}' for: '{clean_query[:50]}...'")
            return {
                "intent": raw_intent,
                "reasoning": reasoning
            }

        logger.warning(
            f" Unknown intent token '{raw_intent}' returned by LLM router. "
            f"Expected one of {valid_intents}. Defaulting to UNSTRUCTURED_RAG."
        )
        return {
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": f"Unrecognized intent token '{raw_intent}'; defaulted to UNSTRUCTURED_RAG."
        }

    except Exception as e:
        logger.warning(f" Exception occurred during query intent classification: {e}")
        logger.info("    Falling back gracefully to UNSTRUCTURED_RAG.")
        return {
            "intent": QueryIntent.UNSTRUCTURED_RAG.value,
            "reasoning": f"Fallback to UNSTRUCTURED_RAG due to classification error: {str(e)}"
        }


if __name__ == "__main__":
    # Test suite for intent classification router
    print("Testing Router Logic (Mock/Local)...\n")
    test_queries = [
        "List all certificates from Germany",
        "How many certificates expire in 2026?",
        "What are the test requirements for section 4?",
        "For German certificates, what is the warranty policy?",
        ""
    ]

    for q in test_queries:
        decision = classify_intent(q)
        print(f"Query: '{q}'")
        print(f"  -> Decision: {decision}\n")
