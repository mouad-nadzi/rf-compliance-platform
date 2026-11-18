"""
core/utils/history.py - Canonical conversation history formatter.

Single source of truth for converting a list of {role, content} turns into
a formatted text block for LLM prompt injection. Bounds total turns and compacts
oversized assistant tables so multi-turn sessions stay fast and prefix-cacheable.
"""

from typing import Dict, List, Optional

DEFAULT_MAX_HISTORY_TURNS = 16
DEFAULT_MAX_CHARS_PER_TURN = 1200


def _compact_turn_content(content: str, max_chars: int = DEFAULT_MAX_CHARS_PER_TURN) -> str:
    """
    Deterministic compaction for oversized history turns (e.g. large Markdown tables).
    Preserves the opening summary/table header and closing note while bounding token growth.
    """
    text = str(content or "").strip()
    if len(text) <= max_chars:
        return text
    head_len = int(max_chars * 0.75)
    tail_len = max_chars - head_len - 40
    if tail_len > 60:
        return f"{text[:head_len].rstrip()}\n... [truncated in history] ...\n{text[-tail_len:].lstrip()}"
    return f"{text[:max_chars].rstrip()} ... [truncated]"


def format_conversation_history(
    history: Optional[List[Dict[str, str]]],
    max_turns: Optional[int] = DEFAULT_MAX_HISTORY_TURNS,
    max_chars_per_turn: int = DEFAULT_MAX_CHARS_PER_TURN,
) -> str:
    """
    Formats a list of {role, content} conversation turns into a compact text
    block suitable for LLM prompt injection and prefix-state caching.

    Args:
        history: list of dicts with 'role' ('user' or 'assistant') and 'content'.
        max_turns: maximum recent turns to include (defaults to 16 = 8 exchanges).
        max_chars_per_turn: maximum characters per individual turn before compaction.

    Returns:
        Formatted string, or empty string if history is empty/None.
    """
    if not history:
        return ""
    turns = list(history)
    if max_turns is not None and len(turns) > max_turns:
        turns = turns[-max_turns:]
    lines = ["--- CONVERSATION HISTORY ---"]
    for turn in turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = _compact_turn_content(turn.get("content", ""), max_chars=max_chars_per_turn)
        if content:
            lines.append(f"{role}: {content}")
    if len(lines) == 1:
        return ""
    lines.append("--- END CONVERSATION HISTORY ---")
    return "\n".join(lines)
