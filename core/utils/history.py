"""
core/utils/history.py - Canonical conversation history formatter.

Single source of truth for converting a list of {role, content} turns into
a formatted text block for LLM prompt injection. All modules that need to
inject conversation history must import from here.
"""

from typing import Dict, List, Optional


def format_conversation_history(
    history: Optional[List[Dict[str, str]]],
    max_turns: Optional[int] = None,
) -> str:
    """
    Formats a list of {role, content} conversation turns into a compact text
    block suitable for LLM prompt injection.

    Args:
        history: list of dicts with 'role' ('user' or 'assistant') and 'content'.
        max_turns: if set, only the last N turns are included (avoids token bloat).

    Returns:
        Formatted string, or empty string if history is empty/None.
    """
    if not history:
        return ""
    turns = list(history)
    if max_turns is not None:
        turns = turns[-max_turns:]
    lines = ["--- CONVERSATION HISTORY ---"]
    for turn in turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append("--- END CONVERSATION HISTORY ---")
    return "\n".join(lines)
