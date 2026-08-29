"""
core/agent/memory.py — Session-Independent Long-Term Memory Engine

Manages persistent cross-session facts, rules, user preferences, and contact profiles
stored in the `agent_memories` PostgreSQL table. Formats memories for prompt injection.
"""

import logging
from typing import Any, Dict, List, Optional
from storage.database import SessionLocal
from schemas.extraction import AgentMemory

logger = logging.getLogger(__name__)


def save_agent_memory(
    memory_key: str,
    fact_text: str,
    source_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Saves a long-term memory fact to PostgreSQL.
    """
    key = str(memory_key or "preference").strip().lower()
    text = str(fact_text or "").strip()
    if not text:
        raise ValueError("Memory fact_text cannot be empty.")

    session = SessionLocal()
    try:
        row = AgentMemory(
            memory_key=key,
            fact_text=text,
            source_session_id=source_session_id,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        logger.info(f" Saved long-term memory id={row.id} [{key}]: '{text[:60]}...'")
        return {
            "id": row.id,
            "memory_key": row.memory_key,
            "fact_text": row.fact_text,
            "source_session_id": row.source_session_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    except Exception as exc:
        session.rollback()
        logger.error(f" Failed to save long-term memory: {exc}")
        raise
    finally:
        session.close()


def get_active_memories(category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetches active long-term memories from database (newest first).
    """
    session = SessionLocal()
    try:
        query = session.query(AgentMemory)
        if category:
            query = query.filter(AgentMemory.memory_key == str(category).strip().lower())
        rows = query.order_by(AgentMemory.created_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "memory_key": r.memory_key,
                "fact_text": r.fact_text,
                "source_session_id": r.source_session_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f" Could not load long-term memories: {exc}")
        return []
    finally:
        session.close()


def delete_agent_memory(memory_id: int) -> bool:
    """
    Deletes a long-term memory record by ID.
    """
    session = SessionLocal()
    try:
        row = session.query(AgentMemory).filter(AgentMemory.id == memory_id).first()
        if not row:
            return False
        session.delete(row)
        session.commit()
        logger.info(f" Deleted long-term memory id={memory_id}.")
        return True
    except Exception as exc:
        session.rollback()
        logger.error(f" Failed to delete long-term memory id={memory_id}: {exc}")
        return False
    finally:
        session.close()


def format_memories_for_prompt(limit: int = 20) -> str:
    """
    Formats active long-term memories as a structured context block for LLM prompt injection.
    """
    memories = get_active_memories(limit=limit)
    if not memories:
        return ""

    lines = ["\n[AGENT LONG-TERM MEMORY (PERSISTENT CROSS-SESSION FACTS & DIRECTIVES)]"]
    for m in memories:
        key = m["memory_key"].upper()
        fact = m["fact_text"]
        lines.append(f"- ({key}): {fact}")
    lines.append("[END LONG-TERM MEMORY]\n")

    return "\n".join(lines)


DEFAULT_IDENTITY_MEMORIES = [
    {
        "memory_key": "identity",
        "fact_text": "You are the Automotive RF Certificate Compliance Assistant, an expert AI regulatory agent. Your mission is to assist compliance engineers in searching, extracting, auditing, and managing automotive telecommunications certificates (FCC, ENACOM, ANATEL, ATT, CE, BNetzA, ICASA, etc.).",
    },
    {
        "memory_key": "tone_and_format",
        "fact_text": "Maintain a direct, precise, executive, evidence-based, and professional tone. Always cite source documents and database records accurately without fabricating claims.",
    },
]


def seed_base_identity_memories() -> None:
    """
    Idempotently seeds default agent identity and behavioral directives into PostgreSQL memory store.
    """
    session = SessionLocal()
    try:
        existing_keys = {m.memory_key for m in session.query(AgentMemory).all()}
        for item in DEFAULT_IDENTITY_MEMORIES:
            if item["memory_key"] not in existing_keys:
                row = AgentMemory(
                    memory_key=item["memory_key"],
                    fact_text=item["fact_text"],
                    source_session_id="system_seed",
                )
                session.add(row)
                logger.info(f" Seeded base memory [{item['memory_key']}]: {item['fact_text'][:60]}...")
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning(f" Could not seed base identity memories: {exc}")
    finally:
        session.close()
