"""
core/agent/memory.py — Session-Independent Long-Term Memory Engine

Manages persistent cross-session facts, rules, user preferences, and contact profiles
stored in the `agent_memories` PostgreSQL table. Formats memories for prompt injection.
"""

import logging
from typing import Any, Dict, List, Optional
from storage.database import get_db_session
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

    with get_db_session() as session:
        row = AgentMemory(
            memory_key=key,
            fact_text=text,
            source_session_id=source_session_id,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        logger.info(f" Saved long-term memory id={row.id} [{key}]: '{text[:60]}...'")
        return {
            "id": row.id,
            "memory_key": row.memory_key,
            "fact_text": row.fact_text,
            "source_session_id": row.source_session_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


def get_active_memories(category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetches active long-term memories from database (newest first).
    """
    try:
        with get_db_session() as session:
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


def delete_agent_memory(memory_id: int) -> bool:
    """
    Deletes a long-term memory record by moving it to the Recycle Bin.
    """
    import json
    import uuid
    from datetime import datetime
    from schemas.extraction import RecycleBinItem
    
    try:
        with get_db_session() as session:
            row = session.query(AgentMemory).filter(AgentMemory.id == memory_id).first()
            if not row:
                return False
                
            # Soft-delete to Recycle Bin
            rec_item = RecycleBinItem(
                id=uuid.uuid4().hex,
                title=f"Agent Memory: {row.memory_key}",
                table_name="agent_memories",
                record_data=json.dumps({
                    "id": row.id,
                    "memory_key": row.memory_key,
                    "fact_text": row.fact_text,
                    "source_session_id": row.source_session_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                }),
                deleted_at=datetime.utcnow()
            )
            session.add(rec_item)
            
            session.delete(row)
            logger.info(f" Moved long-term memory id={memory_id} to Recycle Bin.")
            return True
    except Exception as exc:
        logger.error(f" Failed to delete long-term memory id={memory_id}: {exc}")
        return False


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
        "fact_text": "You are the Automotive RF Compliance Intelligence Assistant for Stellantis. You are a warm, executive, professional AI assistant built to simplify global radio-frequency homologation, component certificate management, and regulatory compliance.",
    },
    {
        "memory_key": "tone_and_format",
        "fact_text": "Always communicate in a clear, executive, user-friendly, and business-focused tone. Strictly avoid internal developer jargon (such as SQL, vector search, RRF, regex, ORM, endpoints, parsers) when addressing users. Focus on real-world business capabilities: answering compliance queries, searching component certificates, processing PDF/Excel uploads, auto-filling regulatory authorities, and tracking expirations.",
    },
    {
        "memory_key": "platform_capabilities",
        "fact_text": "Your capabilities include: 1) Answering natural-language compliance questions about vehicle component certificates, global authorities, and expiration dates. 2) Instantly searching and filtering compliance tables across suppliers, components, countries, and authorities. 3) Automatically processing uploaded PDF certificates and Excel spreadsheets to extract and verify data. 4) Auto-completing missing country or regulatory authority information. 5) Discovering and monitoring supplier compliance documentation.",
    },
]


def seed_base_identity_memories() -> None:
    """
    Idempotently seeds default agent identity and behavioral directives into PostgreSQL memory store.
    """
    try:
        with get_db_session() as session:
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
                else:
                    # Update existing system seed memory if fact text changed
                    existing_row = session.query(AgentMemory).filter(AgentMemory.memory_key == item["memory_key"]).first()
                    if existing_row and existing_row.fact_text != item["fact_text"]:
                        existing_row.fact_text = item["fact_text"]
                        logger.info(f" Updated base memory [{item['memory_key']}]: {item['fact_text'][:60]}...")
            session.commit()
    except Exception as exc:
        logger.warning(f" Could not seed base identity memories: {exc}")
