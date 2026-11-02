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
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        logger.info(f" Saved long-term memory id={row.id} [{key}]: '{text[:60]}...'")
        return {
            "id": row.id,
            "memory_key": row.memory_key,
            "fact_text": row.fact_text,
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


def format_memories_for_prompt(limit: int = 30) -> str:
    """
    Formats active long-term memories as a structured context block for LLM prompt injection.
    Prioritizes core identity, capability, and user rule/preference memories first, followed by
    concise table schema summaries so raw UI column JSON never crowds out core directives.
    """
    import json

    memories = get_active_memories(limit=limit)
    if not memories:
        return ""

    core_memories = []
    schema_memories = []
    for m in memories:
        key = str(m.get("memory_key") or "").strip()
        if key.startswith("table_schema:"):
            schema_memories.append(m)
        else:
            core_memories.append(m)

    # Order core memories chronologically (identity & base directives first, then learned rules/preferences)
    core_memories.sort(key=lambda x: x.get("id") or 0)

    lines = ["\n[AGENT LONG-TERM MEMORY (PERSISTENT CROSS-SESSION FACTS & DIRECTIVES)]"]
    for m in core_memories:
        key = m["memory_key"].upper()
        fact = m["fact_text"]
        lines.append(f"- ({key}): {fact}")

    seen_tables = set()
    try:
        from storage.dynamic_schema import resolve_physical_table_name, get_table_columns
    except Exception:
        resolve_physical_table_name = lambda name: name.lower().replace(" ", "_")
        get_table_columns = lambda name: []

    for m in schema_memories:
        raw_fact = m.get("fact_text") or ""
        try:
            prof = json.loads(raw_fact)
            tbl = str(prof.get("table_name") or m["memory_key"].split(":", 1)[1]).strip()
            tbl_key = resolve_physical_table_name(tbl)
            if tbl_key in seen_tables:
                continue
            seen_tables.add(tbl_key)
            live_cols = get_table_columns(tbl_key)
            if live_cols:
                fields_str = ", ".join(f"{c['column_name']} ({c['data_type']})" for c in live_cols)
            else:
                fields = prof.get("target_fields") or list((prof.get("columns") or {}).keys())
                fields_str = ", ".join(str(f) for f in fields)
            desc = str(prof.get("description") or "").strip()
            lines.append(f"- (TABLE_SCHEMA:{tbl_key}): {desc} Columns: [{fields_str}]")
        except Exception:
            lines.append(f"- ({m['memory_key'].upper()}): {raw_fact}")

    lines.append("[END LONG-TERM MEMORY]\n")
    return "\n".join(lines)


DEFAULT_IDENTITY_MEMORIES = [
    {
        "memory_key": "identity",
        "fact_text": "You are the Automotive RF Compliance Intelligence Assistant for Stellantis. You are a warm, executive, professional AI assistant built to simplify global radio-frequency homologation, component certificate management, and regulatory compliance.",
    },
    {
        "memory_key": "tone_and_format",
        "fact_text": "Always communicate in a clear, executive, user-friendly, and business-focused tone in the same language as the user's message. Strictly avoid internal developer jargon (such as SQL, vector search, RRF, regex, ORM, endpoints, parsers) when addressing users. Focus on real-world business capabilities: answering compliance queries, searching component certificates, inspecting external portals, downloading and processing PDF/Excel documents, auto-filling regulatory authorities, and tracking expirations.",
    },
    {
        "memory_key": "platform_capabilities",
        "fact_text": "Your native capabilities include: 1) Answering natural-language compliance questions about vehicle component certificates, global authorities, and expiration dates. 2) Instantly searching, filtering, creating, and updating compliance tables across suppliers, components, countries, authorities, and responsible engineers. 3) Inspecting external web portals/URLs to discover downloadable certificate documents, downloading PDFs directly from web URLs, and ingesting them into the database via OCR and metadata extraction. 4) Processing uploaded PDF certificates and Excel spreadsheets with header mapping and data standardization. 5) Auto-completing missing country, supplier, or regulatory authority information.",
    },
    {
        "memory_key": "platform_tools",
        "fact_text": "You operate directly through the platform's integrated tools: 1) search_database_sql (structured queries across all compliance tables), 2) search_document_chunks (semantic retrieval over certificate document text), 3) search_hybrid (combined structured and document search), 4) execute_agent_action (check_url for portal discovery, download_documents for direct URL downloading, ingest_to_database for OCR extraction and database persistence, and update_record/delete_record/manage_schema/execute_raw_sql for database management), and 5) Bidirectional Authority-Country-Supplier lookup knowledge base.",
    },
]


def seed_base_identity_memories() -> None:
    """
    Idempotently seeds default agent identity and behavioral directives into PostgreSQL memory store.
    """
    try:
        with get_db_session() as session:
            # Remove legacy redundant key if present so capabilities are consolidated cleanly
            legacy = session.query(AgentMemory).filter(AgentMemory.memory_key == "system_capabilities").first()
            if legacy:
                session.delete(legacy)
            existing_keys = {m.memory_key for m in session.query(AgentMemory).all()}
            for item in DEFAULT_IDENTITY_MEMORIES:
                if item["memory_key"] not in existing_keys:
                    row = AgentMemory(
                        memory_key=item["memory_key"],
                        fact_text=item["fact_text"],
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
