"""
storage/dynamic_schema.py — Dynamic PostgreSQL Schema & Custom Table Engine

Provides safe, parameterized Data Definition Language (DDL) and Data Manipulation
Language (DML) helper functions for creating custom database tables, modifying columns,
and performing dynamic CRUD operations on user-defined tables.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from storage.database import get_db_session, engine

logger = logging.getLogger(__name__)

# Core platform data tables and internal engine tables
CORE_PLATFORM_TABLES = {
    "certificates",
    "authorities",
    "authority_lookups",
    "suppliers",
    "supplier_lookups",
    "sources",
    "agent_memories",
    "notifications",
    "recycle_bin",
    "recycle_bin_records",
}
INTERNAL_ENGINE_TABLES = {
    "alembic_version",
    "certificate_chunks",
    "chat_sessions",
    "chat_messages",
    "agent_actions",
    "workflows",
    "workflow_runs",
}
SYSTEM_TABLES = CORE_PLATFORM_TABLES | INTERNAL_ENGINE_TABLES


# Allowed SQL data types map to prevent SQL injection in DDL statements
ALLOWED_TYPES = {
    "string": "VARCHAR(255)",
    "varchar": "VARCHAR(255)",
    "text": "TEXT",
    "int": "INTEGER",
    "integer": "INTEGER",
    "float": "DOUBLE PRECISION",
    "double": "DOUBLE PRECISION",
    "boolean": "BOOLEAN",
    "bool": "BOOLEAN",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
    "json": "JSONB",
    "jsonb": "JSONB",
}


TABLE_NAME_ALIASES = {
    "rf certificates": "certificates",
    "certificates": "certificates",
    "rf_certificates": "certificates",
    "authorities": "authority_lookups",
    "authority_lookups": "authority_lookups",
    "authority lookups": "authority_lookups",
    "suppliers": "supplier_lookups",
    "supplier_lookups": "supplier_lookups",
    "supplier lookups": "supplier_lookups",
    "sources": "sources",
    "agent memories": "agent_memories",
    "agent_memories": "agent_memories",
}


def resolve_physical_table_name(table_name: str) -> str:
    """Map UI/display table names or aliases to actual PostgreSQL table names."""
    tbl = str(table_name or "").strip().lower()
    if tbl in TABLE_NAME_ALIASES:
        return TABLE_NAME_ALIASES[tbl]
    return sanitize_identifier(tbl)


def sanitize_identifier(identifier: str) -> str:
    """
    Sanitize SQL table or column names to prevent SQL injection.
    Only allows lowercase alphanumeric characters and underscores.
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(identifier or "").strip().lower())
    clean = re.sub(r"^[^a-zA-Z_]+", "", clean)  # Must start with letter or underscore
    if not clean:
        raise ValueError(f"Invalid SQL identifier: '{identifier}'")
    return clean[:63]  # PostgreSQL identifier max length is 63 chars


def sanitize_data_type(data_type: str) -> str:
    """Map user-provided type string to safe SQL data type."""
    raw = str(data_type or "").strip().lower()
    if raw in ALLOWED_TYPES:
        return ALLOWED_TYPES[raw]
    # Check if raw type contains size specifier like VARCHAR(100)
    for key, sql_type in ALLOWED_TYPES.items():
        if raw.startswith(key):
            return sql_type
    return "VARCHAR(255)"


def cleanup_orphaned_schema_memories() -> int:
    """
    Scans `agent_memories` for any `table_schema:<table_name>` entries whose corresponding
    PostgreSQL table no longer exists, and purges them automatically (self-healing reconciliation).
    """
    from schemas.extraction import AgentMemory
    with get_db_session() as db:
        try:
            active_tables_rows = db.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")).fetchall()
            active_tables = {r.table_name.lower() for r in active_tables_rows}

            schema_mems = db.query(AgentMemory).filter(AgentMemory.memory_key.like("table_schema:%")).all()
            deleted_count = 0
            for mem in schema_mems:
                tbl_name = mem.memory_key.split("table_schema:", 1)[1].strip().lower()
                resolved = resolve_physical_table_name(tbl_name)
                if tbl_name not in active_tables and resolved not in active_tables:
                    db.delete(mem)
                    deleted_count += 1
            if deleted_count > 0:
                db.commit()
                logger.info(f"Purged {deleted_count} orphaned table_schema memory entries.")
            return deleted_count
        except Exception as exc:
            db.rollback()
            logger.warning(f"Error during orphaned schema memory cleanup: {exc}")
            return 0


def list_all_user_tables() -> List[Dict[str, Any]]:
    """
    Inspect PostgreSQL information_schema.tables to list all public user tables.
    Automatically purges orphaned schema memories from dropped tables.
    Returns list of dicts with 'table_name', 'is_custom', and 'column_count'.
    """
    cleanup_orphaned_schema_memories()
    excluded_tuple = tuple(INTERNAL_ENGINE_TABLES)
    query = text("""
        SELECT t.table_name,
               count(c.column_name) AS column_count
        FROM information_schema.tables t
        JOIN information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
        WHERE t.table_schema = 'public'
          AND t.table_type = 'BASE TABLE'
          AND t.table_name NOT IN :excluded_tables
        GROUP BY t.table_name
        ORDER BY t.table_name;
    """)
    with get_db_session() as db:
        rows = db.execute(query, {"excluded_tables": excluded_tuple}).fetchall()
        result = []
        for r in rows:
            tbl = r.table_name
            result.append({
                "table_name": tbl,
                "is_custom": tbl not in CORE_PLATFORM_TABLES,
                "column_count": r.column_count,
            })
        return result


def get_table_columns(table_name: str) -> List[Dict[str, str]]:
    """
    Inspect column names and data types for a target table.
    """
    clean_table = resolve_physical_table_name(table_name)
    query = text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :table_name
        ORDER BY ordinal_position;
    """)
    with get_db_session() as db:
        rows = db.execute(query, {"table_name": clean_table}).fetchall()
        return [
            {
                "column_name": r.column_name,
                "data_type": r.data_type,
                "is_nullable": r.is_nullable,
            }
            for r in rows
        ]


def sync_table_schema_memory(table_name: str) -> None:
    """
    Synchronizes the table's schema profile in `agent_memories` (`table_schema:<table_name>`)
    with actual live PostgreSQL columns, keeping target_fields and columns metadata strictly in sync.
    """
    import json
    from datetime import datetime
    from schemas.extraction import AgentMemory

    clean_table = resolve_physical_table_name(table_name)
    raw_columns = get_table_columns(clean_table)
    col_names = [c["column_name"] for c in raw_columns if c["column_name"] not in ("id", "created_at")]

    memory_key = f"table_schema:{clean_table.strip().lower()}"
    with get_db_session() as db:
        try:
            mems = db.query(AgentMemory).filter(AgentMemory.memory_key == memory_key).all()
            if not raw_columns:
                # Table was dropped
                for m in mems:
                    db.delete(m)
                db.commit()
                return

            mem = mems[0] if mems else None
            if len(mems) > 1:
                # Purge duplicate rows
                for extra_mem in mems[1:]:
                    db.delete(extra_mem)

            semantic_map = {}
            if mem:
                try:
                    prof = json.loads(mem.fact_text)
                    semantic_map = prof.get("columns", {})
                except Exception:
                    semantic_map = {}

            # Remove columns that no longer exist in physical table
            semantic_map = {k: v for k, v in semantic_map.items() if k in col_names}

            # Add any new columns with heuristic defaults
            for c in col_names:
                if c not in semantic_map:
                    clow = c.lower()
                    if 'email' in clow:
                        stype = 'email'
                    elif 'link' in clow or 'url' in clow or 'pdf' in clow or 'uri' in clow:
                        stype = 'url'
                    elif 'date' in clow or 'time' in clow or clow.endswith('_at'):
                        stype = 'date'
                    elif 'phone' in clow or 'tel' in clow or 'fax' in clow or 'mobile' in clow:
                        stype = 'phone'
                    elif 'supplier' in clow or 'vendor' in clow or 'manufacturer' in clow:
                        stype = 'supplier'
                    elif 'status' in clow or 'active' in clow or 'priority' in clow or 'badge' in clow:
                        stype = 'badge'
                    elif 'count' in clow or 'score' in clow or 'amount' in clow:
                        stype = 'number'
                    else:
                        stype = 'text'
                    semantic_map[c] = {'semantic_type': stype, 'label': c.replace('_', ' ').title()}

            now_iso = datetime.utcnow().isoformat()
            schema_prof = {
                "table_name": clean_table,
                "description": f"Database table {clean_table}.",
                "target_fields": col_names,
                "columns": semantic_map,
                "registered_at": now_iso,
                "last_updated": now_iso,
            }
            if mem:
                mem.fact_text = json.dumps(schema_prof)
            else:
                new_mem = AgentMemory(memory_key=memory_key, fact_text=json.dumps(schema_prof))
                db.add(new_mem)
            db.commit()
            logger.info(f"Synchronized schema memory for '{clean_table}' with fields {col_names}")
        except Exception as exc:
            db.rollback()
            logger.warning(f"Failed to sync schema memory for '{clean_table}': {exc}")


def create_custom_table(table_name: str, columns: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Dynamically create a new PostgreSQL table with specified column definitions.
    Automatically includes 'id SERIAL PRIMARY KEY' and 'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'.
    """
    clean_table = resolve_physical_table_name(table_name)
    if clean_table in SYSTEM_TABLES:
        raise ValueError(f"Table name '{clean_table}' is a reserved system table.")

    col_defs = ["id SERIAL PRIMARY KEY"]
    added_cols = []

    for col in columns:
        col_name = sanitize_identifier(col.get("name") or col.get("column_name", ""))
        if col_name in ("id", "created_at"):
            continue
        col_type = sanitize_data_type(col.get("type") or col.get("data_type", "string"))
        col_defs.append(f"{col_name} {col_type}")
        added_cols.append(col_name)

    col_defs.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    sql = f"CREATE TABLE IF NOT EXISTS {clean_table} ({', '.join(col_defs)});"

    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Successfully created dynamic table '{clean_table}' with columns {added_cols}")
    return {
        "status": "success",
        "table_name": clean_table,
        "columns": ["id"] + added_cols + ["created_at"],
        "message": f"Table '{clean_table}' created successfully.",
    }


def drop_custom_table(table_name: str) -> Dict[str, Any]:
    """
    Safely drop a dynamic user-created table from PostgreSQL.
    System core tables and engine infrastructure tables cannot be dropped.
    """
    clean_table = resolve_physical_table_name(table_name)
    if clean_table in SYSTEM_TABLES:
        raise ValueError(f"Table '{clean_table}' is a reserved system table and cannot be deleted.")

    sql = f"DROP TABLE IF EXISTS {clean_table} CASCADE;"
    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Successfully dropped dynamic table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "message": f"Table '{clean_table}' dropped successfully.",
    }


def add_columns_to_table(table_name: str, columns: List[Any]) -> Dict[str, Any]:
    """
    Add multiple columns to an existing database table in a single ALTER TABLE statement.
    """
    clean_table = resolve_physical_table_name(table_name)
    clauses = []
    added = []
    for col in columns:
        if isinstance(col, str):
            c_name = col
            c_type = "string"
        elif isinstance(col, dict):
            c_name = col.get("name") or col.get("column_name", "")
            c_type = col.get("type") or col.get("data_type", "string")
        else:
            continue
        clean_col = sanitize_identifier(c_name)
        safe_type = sanitize_data_type(c_type)
        clauses.append(f"ADD COLUMN IF NOT EXISTS {clean_col} {safe_type}")
        added.append({"name": clean_col, "type": safe_type})

    if not clauses:
        raise ValueError("No valid columns provided to add.")

    sql = f"ALTER TABLE {clean_table} {', '.join(clauses)};"
    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Added {len(added)} column(s) to table '{clean_table}': {[c['name'] for c in added]}")
    return {
        "status": "success",
        "table_name": clean_table,
        "added_columns": added,
        "message": f"Successfully added {len(added)} column(s) to table '{clean_table}'.",
    }


def add_column_to_table(table_name: str, column_name: Any, column_type: str = "string") -> Dict[str, Any]:
    """
    Add a new column (or multiple columns) to an existing database table (supports core platform tables and custom tables).
    """
    if isinstance(column_name, (list, tuple)):
        return add_columns_to_table(table_name, list(column_name))

    clean_table = resolve_physical_table_name(table_name)
    clean_column = sanitize_identifier(column_name)
    safe_type = sanitize_data_type(column_type)

    sql = f"ALTER TABLE {clean_table} ADD COLUMN IF NOT EXISTS {clean_column} {safe_type};"
    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Added column '{clean_column}' ({safe_type}) to table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "column_name": clean_column,
        "data_type": safe_type,
        "message": f"Column '{clean_column}' added to table '{clean_table}'.",
    }


def rename_column_in_table(table_name: str, old_column_name: str, new_column_name: str) -> Dict[str, Any]:
    """
    Rename an existing column in a database table.
    """
    clean_table = resolve_physical_table_name(table_name)
    clean_old = sanitize_identifier(old_column_name)
    clean_new = sanitize_identifier(new_column_name)

    if clean_old in ("id", "created_at") or (clean_table == "certificates" and clean_old == "certificate_id"):
        raise ValueError(f"Cannot rename essential system column '{clean_old}'.")

    sql = f"ALTER TABLE {clean_table} RENAME COLUMN {clean_old} TO {clean_new};"
    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Renamed column '{clean_old}' -> '{clean_new}' in table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "old_column": clean_old,
        "new_column": clean_new,
        "message": f"Column '{clean_old}' renamed to '{clean_new}' in table '{clean_table}'.",
    }


def drop_columns_from_table(table_name: str, column_names: Any) -> Dict[str, Any]:
    """
    Drop multiple columns from an existing database table in a single ALTER TABLE statement.
    Supports a list of strings, comma-separated string, or list of dicts.
    """
    clean_table = resolve_physical_table_name(table_name)

    raw_list: List[str] = []
    if isinstance(column_names, str):
        raw_list = [c.strip() for c in column_names.split(",") if c.strip()]
    elif isinstance(column_names, (list, tuple)):
        for item in column_names:
            if isinstance(item, str):
                raw_list.extend([c.strip() for c in item.split(",") if c.strip()])
            elif isinstance(item, dict):
                val = item.get("name") or item.get("column_name", "")
                if val:
                    raw_list.append(str(val).strip())

    if not raw_list:
        raise ValueError("No valid column names provided to drop.")

    clean_columns = []
    for raw in raw_list:
        clean_col = sanitize_identifier(raw)
        if clean_col in ("id", "created_at") or (clean_table == "certificates" and clean_col == "certificate_id"):
            raise ValueError(f"Cannot drop essential column '{clean_col}'.")
        clean_columns.append(clean_col)

    unique_cols = list(dict.fromkeys(clean_columns))
    clauses = [f"DROP COLUMN IF EXISTS {c}" for c in unique_cols]
    sql = f"ALTER TABLE {clean_table} {', '.join(clauses)};"

    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Dropped {len(unique_cols)} column(s) from table '{clean_table}': {unique_cols}")
    return {
        "status": "success",
        "table_name": clean_table,
        "dropped_columns": unique_cols,
        "message": f"Successfully dropped {len(unique_cols)} column(s) from table '{clean_table}'.",
    }


def drop_column_from_table(table_name: str, column_name: Any) -> Dict[str, Any]:
    """
    Drop one or more columns from an existing database table.
    """
    if isinstance(column_name, (list, tuple)) or (isinstance(column_name, str) and "," in column_name):
        return drop_columns_from_table(table_name, column_name)

    clean_table = resolve_physical_table_name(table_name)
    clean_column = sanitize_identifier(column_name)

    if clean_column in ("id", "created_at") or (clean_table == "certificates" and clean_column == "certificate_id"):
        raise ValueError(f"Cannot drop essential column '{clean_column}'.")

    sql = f"ALTER TABLE {clean_table} DROP COLUMN IF EXISTS {clean_column};"
    with engine.begin() as conn:
        conn.execute(text(sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Dropped column '{clean_column}' from table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "column_name": clean_column,
        "message": f"Column '{clean_column}' dropped from table '{clean_table}'.",
    }


def combine_columns_in_table(
    table_name: str,
    target_column: str,
    source_columns: List[str],
    separator: str = " ",
    target_type: str = "string",
    drop_sources: bool = False,
) -> Dict[str, Any]:
    """
    Combine/merge multiple text columns into a target column (e.g. firstname + lastname -> responsible_name),
    and optionally drop the source columns in the same transaction.
    """
    clean_table = resolve_physical_table_name(table_name)
    clean_target = sanitize_identifier(target_column)
    clean_sources = [sanitize_identifier(s) for s in source_columns if s]

    if not clean_sources:
        raise ValueError("source_columns must contain at least one column.")

    safe_target_type = sanitize_data_type(target_type)

    # 1. Add target column if not exists
    add_sql = f"ALTER TABLE {clean_table} ADD COLUMN IF NOT EXISTS {clean_target} {safe_target_type};"

    # 2. Build CONCAT_WS expression
    cols_coalesced = ", ".join([f"COALESCE({c}::text, '')" for c in clean_sources])
    sep_literal = separator.replace("'", "''")
    update_sql = f"UPDATE {clean_table} SET {clean_target} = TRIM(CONCAT_WS('{sep_literal}', {cols_coalesced}));"

    with engine.begin() as conn:
        conn.execute(text(add_sql))
        conn.execute(text(update_sql))

        if drop_sources:
            drop_clauses = []
            for sc in clean_sources:
                if sc != clean_target and sc not in ("id", "created_at", "certificate_id"):
                    drop_clauses.append(f"DROP COLUMN IF EXISTS {sc}")
            if drop_clauses:
                drop_sql = f"ALTER TABLE {clean_table} {', '.join(drop_clauses)};"
                conn.execute(text(drop_sql))

    sync_table_schema_memory(clean_table)
    logger.info(f"Combined {clean_sources} into '{clean_target}' in table '{clean_table}' (drop_sources={drop_sources})")
    return {
        "status": "success",
        "table_name": clean_table,
        "target_column": clean_target,
        "source_columns": clean_sources,
        "drop_sources": drop_sources,
        "message": f"Successfully combined {clean_sources} into '{clean_target}' in table '{clean_table}'.",
    }


def fetch_dynamic_records(table_name: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch all records from a database table (supports core and dynamic tables).
    """
    clean_table = resolve_physical_table_name(table_name)
    cols = [c["column_name"] for c in get_table_columns(clean_table)]
    order_clause = ""
    if "id" in cols:
        order_clause = "ORDER BY id DESC"
    elif "created_at" in cols:
        order_clause = "ORDER BY created_at DESC"
    elif "certificate_id" in cols:
        order_clause = "ORDER BY certificate_id DESC"

    sql = text(f"SELECT * FROM {clean_table} {order_clause} LIMIT :limit;")
    with get_db_session() as db:
        res = db.execute(sql, {"limit": limit})
        keys = list(res.keys())
        rows = []
        for row in res.fetchall():
            d = dict(zip(keys, row))
            for k, v in d.items():
                if hasattr(v, "isoformat"):
                    d[k] = v.isoformat()
            rows.append(d)
        return rows


def insert_dynamic_record(table_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a record into a custom dynamic table.
    """
    clean_table = resolve_physical_table_name(table_name)
    cols = get_table_columns(clean_table)
    valid_cols = [c["column_name"] for c in cols if c["column_name"] not in ("id", "created_at")]
    valid_cols_set = set(valid_cols)
    norm_to_col = {re.sub(r"[^a-zA-Z0-9]", "", c.lower()): c for c in valid_cols}

    col_vals: Dict[str, Any] = {}
    for k, v in record.items():
        if not k or str(k).startswith("_") or str(k).lower() in ("id", "created_at"):
            continue
        try:
            clean_k = sanitize_identifier(k)
        except Exception:
            clean_k = ""
        norm_k = re.sub(r"[^a-zA-Z0-9]", "", str(k).lower())

        target_col = None
        if clean_k and clean_k in valid_cols_set:
            target_col = clean_k
        elif norm_k in norm_to_col:
            target_col = norm_to_col[norm_k]

        if target_col:
            # If not yet set, or if replacing an empty value with a non-empty one
            if target_col not in col_vals or (col_vals[target_col] in (None, "") and v not in (None, "")):
                col_vals[target_col] = v

    if not col_vals:
        raise ValueError("No valid record fields provided matching table columns.")

    keys = list(col_vals.keys())
    col_names_str = ", ".join(keys)
    param_names_str = ", ".join(f":{k}" for k in keys)
    sql = text(f"INSERT INTO {clean_table} ({col_names_str}) VALUES ({param_names_str}) RETURNING id;")

    with get_db_session() as db:
        new_id = db.execute(sql, col_vals).scalar()

    return {"status": "success", "table_name": clean_table, "inserted_id": new_id}


def delete_dynamic_record(table_name: str, record_id: int) -> bool:
    """
    Delete a record by ID from a dynamic table and move it to the Recycle Bin.
    """
    clean_table = sanitize_identifier(table_name)
    from schemas.extraction import RecycleBinItem
    import json, uuid
    from datetime import datetime
    
    with get_db_session() as db:
        fetch_sql = text(f"SELECT * FROM {clean_table} WHERE id = :record_id;")
        row_proxy = db.execute(fetch_sql, {"record_id": int(record_id)}).first()
        if not row_proxy:
            return False
            
        row_dict = dict(row_proxy._mapping)
        
        rec_item = RecycleBinItem(
            id=uuid.uuid4().hex,
            title=f"Record #{record_id} ({clean_table})",
            table_name=f"__TABLE__:{clean_table}",
            record_data=json.dumps(row_dict, default=str),
            deleted_at=datetime.utcnow()
        )
        db.add(rec_item)
        
        sql = text(f"DELETE FROM {clean_table} WHERE id = :record_id;")
        res = db.execute(sql, {"record_id": int(record_id)})
        return res.rowcount > 0


def update_dynamic_record(table_name: str, record_id: int, record: Dict[str, Any]) -> bool:
    """
    Update editable columns of a record in a custom dynamic table.
    """
    clean_table = sanitize_identifier(table_name)
    cols = get_table_columns(clean_table)
    valid_cols = [c["column_name"] for c in cols if c["column_name"] not in ("id", "created_at")]
    valid_cols_set = set(valid_cols)
    norm_to_col = {re.sub(r"[^a-zA-Z0-9]", "", c.lower()): c for c in valid_cols}

    col_vals: Dict[str, Any] = {}
    for k, v in record.items():
        if not k or str(k).startswith("_") or str(k).lower() in ("id", "created_at"):
            continue
        try:
            clean_k = sanitize_identifier(k)
        except Exception:
            clean_k = ""
        norm_k = re.sub(r"[^a-zA-Z0-9]", "", str(k).lower())

        target_col = None
        if clean_k and clean_k in valid_cols_set:
            target_col = clean_k
        elif norm_k in norm_to_col:
            target_col = norm_to_col[norm_k]

        if target_col:
            if target_col not in col_vals or (col_vals[target_col] in (None, "") and v not in (None, "")):
                col_vals[target_col] = v

    if not col_vals:
        return False

    set_clauses = [f"{k} = :{k}" for k in col_vals.keys()]
    params = {"record_id": int(record_id), **col_vals}

    sql = text(f"UPDATE {clean_table} SET {', '.join(set_clauses)} WHERE id = :record_id;")
    with get_db_session() as db:
        res = db.execute(sql, params)
        return res.rowcount > 0


def bulk_insert_dynamic_records(table_name: str, records: List[Dict[str, Any]]) -> int:
    """
    Bulk insert records into a custom dynamic table (e.g. from CSV/Excel import).
    Strictly deduplicates column identifiers to avoid (psycopg2.errors.DuplicateColumn) errors.
    """
    clean_table = sanitize_identifier(table_name)
    cols = get_table_columns(clean_table)
    valid_cols = [c["column_name"] for c in cols if c["column_name"] not in ("id", "created_at")]
    valid_cols_set = set(valid_cols)
    norm_to_col = {re.sub(r"[^a-zA-Z0-9]", "", c.lower()): c for c in valid_cols}

    count = 0
    with get_db_session() as db:
        for r in records:
            col_vals: Dict[str, Any] = {}
            for k, v in r.items():
                if not k or str(k).startswith("_") or str(k).lower() in ("id", "created_at"):
                    continue
                try:
                    clean_k = sanitize_identifier(k)
                except Exception:
                    clean_k = ""
                norm_k = re.sub(r"[^a-zA-Z0-9]", "", str(k).lower())

                target_col = None
                if clean_k and clean_k in valid_cols_set:
                    target_col = clean_k
                elif norm_k in norm_to_col:
                    target_col = norm_to_col[norm_k]

                if target_col:
                    if target_col not in col_vals or (col_vals[target_col] in (None, "") and v not in (None, "")):
                        col_vals[target_col] = v

            if col_vals:
                keys = list(col_vals.keys())
                sql = text(f"INSERT INTO {clean_table} ({', '.join(keys)}) VALUES ({', '.join(':' + k for k in keys)});")
                db.execute(sql, col_vals)
                count += 1
    return count
