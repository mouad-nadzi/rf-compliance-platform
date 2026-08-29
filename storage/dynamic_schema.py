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


def list_all_user_tables() -> List[Dict[str, Any]]:
    """
    Inspect PostgreSQL information_schema.tables to list all public user tables.
    Returns list of dicts with 'table_name', 'is_custom', and 'column_count'.
    """
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
    clean_table = sanitize_identifier(table_name)
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


def create_custom_table(table_name: str, columns: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Dynamically create a new PostgreSQL table with specified column definitions.
    Automatically includes 'id SERIAL PRIMARY KEY' and 'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'.
    """
    clean_table = sanitize_identifier(table_name)
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
    clean_table = sanitize_identifier(table_name)
    if clean_table in SYSTEM_TABLES:
        raise ValueError(f"Table '{clean_table}' is a reserved system table and cannot be deleted.")

    sql = f"DROP TABLE IF EXISTS {clean_table} CASCADE;"
    with engine.begin() as conn:
        conn.execute(text(sql))

    logger.info(f"Successfully dropped dynamic table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "message": f"Table '{clean_table}' dropped successfully.",
    }


def add_column_to_table(table_name: str, column_name: str, column_type: str = "string") -> Dict[str, Any]:
    """
    Add a new column to an existing database table.
    """
    clean_table = sanitize_identifier(table_name)
    clean_column = sanitize_identifier(column_name)
    safe_type = sanitize_data_type(column_type)

    sql = f"ALTER TABLE {clean_table} ADD COLUMN IF NOT EXISTS {clean_column} {safe_type};"
    with engine.begin() as conn:
        conn.execute(text(sql))

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
    clean_table = sanitize_identifier(table_name)
    clean_old = sanitize_identifier(old_column_name)
    clean_new = sanitize_identifier(new_column_name)

    if clean_old in ("id", "created_at"):
        raise ValueError(f"Cannot rename essential system column '{clean_old}'.")

    sql = f"ALTER TABLE {clean_table} RENAME COLUMN {clean_old} TO {clean_new};"
    with engine.begin() as conn:
        conn.execute(text(sql))

    logger.info(f"Renamed column '{clean_old}' -> '{clean_new}' in table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "old_column": clean_old,
        "new_column": clean_new,
        "message": f"Column '{clean_old}' renamed to '{clean_new}' in table '{clean_table}'.",
    }


def drop_column_from_table(table_name: str, column_name: str) -> Dict[str, Any]:
    """
    Drop a column from an existing database table.
    """
    clean_table = sanitize_identifier(table_name)
    clean_column = sanitize_identifier(column_name)

    if clean_column in ("id", "created_at"):
        raise ValueError(f"Cannot drop essential column '{clean_column}'.")

    sql = f"ALTER TABLE {clean_table} DROP COLUMN IF EXISTS {clean_column};"
    with engine.begin() as conn:
        conn.execute(text(sql))

    logger.info(f"Dropped column '{clean_column}' from table '{clean_table}'")
    return {
        "status": "success",
        "table_name": clean_table,
        "column_name": clean_column,
        "message": f"Column '{clean_column}' dropped from table '{clean_table}'.",
    }


def fetch_dynamic_records(table_name: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch all records from a custom dynamic table.
    """
    clean_table = sanitize_identifier(table_name)
    sql = text(f"SELECT * FROM {clean_table} ORDER BY id DESC LIMIT :limit;")
    with get_db_session() as db:
        res = db.execute(sql, {"limit": limit})
        keys = list(res.keys())
        return [dict(zip(keys, row)) for row in res.fetchall()]


def insert_dynamic_record(table_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a record into a custom dynamic table.
    """
    clean_table = sanitize_identifier(table_name)
    cols = get_table_columns(clean_table)
    valid_cols = {c["column_name"] for c in cols if c["column_name"] not in ("id", "created_at")}

    insert_keys = []
    insert_vals = {}
    for k, v in record.items():
        clean_k = sanitize_identifier(k)
        if clean_k in valid_cols:
            insert_keys.append(clean_k)
            insert_vals[clean_k] = v

    if not insert_keys:
        raise ValueError("No valid record fields provided matching table columns.")

    col_names_str = ", ".join(insert_keys)
    param_names_str = ", ".join(f":{k}" for k in insert_keys)
    sql = text(f"INSERT INTO {clean_table} ({col_names_str}) VALUES ({param_names_str}) RETURNING id;")

    with get_db_session() as db:
        new_id = db.execute(sql, insert_vals).scalar()

    return {"status": "success", "table_name": clean_table, "inserted_id": new_id}


def delete_dynamic_record(table_name: str, record_id: int) -> bool:
    """
    Delete a record by ID from a dynamic table.
    """
    clean_table = sanitize_identifier(table_name)
    sql = text(f"DELETE FROM {clean_table} WHERE id = :record_id;")
    with get_db_session() as db:
        res = db.execute(sql, {"record_id": int(record_id)})
        return res.rowcount > 0


def update_dynamic_record(table_name: str, record_id: int, record: Dict[str, Any]) -> bool:
    """
    Update editable columns of a record in a custom dynamic table.
    """
    clean_table = sanitize_identifier(table_name)
    cols = get_table_columns(clean_table)
    valid_cols = {c["column_name"] for c in cols if c["column_name"] not in ("id", "created_at")}

    set_clauses = []
    params = {"record_id": int(record_id)}
    for k, v in record.items():
        clean_k = sanitize_identifier(k)
        if clean_k in valid_cols:
            set_clauses.append(f"{clean_k} = :{clean_k}")
            params[clean_k] = v

    if not set_clauses:
        return False

    sql = text(f"UPDATE {clean_table} SET {', '.join(set_clauses)} WHERE id = :record_id;")
    with get_db_session() as db:
        res = db.execute(sql, params)
        return res.rowcount > 0


def bulk_insert_dynamic_records(table_name: str, records: List[Dict[str, Any]]) -> int:
    """
    Bulk insert records into a custom dynamic table (e.g. from CSV/Excel import).
    """
    clean_table = sanitize_identifier(table_name)
    cols = get_table_columns(clean_table)
    valid_cols = {c["column_name"] for c in cols if c["column_name"] not in ("id", "created_at")}

    count = 0
    with get_db_session() as db:
        for r in records:
            keys = []
            vals = {}
            for k, v in r.items():
                clean_k = sanitize_identifier(k)
                if clean_k in valid_cols:
                    keys.append(clean_k)
                    vals[clean_k] = v
            if keys:
                sql = text(f"INSERT INTO {clean_table} ({', '.join(keys)}) VALUES ({', '.join(':' + k for k in keys)});")
                db.execute(sql, vals)
                count += 1
    return count
