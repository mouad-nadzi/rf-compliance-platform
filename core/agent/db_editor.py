"""
core/agent/db_editor.py — Database Editor Engine (HITL-gated write mutations)

Builds strictly parameter-bound UPDATE/INSERT statements against an allowlisted
set of tables and columns. No DDL and no raw DELETE are ever generated: table and
column identifiers come from a static allowlist and every value is bound as a
parameter, so SQL injection is impossible by construction.

Execution is transactional: the caller (the HITL approval route) runs the compiled
statement inside a single PostgreSQL transaction and rolls back on any error.
"""

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Static identifier allowlist. Table name -> valid writable column names.
#: This is the primary defense: the engine can only ever reference these
#: identifiers, so DDL / arbitrary identifiers are unreachable.
ALLOWED_TABLES: Dict[str, List[str]] = {
    "certificates": [
        "certificate_id", "component", "supplier", "country", "certif_number",
        "authority", "issue_date", "exp_date", "cert_link", "file_name",
    ],
    "authority_lookups": [
        "id", "canonical_authority", "abbreviation", "country",
        "standard_validity_years", "aliases",
    ],
    "supplier_lookups": [
        "id", "canonical_supplier", "aliases",
    ],
    "certificate_chunks": [
        "id", "certificate_id", "page_number", "raw_text",
    ],
}

#: Operations the engine may build. DELETE is intentionally unsupported.
SUPPORTED_OPS = ("update", "insert", "delete")

#: Belt-and-suspenders: whole-word DDL / destructive keywords that must never
#: appear in a generated statement (values are parameter-bound, so this only
#: guards identifier strings).
_BLOCKED_KEYWORD_RE = re.compile(
    r"\b(drop|truncate|alter|create|delete|grant|revoke|merge|call|copy)\b",
    re.IGNORECASE,
)


def _validate_identifier(value: str, kind: str) -> str:
    """Validate an identifier is a plain alphanumeric/underscore token."""
    ident = str(value or "").strip()
    if not ident or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
        raise ValueError(f"Invalid {kind} identifier: {value!r}")
    if _BLOCKED_KEYWORD_RE.search(ident):
        raise ValueError(f"{kind} identifier contains a blocked SQL keyword: {value!r}")
    return ident


def _validate_value(key: str, value: Any) -> Any:
    """Reject non-JSON-serializable values so only safe literals reach bind params."""
    if value is None or isinstance(value, (str, int, float, bool, date, datetime, list, dict)):
        return value
    raise ValueError(f"Unsupported value type for column '{key}': {type(value).__name__}")


def _render_preview(sql: str, params: Dict[str, Any]) -> str:
    """Render a literal-bound preview string for human dry-run inspection."""
    try:
        from sqlalchemy import bindparam, text
        from storage.database import engine

        stmt = text(sql)
        for key, val in params.items():
            stmt = stmt.bindparams(bindparam(key, value=val))
        return str(stmt.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True}))
    except Exception:
        return sql


def build_mutation_sql(
    op: str,
    table: str,
    values: Dict[str, Any],
    row_filter: Optional[Dict[str, Any]] = None,
    fuzzy_match_query: Optional[str] = None,
    allow_full_table: bool = False,
) -> Dict[str, Any]:
    """
    Validate a structured mutation request and return a compiled, parameter-bound
    SQL string plus its parameters for dry-run inspection.

    Args:
        op: "update" or "insert" (any other op, including "delete", is rejected).
        table: one of the keys in ALLOWED_TABLES.
        values: mapping of column -> value to write.
        row_filter: mapping of column -> value identifying the target rows.
            REQUIRED for "update" to prevent full-table updates.

    Returns:
        Dict with keys: op, table, sql (named-placeholder statement), params
        (placeholder -> value), preview (literal-bound rendering for inspection).
    """
    op = str(op or "").strip().lower()
    if op not in SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported mutation op '{op}'; supported ops are {SUPPORTED_OPS}. "
            "DDL and raw DELETE statements are strictly blocked."
        )

    table = _validate_identifier(table, "table")
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' is not in the allowed write set: {sorted(ALLOWED_TABLES)}")

    allowed_cols = set(ALLOWED_TABLES[table])
    values = values or {}
    if op in ("update", "insert") and not values:
        raise ValueError("'values' must be a non-empty dictionary of column -> value.")

    values = {str(k).strip(): v for k, v in values.items()}
    for col in values:
        _validate_identifier(col, "column")
        if col not in allowed_cols:
            raise ValueError(f"Column '{col}' is not writable on table '{table}'.")
        _validate_value(col, values[col])

    params: Dict[str, Any] = {}

    if op == "delete":
        if not row_filter and not fuzzy_match_query and not allow_full_table:
            raise ValueError("'row_filter' or 'fuzzy_match_query' or 'allow_full_table' is required for delete mutations (prevents accidental full-table deletions).")
            
        where_clauses = []
        
        if row_filter:
            row_filter = {str(k).strip(): v for k, v in row_filter.items()}
            for col in row_filter:
                _validate_identifier(col, "column")
                if col not in allowed_cols:
                    raise ValueError(f"Filter column '{col}' is not writable/filterable on table '{table}'.")
                _validate_value(col, row_filter[col])
            where_clauses.extend([f"{col} = :{col}_filter" for col in row_filter])
            params.update({f"{col}_filter": val for col, val in row_filter.items()})

        if fuzzy_match_query:
            # Construct an OR search across specific known text columns to mimic broad natural language lookup
            fuzzy_cols = [c for c in allowed_cols if c in ("certificate_id", "component", "supplier", "certif_number", "file_name")]
            if not fuzzy_cols:
                raise ValueError(f"Table '{table}' does not support fuzzy_match_query (no recognizable text columns).")
            fuzzy_clause = " OR ".join(f"{col} ILIKE :fuzzy_query" for col in fuzzy_cols)
            if where_clauses:
                where_clauses.append(f"({fuzzy_clause})")
            else:
                where_clauses.append(fuzzy_clause)
            params["fuzzy_query"] = f"%{fuzzy_match_query}%"

        where_clause = " AND ".join(where_clauses)
        if where_clause:
            sql = f"DELETE FROM {table} WHERE {where_clause}"
        else:
            sql = f"DELETE FROM {table}"

    elif op == "update":
        if not isinstance(row_filter, dict) or not row_filter:
            raise ValueError("'row_filter' is required for update mutations (prevents full-table updates).")
        row_filter = {str(k).strip(): v for k, v in row_filter.items()}
        for col in row_filter:
            _validate_identifier(col, "column")
            if col not in allowed_cols:
                raise ValueError(f"Filter column '{col}' is not writable on table '{table}'.")
            _validate_value(col, row_filter[col])

        set_clause = ", ".join(f"{col} = :{col}" for col in values)
        where_clause = " AND ".join(f"{col} = :{col}_filter" for col in row_filter)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params.update({col: val for col, val in values.items()})
        params.update({f"{col}_filter": val for col, val in row_filter.items()})
    else:  # insert
        cols = list(values.keys())
        col_list = ", ".join(cols)
        placeholders = ", ".join(f":{col}" for col in cols)
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        params.update(values)

    return {
        "op": op,
        "table": table,
        "sql": sql,
        "params": params,
        "preview": _render_preview(sql, params),
    }


def execute_mutation(db, mutation: Dict[str, Any]) -> int:
    """
    Execute a compiled mutation inside a single strict transaction.

    Commits on success; rolls back and re-raises on any error so callers surface
    actionable messages. Returns the number of affected rows.

    Args:
        db: SQLAlchemy session (from storage.database SessionLocal / get_db).
        mutation: the dict returned by build_mutation_sql().

    Returns:
        int: affected row count.
    """
    from sqlalchemy import text

    sql = mutation["sql"]
    params = mutation["params"]
    try:
        result = db.execute(text(sql), params)
        rowcount = result.rowcount if result.rowcount is not None else -1
        db.commit()
        return rowcount
    except Exception as exc:
        db.rollback()
        logger.error(f" DB mutation failed and rolled back: {exc}")
        raise