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

# Dynamic schema lookups replace static ALLOWED_TABLES

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


def _validate_typed_literal(col: str, value: Any, col_type: str) -> Any:
    """
    Validate that `value` is a literal compatible with the target column's PostgreSQL type.
    Never mutates strings in non-date columns (e.g. component='2008' stays '2008').
    Raises ValueError if `value` is a structured expression or type-incompatible string
    (e.g. 'has 2999', 'before 2025', or bare '9999' on a DATE column) so the caller can
    delegate complex expressions to SQL generation.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        raise ValueError(f"Non-literal expression dict {value!r} for column '{col}' requires SQL generation.")

    ctype = (col_type or "").lower()
    if "date" in ctype or "timestamp" in ctype:
        if isinstance(value, (date, datetime)):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                return s
        raise ValueError(
            f"Column '{col}' has PostgreSQL type '{col_type}' and requires a literal YYYY-MM-DD date (got {value!r})."
        )

    if "int" in ctype or "numeric" in ctype or "double" in ctype or "real" in ctype:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
                return int(s) if "int" in ctype else float(s)
        raise ValueError(
            f"Column '{col}' has PostgreSQL type '{col_type}' and requires a numeric literal (got {value!r})."
        )

    if isinstance(value, str):
        s = value.strip()
        if "%" in s or re.match(r"^\s*(?:I?LIKE|contains?|has|includes?|matching|<|>|!=|>=|<=)\b", s, re.IGNORECASE):
            raise ValueError(f"Column '{col}' filter/value {value!r} contains an operator expression requiring SQL generation.")

    return _validate_value(col, value)


def build_mutation_sql(
    op: str,
    table: str,
    values: Dict[str, Any],
    row_filter: Optional[Dict[str, Any]] = None,
    fuzzy_match_query: Optional[str] = None,
    allow_full_table: bool = False,
) -> Dict[str, Any]:
    """
    Validate a literal key-value mutation request against live PostgreSQL table/column metadata
    and return a compiled, parameter-bound SQL string plus its parameters.
    """
    op = str(op or "").strip().lower()
    if op not in SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported mutation op '{op}'; supported ops are {SUPPORTED_OPS}. "
            "DDL and raw DELETE statements are strictly blocked."
        )

    from storage import dynamic_schema
    table = dynamic_schema.resolve_physical_table_name(str(table or ""))
    table = _validate_identifier(table, "table")

    all_tables = [t["table_name"] for t in dynamic_schema.list_all_user_tables()]
    valid_tables = set(all_tables) | dynamic_schema.CORE_PLATFORM_TABLES

    if table not in valid_tables:
        raise ValueError(f"Table '{table}' is not in the allowed write set: {sorted(valid_tables)}")

    col_meta = dynamic_schema.get_table_columns(table)
    allowed_cols = {c["column_name"] for c in col_meta}
    col_lookup = {c["column_name"].lower(): c["column_name"] for c in col_meta}
    col_types = {c["column_name"]: c["data_type"] for c in col_meta}

    values = dict(values or {})
    if op == "insert":
        # Strip empty auto-managed primary key / timestamp fields so DB defaults apply
        for auto_col in ("id", "created_at"):
            for k in list(values.keys()):
                if k.strip().lower() == auto_col and (values[k] is None or str(values[k]).strip() == ""):
                    values.pop(k, None)
        if table == "certificates":
            has_cert_id = any(
                k.strip().lower() == "certificate_id" and str(v or "").strip()
                for k, v in values.items()
            )
            if not has_cert_id:
                import uuid
                values["certificate_id"] = uuid.uuid4().hex

    if op in ("update", "insert") and not values:
        raise ValueError("'values' must be a non-empty dictionary of column -> value.")

    cleaned_values: Dict[str, Any] = {}
    for raw_col, val in values.items():
        col_raw = str(raw_col).strip()
        _validate_identifier(col_raw, "column")
        col = col_lookup.get(col_raw.lower(), col_raw)
        if col not in allowed_cols:
            raise ValueError(f"Column '{col_raw}' is not writable on table '{table}'.")
        cleaned_values[col] = _validate_typed_literal(col, val, col_types.get(col, "text"))
    values = cleaned_values

    params: Dict[str, Any] = {}

    if op == "delete":
        if not row_filter and not fuzzy_match_query and not allow_full_table:
            raise ValueError("'row_filter' or 'fuzzy_match_query' or 'allow_full_table' is required for delete mutations.")

        where_clauses = []
        if row_filter:
            for raw_col, val in row_filter.items():
                col_raw = str(raw_col).strip()
                _validate_identifier(col_raw, "column")
                col = col_lookup.get(col_raw.lower(), col_raw)
                if col not in allowed_cols:
                    raise ValueError(f"Filter column '{col_raw}' is not filterable on table '{table}'.")
                typed_val = _validate_typed_literal(col, val, col_types.get(col, "text"))
                where_clauses.append(f"{col} = :{col}_filter")
                params[f"{col}_filter"] = typed_val

        if fuzzy_match_query:
            fuzzy_cols = [c for c in allowed_cols if c in ("certificate_id", "component", "supplier", "certif_number", "file_name")]
            if not fuzzy_cols:
                raise ValueError(f"Table '{table}' does not support fuzzy_match_query.")
            fuzzy_clause = " OR ".join(f"{col} ILIKE :fuzzy_query" for col in fuzzy_cols)
            where_clauses.append(f"({fuzzy_clause})" if where_clauses else fuzzy_clause)
            params["fuzzy_query"] = f"%{fuzzy_match_query}%"

        where_clause = " AND ".join(where_clauses)
        sql = f"DELETE FROM {table} WHERE {where_clause}" if where_clause else f"DELETE FROM {table}"

    elif op == "update":
        if not row_filter and not fuzzy_match_query and not allow_full_table:
            raise ValueError("'row_filter' or 'fuzzy_match_query' or 'allow_full_table' is required for update mutations.")

        where_clauses = []
        if row_filter:
            for raw_col, val in row_filter.items():
                col_raw = str(raw_col).strip()
                _validate_identifier(col_raw, "column")
                col = col_lookup.get(col_raw.lower(), col_raw)
                if col not in allowed_cols:
                    raise ValueError(f"Filter column '{col_raw}' is not filterable on table '{table}'.")
                typed_val = _validate_typed_literal(col, val, col_types.get(col, "text"))
                where_clauses.append(f"{col} = :{col}_filter")
                params[f"{col}_filter"] = typed_val

        if fuzzy_match_query:
            fuzzy_cols = [c for c in allowed_cols if c in ("certificate_id", "component", "supplier", "certif_number", "file_name")]
            if not fuzzy_cols:
                raise ValueError(f"Table '{table}' does not support fuzzy_match_query.")
            fuzzy_clause = " OR ".join(f"{col} ILIKE :fuzzy_query" for col in fuzzy_cols)
            where_clauses.append(f"({fuzzy_clause})" if where_clauses else fuzzy_clause)
            params["fuzzy_query"] = f"%{fuzzy_match_query}%"

        set_clause = ", ".join(f"{col} = :{col}" for col in values)
        where_clause = " AND ".join(where_clauses)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}" if where_clause else f"UPDATE {table} SET {set_clause}"
        params.update(values)
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