"""
core/rag/sql_engine.py — Text-to-SQL Engine for the METADATA_QUERY Path.

Executes the structured-query side of the Dual-Path architecture:

  1. Injects the live `certificates` table schema into a highly constrained
     system prompt (schema-injection prompting).
  2. Uses the local LLM facade (core.llm.generate_json) to translate a natural
     language question into a single PostgreSQL SELECT statement.
  3. Safely executes the statement against the provided SQLAlchemy session
     (SELECT-only validation + read-only transaction enforcement).
  4. Formats the raw results in Python and synthesizes a natural-language answer
     via the LLM, falling back to a deterministic Python-formatted summary.

Runs 100% locally via the BaseLLMEngine registry pattern. Read access only.
"""

import json
import logging
import re
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from core.prompts import (
    SQL_SYSTEM_PROMPT_TEMPLATE,
    SQL_RESULT_SYNTHESIS_PROMPT,
    config_sql_system_prompt,
    config_sql_result_synthesis_prompt,
)
from schemas.extraction import CertificateMetadata

logger = logging.getLogger(__name__)

#: Graceful message returned whenever the database cannot process the request.
FALLBACK_MESSAGE = (
    "The database could not process this specific filtering request. "
    "Please rephrase the question or check the certificate metadata attributes."
)

#: Hard caps applied to generated queries and forwarded results.
MAX_ROWS_IN_SQL = 100
MAX_ROWS_FOR_LLM = 8

_SCHEMA_DESC_CACHE: List[str] = []


def invalidate_schema_cache() -> None:
    """Clears the cached PostgreSQL schema description when tables or columns change."""
    _SCHEMA_DESC_CACHE.clear()

#: Data-definition / data-mutation tokens forbidden in generated SQL.
_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"MERGE|CALL|COPY|REPLACE|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)
#: Session-altering statements that must never appear.
_SETUP_STATEMENT_RE = re.compile(r"\bSET\s+(ROLE|SESSION|TRANSACTION)\b", re.IGNORECASE)
#: A query must begin with SELECT or WITH (CTE) only.
_SELECT_PREFIX_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
#: Detects multi-statement injection (a semicolon followed by more tokens).
_MULTI_STATEMENT_RE = re.compile(r";\s*\S")


def build_schema_description() -> str:
    """
    Derives a concise multi-table PostgreSQL schema description from the live database,
    cached in memory and invalidated on DDL changes.
    """
    if _SCHEMA_DESC_CACHE:
        return _SCHEMA_DESC_CACHE[0]

    table_docs = {
        "certificates": "Regulatory RF homologation certificates and approval records.",
        "components": "Vehicle components master list with frequencies, model names, supplier, DocInfo links, and assigned responsible engineer email/ID.",
        "responsibles": "Directory of engineering and homologation responsible personnel (contact details, email, phone, mission comments).",
        "countries": "Country regulatory priorities (stla_current_priority), ISO country codes, and market status comments.",
        "authority_lookups": "Regulatory authorities (e.g. ENACOM, ANATEL, FCC, IFT), country, validity period in years, and aliases.",
        "supplier_lookups": "Canonical automotive suppliers and manufacturers (e.g. BOSCH, CONTINENTAL, MARELLI, VALEO).",
        "sources": "Monitored compliance portals, document URLs, and cookie credentials.",
        "agent_memories": "Cross-session user directives, rules, and preferences learned by the assistant."
    }

    lines = []
    try:
        from storage.dynamic_schema import list_all_user_tables, get_table_columns
        user_tables = list_all_user_tables()
        for t in user_tables:
            tbl_name = t["table_name"]
            doc = table_docs.get(tbl_name, "User data table.")
            cols = get_table_columns(tbl_name)
            col_parts = []
            for c in cols:
                col_name = c["column_name"]
                data_type = c["data_type"]
                pk_tag = " PK" if col_name in ("id", "certificate_id") and c["is_nullable"] != "YES" else ""
                col_parts.append(f"`{col_name}` ({data_type}{pk_tag})")
            lines.append(f"- `{tbl_name}` ({doc}): {', '.join(col_parts)}")
    except Exception as exc:
        logger.warning(f"Dynamic schema reflection fallback: {exc}")
        lines = [
            "- `certificates` (Regulatory RF certificates): `certificate_id` (varchar PK), `component` (varchar), `supplier` (varchar), `country` (varchar), `certif_number` (varchar), `authority` (varchar), `issue_date` (date), `exp_date` (date), `cert_link` (varchar), `file_name` (varchar)",
            "- `components` (Vehicle components): `id` (integer PK), `componentname` (varchar), `modelname` (varchar), `supplier` (varchar), `freq` (varchar), `id_responsible` (varchar), `docinfo_link` (varchar)",
            "- `responsibles` (Responsible engineers): `id` (integer PK), `firstname` (varchar), `lastname` (varchar), `phone` (varchar), `email` (varchar), `mission_comment` (varchar)",
            "- `authority_lookups` (Regulatory authorities): `id` (integer PK), `canonical_authority` (varchar), `abbreviation` (varchar), `country` (varchar), `standard_validity_years` (varchar)",
            "- `supplier_lookups` (Supplier aliases): `id` (integer PK), `canonical_supplier` (varchar), `aliases` (jsonb)",
            "- `countries` (Market priorities): `id` (integer PK), `country` (varchar), `country_code` (varchar), `stla_current_priority` (varchar)",
        ]

    result = "\n".join(lines).strip()
    if lines:
        _SCHEMA_DESC_CACHE[:] = [result]
    return result


def build_sql_system_prompt() -> str:
    """
    Builds the constrained Text-to-SQL system prompt with the live schema and long-term memories injected.
    """
    return config_sql_system_prompt(schema=build_schema_description())


def _strip_sql_comments(sql: str) -> str:
    """Removes `-- line` and `/* block */` comments so keywords cannot hide inside them."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\r\n]*", " ", sql)
    return sql


def _validate_sql(sql: str) -> bool:
    """
    Validates that a generated SQL string is a single, read-only SELECT statement.

    Checks (in order):
      1. Non-empty input.
      2. No comment-masked content.
      3. No multi-statement injection (`;` followed by more tokens).
      4. No forbidden data-definition / data-mutation keywords.
      5. No session-altering SET statements.
      6. Statement begins with SELECT or WITH.

    Args:
        sql (str): The raw SQL string generated by the LLM.

    Returns:
        bool: True if the statement is safe to execute, False otherwise.
    """
    if not sql or not str(sql).strip():
        return False

    stripped = str(sql).strip().rstrip(";").strip()
    sanitized = _strip_sql_comments(stripped)

    if _MULTI_STATEMENT_RE.search(sanitized):
        logger.warning("  Rejected SQL: multi-statement payload detected.")
        return False
    if _FORBIDDEN_SQL_KEYWORDS.search(sanitized):
        logger.warning("  Rejected SQL: forbidden DDL/DML keyword detected.")
        return False
    if _SETUP_STATEMENT_RE.search(sanitized):
        logger.warning("  Rejected SQL: session-altering SET statement detected.")
        return False
    if not _SELECT_PREFIX_RE.match(sanitized):
        logger.warning("  Rejected SQL: statement does not begin with SELECT/WITH.")
        return False

    return True


def _ensure_row_cap(sql: str) -> str:
    """
    Appends a hard LIMIT cap to queries that do not already specify one, preventing
    unbounded result sets from the generated SQL.
    """
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.strip().rstrip(';').strip()} LIMIT {MAX_ROWS_IN_SQL}"


def _parse_json_object(raw_json: str) -> Dict[str, Any]:
    """
    Parses a raw model response into a dictionary, tolerating markdown code
    fences and surrounding prose (regex fallback on the first JSON object).
    """
    if not raw_json:
        return {}

    cleaned_str = raw_json.strip()
    if cleaned_str.startswith("```"):
        lines = cleaned_str.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_str = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned_str, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def generate_sql_query(user_query: str, history_text: str = "", error_feedback: str = "") -> str:
    """
    Translates a natural language question into a PostgreSQL SELECT statement
    using the local LLM facade with schema-injection prompting.

    Args:
        user_query (str): The user's natural language question.
        history_text (str): Optional prior conversation history block.
        error_feedback (str): Optional previous SQL execution error for self-correction.

    Returns:
        str: The generated SQL statement, or an empty string if generation failed.
    """
    from core.llm import generate_json

    history_block = f"{history_text}\n\n" if history_text else ""
    error_block = f"PREVIOUS EXECUTION ERROR TO FIX:\n{error_feedback}\n\n" if error_feedback else ""
    raw_json_response = generate_json(
        system_prompt=build_sql_system_prompt(),
        user_prompt=(
            f"{history_block}{error_block}USER QUESTION: {user_query}\n\n"
            f"Return ONLY the raw JSON output matching the schema."
        ),
        disable_thinking=True,
        max_tokens=384,
        response_prefix='{"sql": "',
        hop_name="sql_gen",
    )

    parsed = _parse_json_object(raw_json_response)
    sql = str(parsed.get("sql", "")).strip()

    # Strip accidental SQL code fences embedded inside the JSON "sql" field.
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"\s*```$", "", sql).strip()

    return sql


def _enforce_read_only(db_session) -> None:
    """
    Defense-in-depth: marks the current transaction as READ ONLY on PostgreSQL so
    any mutating statement fails at the database level even if validation is bypassed.

    Non-fatal: if the transaction already has activity, the read-only hint is skipped.
    """
    try:
        connection = db_session.connection()
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
    except Exception as e:
        logger.warning(f"  Could not enforce read-only transaction: {e}")


def _cell_to_text(value: Any) -> str:
    """Converts a single DB cell value into a display-friendly string."""
    if value is None or str(value).strip() == "":
        return "Unassigned"
    if hasattr(value, "isoformat"):
        val_str = value.isoformat()
        return "Indefinite" if val_str.startswith("2999-") else val_str
    if isinstance(value, (list, tuple)):
        return repr(value)
    val_str = str(value)
    return "Indefinite" if val_str == "2999-01-01" else val_str


def _build_markdown_table(rows: List[Dict[str, Any]], columns: List[str], max_display: int = 15) -> str:
    """
    Deterministically renders PostgreSQL result rows into a clean GFM Markdown table.
    """
    if not rows or not columns:
        return ""
    lines = []
    header = "| " + " | ".join(col.replace("_", " ").title() for col in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(separator)
    for row in rows[:max_display]:
        row_str = "| " + " | ".join(_cell_to_text(row.get(col)).replace("|", "\\|").replace("\n", " ") for col in columns) + " |"
        lines.append(row_str)
    if len(rows) > max_display:
        lines.append(f"\n*(Showing top {max_display} of {len(rows)} total matching records)*")
    return "\n".join(lines)


def _format_results(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """
    Formats raw query results into a compact text summary for the LLM executive summary hop.
    """
    display_rows = rows[:MAX_ROWS_FOR_LLM]

    lines = [f"Total rows returned: {len(rows)}"]
    if display_rows:
        lines.append("Columns: " + ", ".join(columns))
        for idx, row in enumerate(display_rows, start=1):
            pairs = ", ".join(f"{key}={_cell_to_text(value)}" for key, value in row.items())
            lines.append(f"Row {idx}: {pairs}")
        if len(rows) > MAX_ROWS_FOR_LLM:
            lines.append(f"(Showing top {MAX_ROWS_FOR_LLM} of {len(rows)} rows for summary context; full table is rendered automatically.)")

    return "\n".join(lines)


def _synthesize_answer(
    user_query: str,
    sql: str,
    rows: List[Dict[str, Any]],
    columns: List[str],
    history_text: str = "",
    disable_thinking: bool = True,
) -> str:
    """
    Synthesizes a concise 1-2 sentence executive summary via the LLM and appends
    the deterministic Markdown table for multi-row / multi-column results.
    """
    if not rows:
        return "No matching records were found in the database for your query."

    formatted_results = _format_results(rows, columns)
    history_block = f"{history_text}\n\n" if history_text else ""
    user_prompt = (
        f"{history_block}"
        f"USER QUESTION: {user_query}\n\n"
        f"EXECUTED SQL:\n{sql}\n\n"
        f"QUERY RESULTS:\n{formatted_results}\n\n"
        "Return ONLY the raw JSON output matching the schema."
    )

    needs_table = len(rows) >= 2 or (len(rows) == 1 and len(columns) >= 2)
    table_md = _build_markdown_table(rows, columns) if needs_table else ""

    try:
        from core.llm import generate_json

        raw_json_response = generate_json(
            system_prompt=config_sql_result_synthesis_prompt(),
            user_prompt=user_prompt,
            disable_thinking=disable_thinking,
            max_tokens=180,
            response_prefix='{"answer": "',
            hop_name="sql_summary",
        )

        parsed = _parse_json_object(raw_json_response)
        answer = str(parsed.get("answer", "")).strip()
        if answer:
            if table_md and "| ---" not in answer:
                return f"{answer}\n\n{table_md}"
            return answer
        logger.warning("  Empty answer returned by SQL result synthesis LLM.")
    except Exception as e:
        logger.warning(f"  SQL result synthesis failed: {e}")

    return _format_python_answer(user_query, rows, columns)


def _format_python_answer(user_query: str, rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """
    Deterministic fallback answer formatter used when LLM synthesis is unavailable.
    Formats results as an executive summary followed by a clean Markdown table.
    """
    if not rows:
        return "No matching records were found in the database for your query."

    if len(columns) == 1 and len(rows) == 1:
        return f"The result for your query is **{_cell_to_text(rows[0].get(columns[0]))}**."

    table_md = _build_markdown_table(rows, columns)
    return f"Found **{len(rows)}** record(s) matching your request:\n\n{table_md}"


def _stream_synthesize_answer(
    user_query: str,
    sql: str,
    rows: List[Dict[str, Any]],
    columns: List[str],
    history_text: str = "",
    disable_thinking: bool = True,
):
    """
    Generator: streams the concise 1-2 sentence executive summary token-by-token,
    and immediately emits the deterministic Markdown table for tabular results.
    """
    if not rows:
        yield "No matching records were found in the database for your query."
        return

    formatted_results = _format_results(rows, columns)
    history_block = f"{history_text}\n\n" if history_text else ""
    user_prompt = (
        f"{history_block}"
        f"USER QUESTION: {user_query}\n\n"
        f"EXECUTED SQL:\n{sql}\n\n"
        f"QUERY RESULTS:\n{formatted_results}\n\n"
        "Return ONLY the raw JSON output matching the schema."
    )

    needs_table = len(rows) >= 2 or (len(rows) == 1 and len(columns) >= 2)
    table_md = _build_markdown_table(rows, columns) if needs_table else ""

    buffer = ""
    last_answer = ""
    try:
        from core.llm import generate_stream
        from core.rag.qa import extract_answer_value

        for token in generate_stream(
            system_prompt=config_sql_result_synthesis_prompt(),
            user_prompt=user_prompt,
            disable_thinking=disable_thinking,
            max_tokens=180,
            response_prefix='{"answer": "',
            hop_name="sql_summary",
        ):
            buffer += token
            current = extract_answer_value(buffer)
            if current and len(current) > len(last_answer):
                delta = current[len(last_answer):]
                last_answer = current
                if delta:
                    yield delta
        if last_answer:
            if table_md and "| ---" not in last_answer:
                yield f"\n\n{table_md}"
            return
        logger.warning("  Empty answer returned by streamed SQL result synthesis.")
    except Exception as e:
        logger.warning(f"  SQL result synthesis streaming failed: {e}")

    yield _format_python_answer(user_query, rows, columns)


def execute_sql_rows(
    user_query: str,
    db_session=None,
    history_text: str = "",
):
    """
    Translates `user_query` to SQL and executes it safely (up to 3 retries).
    Returns `(sql, rows, columns, error_feedback)`.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        return "", [], [], "Empty query"

    MAX_SQL_RETRIES = 3
    error_feedback = ""
    sql = ""
    rows: List[Dict[str, Any]] = []
    columns: List[str] = []

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal
        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        for attempt in range(MAX_SQL_RETRIES):
            try:
                sql = generate_sql_query(clean_query, history_text=history_text, error_feedback=error_feedback)
            except Exception as e:
                logger.warning(f"  Text-to-SQL generation failed: {e}")
                error_feedback = f"Text-to-SQL generation failed: {e}"
                continue

            if not sql or not _validate_sql(sql):
                logger.warning(f"  Generated SQL failed validation. SQL: {sql!r}")
                error_feedback = "Generated SQL failed validation (e.g. invalid syntax, mutating keywords, or multiple statements)."
                continue

            sql = _ensure_row_cap(sql)
            logger.info(f" Executing generated SQL for METADATA_QUERY (attempt {attempt + 1}/{MAX_SQL_RETRIES}): {sql}")

            try:
                _enforce_read_only(db_session)
                result = db_session.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(row) for row in result.mappings().all()]
                error_feedback = ""
                break
            except SQLAlchemyError as e:
                logger.warning(f"  SQLAlchemy error during metadata query execution: {e}")
                error_feedback = f"PostgreSQL execution failed with error: {e}"
                try:
                    db_session.rollback()
                except Exception as rollback_err:
                    logger.warning(f"  Rollback failed: {rollback_err}")
                continue
            except Exception as e:
                logger.warning(f"  Unexpected error during metadata query execution: {e}")
                error_feedback = f"Unexpected error: {e}"
                try:
                    db_session.rollback()
                except Exception as rollback_err:
                    logger.warning(f"  Rollback failed: {rollback_err}")
                continue
    finally:
        if close_session_on_exit:
            db_session.close()

    return sql, rows, columns, error_feedback


def execute_sql_structured_context(user_query: str, db_session=None, history_text: str = "") -> str:
    """
    Executes Text-to-SQL and returns compact structured facts WITHOUT an intermediate
    LLM synthesis hop (used by HYBRID_QUERY so synthesis only happens once).
    """
    sql, rows, columns, error_feedback = execute_sql_rows(
        user_query, db_session=db_session, history_text=history_text
    )
    if error_feedback or not sql:
        return ""
    if not rows:
        return "No matching database rows found."
    table_md = _build_markdown_table(rows, columns, max_display=15)
    return f"Executed SQL: {sql}\nTotal matching rows: {len(rows)}\n\n{table_md}"


def stream_metadata_answer(
    user_query: str,
    db_session=None,
    history_text: str = "",
    disable_thinking: bool = True,
):
    """
    Generator: runs the full METADATA_QUERY pipeline and yields the final
    natural-language answer as progressive token chunks.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        logger.warning("Empty or blank user query provided to stream_metadata_answer.")
        yield FALLBACK_MESSAGE
        return

    sql, rows, columns, error_feedback = execute_sql_rows(
        clean_query, db_session=db_session, history_text=history_text
    )
    if error_feedback:
        logger.warning(f"  Exhausted SQL retries. Last error: {error_feedback}")
        yield FALLBACK_MESSAGE
        return

    yield from _stream_synthesize_answer(
        clean_query, sql, rows, columns, history_text=history_text, disable_thinking=disable_thinking
    )


def execute_metadata_query(
    user_query: str,
    db_session,
    history_text: str = "",
    disable_thinking: bool = True,
) -> str:
    """
    Full METADATA_QUERY execution pipeline.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        logger.warning("Empty or blank user query provided to execute_metadata_query.")
        return FALLBACK_MESSAGE

    sql, rows, columns, error_feedback = execute_sql_rows(
        clean_query, db_session=db_session, history_text=history_text
    )
    if error_feedback:
        logger.warning(f"  Exhausted SQL retries. Last error: {error_feedback}")
        return FALLBACK_MESSAGE

    return _synthesize_answer(
        clean_query, sql, rows, columns, history_text=history_text, disable_thinking=disable_thinking
    )


if __name__ == "__main__":
    # Offline validation suite (no live database required).
    print("Testing Text-to-SQL Engine Validation & Schema Injection...\n")

    print("Derived PostgreSQL Schema (from ORM):")
    print("-" * 60)
    print(build_schema_description())
    print("-" * 60)
    print()

    test_cases = [
        ("SELECT COUNT(*) FROM certificates WHERE EXTRACT(YEAR FROM exp_date) = 2026;", True),
        ("SELECT * FROM certificates WHERE country = 'Germany';", True),
        ("WITH recent AS (SELECT * FROM certificates WHERE issue_date >= '2024-01-01') SELECT * FROM recent;", True),
        ("DROP TABLE certificates;", False),
        ("DELETE FROM certificates;", False),
        ("UPDATE certificates SET country = 'France';", False),
        ("SELECT * FROM certificates; DROP TABLE certificates;", False),
        ("INSERT INTO certificates (certificate_id) VALUES ('x');", False),
        ("SELECT * FROM certificates -- DROP TABLE certificates", True),
        ("ALTER TABLE certificates ADD COLUMN evil TEXT;", False),
        ("SELECT * FROM pg_user;", True),
        ("", False),
        (None, False),
    ]

    all_passed = True
    for raw_sql, expected in test_cases:
        result = _validate_sql(raw_sql)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  [{status}] validate({raw_sql!r}) -> {result} (expected {expected})")

    print()
    print("Row cap enforcement:")
    print(f"  {_ensure_row_cap('SELECT * FROM certificates')}")
    print(f"  {_ensure_row_cap('SELECT * FROM certificates LIMIT 5;')}")
    print()

    if all_passed:
        print(" All SQL validation tests passed.")
    else:
        print(" Some SQL validation tests failed.")
