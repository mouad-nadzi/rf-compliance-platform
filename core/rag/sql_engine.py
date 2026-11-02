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
MAX_ROWS_FOR_LLM = 50

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
    Derives a complete, multi-table PostgreSQL schema description directly from
    the live database and schema definitions.
    
    Includes all core tables (certificates, components, responsibles, countries,
    authority_lookups, supplier_lookups, sources, agent_memories) and user custom tables,
    annotating column types, nullability, and primary keys.
    """
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
            lines.append(f"Table: `{tbl_name}` — {doc}")
            lines.append("Columns:")
            cols = get_table_columns(tbl_name)
            for c in cols:
                col_name = c["column_name"]
                data_type = c["data_type"]
                nullability = "NULLABLE" if c["is_nullable"] == "YES" else "NOT NULL"
                if col_name in ("id", "certificate_id") and not (c["is_nullable"] == "YES"):
                    nullability = "PRIMARY KEY"
                lines.append(f"  - `{col_name}` {data_type} {nullability}")
            lines.append("")
    except Exception as exc:
        logger.warning(f"Dynamic schema reflection fallback: {exc}")
        lines = [
            "Table: `certificates` — Regulatory RF homologation certificates and approval records.",
            "Columns:",
            "  - `certificate_id` character varying PRIMARY KEY",
            "  - `component` character varying NULLABLE -- Normalized component/model code",
            "  - `supplier` character varying NULLABLE -- Normalized supplier name",
            "  - `country` character varying NULLABLE -- Jurisdiction country",
            "  - `certif_number` character varying NULLABLE -- Certificate number",
            "  - `authority` character varying NULLABLE -- Regulatory body acronym (e.g. ENACOM, ANATEL)",
            "  - `issue_date` date NULLABLE -- Issuance date (YYYY-MM-DD)",
            "  - `exp_date` date NULLABLE -- Expiration date (YYYY-MM-DD)",
            "  - `cert_link` character varying NULLABLE -- Document link",
            "  - `file_name` character varying NULLABLE",
            "",
            "Table: `components` — Vehicle components master list with frequencies and assigned responsible engineer.",
            "Columns:",
            "  - `id` integer PRIMARY KEY",
            "  - `componentname` character varying NULLABLE -- Component code/name (e.g. RTBM SH, VSM, AIDA)",
            "  - `modelname` character varying NULLABLE -- Model identifier",
            "  - `supplier` character varying NULLABLE -- Component supplier",
            "  - `freq` character varying NULLABLE -- Operating frequency (e.g. 125 kHz, 2.4 GHz)",
            "  - `id_responsible` character varying NULLABLE -- Responsible engineer email or id (e.g. fatimaezzhara.farkaoui@external.stellantis.com)",
            "  - `docinfo_link` character varying NULLABLE -- Link to DocInfo fiche",
            "",
            "Table: `responsibles` — Responsible engineers directory.",
            "Columns:",
            "  - `id` integer PRIMARY KEY",
            "  - `firstname` character varying NULLABLE",
            "  - `lastname` character varying NULLABLE",
            "  - `phone` character varying NULLABLE",
            "  - `email` character varying NULLABLE -- Contact email address",
            "  - `mission_comment` character varying NULLABLE",
            "",
            "Table: `authority_lookups` — Regulatory authorities.",
            "Columns:",
            "  - `id` integer PRIMARY KEY",
            "  - `canonical_authority` character varying NOT NULL",
            "  - `abbreviation` character varying NULLABLE",
            "  - `country` character varying NOT NULL",
            "  - `standard_validity_years` character varying NULLABLE",
            "",
            "Table: `supplier_lookups` — Supplier aliases.",
            "Columns:",
            "  - `id` integer PRIMARY KEY",
            "  - `canonical_supplier` character varying NOT NULL",
            "  - `aliases` jsonb NOT NULL",
            "",
            "Table: `countries` — Market priorities.",
            "Columns:",
            "  - `id` integer PRIMARY KEY",
            "  - `country` character varying NULLABLE",
            "  - `country_code` character varying NULLABLE",
            "  - `stla_current_priority` character varying NULLABLE",
        ]

    return "\n".join(lines).strip()


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
    if value is None:
        return "NULL"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return repr(value)
    return str(value)


def _format_results(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """
    Formats raw query results into a compact text table for LLM synthesis.
    """
    display_rows = rows[:MAX_ROWS_FOR_LLM]

    lines = [f"Total rows returned: {len(rows)}"]
    if display_rows:
        lines.append("Columns: " + ", ".join(columns))
        for idx, row in enumerate(display_rows, start=1):
            pairs = ", ".join(f"{key}={_cell_to_text(value)}" for key, value in row.items())
            lines.append(f"Row {idx}: {pairs}")
        if len(rows) > MAX_ROWS_FOR_LLM:
            lines.append(f"(Only the first {MAX_ROWS_FOR_LLM} rows are shown.)")

    return "\n".join(lines)


def _synthesize_answer(user_query: str, sql: str, rows: List[Dict[str, Any]], columns: List[str], history_text: str = "") -> str:
    """
    Feeds the raw query results back to the local LLM to synthesize a natural
    language answer. Falls back to a deterministic Python-formatted summary if
    the LLM call or JSON parsing fails.
    """
    formatted_results = _format_results(rows, columns)
    history_block = f"{history_text}\n\n" if history_text else ""
    user_prompt = (
        f"{history_block}"
        f"USER QUESTION: {user_query}\n\n"
        f"EXECUTED SQL:\n{sql}\n\n"
        f"QUERY RESULTS:\n{formatted_results}\n\n"
        "Return ONLY the raw JSON output matching the schema."
    )

    try:
        from core.llm import generate_json

        raw_json_response = generate_json(
            system_prompt=config_sql_result_synthesis_prompt(),
            user_prompt=user_prompt,
            disable_thinking=True,
        )

        parsed = _parse_json_object(raw_json_response)
        answer = str(parsed.get("answer", "")).strip()
        if answer:
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

    lines = [f"Found **{len(rows)}** record(s) matching your request:\n"]
    header = "| " + " | ".join(col.replace("_", " ").title() for col in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(separator)
    for row in rows[:15]:
        row_str = "| " + " | ".join(_cell_to_text(row.get(col)) for col in columns) + " |"
        lines.append(row_str)
    if len(rows) > 15:
        lines.append(f"\n*(Showing top 15 of {len(rows)} total records)*")
    return "\n".join(lines)


def _stream_synthesize_answer(user_query: str, sql: str, rows: List[Dict[str, Any]], columns: List[str], history_text: str = ""):
    """
    Generator: streams the natural-language answer synthesis token-by-token.

    Builds the same prompt as _synthesize_answer but decodes incrementally and
    surfaces only the growing JSON 'answer' field. Falls back to the deterministic
    Python-formatted answer (yielded once) if streaming or parsing fails.
    """
    formatted_results = _format_results(rows, columns)
    history_block = f"{history_text}\n\n" if history_text else ""
    user_prompt = (
        f"{history_block}"
        f"USER QUESTION: {user_query}\n\n"
        f"EXECUTED SQL:\n{sql}\n\n"
        f"QUERY RESULTS:\n{formatted_results}\n\n"
        "Return ONLY the raw JSON output matching the schema."
    )

    buffer = ""
    last_answer = ""
    try:
        from core.llm import generate_stream
        from core.rag.qa import extract_answer_value

        for token in generate_stream(
            system_prompt=config_sql_result_synthesis_prompt(),
            user_prompt=user_prompt,
            disable_thinking=True,
        ):
            buffer += token
            current = extract_answer_value(buffer)
            if current and len(current) > len(last_answer):
                delta = current[len(last_answer):]
                last_answer = current
                if delta:
                    yield delta
        if last_answer:
            return
        logger.warning("  Empty answer returned by streamed SQL result synthesis.")
    except Exception as e:
        logger.warning(f"  SQL result synthesis streaming failed: {e}")

    yield _format_python_answer(user_query, rows, columns)


def stream_metadata_answer(user_query: str, db_session=None, history_text: str = ""):
    """
    Generator: runs the full METADATA_QUERY pipeline and yields the final
    natural-language answer as progressive token chunks.

    SQL translation + execution are synchronous (fast); the final answer
    synthesis is streamed so the UI shows live output generation.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        logger.warning("Empty or blank user query provided to stream_metadata_answer.")
        yield FALLBACK_MESSAGE
        return

    MAX_SQL_RETRIES = 3
    error_feedback = ""
    sql = ""
    rows = []
    columns = []

    # 2. Open a local session if the caller did not supply one
    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal
        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        for attempt in range(MAX_SQL_RETRIES):
            # 1. LLM SQL translation
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

            # 3. Safe execution with graceful fallback
            try:
                _enforce_read_only(db_session)
                result = db_session.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(row) for row in result.mappings().all()]
                error_feedback = ""
                break # Success, exit retry loop
            except SQLAlchemyError as e:
                logger.warning(f"  SQLAlchemy error during metadata query execution: {e}")
                error_feedback = f"PostgreSQL execution failed with error: {e}"
                try:
                    db_session.rollback()
                except Exception as rollback_err:
                    logger.warning(f"  Rollback failed: {rollback_err}")
                continue # Retry
            except Exception as e:
                logger.warning(f"  Unexpected error during metadata query execution: {e}")
                error_feedback = f"Unexpected error: {e}"
                try:
                    db_session.rollback()
                except Exception as rollback_err:
                    logger.warning(f"  Rollback failed: {rollback_err}")
                continue # Retry

        if error_feedback:
            # If we exhausted retries and still have an error, yield fallback
            logger.warning(f"  Exhausted {MAX_SQL_RETRIES} SQL retries. Last error: {error_feedback}")
            yield FALLBACK_MESSAGE
            return

    finally:
        if close_session_on_exit:
            db_session.close()

    # 4. Stream the natural-language answer synthesis
    yield from _stream_synthesize_answer(
        clean_query, sql, rows, columns, history_text=history_text
    )


def execute_metadata_query(user_query: str, db_session, history_text: str = "") -> str:
    """
    Full METADATA_QUERY execution pipeline:

      1. Empty-input guard.
      2. LLM translates the question into SQL (schema-injected system prompt).
      3. SELECT-only validation + row cap + read-only transaction enforcement.
      4. Safe execution against the provided SQLAlchemy session.
      5. Natural-language answer synthesis with deterministic fallback.

    Args:
        user_query (str): The natural language question asked by the user.
        db_session: A SQLAlchemy session (or the storage.database scoped session).
            If None, a fresh session is opened and closed automatically.
        history_text (str): Optional prior conversation history block.

    Returns:
        str: A natural language answer, or a graceful fallback message on failure.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        logger.warning("Empty or blank user query provided to execute_metadata_query.")
        return FALLBACK_MESSAGE

    MAX_SQL_RETRIES = 3
    error_feedback = ""
    sql = ""
    rows = []
    columns = []

    # 2. Open a local session if the caller did not supply one
    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal
        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        for attempt in range(MAX_SQL_RETRIES):
            # 1. LLM SQL translation
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

            # 3. Safe execution with graceful fallback
            try:
                _enforce_read_only(db_session)
                result = db_session.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(row) for row in result.mappings().all()]
                error_feedback = ""
                break # Success, exit retry loop
            except SQLAlchemyError as e:
                logger.warning(f"  SQLAlchemy error during metadata query execution: {e}")
                error_feedback = f"PostgreSQL execution failed with error: {e}"
                try:
                    db_session.rollback()
                except Exception as rollback_err:
                    logger.warning(f"  Rollback failed: {rollback_err}")
                continue # Retry
            except Exception as e:
                logger.warning(f"  Unexpected error during metadata query execution: {e}")
                error_feedback = f"Unexpected error: {e}"
                try:
                    db_session.rollback()
                except Exception as rollback_err:
                    logger.warning(f"  Rollback failed: {rollback_err}")
                continue # Retry

        if error_feedback:
            # If we exhausted retries and still have an error, yield fallback
            logger.warning(f"  Exhausted {MAX_SQL_RETRIES} SQL retries. Last error: {error_feedback}")
            return FALLBACK_MESSAGE

    finally:
        if close_session_on_exit:
            db_session.close()

    # 4. Synthesize a natural-language answer from the raw results
    return _synthesize_answer(clean_query, sql, rows, columns, history_text=history_text)


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
