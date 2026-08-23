"""
storage/backup.py — Portable database dump utility using pg_dump.
"""

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional
from config import BASE_DIR, POSTGRES_DB, POSTGRES_USER, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

# Portable backup path relative to project root
DEFAULT_BACKUP_PATH = Path(BASE_DIR) / "storage" / "db_backup.sql"


def export_database_to_sql(output_path: Optional[Path] = None) -> bool:
    """
    Executes a pg_dump to export the PostgreSQL database to a clean, portable .sql file.
    Uses subprocess.run with strict error handling.
    """
    target_path = Path(output_path or DEFAULT_BACKUP_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve connection details from environment or defaults
    db_name = os.getenv("POSTGRES_DB", POSTGRES_DB)
    db_user = os.getenv("POSTGRES_USER", POSTGRES_USER)
    db_host = os.getenv("POSTGRES_HOST", POSTGRES_HOST)
    db_port = str(os.getenv("POSTGRES_PORT", str(POSTGRES_PORT)))

    cmd = [
        "pg_dump",
        "-h",
        db_host,
        "-p",
        db_port,
        "-U",
        db_user,
        "-d",
        db_name,
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        "-f",
        str(target_path),
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = os.getenv("POSTGRES_PASSWORD", POSTGRES_PASSWORD)

    try:
        # Run pg_dump synchronously within this function
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env=env,
        )
        logger.info(f"✅ Database successfully dumped to {target_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ pg_dump execution failed (exit code {e.returncode}): {e.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("⚠️ 'pg_dump' CLI utility not found in system PATH. Backup skipped.")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during database backup: {str(e)}")
        return False


def trigger_async_backup(output_path: Optional[Path] = None) -> None:
    """Triggers database export in a background thread to prevent blocking ingestion."""
    thread = threading.Thread(
        target=export_database_to_sql,
        args=(output_path,),
        daemon=True,
        name="DB-Backup-Worker",
    )
    thread.start()


def restore_database_from_sql(input_path: Optional[Path] = None) -> bool:
    """
    Executes psql to restore the PostgreSQL database from a clean .sql file.
    Uses subprocess.run with strict error handling.
    """
    target_path = Path(input_path or DEFAULT_BACKUP_PATH)
    if not target_path.exists():
        logger.warning(f"Backup file '{target_path}' does not exist. Restoration skipped.")
        return False

    db_name = os.getenv("POSTGRES_DB", POSTGRES_DB)
    db_user = os.getenv("POSTGRES_USER", POSTGRES_USER)
    db_host = os.getenv("POSTGRES_HOST", POSTGRES_HOST)
    db_port = str(os.getenv("POSTGRES_PORT", str(POSTGRES_PORT)))

    cmd = [
        "psql",
        "-h",
        db_host,
        "-p",
        db_port,
        "-U",
        db_user,
        "-d",
        db_name,
        "-f",
        str(target_path),
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = os.getenv("POSTGRES_PASSWORD", POSTGRES_PASSWORD)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            env=env,
        )
        logger.info(f"Database successfully restored from {target_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"psql restoration failed (exit code {e.returncode}): {e.stderr[:300]}")
        return False
    except FileNotFoundError:
        logger.warning("'psql' CLI utility not found in system PATH. Restoration skipped.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database restoration: {str(e)}")
        return False
