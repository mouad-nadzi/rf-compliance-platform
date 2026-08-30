"""
storage/database.py — PostgreSQL & PGVector Database Initialization & Session Management
"""

from contextlib import contextmanager
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from server import config

logger = logging.getLogger(__name__)

# SQLAlchemy Engine & Base Setup
engine = create_engine(
    config.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session_factory = scoped_session(SessionLocal)

Base = declarative_base()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions. Handles commit on success and rollback on failure.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_db():
    """
    FastAPI dependency yielding a raw database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> bool:
    """
    Initialize database, enable vector extension, clear stale ORM metadata, drop legacy unique constraints, and seed lookups.

    Returns:
        bool: True if initialization succeeded, False if connection/execution failed.
    """
    try:
        import schemas.extraction
        import storage.models

        logger.info(f"Connecting to database at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}...")
        with engine.connect() as conn:
            # Enable pgvector extension
            logger.info("Enabling pgvector extension if not exists...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(text("ALTER TABLE IF EXISTS authority_lookups DROP CONSTRAINT IF EXISTS authority_lookups_canonical_authority_key;"))
            conn.execute(text("DROP INDEX IF EXISTS authority_lookups_canonical_authority_key;"))
            conn.execute(text("ALTER TABLE IF EXISTS sources ADD COLUMN IF NOT EXISTS cookie_header TEXT;"))
            conn.execute(text("ALTER TABLE IF EXISTS certificates ADD COLUMN IF NOT EXISTS cert_link VARCHAR(500);"))
            conn.commit()

        # Create tables
        logger.info("Creating database tables if not exist...")
        from schemas.extraction import CertificateMetadata, Source, AgentMemory, RecycleBinItem
        Base.metadata.create_all(bind=engine)

        # Seed lookup tables from knowledge assets if needed
        try:
            from storage.seed_lookups import seed_lookup_tables
            seed_lookup_tables()
        except Exception as seed_err:
            logger.warning(f"Lookup table seeding skipped/failed: {seed_err}")

        # Seed base identity memories into long-term memory store
        try:
            from core.agent.memory import seed_base_identity_memories
            seed_base_identity_memories()
        except Exception as mem_err:
            logger.warning(f"Base memory seeding skipped/failed: {mem_err}")

        logger.info("Database initialized successfully with pgvector & lookup support.")
        return True

    except Exception as e:
        logger.error(f" Failed to initialize database: {e}")
        return False
