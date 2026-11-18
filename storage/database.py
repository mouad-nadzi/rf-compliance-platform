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
            conn.execute(text("ALTER TABLE IF EXISTS certificates ALTER COLUMN certificate_id SET DEFAULT md5(random()::text || clock_timestamp()::text);"))
            conn.execute(text("ALTER TABLE IF EXISTS certificates ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;"))
            conn.execute(text("ALTER TABLE IF EXISTS chat_sessions ADD COLUMN IF NOT EXISTS user_id VARCHAR(64);"))
            conn.commit()

        # Create tables
        logger.info("Creating database tables if not exist...")
        from schemas.extraction import CertificateMetadata, Source, AgentMemory, RecycleBinItem, UserAccount
        Base.metadata.create_all(bind=engine)

        # Bootstrap unique Administrator account if none exists and backfill legacy chat sessions
        try:
            from server.auth import hash_password
            with get_db_session() as session:
                admin_row = session.query(UserAccount).filter(UserAccount.role == "admin").first()
                if admin_row is None:
                    admin_email = str(config.ADMIN_EMAIL or "mouadnadzi3@gmail.com").strip().lower()
                    admin_name = str(config.ADMIN_NAME or "Mouad Nadzi").strip()
                    admin_row = UserAccount(
                        email=admin_email,
                        full_name=admin_name,
                        role="admin",
                        password_hash=hash_password(config.ADMIN_INITIAL_PASSWORD),
                        status="active",
                    )
                    session.add(admin_row)
                    session.flush()
                    logger.info(f"Bootstrapped unique Administrator account: {admin_email}")
                session.execute(
                    text("UPDATE chat_sessions SET user_id = :uid WHERE user_id IS NULL"),
                    {"uid": admin_row.id},
                )
        except Exception as auth_err:
            logger.warning(f"Administrator bootstrap skipped/failed: {auth_err}")

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
