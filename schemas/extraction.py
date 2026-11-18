"""
schemas/extraction.py — Data Extraction Schemas & Database ORM Models

Defines:
1. Pydantic v2 models (CertificateExtractionSchema) used to structure text output
   extracted by the LLM from raw Markdown.
2. SQLAlchemy ORM models (CertificateMetadata, CertificateChunk) for PostgreSQL + pgvector persistence.
"""

from typing import Optional
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from sqlalchemy import Column, String, Text, Integer, Date, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from storage.database import Base


class CertificateExtractionSchema(BaseModel):
    component: str = Field(
        ...,
        description="The component model code or equipment identifier (e.g., IM3C, SD1A, VSM-125kHz, F5CP12).",
    )
    supplier: str = Field(
        ...,
        description="The foreign manufacturer, OEM, or global brand (e.g., VALEO, BOSCH, APTIV, FIH Mobile Limited). Excludes local legal applicants/representatives.",
    )
    country: Optional[str] = Field(
        None,
        description="The country of regulatory jurisdiction (e.g., Argentina, Bolivia, Brazil).",
    )
    certif_number: str = Field(
        ...,
        description="The primary certificate, disposition, or homologation number (e.g., H-22392, 425/2025, DEKRA-00245-23).",
    )
    authority: str = Field(
        ...,
        description="The issuing authority or regulatory body (e.g., ENTE NACIONAL DE COMUNICACIONES, ATT, ANATEL).",
    )
    issue_date: Optional[str] = Field(
        None,
        description="Date of issuance in YYYY-MM-DD format.",
    )
    exp_date: Optional[str] = Field(
        None,
        description="Date of expiration in YYYY-MM-DD format (explicit or computed).",
    )
    cert_link: Optional[str] = Field(
        None,
        description="URL or hyperlink to the official certificate document or regulatory authority page.",
    )


class CertificateMetadata(Base):
    """
    SQLAlchemy ORM model representing normalized certificate metadata stored in PostgreSQL.
    
    Fields accept normalized English values for consistent relational querying:
    Component, Supplier, Country, Certif Number, Authority, Issue Date, Exp Date.
    """
    __tablename__ = "certificates"

    certificate_id = Column(String(64), primary_key=True, index=True, comment="Unique identifier for the certificate record")
    component = Column(String(255), nullable=True, index=True, comment="Normalized Component identifier")
    supplier = Column(String(255), nullable=True, index=True, comment="Normalized Supplier/manufacturer name")
    country = Column(String(100), nullable=True, index=True, comment="Normalized English Country name")
    certif_number = Column(String(255), nullable=True, index=True, comment="Normalized Certif Number / certification ID")
    authority = Column(String(255), nullable=True, index=True, comment="Normalized Authority / certification body")
    issue_date = Column(Date, nullable=True, comment="Certificate Issue Date")
    exp_date = Column(Date, nullable=True, comment="Certificate Exp Date")
    cert_link = Column(Text, nullable=True, comment="URL link to the official certificate or regulatory page")
    file_name = Column(String(255), nullable=True, comment="Source document file name")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Timestamp of database record insertion")

    # Relationship to chunks
    chunks = relationship("CertificateChunk", back_populates="certificate", cascade="all, delete-orphan")


class CertificateChunk(Base):
    """
    SQLAlchemy ORM model representing document text chunks and vector embeddings.
    """
    __tablename__ = "certificate_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    certificate_id = Column(String(64), ForeignKey("certificates.certificate_id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=True, comment="1-based page number reference")
    raw_text = Column(Text, nullable=False, comment="Raw markdown chunk text")
    embedding = Column(Vector(1024), nullable=True, comment="1024-dimensional dense vector embedding (BAAI/bge-m3 pgvector)")

    # Relationship back to certificate metadata
    certificate = relationship("CertificateMetadata", back_populates="chunks")


class ChatSession(Base):
    """
    SQLAlchemy ORM model for persisted chat sessions (survive backend restarts).
    """
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True, index=True, comment="Session identifier (sess_<hex>)")
    user_id = Column(String(64), nullable=True, index=True, comment="Owning user_id for per-user session isolation")
    title = Column(String(255), nullable=False, default="", comment="Session title derived from first question")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, comment="Session creation timestamp")
    frozen = Column(Boolean, nullable=False, default=False, comment="True when the context window budget is exhausted")

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    """
    SQLAlchemy ORM model for a single chat turn inside a persisted session.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(32), nullable=False, comment="'user' or 'assistant'")
    content = Column(Text, nullable=False, default="", comment="Message body text")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, comment="Message timestamp")

    session = relationship("ChatSession", back_populates="messages")


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True, comment="Source row id")
    url = Column(Text, nullable=False, comment="Portal/database URL to check for documents")
    description = Column(String(255), nullable=True, comment="Optional human-readable label")
    active = Column(Boolean, nullable=False, default=True, comment="Whether the autonomous scraper checks this source")
    cookie_header = Column(Text, nullable=True, comment="Optional HTTP cookie/token header for authenticated portals")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Timestamp of source record insertion")


class AgentMemory(Base):
    __tablename__ = "agent_memories"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True, comment="Memory row id")
    memory_key = Column(String(100), unique=True, nullable=False, default="preference", comment="Category key: preference, rule, contact, portal_note")
    fact_text = Column(Text, nullable=False, comment="Persisted long-term fact or user directive")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Timestamp when memory was recorded")


class RecycleBinItem(Base):
    __tablename__ = "recycle_bin"
    __table_args__ = {"extend_existing": True}

    id = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex, comment="Recycle bin item primary key")
    title = Column(String(255), nullable=False, comment="Display title for deleted item")
    table_name = Column(String(100), nullable=False, comment="Original table name")
    record_data = Column(Text, nullable=False, comment="JSON encoded record payload")
    deleted_at = Column(DateTime, default=datetime.utcnow, comment="Timestamp of deletion")


class NotificationItem(Base):
    __tablename__ = "notifications"
    __table_args__ = {"extend_existing": True}

    id = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex, comment="Notification item primary key")
    title = Column(String(255), nullable=False, comment="Short notification title")
    message = Column(Text, nullable=False, comment="Notification description text")
    category = Column(String(50), default="standardization", comment="Category: standardization, import, system, expiry_alert")
    is_read = Column(Boolean, default=False, comment="Whether notification has been read")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Notification timestamp")


class ExpiryAlertLog(Base):
    """
    Tracks dispatched or staged expiration alerts per (certificate_id, exp_date, alert_window_days)
    to guarantee idempotency across scheduled background scans.
    """
    __tablename__ = "expiry_alert_logs"
    __table_args__ = {"extend_existing": True}

    id = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex, comment="Alert log primary key")
    certificate_id = Column(String(64), nullable=False, index=True, comment="Referenced certificate_id")
    certif_number = Column(String(255), nullable=True, comment="Certificate number at alert time")
    component = Column(String(255), nullable=True, comment="Component name at alert time")
    country = Column(String(100), nullable=True, comment="Country at alert time")
    exp_date = Column(Date, nullable=False, comment="Certificate expiration date at alert time")
    alert_window_days = Column(Integer, nullable=False, comment="Lead-time threshold in days (e.g. 30, 60, 90)")
    recipient_email = Column(String(255), nullable=True, comment="Target responsible or fallback email")
    proposal_id = Column(String(64), nullable=True, comment="Associated HITL SEND_EMAIL proposal_id, if staged")
    alerted_at = Column(DateTime, default=datetime.utcnow, comment="Timestamp when alert was generated")


class UserAccount(Base):
    """
    Platform user account for 3-tier RBAC:
      - 'admin': Unique Administrator (user management, invite links, sender email config, full access)
      - 'editor': Allow-to-Edit user (database writes, ingestion, automations, email approvals)
      - 'viewer': Read-Only user (view tables, run Q&A chat; no mutations or automation triggers)
    """
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id = Column(String(64), primary_key=True, default=lambda: f"usr_{uuid.uuid4().hex[:12]}", comment="User primary key")
    email = Column(String(255), unique=True, nullable=False, index=True, comment="Canonical lowercase user email")
    full_name = Column(String(255), nullable=False, comment="User display name")
    role = Column(String(32), nullable=False, default="viewer", comment="Role: admin | editor | viewer")
    password_hash = Column(String(255), nullable=True, comment="PBKDF2-SHA256 salted password hash; NULL while invited")
    status = Column(String(32), nullable=False, default="invited", comment="Account status: active | invited")
    invite_token = Column(String(128), nullable=True, index=True, comment="Single-use onboarding invitation token")
    invite_expires_at = Column(DateTime, nullable=True, comment="Invitation expiration timestamp (UTC)")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Account creation timestamp")
    last_login_at = Column(DateTime, nullable=True, comment="Last successful login timestamp")
