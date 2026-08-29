"""
schemas/extraction.py — Data Extraction Schemas & Database ORM Models

Defines:
1. Pydantic v2 models (CertificateExtractionSchema) used to structure text output
   extracted by the LLM from raw Markdown.
2. SQLAlchemy ORM models (CertificateMetadata, CertificateChunk) for PostgreSQL + pgvector persistence.
"""

from typing import Optional
from datetime import datetime
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
    memory_key = Column(String(100), nullable=False, default="preference", comment="Category key: preference, rule, contact, portal_note")
    fact_text = Column(Text, nullable=False, comment="Persisted long-term fact or user directive")
    source_session_id = Column(String(100), nullable=True, comment="Optional session ID where memory originated")
    created_at = Column(DateTime, default=datetime.utcnow, comment="Timestamp when memory was recorded")
