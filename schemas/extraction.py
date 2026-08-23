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

from sqlalchemy import Column, String, Text, Integer, Date, DateTime, ForeignKey
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
