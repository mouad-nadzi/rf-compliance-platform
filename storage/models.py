"""
storage/models.py — ORM Lookup Models for Metadata Normalization & Enrichment
"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from storage.database import Base


class AuthorityLookup(Base):
    __tablename__ = "authority_lookups"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_authority = Column(String(255), nullable=False)
    abbreviation = Column(String(100), nullable=True)
    country = Column(String(100), nullable=False)
    standard_validity_years = Column(Integer, nullable=True)
    aliases = Column(JSONB, default=list, nullable=False)


class SupplierLookup(Base):
    __tablename__ = "supplier_lookups"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_supplier = Column(String(255), nullable=False)
    aliases = Column(JSONB, default=list, nullable=False)
