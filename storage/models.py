"""
storage/models.py — ORM Lookup Models for Metadata Normalization & Enrichment
"""

from typing import Any, Optional
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from storage.database import Base

# Tiers for standard_validity_years in knowledge/authorities.json:
#   - numeric string (e.g. "3")  -> determined term
#   - "infinite"                 -> non-expiring certificate
#   - None / null                -> variable / context-dependent term
INFINITE_VALIDITY = "infinite"


def normalize_validity_years(value: Any) -> Optional[str]:
    """
    Normalize a raw standard_validity_years value into its canonical string form.

    Returns:
        - "infinite" for non-expiring terms
        - a plain numeric string (e.g. "3") for determined terms
        - None for variable / context-dependent terms (and empty values)
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("none", "null", "nan", "n/a"):
        return None
    if s.lower() == INFINITE_VALIDITY:
        return INFINITE_VALIDITY
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


class AuthorityLookup(Base):
    __tablename__ = "authority_lookups"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_authority = Column(String(255), nullable=False)
    abbreviation = Column(String(100), nullable=True)
    country = Column(String(100), nullable=False)
    standard_validity_years = Column(String(20), nullable=True)
    aliases = Column(JSONB, default=list, nullable=False)


class SupplierLookup(Base):
    __tablename__ = "supplier_lookups"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_supplier = Column(String(255), nullable=False)
    aliases = Column(JSONB, default=list, nullable=False)
