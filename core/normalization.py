"""
core/normalization.py — Unified RF Certificate Normalization Engine

Provides deterministic, centralized normalization rules for all ingestion pathways:
- PDF Extraction Pipeline
- Batch Spreadsheet / Excel Ingestion
- Single Certificate API Saves
- AI Diffing Standardization
- Database Seed & Migration Passes
- Recycle Bin Recovery

Enforces:
1. Empty cell invariance (no placeholder strings like 'N/A', 'Unknown', '—').
2. Deterministic Country Canonicalization (e.g., 'USA' -> 'United States').
3. Deterministic Supplier Canonicalization (alias lookup).
4. Deterministic Authority Canonicalization (abbreviation & alias lookup).
5. Bidirectional Auto-Fill (Authority <-> Country, Component -> Supplier).
6. ISO 8601 Date Parsing & Perpetual Validity resolution.
"""

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Non-expiry / Infinite constants
NO_EXPIRY_DATE = date(2999, 1, 1)
NO_EXPIRY_DISPLAY = "Infinite"
NO_EXPIRY_MARKERS = (
    "perpetual",
    "infinite",
    "indefinite",
    "permanent",
    "no expiry",
    "no expiration",
    "2999-01-01",
    "2999/01/01",
    "9999-01-01",
    "9999/01/01",
)

PLACEHOLDER_STRINGS = {
    "—",
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not found",
    "not available",
    "undefined",
    "?",
    "??",
}

# Explicit Country Canonicalization Map (case-insensitive lookup key)
COUNTRY_CANONICAL_MAP: Dict[str, str] = {
    # United States
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s": "United States",
    "united states of america": "United States",
    "united states": "United States",
    # United Kingdom
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.k": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "united kingdom": "United Kingdom",
    "united kingdom (uk)": "United Kingdom",
    # United Arab Emirates
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "u.a.e": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    # South Korea
    "south korea": "South Korea",
    "korea, republic of": "South Korea",
    "republic of korea": "South Korea",
    "korea (south)": "South Korea",
    "south-korea": "South Korea",
    "korea": "South Korea",
    # Netherlands
    "netherlands": "Netherlands",
    "the netherlands": "Netherlands",
    "netherland": "Netherlands",
    "holland": "Netherlands",
    # Czech Republic
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    # Russia
    "russia": "Russian Federation",
    "russian federation": "Russian Federation",
    # China / Taiwan / Hong Kong
    "prc": "China",
    "p.r. china": "China",
    "people's republic of china": "China",
    "peoples republic of china": "China",
    "china": "China",
    "taiwan": "Taiwan",
    "taiwan (roc)": "Taiwan",
    "taiwan, province of china": "Taiwan",
    "hong kong": "Hong Kong",
    "hong kong sar": "Hong Kong",
    # Vietnam
    "vietnam": "Vietnam",
    "viet nam": "Vietnam",
    # Ivory Coast
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    # Myanmar
    "myanmar": "Union of Myanmar",
    "burma": "Union of Myanmar",
    "union of myanmar": "Union of Myanmar",
    # Congo
    "congo": "Congo",
    "republic of the congo": "Congo",
    "congo republic": "Congo",
    "congo-brazzaville": "Congo",
    "dr congo": "Rep. Dem. Congo",
    "drc": "Rep. Dem. Congo",
    "democratic republic of the congo": "Rep. Dem. Congo",
    "rep. dem. congo": "Rep. Dem. Congo",
    "congo-kinshasa": "Rep. Dem. Congo",
    # Eswatini
    "eswatini": "Eswatini (Swaziland)",
    "swaziland": "Eswatini (Swaziland)",
    "eswatini (swaziland)": "Eswatini (Swaziland)",
    # Territories & Jurisdictions
    "puerto rico": "Puerto Rico (USA)",
    "puerto rico (usa)": "Puerto Rico (USA)",
    "montserrat": "Montserrat (UK)",
    "montserrat (uk)": "Montserrat (UK)",
    "turks and caicos islands": "Turks and Caicos Islands (UK)",
    "turks and caicos islands (uk)": "Turks and Caicos Islands (UK)",
    "curaçao": "Curaçao (Netherland)",
    "curacao": "Curaçao (Netherland)",
    "curaçao (netherland)": "Curaçao (Netherland)",
    "bonaire, saint eustatius and saba": "Bonaire, Saint Eustatius and Saba (Netherland)",
    "bonaire, saint eustatius and saba (netherland)": "Bonaire, Saint Eustatius and Saba (Netherland)",
    "french polynesia": "French Polynesia (France PTOM)",
    "french polynesia (france ptom)": "French Polynesia (France PTOM)",
    "new caledonia": "New Caledonia (France PTOM)",
    "new caledonia (france ptom)": "New Caledonia (France PTOM)",
    "wallis and futuna": "Wallis and Futuna Islands (France)",
    "wallis and futuna islands (france)": "Wallis and Futuna Islands (France)",
    "saint-barthelemy": "Saint-Barthelemy (France)",
    "saint-barthelemy (france)": "Saint-Barthelemy (France)",
    "saint barthelemy": "Saint-Barthelemy (France)",
    "french southern and antarctic lands": "French Southern and Antartic lands (France)",
    "french southern and antartic lands (france)": "French Southern and Antartic lands (France)",
    "martinique": "Martinique (France DROM)",
    "martinique (france drom)": "Martinique (France DROM)",
    "tokelau": "Tokelau (New Zealand)",
    "tokelau (new zealand)": "Tokelau (New Zealand)",
}

FRENCH_MONTH_MAP: Dict[str, str] = {
    "janv": "jan",
    "janvier": "jan",
    "févr": "feb",
    "fevr": "feb",
    "fév": "feb",
    "fev": "feb",
    "février": "feb",
    "fevrier": "feb",
    "mars": "mar",
    "avr": "apr",
    "avril": "apr",
    "mai": "may",
    "juin": "jun",
    "juil": "jul",
    "jui": "jul",
    "juillet": "jul",
    "août": "aug",
    "aout": "aug",
    "sept": "sep",
    "septembre": "sep",
    "oct": "oct",
    "octobre": "oct",
    "nov": "nov",
    "novembre": "nov",
    "déc": "dec",
    "dec": "dec",
    "décembre": "dec",
    "decembre": "dec",
}


def clean_string_field(val: Any) -> str:
    """Cleans whitespace and reduces placeholder tokens to empty strings."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in PLACEHOLDER_STRINGS:
        return ""
    return s


def is_no_expiry(val: Any) -> bool:
    """Checks whether a value indicates non-expiring/perpetual validity."""
    if not val:
        return False
    s = str(val).strip().lower()
    return s in NO_EXPIRY_MARKERS or s.startswith("2999") or s.startswith("9999")


def normalize_country(raw_country: Optional[str], known_countries: Optional[Dict[str, str]] = None) -> str:
    """
    Standardizes a country name using explicit canonical mappings and lookup tables.
    Returns empty string if country is missing or placeholder.
    """
    clean = clean_string_field(raw_country)
    if not clean:
        return ""

    key = clean.lower()

    # 1. Direct match in explicit canonical map
    if key in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[key]

    # 2. Known countries map from AuthorityLookup DB
    if known_countries and key in known_countries:
        return known_countries[key]

    return clean


def parse_date(date_val: Any) -> Optional[date]:
    """
    Converts arbitrary date strings, numbers, or date objects to a standard Python date object.
    Returns NO_EXPIRY_DATE (2999-01-01) for perpetual dates.
    """
    if not date_val:
        return None
    if isinstance(date_val, date) and not isinstance(date_val, datetime):
        if date_val.year >= 2900:
            return NO_EXPIRY_DATE
        return date_val
    if isinstance(date_val, datetime):
        if date_val.year >= 2900:
            return NO_EXPIRY_DATE
        return date_val.date()

    s = clean_string_field(date_val)
    if not s:
        return None

    if is_no_expiry(s):
        return NO_EXPIRY_DATE

    # Excel serial number float/int check
    if s.replace(".", "", 1).isdigit():
        try:
            num = float(s)
            if 1000 < num < 100000:
                from datetime import timedelta
                return (datetime(1899, 12, 30) + timedelta(days=num)).date()
        except Exception:
            pass

    # YYYY-MM-DD prefix check
    if len(s) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", s[:10]):
        try:
            parsed = datetime.strptime(s[:10], "%Y-%m-%d").date()
            if parsed.year >= 2900:
                return NO_EXPIRY_DATE
            return parsed
        except ValueError:
            pass

    # French month translation
    s_lower = s.lower()
    for fr, en in FRENCH_MONTH_MAP.items():
        if fr in s_lower:
            s_lower = re.sub(r"(?<=[\d\s\-_])" + re.escape(fr) + r"(?=[\d\s\-_]|$)", en, s_lower)
            break

    # dateutil parser
    try:
        from dateutil import parser as date_parser
        dt = date_parser.parse(s_lower, dayfirst=True)
        if dt.year >= 2900:
            return NO_EXPIRY_DATE
        return dt.date()
    except Exception:
        pass

    # Standard format iteration
    for fmt in (
        "%d-%b-%y",
        "%d-%b-%Y",
        "%b-%d-%y",
        "%b-%d-%Y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d-%m-%y",
    ):
        try:
            parsed = datetime.strptime(s, fmt).date()
            if parsed.year >= 2900:
                return NO_EXPIRY_DATE
            return parsed
        except Exception:
            pass

    return None


def format_date_iso(d: Optional[date]) -> Optional[str]:
    """Formats a date to YYYY-MM-DD or Infinite string, preserving None for empty."""
    if d is None:
        return None
    if d.year >= 2900:
        return "Infinite"
    return d.strftime("%Y-%m-%d")


def get_lookup_caches(db: Optional[Session] = None) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, Any], Dict[str, str]]:
    """
    Builds in-memory lookup indices from PostgreSQL for fast resolution:
    1. country_to_auth_map (lower country -> authority abbreviation or canonical)
    2. auth_to_country_map (lower auth -> canonical country)
    3. alias_to_supp_map (lower supp/alias -> canonical supplier)
    4. auth_validity_map (lower auth -> standard_validity_years)
    5. known_countries (lower country -> canonical DB country name)
    """
    country_to_auth_map: Dict[str, str] = {}
    auth_to_country_map: Dict[str, str] = {}
    alias_to_supp_map: Dict[str, str] = {}
    auth_validity_map: Dict[str, Any] = {}
    known_countries: Dict[str, str] = {}

    if db is None:
        return country_to_auth_map, auth_to_country_map, alias_to_supp_map, auth_validity_map, known_countries

    try:
        from storage.models import AuthorityLookup, SupplierLookup

        authorities = db.query(AuthorityLookup).all()
        for al in authorities:
            c = (al.country or "").strip()
            if c:
                canon_country = COUNTRY_CANONICAL_MAP.get(c.lower(), c)
                known_countries[c.lower()] = canon_country
                known_countries[canon_country.lower()] = canon_country

                auth_target = al.abbreviation if al.abbreviation else al.canonical_authority
                country_to_auth_map[c.lower()] = auth_target
                country_to_auth_map[canon_country.lower()] = auth_target

                if al.abbreviation:
                    auth_to_country_map[al.abbreviation.strip().lower()] = canon_country
                if al.canonical_authority:
                    auth_to_country_map[al.canonical_authority.strip().lower()] = canon_country
                if isinstance(al.aliases, list):
                    for alias in al.aliases:
                        if alias:
                            auth_to_country_map[alias.strip().lower()] = canon_country

                if al.standard_validity_years:
                    val = al.standard_validity_years
                    if al.abbreviation:
                        auth_validity_map[al.abbreviation.strip().lower()] = val
                    if al.canonical_authority:
                        auth_validity_map[al.canonical_authority.strip().lower()] = val

        suppliers = db.query(SupplierLookup).all()
        for sl in suppliers:
            if sl.canonical_supplier:
                canon = sl.canonical_supplier.strip()
                alias_to_supp_map[canon.lower()] = canon
                if isinstance(sl.aliases, list):
                    for alias in sl.aliases:
                        if alias:
                            alias_to_supp_map[alias.strip().lower()] = canon

    except Exception as e:
        logger.warning(f"Error building lookup caches: {e}")

    return country_to_auth_map, auth_to_country_map, alias_to_supp_map, auth_validity_map, known_countries


def normalize_certificate_fields(
    raw_component: Optional[str],
    raw_supplier: Optional[str],
    raw_country: Optional[str],
    raw_certif_number: Optional[str],
    raw_authority: Optional[str],
    raw_issue_date: Any,
    raw_exp_date: Any,
    raw_cert_link: Optional[str] = None,
    raw_file_name: Optional[str] = None,
    db: Optional[Session] = None,
    caches: Optional[Tuple] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Single unified normalization pass for all certificate attributes.

    Returns:
        (normalized_dict, autofilled_descriptions_list)
    """
    if caches:
        country_to_auth_map, auth_to_country_map, alias_to_supp_map, auth_validity_map, known_countries = caches
    else:
        country_to_auth_map, auth_to_country_map, alias_to_supp_map, auth_validity_map, known_countries = get_lookup_caches(db)

    autofilled: List[str] = []

    # 1. Clean raw string values (empty string for placeholders)
    component = clean_string_field(raw_component)
    supplier = clean_string_field(raw_supplier)
    country = clean_string_field(raw_country)
    certif_number = clean_string_field(raw_certif_number)
    authority = clean_string_field(raw_authority)
    cert_link = clean_string_field(raw_cert_link)
    file_name = clean_string_field(raw_file_name)

    # 2. Country Canonicalization
    if country:
        normalized_c = normalize_country(country, known_countries)
        country = normalized_c

    # 3. Supplier Canonicalization
    if supplier:
        supp_key = supplier.lower()
        if supp_key in alias_to_supp_map:
            supplier = alias_to_supp_map[supp_key]
        else:
            for alias_key, canon_val in alias_to_supp_map.items():
                if len(alias_key) >= 3 and alias_key in supp_key:
                    supplier = canon_val
                    break

    # 4. Bidirectional Country <-> Authority Auto-Fill
    if not country and authority:
        auth_key = authority.lower()
        if auth_key in auth_to_country_map:
            country = auth_to_country_map[auth_key]
            autofilled.append(f'Country: "{country}" (derived from Authority "{authority}")')

    if not authority and country:
        ctry_key = country.lower()
        if ctry_key in country_to_auth_map:
            authority = country_to_auth_map[ctry_key]
            autofilled.append(f'Authority: "{authority}" (derived from Country "{country}")')

    # 5. Component -> Supplier Auto-Fill
    if not supplier and component:
        comp_key = component.lower()
        for alias_key, canon_val in alias_to_supp_map.items():
            if len(alias_key) >= 3 and alias_key in comp_key:
                supplier = canon_val
                autofilled.append(f'Supplier: "{supplier}" (derived from Component "{component}")')
                break

    # 6. Date Normalization
    parsed_issue = parse_date(raw_issue_date)
    parsed_exp = parse_date(raw_exp_date)

    # 7. Standard Validity / Expiration Date Derivation from Authority
    if parsed_exp is None and authority:
        auth_key = authority.lower()
        validity = auth_validity_map.get(auth_key)
        if validity:
            validity_str = str(validity).strip().lower()
            if validity_str == "infinite":
                parsed_exp = NO_EXPIRY_DATE
                autofilled.append(f'Expiration Date: "Infinite" (derived from Authority "{authority}")')
            elif parsed_issue:
                try:
                    val_years = int(float(validity_str))
                    try:
                        from dateutil.relativedelta import relativedelta
                        parsed_exp = parsed_issue + relativedelta(years=val_years)
                    except ImportError:
                        parsed_exp = parsed_issue.replace(year=parsed_issue.year + val_years)
                    autofilled.append(f'Expiration Date: "{parsed_exp.strftime("%Y-%m-%d")}" (derived from Authority validity: {val_years} years)')
                except Exception:
                    pass

    normalized = {
        "component": component or None,
        "supplier": supplier or None,
        "country": country or None,
        "certif_number": certif_number or None,
        "authority": authority or None,
        "issue_date": parsed_issue,
        "exp_date": parsed_exp,
        "issue_date_iso": format_date_iso(parsed_issue),
        "exp_date_iso": format_date_iso(parsed_exp),
        "cert_link": cert_link or None,
        "file_name": file_name or None,
    }

    return normalized, autofilled


def normalize_certificate_dict(
    payload: Dict[str, Any],
    db: Optional[Session] = None,
    caches: Optional[Tuple] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Adapter to normalize any dictionary representation of a certificate record
    (e.g., from CSV row, JSON API payload, database row dict).
    """
    raw_comp = payload.get("component") or payload.get("Component") or payload.get("name") or ""
    raw_supp = payload.get("supplier") or payload.get("Supplier") or payload.get("canonical_supplier") or ""
    raw_ctry = payload.get("country") or payload.get("Country") or ""
    raw_cert = payload.get("certif_number") or payload.get("Certif Number") or payload.get("certif_no") or payload.get("H-27922") or ""
    raw_auth = payload.get("authority") or payload.get("Authority") or payload.get("canonical_authority") or ""
    raw_issue = payload.get("issue_date") or payload.get("Issue Date") or payload.get("issueDate") or ""
    raw_exp = (
        payload.get("exp_date")
        or payload.get("expiration_date")
        or payload.get("Expiration Date")
        or payload.get("Exp Date")
        or payload.get("expDate")
        or ""
    )
    raw_link = payload.get("cert_link") or payload.get("PDF Document Link") or payload.get("url") or ""
    raw_file = payload.get("file_name") or payload.get("File Name") or ""

    return normalize_certificate_fields(
        raw_component=raw_comp,
        raw_supplier=raw_supp,
        raw_country=raw_ctry,
        raw_certif_number=raw_cert,
        raw_authority=raw_auth,
        raw_issue_date=raw_issue,
        raw_exp_date=raw_exp,
        raw_cert_link=raw_link,
        raw_file_name=raw_file,
        db=db,
        caches=caches,
    )
