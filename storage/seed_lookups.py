import json
from pathlib import Path
from sqlalchemy import text
from storage.database import get_db_session, engine, Base
from storage.models import AuthorityLookup, SupplierLookup, normalize_validity_years

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "lookups"


def seed_lookup_tables():
    # Ensure any stale constraint is dropped cleanly in SQL database and add abbreviation column if missing
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE IF EXISTS authority_lookups DROP CONSTRAINT IF EXISTS authority_lookups_canonical_authority_key;"))
            conn.execute(text("DROP INDEX IF EXISTS authority_lookups_canonical_authority_key;"))
            conn.execute(text("ALTER TABLE IF EXISTS authority_lookups ADD COLUMN IF NOT EXISTS abbreviation VARCHAR(100);"))
            # Migrate the legacy integer column to the three-tier string representation.
            conn.execute(text(
                "ALTER TABLE authority_lookups ALTER COLUMN standard_validity_years "
                "TYPE VARCHAR(20) USING standard_validity_years::text;"
            ))
            conn.commit()
    except Exception:
        pass

    Base.metadata.create_all(bind=engine)

    authorities_path = KNOWLEDGE_DIR / "authorities.json"
    suppliers_path = KNOWLEDGE_DIR / "suppliers.json"

    with get_db_session() as db:
        # Seed Authorities (idempotent upsert keyed on canonical_authority + country)
        if authorities_path.exists():
            with open(authorities_path, "r", encoding="utf-8") as f:
                authorities_data = json.load(f)

            existing = db.query(AuthorityLookup).all()
            existing_map = {
                ((a.canonical_authority or "").strip() + "|" + (a.country or "").strip()).lower(): a
                for a in existing
            }
            inserted = 0
            updated = 0
            for item in authorities_data:
                canonical = item.get("canonical_authority") or ""
                country = item.get("country") or ""
                key = (canonical + "|" + country).lower()
                abbr = item.get("abbreviation") or (item["aliases"][0] if item.get("aliases") else item["canonical_authority"])
                validity = normalize_validity_years(item.get("standard_validity_years"))
                record = existing_map.get(key)
                if record is None:
                    db.add(
                        AuthorityLookup(
                            canonical_authority=canonical,
                            abbreviation=abbr,
                            country=country,
                            standard_validity_years=validity,
                            aliases=item.get("aliases", []),
                        )
                    )
                    inserted += 1
                else:
                    record.abbreviation = abbr
                    record.standard_validity_years = validity
                    record.aliases = item.get("aliases", [])
                    updated += 1
            db.commit()
            print(f"Loaded authorities from JSON ({inserted} inserted, {updated} updated).")

        # Seed Suppliers (idempotent upsert)
        if suppliers_path.exists():
            with open(suppliers_path, "r", encoding="utf-8") as f:
                suppliers_data = json.load(f)

            existing_supps = db.query(SupplierLookup).all()
            existing_supp_map = { (s.canonical_supplier or "").strip().lower(): s for s in existing_supps }
            inserted_supp = 0
            updated_supp = 0

            for item in suppliers_data:
                canon = item.get("canonical_supplier") or ""
                key = canon.strip().lower()
                aliases = item.get("aliases", [])
                rec = existing_supp_map.get(key)
                if rec is None:
                    db.add(SupplierLookup(canonical_supplier=canon, aliases=aliases))
                    inserted_supp += 1
                else:
                    rec.aliases = aliases
                    updated_supp += 1
            db.commit()
            print(f"Loaded suppliers from JSON ({inserted_supp} inserted, {updated_supp} updated).")

        # Standardize existing uncanonicalized supplier strings and auto-fill empty authority/country in certificates table
        try:
            from schemas.extraction import CertificateMetadata
            
            # Pre-cache AuthorityLookup and SupplierLookup maps for fast in-memory matching
            authority_lookups = db.query(AuthorityLookup).all()
            country_to_auth_map = {}
            auth_to_country_map = {}
            for al in authority_lookups:
                if al.country:
                    auth_target = al.abbreviation if al.abbreviation else al.canonical_authority
                    country_to_auth_map[al.country.strip().lower()] = auth_target
                    if al.abbreviation:
                        auth_to_country_map[al.abbreviation.strip().lower()] = al.country
                    if al.canonical_authority:
                        auth_to_country_map[al.canonical_authority.strip().lower()] = al.country
                    if isinstance(al.aliases, list):
                        for alias in al.aliases:
                            if alias:
                                auth_to_country_map[alias.strip().lower()] = al.country

            supplier_lookups = db.query(SupplierLookup).all()
            alias_to_canon = {}
            for sl in supplier_lookups:
                if sl.canonical_supplier:
                    alias_to_canon[sl.canonical_supplier.strip().lower()] = sl.canonical_supplier
                    for alias in (sl.aliases or []):
                        if alias:
                            alias_to_canon[alias.strip().lower()] = sl.canonical_supplier

            certs = db.query(CertificateMetadata).all()
            updated_cert_suppliers = 0
            autofilled_auth_country = 0
            cleaned_placeholders = 0

            for cert in certs:
                # 1. Clean legacy placeholder strings ("N/A", "Unknown", "—")
                if cert.country in ('—', 'N/A', 'Unknown'):
                    cert.country = ""
                    cleaned_placeholders += 1
                if cert.authority in ('—', 'N/A', 'Unknown'):
                    cert.authority = ""
                    cleaned_placeholders += 1
                if cert.supplier in ('—', 'N/A', 'Unknown'):
                    cert.supplier = ""
                    cleaned_placeholders += 1

                # 2. Supplier Canonicalization
                if cert.supplier:
                    supp_key = cert.supplier.strip().lower()
                    canon_target = alias_to_canon.get(supp_key)
                    if not canon_target:
                        for alias_key, target in alias_to_canon.items():
                            if len(alias_key) >= 3 and alias_key in supp_key:
                                canon_target = target
                                break
                    if canon_target and cert.supplier != canon_target:
                        cert.supplier = canon_target
                        updated_cert_suppliers += 1

                # 3. Derive Country from Authority if Country is empty
                if not (cert.country or "").strip() and (cert.authority or "").strip():
                    auth_key = cert.authority.strip().lower()
                    if auth_key in auth_to_country_map:
                        cert.country = auth_to_country_map[auth_key]
                        autofilled_auth_country += 1

                # 4. Derive Authority from Country if Authority is empty
                if not (cert.authority or "").strip() and (cert.country or "").strip():
                    ctry_key = cert.country.strip().lower()
                    if ctry_key in country_to_auth_map:
                        cert.authority = country_to_auth_map[ctry_key]
                        autofilled_auth_country += 1

                # 5. Derive Supplier from Component if Supplier is empty
                if not (cert.supplier or "").strip() and (cert.component or "").strip():
                    comp_key = cert.component.strip().lower()
                    for alias_key, target_supp in alias_to_canon.items():
                        if len(alias_key) >= 3 and alias_key in comp_key:
                            cert.supplier = target_supp
                            updated_cert_suppliers += 1
                            break

            if updated_cert_suppliers > 0 or autofilled_auth_country > 0 or cleaned_placeholders > 0:
                db.commit()
                print(f"Database Auto-Fill & Standardization Summary:")
                print(f" - Standardized/Derived suppliers: {updated_cert_suppliers} records")
                print(f" - Auto-filled Authority/Country: {autofilled_auth_country} records")
                print(f" - Cleaned placeholder strings: {cleaned_placeholders} fields")
        except Exception as ex:
            print(f"Supplier/Authority standardization cleanup warning: {ex}")

        print("Lookup tables verified and seeded successfully.")


if __name__ == "__main__":
    seed_lookup_tables()
