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

        # Unified standardization and country/supplier/authority migration in certificates table
        try:
            from schemas.extraction import CertificateMetadata
            from core.normalization import normalize_certificate_fields, get_lookup_caches

            caches = get_lookup_caches(db)
            certs = db.query(CertificateMetadata).all()
            updated_count = 0
            country_standardized = 0
            supplier_standardized = 0
            autofilled_auth_country = 0
            cleaned_placeholders = 0

            for cert in certs:
                orig_country = cert.country or ""
                orig_supplier = cert.supplier or ""
                orig_authority = cert.authority or ""

                norm_dict, autofilled = normalize_certificate_fields(
                    raw_component=cert.component,
                    raw_supplier=cert.supplier,
                    raw_country=cert.country,
                    raw_certif_number=cert.certif_number,
                    raw_authority=cert.authority,
                    raw_issue_date=cert.issue_date,
                    raw_exp_date=cert.exp_date,
                    raw_cert_link=cert.cert_link,
                    raw_file_name=cert.file_name,
                    db=db,
                    caches=caches,
                )

                changed = False
                new_country = norm_dict.get("country") or ""
                new_supplier = norm_dict.get("supplier") or ""
                new_authority = norm_dict.get("authority") or ""
                new_issue_date = norm_dict.get("issue_date")
                new_exp_date = norm_dict.get("exp_date")

                if (cert.country or "") != new_country:
                    if orig_country in ("—", "N/A", "Unknown", "null", "None"):
                        cleaned_placeholders += 1
                    elif orig_country and new_country:
                        country_standardized += 1
                    cert.country = new_country
                    changed = True

                if (cert.supplier or "") != new_supplier:
                    if orig_supplier in ("—", "N/A", "Unknown", "null", "None"):
                        cleaned_placeholders += 1
                    elif orig_supplier and new_supplier:
                        supplier_standardized += 1
                    cert.supplier = new_supplier
                    changed = True

                if (cert.authority or "") != new_authority:
                    if orig_authority in ("—", "N/A", "Unknown", "null", "None"):
                        cleaned_placeholders += 1
                    cert.authority = new_authority
                    changed = True

                if cert.issue_date != new_issue_date and new_issue_date is not None:
                    cert.issue_date = new_issue_date
                    changed = True

                if cert.exp_date != new_exp_date and new_exp_date is not None:
                    cert.exp_date = new_exp_date
                    changed = True

                if autofilled:
                    autofilled_auth_country += 1

                if changed:
                    updated_count += 1

            if updated_count > 0:
                db.commit()
                print(f"Database Normalization & Standardization Summary:")
                print(f" - Total certificate records updated: {updated_count}")
                print(f" - Countries standardized (e.g. USA -> United States): {country_standardized} records")
                print(f" - Suppliers standardized/derived: {supplier_standardized} records")
                print(f" - Auto-filled Authority/Country/Date: {autofilled_auth_country} records")
                print(f" - Cleaned placeholder strings: {cleaned_placeholders} fields")
        except Exception as ex:
            print(f"Standardization migration cleanup warning: {ex}")

        print("Lookup tables verified and seeded successfully.")


if __name__ == "__main__":
    seed_lookup_tables()
