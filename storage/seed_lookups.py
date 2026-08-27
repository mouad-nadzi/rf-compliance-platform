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

        # Seed Suppliers
        if suppliers_path.exists() and db.query(SupplierLookup).count() == 0:
            with open(suppliers_path, "r", encoding="utf-8") as f:
                suppliers_data = json.load(f)
                for item in suppliers_data:
                    try:
                        db.add(
                            SupplierLookup(
                                canonical_supplier=item["canonical_supplier"],
                                aliases=item.get("aliases", []),
                            )
                        )
                        db.commit()
                    except Exception:
                        db.rollback()
            print("Loaded suppliers from JSON.")

        print("Lookup tables verified and seeded successfully.")


if __name__ == "__main__":
    seed_lookup_tables()
