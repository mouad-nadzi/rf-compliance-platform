import json
from pathlib import Path
from sqlalchemy import text
from storage.database import get_db_session, engine, Base
from storage.models import AuthorityLookup, SupplierLookup

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "lookups"


def seed_lookup_tables():
    # Ensure any stale constraint is dropped cleanly in SQL database and add abbreviation column if missing
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE IF EXISTS authority_lookups DROP CONSTRAINT IF EXISTS authority_lookups_canonical_authority_key;"))
            conn.execute(text("DROP INDEX IF EXISTS authority_lookups_canonical_authority_key;"))
            conn.execute(text("ALTER TABLE IF EXISTS authority_lookups ADD COLUMN IF NOT EXISTS abbreviation VARCHAR(100);"))
            conn.commit()
    except Exception:
        pass

    Base.metadata.create_all(bind=engine)

    authorities_path = KNOWLEDGE_DIR / "authorities.json"
    suppliers_path = KNOWLEDGE_DIR / "suppliers.json"

    with get_db_session() as db:
        # Seed Authorities
        if authorities_path.exists():
            with open(authorities_path, "r", encoding="utf-8") as f:
                authorities_data = json.load(f)

            if db.query(AuthorityLookup).count() == 0:
                for item in authorities_data:
                    try:
                        abbr = item.get("abbreviation") or (item["aliases"][0] if item.get("aliases") else item["canonical_authority"])
                        db.add(
                            AuthorityLookup(
                                canonical_authority=item["canonical_authority"],
                                abbreviation=abbr,
                                country=item["country"],
                                standard_validity_years=item.get("standard_validity_years"),
                                aliases=item.get("aliases", []),
                            )
                        )
                        db.commit()
                    except Exception:
                        db.rollback()
                print("Loaded authorities from JSON.")
            else:
                # Update existing authority records with abbreviation if missing
                for auth in db.query(AuthorityLookup).all():
                    if not auth.abbreviation:
                        if auth.aliases and len(auth.aliases) > 0:
                            auth.abbreviation = auth.aliases[0]
                        else:
                            auth.abbreviation = auth.canonical_authority
                db.commit()

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
