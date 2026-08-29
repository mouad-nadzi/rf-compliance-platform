"""
tests/test_dynamic_schema.py — Unit Tests for Dynamic Schema Engine
"""

import unittest
from storage import dynamic_schema


class TestDynamicSchemaEngine(unittest.TestCase):
    def test_dynamic_table_lifecycle(self):
        table_name = "test_vendor_audits"

        # 1. Create custom table
        create_res = dynamic_schema.create_custom_table(
            table_name=table_name,
            columns=[
                {"name": "vendor_code", "type": "string"},
                {"name": "score", "type": "float"},
            ],
        )
        self.assertEqual(create_res["status"], "success")

        # 2. Get table columns
        cols = dynamic_schema.get_table_columns(table_name)
        col_names = [c["column_name"] for c in cols]
        self.assertIn("vendor_code", col_names)
        self.assertIn("score", col_names)

        # 3. Add column
        add_res = dynamic_schema.add_column_to_table(table_name, "notes", "text")
        self.assertEqual(add_res["status"], "success")

        cols_after_add = [c["column_name"] for c in dynamic_schema.get_table_columns(table_name)]
        self.assertIn("notes", cols_after_add)

        # 4. Insert dynamic record
        insert_res = dynamic_schema.insert_dynamic_record(
            table_name,
            {"vendor_code": "VND-101", "score": 98.5, "notes": "Audited cleanly"},
        )
        self.assertEqual(insert_res["status"], "success")
        inserted_id = insert_res["inserted_id"]

        # 5. Fetch dynamic records
        records = dynamic_schema.fetch_dynamic_records(table_name)
        self.assertTrue(len(records) >= 1)
        self.assertEqual(records[0]["vendor_code"], "VND-101")

        # 6. Drop column
        drop_res = dynamic_schema.drop_column_from_table(table_name, "notes")
        self.assertEqual(drop_res["status"], "success")

        # 7. Delete dynamic record
        deleted = dynamic_schema.delete_dynamic_record(table_name, inserted_id)
        self.assertTrue(deleted)

        # 8. List all tables
        tables = dynamic_schema.list_all_user_tables()
        table_names = [t["table_name"] for t in tables]
        self.assertIn(table_name, table_names)

        # 9. Drop custom table
        drop_tbl_res = dynamic_schema.drop_custom_table(table_name)
        self.assertEqual(drop_tbl_res["status"], "success")

        tables_after_drop = dynamic_schema.list_all_user_tables()
        table_names_after = [t["table_name"] for t in tables_after_drop]
        self.assertNotIn(table_name, table_names_after)


if __name__ == "__main__":
    unittest.main()
