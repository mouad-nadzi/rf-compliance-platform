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


    def test_bulk_operations_and_combine(self):
        from core.agent.tools import ManageSchemaTool
        table_name = "test_bulk_schema_table"

        # 1. Create table with extra columns
        create_res = dynamic_schema.create_custom_table(
            table_name=table_name,
            columns=[
                {"name": "firstname", "type": "string"},
                {"name": "lastname", "type": "string"},
                {"name": "component", "type": "string"},
                {"name": "country", "type": "string"},
                {"name": "phone", "type": "string"},
                {"name": "fax", "type": "string"},
                {"name": "temp_notes", "type": "string"},
            ],
        )
        self.assertEqual(create_res["status"], "success")

        # 2. Insert record
        insert_res = dynamic_schema.insert_dynamic_record(
            table_name,
            {
                "firstname": "John",
                "lastname": "Doe",
                "component": "Radar B3",
                "country": "Germany",
                "phone": "+49123456",
                "fax": "+49654321",
                "temp_notes": "Internal note",
            },
        )
        self.assertEqual(insert_res["status"], "success")

        # 3. Test combine_columns (firstname + lastname -> responsible_name, drop sources)
        combine_res = dynamic_schema.combine_columns_in_table(
            table_name=table_name,
            target_column="responsible_name",
            source_columns=["firstname", "lastname"],
            drop_sources=True,
        )
        self.assertEqual(combine_res["status"], "success")

        # Verify combined value
        records = dynamic_schema.fetch_dynamic_records(table_name)
        self.assertEqual(records[0]["responsible_name"], "John Doe")
        cols_after_combine = [c["column_name"] for c in dynamic_schema.get_table_columns(table_name)]
        self.assertIn("responsible_name", cols_after_combine)
        self.assertNotIn("firstname", cols_after_combine)
        self.assertNotIn("lastname", cols_after_combine)

        # 4. Test bulk drop via drop_columns_from_table
        drop_bulk_res = dynamic_schema.drop_columns_from_table(table_name, ["phone", "fax", "temp_notes"])
        self.assertEqual(drop_bulk_res["status"], "success")
        self.assertEqual(len(drop_bulk_res["dropped_columns"]), 3)

        cols_after_drop = [c["column_name"] for c in dynamic_schema.get_table_columns(table_name)]
        self.assertNotIn("phone", cols_after_drop)
        self.assertNotIn("fax", cols_after_drop)
        self.assertNotIn("temp_notes", cols_after_drop)

        # 5. Test ManageSchemaTool tool execution with bulk actions
        tool = ManageSchemaTool()
        
        # Test tool bulk add_columns
        tool_add_res = tool.execute(
            action="add_columns",
            table_name=table_name,
            columns=[
                {"name": "email", "type": "string"},
                {"name": "exp_date", "type": "date"},
            ],
        )
        self.assertEqual(tool_add_res["status"], "success")

        cols_after_tool_add = [c["column_name"] for c in dynamic_schema.get_table_columns(table_name)]
        self.assertIn("email", cols_after_tool_add)
        self.assertIn("exp_date", cols_after_tool_add)

        # Test tool bulk drop_columns with comma-separated column_name string
        tool_drop_res = tool.execute(
            action="drop_columns",
            table_name=table_name,
            column_name="email, exp_date",
        )
        self.assertEqual(tool_drop_res["status"], "success")

        # Clean up
        dynamic_schema.drop_custom_table(table_name)


if __name__ == "__main__":
    unittest.main()
