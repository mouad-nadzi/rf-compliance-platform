"""
tests/test_memory_engine.py — Unit & Integration Tests for Session-Independent Long-Term Memory Engine
"""

import unittest
from core.agent.memory import (
    save_agent_memory,
    get_active_memories,
    delete_agent_memory,
    format_memories_for_prompt,
)


class TestMemoryEngine(unittest.TestCase):

    def test_memory_lifecycle(self):
        """Verify memory creation, listing, prompt formatting, and deletion."""
        # 1. Save memory
        fact = "Test Directive: Stellantis certificates require 60-day expiry warnings."
        mem = save_agent_memory("rule", fact)
        self.assertIsNotNone(mem.get("id"))
        self.assertEqual(mem["memory_key"], "rule")
        self.assertEqual(mem["fact_text"], fact)
        mem_id = mem["id"]

        # 2. Get active memories
        active = get_active_memories(category="rule")
        found = [m for m in active if m["id"] == mem_id]
        self.assertTrue(len(found) > 0)

        # 3. Format prompt block
        prompt_block = format_memories_for_prompt()
        self.assertIn("AGENT LONG-TERM MEMORY", prompt_block)
        self.assertIn("Stellantis certificates require 60-day expiry warnings", prompt_block)

        # 4. Delete memory
        deleted = delete_agent_memory(mem_id)
        self.assertTrue(deleted)

        # 5. Verify gone
        after = get_active_memories(category="rule")
        found_after = [m for m in after if m["id"] == mem_id]
        self.assertEqual(len(found_after), 0)


if __name__ == "__main__":
    unittest.main()
