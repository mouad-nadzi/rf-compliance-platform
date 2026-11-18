"""
tests/test_session_and_performance.py — Unit Tests for Prefix Caching Architecture,
Bounded Conversation History, Deterministic SQL Table Rendering, and Session-Scoped Proposals.
"""

import tempfile
import unittest

from core.llm.qwen3_8_27b import Qwen3_8_27BEngine
from core.prompts import (
    config_unified_agent_system_prompt,
    config_sql_system_prompt,
    config_sql_result_synthesis_prompt,
    config_qa_system_prompt,
)
from core.rag.sql_engine import _build_markdown_table
from core.utils.history import format_conversation_history
from core.agent.proposals import ProposalManager


class TestSessionAndPerformanceArchitecture(unittest.TestCase):

    def test_shared_memory_prefix_at_start_of_all_prompts(self):
        """All config_* prompts must place [AGENT LONG-TERM MEMORY] at token 0 so prefix checkpoints match."""
        p_router = config_unified_agent_system_prompt()
        p_sql = config_sql_system_prompt("schema_stub")
        p_synth = config_sql_result_synthesis_prompt()
        p_qa = config_qa_system_prompt()

        end_marker = "[END LONG-TERM MEMORY]"
        self.assertIn(end_marker, p_router)
        prefix_end = p_router.find(end_marker) + len(end_marker)
        shared_prefix = p_router[:prefix_end]

        self.assertTrue(p_sql.startswith(shared_prefix))
        self.assertTrue(p_synth.startswith(shared_prefix))
        self.assertTrue(p_qa.startswith(shared_prefix))

    def test_qwen38_prompt_segment_splitting_and_thinking_header(self):
        """Qwen3_8_27BEngine must split memory + history turns into append-only prefix segments and use <think>\\n\\n</think>\\n\\n for disable_thinking=True."""
        engine = Qwen3_8_27BEngine()
        sys_p = config_unified_agent_system_prompt()
        hist = format_conversation_history([
            {"role": "user", "content": "How many certificates from Brazil?"},
            {"role": "assistant", "content": "There are 112 certificates."},
        ])
        user_p = f"{hist}\n\nUSER MESSAGE: and from Argentina?"

        segs_1, suffix_1 = engine._split_prompt_segments(
            system_prompt=sys_p,
            user_prompt=user_p,
            disable_thinking=True,
            response_prefix='{"tool": "',
        )
        # Base memory segment + 2 history turn segments = 3 prefix segments
        self.assertEqual(len(segs_1), 3)
        self.assertIn("[END LONG-TERM MEMORY]", segs_1[0])
        self.assertIn("User: How many certificates from Brazil?", segs_1[1])
        self.assertIn("Assistant: There are 112 certificates.", segs_1[2])
        self.assertIn('<|im_start|>assistant\n<think>\n\n</think>\n\n{"tool": "', suffix_1)

        # Hop 2 (SQL generator) with the same history must produce the EXACT same prefix segments
        segs_2, _ = engine._split_prompt_segments(
            system_prompt=config_sql_system_prompt("schema"),
            user_prompt=f"{hist}\n\nUSER QUESTION: and from Argentina?",
            disable_thinking=True,
            response_prefix='{"sql": "',
        )
        self.assertEqual(segs_1, segs_2)

    def test_bounded_history_compaction(self):
        """Oversized turns and excess turn counts must be deterministically bounded."""
        huge_table = "Summary line\n" + ("| A | B |\n" * 400)
        history = [
            {"role": "user", "content": f"Question {i}"} if i % 2 == 0 else {"role": "assistant", "content": huge_table}
            for i in range(24)
        ]
        formatted = format_conversation_history(history, max_turns=6, max_chars_per_turn=500)
        self.assertNotIn("Question 0", formatted)
        self.assertIn("Question 22", formatted)
        self.assertIn("[truncated in history]", formatted)

    def test_deterministic_markdown_table_rendering(self):
        """_build_markdown_table must render GFM tables with clean headers and Indefinite sentinel handling."""
        rows = [
            {"component": "VSM", "supplier": "Continental", "exp_date": "2999-01-01"},
            {"component": "IM3C", "supplier": "Valeo", "exp_date": "2026-08-15"},
        ]
        md = _build_markdown_table(rows, ["component", "supplier", "exp_date"])
        self.assertIn("| Component | Supplier | Exp Date |", md)
        self.assertIn("| VSM | Continental | Indefinite |", md)
        self.assertIn("| IM3C | Valeo | 2026-08-15 |", md)

    def test_session_scoped_proposals(self):
        """Proposals created in session A must not leak into session B's pending list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProposalManager(directory=tmpdir)
            p_a = pm.create_proposal(
                "AGENT_ACTION",
                {"action": "delete_record", "description": "Delete in A"},
                session_id="sess_a",
            )
            p_b = pm.create_proposal(
                "AGENT_ACTION",
                {"action": "delete_record", "description": "Delete in B"},
                session_id="sess_b",
            )

            pending_a = pm.list_pending_proposals(session_id="sess_a")
            pending_b = pm.list_pending_proposals(session_id="sess_b")

            self.assertEqual([p["proposal_id"] for p in pending_a], [p_a["proposal_id"]])
            self.assertEqual([p["proposal_id"] for p in pending_b], [p_b["proposal_id"]])


if __name__ == "__main__":
    unittest.main()
