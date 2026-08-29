"""
tests/test_dag_engine.py — Unit & Integration Tests for DAG Workflow Engine
"""

import unittest
from core.workflow.dag_engine import (
    DAGNode,
    DAGWorkflow,
    DAGExecutor,
    execute_dag_workflow,
)


class TestDAGEngine(unittest.TestCase):

    def test_topological_sorting_valid_dag(self):
        """Verify Kahn's algorithm correctly sorts nodes in dependency order."""
        nodes = [
            DAGNode(node_id="node_c", tool_name="data_converter", depends_on=["node_b"]),
            DAGNode(node_id="node_a", tool_name="web_downloader", depends_on=[]),
            DAGNode(node_id="node_b", tool_name="filter_data", depends_on=["node_a"]),
        ]
        wf = DAGWorkflow(workflow_id="wf_test_valid", title="Valid DAG Test", description="Test DAG", nodes=nodes)
        executor = DAGExecutor(wf)
        order = executor.compute_topological_order()

        self.assertEqual(order, ["node_a", "node_b", "node_c"])

    def test_cycle_detection_invalid_dag(self):
        """Verify cycle detection raises ValueError on cyclic graph (A -> B -> A)."""
        nodes = [
            DAGNode(node_id="node_a", tool_name="web_downloader", depends_on=["node_b"]),
            DAGNode(node_id="node_b", tool_name="filter_data", depends_on=["node_a"]),
        ]
        wf = DAGWorkflow(workflow_id="wf_test_cycle", title="Cycle DAG Test", description="Cyclic DAG", nodes=nodes)
        executor = DAGExecutor(wf)

        with self.assertRaises(ValueError):
            executor.compute_topological_order()

    def test_dag_execution_state_wiring(self):
        """Verify data_converter node executes cleanly receiving input data."""
        nodes = [
            DAGNode(
                node_id="n1",
                tool_name="filter_data",
                params={
                    "data": [
                        {"supplier": "Espressif", "status": "active"},
                        {"supplier": "Other", "status": "inactive"},
                    ],
                    "field": "supplier",
                    "operator": "==",
                    "value": "Espressif",
                },
            ),
            DAGNode(
                node_id="n2",
                tool_name="data_converter",
                params={"format": "json"},
                depends_on=["n1"],
            ),
        ]
        wf = DAGWorkflow(workflow_id="wf_test_exec", title="Exec DAG Test", description="Exec test", nodes=nodes)
        res = execute_dag_workflow(wf)

        self.assertEqual(res.status, "success")
        self.assertIn("n1", res.executed_nodes)
        self.assertIn("n2", res.executed_nodes)
        self.assertEqual(res.node_outputs["n1"]["count"], 1)
        self.assertIn("Espressif", res.node_outputs["n2"])


if __name__ == "__main__":
    unittest.main()
