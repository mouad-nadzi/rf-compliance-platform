"""
core/workflow/dag_engine.py — Modular DAG (Directed Acyclic Graph) Workflow Engine

Provides graph data structures, topological dependency sorting with cycle detection,
dynamic inter-node state passing, and a production-grade execution runner.
"""

from collections import deque
from dataclasses import dataclass, field, asdict
import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

WORKFLOWS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "workflows",
)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class DAGNode:
    node_id: str
    tool_name: str  # e.g., "web_downloader", "discover_pdfs", "glm_ocr", "extract_metadata", "persist_database", "filter_data", "email_drafting", "data_converter"
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # list of node_ids that must complete first

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGNode":
        return cls(
            node_id=str(data.get("node_id") or uuid.uuid4().hex[:8]),
            tool_name=str(data.get("tool_name", "web_downloader")),
            description=str(data.get("description", "")),
            params=dict(data.get("params", {})),
            depends_on=list(data.get("depends_on", [])),
        )


@dataclass
class DAGWorkflow:
    workflow_id: str
    title: str
    description: str
    nodes: List[DAGNode] = field(default_factory=list)
    enabled: bool = True
    require_approval: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "title": self.title,
            "description": self.description,
            "enabled": self.enabled,
            "require_approval": self.require_approval,
            "nodes": [n.to_dict() for n in self.nodes],
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGWorkflow":
        nodes_raw = data.get("nodes", [])
        nodes = [DAGNode.from_dict(n) for n in nodes_raw]
        return cls(
            workflow_id=str(data.get("workflow_id") or f"wf_{uuid.uuid4().hex[:8]}"),
            title=str(data.get("title", "Untitled Workflow")),
            description=str(data.get("description", "")),
            enabled=bool(data.get("enabled", True)),
            require_approval=bool(data.get("require_approval", False)),
            nodes=nodes,
            custom_params=dict(data.get("custom_params", {})),
        )


@dataclass
class DAGRunResult:
    run_id: str
    workflow_id: str
    status: str  # "success", "failed", "paused_hitl"
    executed_nodes: List[str] = field(default_factory=list)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DAGExecutor:
    """
    Topological execution runner for DAG workflows.
    Validates graph acyclicity, passes outputs between dependent nodes,
    and runs concrete tool handlers safely.
    """

    def __init__(self, workflow: DAGWorkflow):
        self.workflow = workflow
        self.nodes_map: Dict[str, DAGNode] = {n.node_id: n for n in workflow.nodes}

    def compute_topological_order(self) -> List[str]:
        """
        Computes topological execution order using Kahn's algorithm.
        Raises ValueError if a cycle is detected.
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes_map}
        adj_list: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes_map}

        for node_id, node in self.nodes_map.items():
            for parent_id in node.depends_on:
                if parent_id in self.nodes_map:
                    adj_list[parent_id].append(node_id)
                    in_degree[node_id] += 1
                else:
                    logger.warning(f" Node '{node_id}' references unknown parent dependency '{parent_id}'. Skipping parent.")

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes_map):
            raise ValueError(f"Cycle detected in workflow DAG '{self.workflow.workflow_id}'!")

        return order

    def execute(self, status_callback: Optional[Callable[[str, str], None]] = None) -> DAGRunResult:
        """
        Executes all nodes in topological order, wiring outputs between parent and child nodes.
        """
        start_time = time.time()
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        result = DAGRunResult(
            run_id=run_id,
            workflow_id=self.workflow.workflow_id,
            status="success",
        )

        try:
            order = self.compute_topological_order()
        except Exception as exc:
            result.status = "failed"
            result.errors["graph"] = str(exc)
            return result

        context: Dict[str, Any] = {}

        for node_id in order:
            node = self.nodes_map[node_id]
            if status_callback:
                status_callback("workflow_step", f"Executing node '{node_id}' ({node.tool_name})...")

            # Merge node static params with context from upstream parent dependencies
            effective_params = dict(node.params)
            for parent_id in node.depends_on:
                parent_out = result.node_outputs.get(parent_id)
                if isinstance(parent_out, dict):
                    # Wire standard keys if missing in node params
                    if "file_path" in parent_out and "file_path" not in effective_params:
                        effective_params["file_path"] = parent_out["file_path"]
                    if "path" in parent_out and "file_path" not in effective_params:
                        effective_params["file_path"] = parent_out["path"]
                    if "url" in parent_out and "url" not in effective_params:
                        effective_params["url"] = parent_out["url"]
                    if "data" in parent_out and "data" not in effective_params:
                        effective_params["data"] = parent_out["data"]
                    if "verified_urls" in parent_out and "verified_urls" not in effective_params:
                        effective_params["verified_urls"] = parent_out["verified_urls"]

            try:
                out = self._dispatch_node_tool(node.tool_name, effective_params)
                result.executed_nodes.append(node_id)
                result.node_outputs[node_id] = out
            except Exception as node_exc:
                logger.error(f" Error executing node '{node_id}' ({node.tool_name}): {node_exc}")
                result.status = "failed"
                result.errors[node_id] = str(node_exc)
                break

        result.execution_time_seconds = round(time.time() - start_time, 3)
        return result

    def _dispatch_node_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Route tool_name to concrete implementation."""
        from core.agent.tools import WebDownloaderTool, DataConverterTool, EmailDraftingTool

        if tool_name == "web_downloader":
            t = WebDownloaderTool()
            return t.execute(**params)

        elif tool_name == "data_converter":
            t = DataConverterTool()
            return t.execute(**params)

        elif tool_name == "email_drafting":
            t = EmailDraftingTool()
            return t.execute(**params)

        elif tool_name == "discover_pdfs":
            from core.agent.scraper import discover_pdf_urls
            from storage.database import SessionLocal
            url = str(params.get("url") or "").strip()
            session = SessionLocal()
            try:
                res = discover_pdf_urls(url, db_session=session, cookie_header=params.get("cookie_header"))
                return {"verified_urls": res.verified_urls, "skipped_existing": res.skipped_existing}
            finally:
                session.close()

        elif tool_name == "glm_ocr":
            from core.ocr.glm_ocr import run_glm_ocr
            file_path = str(params.get("file_path") or "").strip()
            ocr_res = run_glm_ocr(file_path)
            return {"ocr_text": ocr_res.get("text", ""), "file_path": file_path}

        elif tool_name == "extract_metadata":
            from core.extractor import extract_certificate_data
            ocr_text = str(params.get("ocr_text") or params.get("text") or "").strip()
            metadata = extract_certificate_data(ocr_text)
            return {"metadata": metadata.model_dump() if hasattr(metadata, "model_dump") else str(metadata)}

        elif tool_name == "persist_database":
            from core.extractor import save_certificate_to_db, extract_certificate_data
            metadata_raw = params.get("metadata")
            file_path = str(params.get("file_path") or "").strip()
            url = str(params.get("url") or "").strip()
            if isinstance(metadata_raw, str):
                metadata = extract_certificate_data(metadata_raw)
            else:
                metadata = metadata_raw
            if metadata:
                rec = save_certificate_to_db(metadata, cert_link=url, file_name=os.path.basename(file_path))
                return {"certificate_id": getattr(rec, "certificate_id", None), "status": "persisted"}
            return {"status": "skipped"}

        elif tool_name == "filter_data":
            data = params.get("data", [])
            field_name = params.get("field")
            operator = str(params.get("operator", "==")).strip()
            value = params.get("value")

            if not isinstance(data, list):
                return {"data": []}

            filtered = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                val = row.get(field_name)
                if operator == "==" and str(val) == str(value):
                    filtered.append(row)
                elif operator == "!=" and str(val) != str(value):
                    filtered.append(row)
                elif operator == ">" and float(val or 0) > float(value or 0):
                    filtered.append(row)
                elif operator == "<" and float(val or 0) < float(value or 0):
                    filtered.append(row)
                elif operator in ("contains", "in") and str(value).lower() in str(val).lower():
                    filtered.append(row)
            return {"data": filtered, "count": len(filtered)}

        else:
            raise ValueError(f"Unknown DAG node tool: '{tool_name}'")


def execute_dag_workflow(workflow: DAGWorkflow, status_callback: Optional[Callable[[str, str], None]] = None) -> DAGRunResult:
    executor = DAGExecutor(workflow)
    return executor.execute(status_callback=status_callback)
