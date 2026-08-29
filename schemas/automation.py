"""
schemas/automation.py — Automation Configuration Data Model

Defines the standardized AutomationConfig schema used across background workers,
API endpoints, and UI rendering components. Supports both static system jobs
(e.g., Autonomous Scheduler) and dynamic agent-generated custom automations.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AutomationConfig:
    id: str
    title: str
    description: str
    enabled: bool = True
    require_approval: bool = True
    interval_hours: int = 24
    cron_expression: Optional[str] = None
    action_type: str = "pdf_discovery_and_ingest"
    target_urls: List[str] = field(default_factory=list)
    running: bool = False
    next_run_time: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutomationConfig":
        """Instantiate AutomationConfig from a dict with defaults."""
        return cls(
            id=str(data.get("id", "autonomous_scheduler")),
            title=str(data.get("title", "Autonomous Scheduler Configuration")),
            description=str(
                data.get(
                    "description",
                    "Configure background discovery intervals and staged approval behavior for the autonomous ingestion engine.",
                )
            ),
            enabled=bool(data.get("enabled", True)),
            require_approval=bool(data.get("require_approval", True)),
            interval_hours=max(1, int(data.get("interval_hours", 24))),
            cron_expression=data.get("cron_expression"),
            action_type=str(data.get("action_type", "pdf_discovery_and_ingest")),
            target_urls=list(data.get("target_urls", [])),
            running=bool(data.get("running", False)),
            next_run_time=data.get("next_run_time"),
            custom_params=dict(data.get("custom_params", {})),
        )
