"""
core/agent/proposals.py — HITL Proposal Staging Manager

Stages side-effectful actions (database edits, email dispatches) as PENDING
proposals that require explicit human approval before execution. Proposals
persist as JSON files under data/agent/proposals/ so they survive restarts and
remain auditable.

Status flow: PENDING -> APPROVED | REJECTED. Nothing is executed until approved.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from core.agent.tools import AGENT_DATA_DIR

logger = logging.getLogger(__name__)

#: Directory where staged proposals are persisted.
PROPOSALS_DIR: str = os.path.join(AGENT_DATA_DIR, "proposals")

#: Valid proposal types.
VALID_TYPES = ("DB_EDIT", "SEND_EMAIL", "INGEST_DOCUMENT", "AGENT_ACTION")
#: Valid statuses.
VALID_STATUSES = ("PENDING", "APPROVED", "REJECTED")
#: Allowed transitions (nothing can leave APPROVED / REJECTED).
ALLOWED_TRANSITIONS: Dict[str, tuple] = {
    "PENDING": ("APPROVED", "REJECTED"),
    "APPROVED": (),
    "REJECTED": (),
}


class ProposalManager:
    """
    File-backed store for staged HITL proposals.

    Each proposal is one JSON file keyed by its UUID proposal_id. Methods are
    synchronous and self-contained (no external service dependencies).
    """

    def __init__(self, directory: Optional[str] = None) -> None:
        self.dir: str = directory or PROPOSALS_DIR
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, proposal_id: str) -> str:
        return os.path.join(self.dir, f"{proposal_id}.json")

    def _write(self, proposal: Dict[str, Any]) -> None:
        path = self._path(proposal["proposal_id"])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(proposal, fh, indent=2, ensure_ascii=False, default=str)

    def create_proposal(
        self,
        proposal_type: str,
        payload: Dict[str, Any],
        sql_preview: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage a new PENDING proposal.
        """
        if proposal_type not in VALID_TYPES:
            raise ValueError(f"Invalid proposal type '{proposal_type}'; must be one of {VALID_TYPES}.")

        resolved_session_id = session_id or (payload.get("session_id") if isinstance(payload, dict) else None)
        proposal_id = uuid.uuid4().hex[:12]
        proposal = {
            "id": proposal_id,
            "proposal_id": proposal_id,
            "type": proposal_type,
            "session_id": resolved_session_id,
            "payload": payload,
            "status": "PENDING",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sql_preview": sql_preview,
            "reasoning": sql_preview or (payload.get("description") if isinstance(payload, dict) else None) or "",
        }
        self._write(proposal)
        logger.info(f" Staged {proposal_type} proposal {proposal_id} (session={resolved_session_id}).")
        return self._enrich(proposal)

    def _enrich(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure proposal dict always exposes id, proposal_id, reasoning, and hydrated email draft details."""
        pid = proposal.get("proposal_id") or proposal.get("id") or ""
        proposal["proposal_id"] = pid
        proposal["id"] = pid
        payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}

        if proposal.get("type") == "SEND_EMAIL":
            draft_id = payload.get("draft_id")
            if draft_id and not payload.get("body_content"):
                try:
                    from core.agent.tools import DRAFTS_DIR
                    draft_path = os.path.join(DRAFTS_DIR, f"{draft_id}.json")
                    if os.path.exists(draft_path):
                        with open(draft_path, encoding="utf-8") as fh:
                            draft_data = json.load(fh)
                        payload["recipient"] = payload.get("recipient") or draft_data.get("recipient")
                        payload["subject"] = payload.get("subject") or draft_data.get("subject")
                        payload["body_content"] = draft_data.get("body_content", "")
                        proposal["payload"] = payload
                except Exception:
                    pass

        if not proposal.get("reasoning"):
            if proposal.get("type") == "SEND_EMAIL" and (payload.get("recipient") or payload.get("subject")):
                proposal["reasoning"] = (
                    f"To: {payload.get('recipient', 'N/A')} | Subject: {payload.get('subject', 'N/A')}"
                )
            else:
                proposal["reasoning"] = (
                    proposal.get("sql_preview")
                    or payload.get("description")
                    or payload.get("source_url")
                    or proposal.get("type", "")
                )
        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Return the proposal dict, or None if it does not exist."""
        path = self._path(proposal_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return self._enrich(json.load(fh))
        except Exception as exc:
            logger.warning(f" Could not read proposal {proposal_id}: {exc}")
            return None

    def list_pending_proposals(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return proposals currently in the PENDING state, scoped to session_id when provided."""
        pending = [p for p in self._list_all() if p.get("status") == "PENDING"]
        if session_id:
            return [
                p for p in pending
                if p.get("session_id") == session_id or p.get("session_id") is None
            ]
        return pending

    def list_all_proposals(self) -> List[Dict[str, Any]]:
        """Return all proposals, newest first."""
        return self._list_all()

    def _list_all(self) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        if not os.path.isdir(self.dir):
            return proposals
        for name in os.listdir(self.dir):
            if not name.endswith(".json"):
                continue
            proposal = self.get_proposal(name[:-5])
            if proposal:
                proposals.append(proposal)
        proposals.sort(key=lambda p: p.get("created_at", ""), reverse=True)
        return proposals

    def update_status(self, proposal_id: str, status: str) -> Dict[str, Any]:
        """
        Transition a proposal to a new status (enforcing allowed transitions).

        Raises:
            KeyError: proposal not found.
            ValueError: invalid status or disallowed transition.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'; must be one of {VALID_STATUSES}.")

        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Proposal {proposal_id} not found.")

        current = proposal.get("status", "PENDING")
        if status not in ALLOWED_TRANSITIONS.get(current, ()):
            raise ValueError(f"Cannot transition proposal {proposal_id} from '{current}' to '{status}'.")

        proposal["status"] = status
        self._write(proposal)
        logger.info(f" Proposal {proposal_id} status -> {status}.")
        return proposal


#: Shared singleton for the FastAPI routes.
proposal_manager = ProposalManager()