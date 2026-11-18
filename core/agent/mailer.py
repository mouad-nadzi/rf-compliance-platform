"""
core/agent/mailer.py — Live SMTP Email Dispatcher

Provides a single shared, environment-configured SMTP transport for dispatching
approved `SEND_EMAIL` drafts and automated expiry alerts. Credentials are read
strictly from environment variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`,
`SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_USE_TLS`) and never hardcoded.
"""

from datetime import datetime
from email.message import EmailMessage
import json
import logging
import os
import smtplib
import ssl
from typing import Any, Dict

logger = logging.getLogger(__name__)


def send_email_via_smtp(recipient: str, subject: str, body_content: str) -> Dict[str, Any]:
    """
    Send a plain-text email over SMTP (STARTTLS on port 587 or SSL on port 465).
    Raises RuntimeError if SMTP credentials are missing or delivery fails.
    """
    from server.config import (
        SMTP_FROM,
        SMTP_HOST,
        SMTP_PASSWORD,
        SMTP_PORT,
        SMTP_USER,
        SMTP_USE_TLS,
    )

    to_addr = str(recipient or "").strip()
    if not to_addr:
        raise ValueError("Cannot send email: recipient address is empty.")

    # Prefer Administrator-configured sender credentials from the Alerting Automation Card
    cfg_sender = ""
    cfg_password = ""
    try:
        from core.agent.worker import _load_alerting_config
        alert_cfg = _load_alerting_config()
        cfg_sender = str(alert_cfg.get("sender_email") or "").strip()
        cfg_password = str(alert_cfg.get("smtp_password") or "").strip()
    except Exception:
        pass

    host = str(SMTP_HOST or "").strip()
    port = int(SMTP_PORT or 587)
    user = cfg_sender or str(SMTP_USER or "").strip()
    password = cfg_password or str(SMTP_PASSWORD or "").strip()
    from_addr = cfg_sender or str(SMTP_FROM or user).strip()

    if not host or not user or not password:
        raise RuntimeError(
            "Sender Email / SMTP credentials are not configured. "
            "The Administrator can set the Sender Email and App Password on the Alerting Automation Card."
        )

    msg = EmailMessage()
    msg["Subject"] = str(subject or "(No Subject)")
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(str(body_content or ""))

    context = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20, context=context) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                if SMTP_USE_TLS:
                    server.starttls(context=context)
                    server.ehlo()
                server.login(user, password)
                server.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            f"SMTP authentication failed for '{user}' on {host}:{port}. "
            "For Gmail, use a 16-character App Password (not your regular account password)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"SMTP delivery failed to '{to_addr}' via {host}:{port}: {exc}") from exc

    logger.info(f"SMTP email dispatched to '{to_addr}' via {host}:{port} (subject='{subject}').")
    return {
        "recipient": to_addr,
        "from": from_addr,
        "subject": subject,
        "host": host,
        "port": port,
    }


def dispatch_email_draft(draft_path: str) -> Dict[str, Any]:
    """
    Read a staged draft JSON file, transmit it via SMTP, and mark it dispatched.
    """
    if not draft_path or not os.path.exists(draft_path):
        raise FileNotFoundError(f"Email draft file not found: {draft_path}")

    with open(draft_path, encoding="utf-8") as fh:
        draft = json.load(fh)

    smtp_info = send_email_via_smtp(
        recipient=draft.get("recipient", ""),
        subject=draft.get("subject", ""),
        body_content=draft.get("body_content", ""),
    )

    draft["status"] = "dispatched"
    draft["dispatched_at"] = datetime.utcnow().isoformat()
    with open(draft_path, "w", encoding="utf-8") as fh:
        json.dump(draft, fh, indent=2, ensure_ascii=False)

    return {
        "draft_id": draft.get("draft_id"),
        "recipient": smtp_info["recipient"],
        "subject": smtp_info["subject"],
        "message": f"Email sent to {smtp_info['recipient']}.",
    }
