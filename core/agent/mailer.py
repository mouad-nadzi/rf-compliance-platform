"""
core/agent/mailer.py — Live Email Dispatcher (Google OAuth 2.0 Gmail REST API + SMTP Fallback)

Primary modern transport:
  - Google OAuth 2.0 + Gmail REST API (https://gmail.googleapis.com/gmail/v1/users/me/messages/send)
    using an Admin-authorized refresh token over HTTPS (no password stored).
Fallback transport:
  - Standard SMTP (STARTTLS on port 587 or SSL on port 465).
"""

import base64
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import html
import json
import logging
import os
import smtplib
import ssl
import time
from typing import Any, Dict, Tuple
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GOOGLE_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/gmail.send "
    "https://www.googleapis.com/auth/userinfo.email"
)


def _get_google_oauth_client() -> Tuple[str, str]:
    """Return effective (client_id, client_secret) from alerting_config.json or server.config."""
    from core.agent.worker import _load_alerting_config
    from server.config import GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET

    cfg = _load_alerting_config()
    client_id = str(cfg.get("google_client_id") or GOOGLE_OAUTH_CLIENT_ID or "").strip()
    client_secret = str(cfg.get("google_client_secret") or GOOGLE_OAUTH_CLIENT_SECRET or "").strip()
    return client_id, client_secret


def build_google_oauth_url(redirect_uri: str, state: str) -> str:
    """Build the Google OAuth 2.0 authorization URL requesting offline Gmail send access."""
    client_id, client_secret = _get_google_oauth_client()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Google OAuth Client ID and Client Secret are not configured. "
            "Paste them under 'Google OAuth App Credentials' on the Alerting card first."
        )
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_OAUTH_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_google_oauth_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    """Exchange a Google OAuth authorization code for tokens and persist the connected Gmail account."""
    from core.agent.worker import _load_alerting_config, _save_alerting_config

    client_id, client_secret = _get_google_oauth_client()
    if not client_id or not client_secret:
        raise RuntimeError("Google OAuth Client ID and Client Secret are missing.")

    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    if not resp.ok:
        raise RuntimeError(f"Google token exchange failed ({resp.status_code}): {resp.text}")

    token_data = resp.json()
    access_token = str(token_data.get("access_token") or "").strip()
    refresh_token = str(token_data.get("refresh_token") or "").strip()
    expires_in = int(token_data.get("expires_in") or 3600)

    if not access_token:
        raise RuntimeError("Google OAuth response did not include an access_token.")

    userinfo_resp = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if not userinfo_resp.ok:
        raise RuntimeError(f"Could not fetch Google account email ({userinfo_resp.status_code}): {userinfo_resp.text}")

    email = str(userinfo_resp.json().get("email") or "").strip().lower()
    if not email:
        raise RuntimeError("Google userinfo did not return an email address.")

    cfg = _load_alerting_config()
    cfg["google_access_token"] = access_token
    if refresh_token:
        cfg["google_refresh_token"] = refresh_token
    cfg["google_token_expires_at"] = time.time() + max(60, expires_in - 60)
    cfg["google_connected_email"] = email
    cfg["sender_email"] = email
    _save_alerting_config(cfg)

    logger.info(f"Google OAuth Gmail connected for '{email}'.")
    return {"email": email, "has_refresh_token": bool(cfg.get("google_refresh_token"))}


def disconnect_google_oauth() -> None:
    """Clear stored Google OAuth tokens from alerting_config.json."""
    from core.agent.worker import _load_alerting_config, _save_alerting_config

    cfg = _load_alerting_config()
    cfg["google_refresh_token"] = ""
    cfg["google_access_token"] = ""
    cfg["google_token_expires_at"] = 0.0
    cfg["google_connected_email"] = ""
    _save_alerting_config(cfg)
    logger.info("Disconnected Google OAuth Gmail sender.")


def _get_valid_gmail_access_token() -> Tuple[str, str]:
    """Return a valid (access_token, sender_email), refreshing via refresh_token if expired."""
    from core.agent.worker import _load_alerting_config, _save_alerting_config

    cfg = _load_alerting_config()
    client_id, client_secret = _get_google_oauth_client()
    refresh_token = str(cfg.get("google_refresh_token") or "").strip()
    sender_email = str(cfg.get("google_connected_email") or cfg.get("sender_email") or "").strip()

    if not refresh_token or not client_id or not client_secret:
        raise RuntimeError("Google OAuth is not connected.")

    access_token = str(cfg.get("google_access_token") or "").strip()
    expires_at = float(cfg.get("google_token_expires_at") or 0.0)

    if access_token and time.time() < expires_at:
        return access_token, sender_email

    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Google OAuth token refresh failed ({resp.status_code}). "
            "Reconnect Gmail via Google OAuth on the Alerting card."
        )

    data = resp.json()
    new_token = str(data.get("access_token") or "").strip()
    expires_in = int(data.get("expires_in") or 3600)
    if not new_token:
        raise RuntimeError("Google token refresh did not return an access_token.")

    cfg["google_access_token"] = new_token
    cfg["google_token_expires_at"] = time.time() + max(60, expires_in - 60)
    _save_alerting_config(cfg)
    return new_token, sender_email


def _build_email_message(
    sender_email: str,
    recipient: str,
    subject: str,
    body_content: str,
    attachment_path: str | None = None,
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = str(subject or "(No Subject)")
    if sender_email:
        msg["From"] = sender_email
    msg["To"] = recipient
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender_email.split("@")[-1] if "@" in sender_email else None)

    plain_text = str(body_content or "")
    msg.set_content(plain_text)

    escaped_html = "<br>\n".join(html.escape(line) for line in plain_text.splitlines())
    html_body = (
        "<!doctype html><html><body style=\"font-family:Segoe UI,Arial,sans-serif;color:#0f172a;font-size:14px;line-height:1.55;\">"
        f"<div style=\"max-width:640px;padding:16px 20px;border:1px solid #e2e8f0;border-radius:8px;\">{escaped_html}</div>"
        "</body></html>"
    )
    msg.add_alternative(html_body, subtype="html")

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as fh:
            data = fh.read()
        filename = os.path.basename(attachment_path)
        subtype = "pdf" if filename.lower().endswith(".pdf") else "octet-stream"
        maintype = "application"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

    return msg


def send_email_via_gmail_api(
    recipient: str,
    subject: str,
    body_content: str,
    attachment_path: str | None = None,
) -> Dict[str, Any]:
    """Send an email over HTTPS using the connected Google OAuth Gmail REST API."""
    to_addr = str(recipient or "").strip()
    if not to_addr:
        raise ValueError("Cannot send email: recipient address is empty.")

    access_token, sender_email = _get_valid_gmail_access_token()
    msg = _build_email_message(sender_email, to_addr, subject, body_content, attachment_path=attachment_path)

    raw_b64 = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    resp = requests.post(
        GMAIL_SEND_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={"raw": raw_b64},
        timeout=20,
    )
    if not resp.ok:
        raise RuntimeError(f"Gmail API send failed ({resp.status_code}): {resp.text}")

    gmail_resp_json = resp.json() if resp.content else {}
    gmail_msg_id = gmail_resp_json.get("id")
    logger.info(
        f"Gmail API email sent from '{sender_email}' to '{to_addr}' "
        f"(subject='{subject}', attachment={bool(attachment_path)}, gmail_id={gmail_msg_id})."
    )
    return {
        "recipient": to_addr,
        "from": sender_email,
        "subject": subject,
        "attachment": os.path.basename(attachment_path) if attachment_path else None,
        "gmail_message_id": gmail_msg_id,
        "transport": "gmail_oauth_api",
    }


def send_email_via_smtp(
    recipient: str,
    subject: str,
    body_content: str,
    attachment_path: str | None = None,
) -> Dict[str, Any]:
    """
    Send an email using Google OAuth 2.0 + Gmail REST API when connected,
    falling back to SMTP (STARTTLS on port 587 or SSL on port 465) if configured.
    """
    from core.agent.worker import _load_alerting_config
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

    # Priority 1: Connected Google OAuth 2.0 + Gmail REST API
    cfg = _load_alerting_config()
    client_id, client_secret = _get_google_oauth_client()
    if cfg.get("google_refresh_token") and client_id and client_secret:
        return send_email_via_gmail_api(to_addr, subject, body_content, attachment_path=attachment_path)

    # Priority 2: SMTP credentials (from Admin card or .env)
    cfg_sender = str(cfg.get("sender_email") or "").strip()
    cfg_password = str(cfg.get("smtp_password") or "").strip()

    host = str(SMTP_HOST or "").strip()
    port = int(SMTP_PORT or 587)
    user = cfg_sender or str(SMTP_USER or "").strip()
    password = cfg_password or str(SMTP_PASSWORD or "").strip()
    if "gmail.com" in host.lower():
        password = password.replace(" ", "")
    from_addr = cfg_sender or str(SMTP_FROM or user).strip()

    if not host or not user or not password:
        raise RuntimeError(
            "Email sender is not connected. The Administrator can click "
            "'Connect Gmail via Google OAuth' on the Alerting Automation Card."
        )

    msg = _build_email_message(from_addr, to_addr, subject, body_content, attachment_path=attachment_path)

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
            "Connect Gmail via Google OAuth or use a 16-letter Google App Password."
        ) from exc
    except smtplib.SMTPServerDisconnected as exc:
        raise RuntimeError(
            f"SMTP server {host}:{port} closed the connection during authentication for '{user}'. "
            "Use 'Connect Gmail via Google OAuth' on the Alerting card (recommended) or a 16-letter Google App Password."
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
        "transport": "smtp",
    }


def dispatch_email_draft(draft_path: str) -> Dict[str, Any]:
    """
    Read a staged draft JSON file, transmit it via Gmail API or SMTP, and mark it dispatched.
    """
    if not draft_path or not os.path.exists(draft_path):
        raise FileNotFoundError(f"Email draft file not found: {draft_path}")

    with open(draft_path, encoding="utf-8") as fh:
        draft = json.load(fh)

    info = send_email_via_smtp(
        recipient=draft.get("recipient", ""),
        subject=draft.get("subject", ""),
        body_content=draft.get("body_content", ""),
        attachment_path=draft.get("attachment_path"),
    )

    draft["status"] = "dispatched"
    draft["transport"] = info.get("transport", "smtp")
    draft["dispatched_at"] = datetime.utcnow().isoformat()
    with open(draft_path, "w", encoding="utf-8") as fh:
        json.dump(draft, fh, indent=2, ensure_ascii=False)

    return {
        "draft_id": draft.get("draft_id"),
        "recipient": info["recipient"],
        "subject": info["subject"],
        "transport": info.get("transport"),
        "message": f"Email sent to {info['recipient']}.",
    }
