"""
server/auth.py — Authentication, 3-Tier RBAC & Agent User Context

Implements zero-dependency cryptographic password hashing (PBKDF2-SHA256) and
stateless HMAC-SHA256 session tokens transported via HttpOnly cookies (supporting
native EventSource SSE streams) or Authorization Bearer headers.

Roles:
  - admin: Unique Administrator (full access, user management, invite links, sender email config)
  - editor: Allow-to-Edit user (database writes, ingestion, automations, email approvals)
  - viewer: Read-Only user (view tables and run read-only Q&A chat)
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import Request
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

VALID_ROLES = ("admin", "editor", "viewer")
INVITABLE_ROLES = ("editor", "viewer")
AUTH_COOKIE_NAME = "rf_access_token"
_PBKDF2_ITERATIONS = 260_000

_CACHED_SECRET: Optional[str] = None


def _get_signing_secret() -> bytes:
    """Return a persistent HMAC signing secret from env or data/agent/.auth_secret."""
    global _CACHED_SECRET
    if _CACHED_SECRET:
        return _CACHED_SECRET.encode("utf-8")

    from server.config import AUTH_SECRET_KEY, BASE_DIR

    env_secret = str(AUTH_SECRET_KEY or "").strip()
    if env_secret:
        _CACHED_SECRET = env_secret
        return env_secret.encode("utf-8")

    secret_path = os.path.join(BASE_DIR, "data", "agent", ".auth_secret")
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    if os.path.exists(secret_path):
        try:
            with open(secret_path, encoding="utf-8") as fh:
                stored = fh.read().strip()
            if stored:
                _CACHED_SECRET = stored
                return stored.encode("utf-8")
        except Exception:
            pass

    generated = secrets.token_urlsafe(48)
    with open(secret_path, "w", encoding="utf-8") as fh:
        fh.write(generated)
    _CACHED_SECRET = generated
    return generated.encode("utf-8")


def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-SHA256 with a random 16-byte salt."""
    raw = str(password or "")
    if len(raw) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", raw.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored_hash: Optional[str]) -> bool:
    """Verify a plaintext password against a stored PBKDF2-SHA256 hash."""
    if not password or not stored_hash:
        return False
    try:
        parts = stored_hash.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = parts[2]
        expected_hex = parts[3]
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def create_access_token(user_id: str, email: str, role: str, ttl_hours: Optional[int] = None) -> str:
    """Create a signed HMAC-SHA256 session token."""
    from server.config import AUTH_TOKEN_TTL_HOURS

    ttl = int(ttl_hours if ttl_hours is not None else AUTH_TOKEN_TTL_HOURS)
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": now,
        "exp": now + max(1, ttl) * 3600,
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
    sig = hmac.new(_get_signing_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    return f"{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a session token; returns payload dict or None if invalid/expired."""
    if not token or "." not in token:
        return None
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        expected_sig = hmac.new(_get_signing_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
        expected_b64 = base64.urlsafe_b64encode(expected_sig).decode("ascii").rstrip("=")
        if not hmac.compare_digest(sig_b64, expected_b64):
            return None
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def extract_token_from_request(request: Request) -> Optional[str]:
    """Extract token from Authorization: Bearer header or HttpOnly session cookie."""
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        tok = auth_header[7:].strip()
        if tok:
            return tok
    cookie_tok = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_tok:
        return cookie_tok.strip()
    return None


def get_authenticated_user(request: Request, db: Session) -> Optional[Dict[str, Any]]:
    """Resolve and verify the currently authenticated active user from the request."""
    from schemas.extraction import UserAccount

    token = extract_token_from_request(request)
    if not token:
        return None
    claims = decode_access_token(token)
    if not claims:
        return None
    user_id = claims.get("sub")
    if not user_id:
        return None

    user = db.query(UserAccount).filter(UserAccount.id == user_id, UserAccount.status == "active").first()
    if not user:
        return None
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "status": user.status,
    }


def format_user_context_for_prompt(user: Optional[Dict[str, Any]]) -> str:
    """
    Build a concise user-session & permission block injected at the tail of user
    prompts (preserving the global 1014-token [AGENT LONG-TERM MEMORY] prefix cache).
    """
    if not user:
        return ""
    role = str(user.get("role") or "viewer").lower()
    name = str(user.get("full_name") or user.get("email") or "User")
    email = str(user.get("email") or "")

    if role == "admin":
        role_label = "Administrator (Unique Admin)"
        rights = (
            "Full platform access: read & query compliance data, insert/update/delete records, "
            "modify schemas, run/configure automations, approve/reject email & DB proposals, "
            "configure the alert Sender Email, and manage platform users."
        )
    elif role == "editor":
        role_label = "Editor (Allow to Edit)"
        rights = (
            "Can read & query compliance data, insert/update/delete database records, "
            "ingest documents, run/schedule automations, and approve/reject proposals. "
            "CANNOT manage users or modify the global alert Sender Email configuration (Administrator only)."
        )
    else:
        role_label = "Viewer (Read-Only)"
        rights = (
            "Read-only access: can ask compliance questions, search certificates, and view tables. "
            "CANNOT create, update, or delete records, modify schemas, ingest documents, "
            "trigger automations, approve/reject proposals, or send emails."
        )

    return (
        f"[ACTIVE USER SESSION & PERMISSIONS]\n"
        f"- User: {name} ({email})\n"
        f"- Role: {role_label}\n"
        f"- Permissions: {rights}\n"
    )
