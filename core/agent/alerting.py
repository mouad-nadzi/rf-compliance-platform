"""
core/agent/alerting.py — Automated Certificate Expiry Alerting Engine

Queries the canonical `certificates` table dynamically joined with `components`,
`responsibles`, and `countries` to detect certificates approaching expiration
within a configurable lead-time window (e.g. 30, 60, or 90 days).

Emits in-app notifications (`notifications` table), drafts consolidated renewal
alert emails per responsible engineer (or fallback admin for unassigned
components) via `EmailDraftingTool`, optionally stages HITL `SEND_EMAIL`
proposals, and records alert state in `expiry_alert_logs` for idempotency.
"""

from collections import defaultdict
from datetime import date, datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _clamp_alert_days(days: int) -> int:
    return max(1, min(365, int(days)))


def query_expiring_certificates(
    db_session: Session,
    alert_days_before: int = 30,
) -> List[Dict[str, Any]]:
    """
    Query `certificates` as the canonical source of truth, dynamically enriched
    with responsible engineer contact info (`components` -> `responsibles`) and
    market priority (`countries`).
    """
    window_days = _clamp_alert_days(alert_days_before)

    sql = text(
        """
        SELECT
            c.certificate_id,
            c.component,
            c.supplier,
            c.country,
            c.certif_number,
            c.authority,
            c.issue_date,
            c.exp_date,
            c.cert_link,
            (c.exp_date - CURRENT_DATE) AS days_remaining,
            comp.id_responsible AS component_owner_email,
            comp.docinfo_link,
            r.firstname,
            r.lastname,
            COALESCE(
                NULLIF(TRIM(r.email), ''),
                NULLIF(TRIM(comp.id_responsible), '')
            ) AS responsible_email,
            r.phone AS responsible_phone,
            cnt.stla_current_priority AS market_priority
        FROM certificates c
        LEFT JOIN LATERAL (
            SELECT id_responsible, docinfo_link
            FROM components comp
            WHERE LOWER(TRIM(c.component)) = LOWER(TRIM(comp.componentname))
            ORDER BY comp.id DESC
            LIMIT 1
        ) comp ON TRUE
        LEFT JOIN LATERAL (
            SELECT firstname, lastname, email, phone
            FROM responsibles r
            WHERE LOWER(TRIM(comp.id_responsible)) = LOWER(TRIM(r.email))
            ORDER BY r.id DESC
            LIMIT 1
        ) r ON TRUE
        LEFT JOIN LATERAL (
            SELECT stla_current_priority
            FROM countries cnt
            WHERE LOWER(TRIM(c.country)) = LOWER(TRIM(cnt.country))
            ORDER BY cnt.id DESC
            LIMIT 1
        ) cnt ON TRUE
        WHERE c.exp_date IS NOT NULL
          AND c.exp_date >= CURRENT_DATE
          AND c.exp_date <= CURRENT_DATE + make_interval(days => :alert_days)
        ORDER BY c.exp_date ASC, c.country ASC, c.component ASC
        """
    )

    rows = db_session.execute(sql, {"alert_days": window_days}).mappings().all()
    results: List[Dict[str, Any]] = []
    for row in rows:
        first = (row.get("firstname") or "").strip()
        last = (row.get("lastname") or "").strip()
        full_name = f"{first} {last}".strip() or None
        exp_val = row.get("exp_date")
        if isinstance(exp_val, datetime):
            exp_val = exp_val.date()
        results.append(
            {
                "certificate_id": row.get("certificate_id"),
                "component": row.get("component") or "",
                "supplier": row.get("supplier") or "",
                "country": row.get("country") or "",
                "certif_number": row.get("certif_number") or "",
                "authority": row.get("authority") or "",
                "exp_date": exp_val,
                "cert_link": row.get("cert_link") or "",
                "days_remaining": int(row.get("days_remaining") or 0),
                "responsible_name": full_name,
                "responsible_email": (row.get("responsible_email") or "").strip() or None,
                "responsible_phone": (row.get("responsible_phone") or "").strip() or None,
                "docinfo_link": (row.get("docinfo_link") or "").strip() or None,
                "market_priority": (row.get("market_priority") or "").strip() or None,
            }
        )
    return results


def _build_email_body(
    recipient_name: Optional[str],
    certs: List[Dict[str, Any]],
    alert_days_before: int,
    is_fallback: bool,
) -> Tuple[str, str]:
    """Build concise subject and body matching the Stellantis renewal alert template."""
    if is_fallback:
        subject = "URGENT! RF Certification Renewal Alert (Unassigned Components)"
        greeting = "Dear Compliance Administration Team,"
    else:
        subject = "URGENT! RF Certification Renewal Alert"
        greeting = f"Dear {recipient_name or 'Responsible Engineer'},"

    lines = [
        greeting,
        "",
        "Please find attached the list of certificates that need to be renewed as soon as possible.",
        "Please ensure that the renewals are provided before the expiration dates.",
        "",
        "-- Stellantis RF Compliance Platform",
    ]
    return subject, "\n".join(lines)


def _mark_draft_dispatched(draft_path: str) -> None:
    if not draft_path or not os.path.exists(draft_path):
        return
    from core.agent.mailer import dispatch_email_draft

    dispatch_email_draft(draft_path)


def run_expiry_alert_scan(
    alert_days_before: int = 30,
    require_approval: bool = True,
    fallback_email: Optional[str] = None,
    force_rescan: bool = False,
    db_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Execute an expiration alert scan against the live `certificates` table.

    Deduplicates against `expiry_alert_logs` on `(certificate_id, exp_date, alert_window_days)`
    unless `force_rescan=True`. Emits in-app notifications, stages/dispatches
    consolidated email drafts per responsible engineer (plus fallback admin for
    unassigned components), and logs all emitted alerts.
    """
    from core.agent.proposals import proposal_manager
    from core.agent.tools import EmailDraftingTool
    from schemas.extraction import ExpiryAlertLog, NotificationItem
    from server.config import EXPIRY_ALERT_FALLBACK_EMAIL
    from storage.database import SessionLocal

    window_days = _clamp_alert_days(alert_days_before)
    effective_fallback = (fallback_email or EXPIRY_ALERT_FALLBACK_EMAIL or "").strip()

    owns_session = db_session is None
    session = db_session or SessionLocal()

    try:
        all_matches = query_expiring_certificates(
            session,
            alert_days_before=window_days,
        )

        if not force_rescan and all_matches:
            existing_rows = (
                session.query(ExpiryAlertLog.certificate_id, ExpiryAlertLog.exp_date)
                .filter(ExpiryAlertLog.alert_window_days == window_days)
                .all()
            )
            already_alerted = {(r.certificate_id, r.exp_date) for r in existing_rows}
            new_matches = [
                m for m in all_matches if (m["certificate_id"], m["exp_date"]) not in already_alerted
            ]
        else:
            new_matches = list(all_matches)

        skipped_count = len(all_matches) - len(new_matches)

        # Group new matches by target email (responsible engineer or fallback admin)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        unassigned_count = 0

        for cert in new_matches:
            days_rem = cert["days_remaining"]
            if days_rem < 0:
                title = f"Certificate Expired ({abs(days_rem)}d ago): {cert['component']} ({cert['country']})"
            elif days_rem == 0:
                title = f"Certificate Expires Today: {cert['component']} ({cert['country']})"
            else:
                title = f"Certificate Expiring in {days_rem}d: {cert['component']} ({cert['country']})"

            resp_email = cert.get("responsible_email")
            if resp_email:
                resp_desc = (
                    f"{cert['responsible_name']} ({resp_email})"
                    if cert.get("responsible_name")
                    else resp_email
                )
            else:
                unassigned_count += 1
                resp_desc = f"Unassigned (routed to fallback: {effective_fallback})" if effective_fallback else "Unassigned"

            exp_str = cert["exp_date"].isoformat() if isinstance(cert["exp_date"], date) else str(cert["exp_date"])
            msg = (
                f"Cert #{cert['certif_number'] or 'N/A'} ({cert['authority'] or 'N/A'}, "
                f"{cert['supplier'] or 'N/A'}) expires on {exp_str}. Responsible: {resp_desc}."
            )

            session.add(
                NotificationItem(
                    id=uuid.uuid4().hex,
                    title=title[:255],
                    message=msg,
                    category="expiry_alert",
                    is_read=False,
                    created_at=datetime.utcnow(),
                )
            )

            target_email = resp_email or effective_fallback
            if target_email:
                grouped[target_email].append(cert)

        email_tool = EmailDraftingTool()
        staged_proposal_ids: List[str] = []
        cert_to_proposal: Dict[str, Optional[str]] = {}
        emails_drafted = 0

        for target_email, certs in grouped.items():
            is_fallback = all(not c.get("responsible_email") for c in certs)
            recipient_name = next((c.get("responsible_name") for c in certs if c.get("responsible_name")), None)
            subject, body = _build_email_body(recipient_name, certs, window_days, is_fallback=is_fallback)

            from core.agent.report_pdf import generate_renewal_alert_pdf
            pdf_path = generate_renewal_alert_pdf(
                certs=certs,
                recipient_name=recipient_name or ("Compliance Administration Team" if is_fallback else "Responsible Engineer"),
                alert_days_before=window_days,
            )

            draft_res = email_tool.execute(
                recipient=target_email,
                subject=subject,
                body_content=body,
                attachment_path=pdf_path,
            )
            emails_drafted += 1
            draft_id = draft_res["draft_id"]

            proposal_id: Optional[str] = None
            if require_approval:
                cert_summaries = [
                    {
                        "component": c.get("component"),
                        "country": c.get("country"),
                        "certif_number": c.get("certif_number"),
                        "exp_date": c["exp_date"].isoformat() if isinstance(c.get("exp_date"), date) else str(c.get("exp_date")),
                        "days_remaining": c.get("days_remaining"),
                    }
                    for c in certs
                ]
                proposal = proposal_manager.create_proposal(
                    "SEND_EMAIL",
                    {
                        "draft_id": draft_id,
                        "recipient": target_email,
                        "subject": subject,
                        "body_content": body,
                        "attachment_path": pdf_path,
                        "attachment_name": os.path.basename(pdf_path),
                        "certificate_count": len(certs),
                        "alert_window_days": window_days,
                        "certificates": cert_summaries,
                    },
                    sql_preview=f"To: {target_email} | {subject}",
                )
                proposal_id = proposal["proposal_id"]
                staged_proposal_ids.append(proposal_id)
            else:
                _mark_draft_dispatched(draft_res.get("path", ""))

            for c in certs:
                cert_to_proposal[c["certificate_id"]] = proposal_id

        for cert in new_matches:
            target_email = cert.get("responsible_email") or effective_fallback or None
            session.add(
                ExpiryAlertLog(
                    id=uuid.uuid4().hex,
                    certificate_id=cert["certificate_id"],
                    certif_number=cert.get("certif_number"),
                    component=cert.get("component"),
                    country=cert.get("country"),
                    exp_date=cert["exp_date"],
                    alert_window_days=window_days,
                    recipient_email=target_email,
                    proposal_id=cert_to_proposal.get(cert["certificate_id"]),
                    alerted_at=datetime.utcnow(),
                )
            )

        session.commit()

        summary = {
            "alert_window_days": window_days,
            "total_in_window": len(all_matches),
            "already_alerted_skipped": skipped_count,
            "new_alerts_created": len(new_matches),
            "notifications_sent": len(new_matches),
            "emails_drafted": emails_drafted,
            "proposals_staged": len(staged_proposal_ids),
            "proposal_ids": staged_proposal_ids,
            "unassigned_count": unassigned_count,
        }
        logger.info(
            f"Expiry alert scan completed (window={window_days}d): "
            f"in_window={len(all_matches)}, new={len(new_matches)}, skipped={skipped_count}, "
            f"emails_drafted={emails_drafted}, proposals_staged={len(staged_proposal_ids)}"
        )
        return summary
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()
