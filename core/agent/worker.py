"""
core/agent/worker.py — Autonomous Background Task Scheduler

Runs agentic tools on a schedule (e.g. periodic document ingestion) via
APScheduler's AsyncIOScheduler, fully isolated from the FastAPI request path so
long-running autonomous work never blocks the event loop.

Best-effort execution: every scheduled job wraps its work in try/except and never
raises, so a background tool failure can never crash the main FastAPI server.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.agent.tools import AGENT_DATA_DIR

logger = logging.getLogger(__name__)

#: Default interval between autonomous ingestion runs (seconds). 24 hours.
DEFAULT_INGESTION_INTERVAL_SECONDS: int = 24 * 60 * 60

#: Module-level scheduler singleton shared with the FastAPI lifespan.
scheduler = AsyncIOScheduler()

#: APScheduler job id for the autonomous ingestion job.
JOB_ID: str = "autonomous_ingestion"
#: APScheduler job id for the certificate expiry alerting scanner.
ALERT_JOB_ID: str = "expiry_alert_scanner"

#: Persisted scheduler configuration (survives restarts; editable from CONTROL page).
SCHEDULER_CONFIG_FILE: str = os.path.join(AGENT_DATA_DIR, "scheduler_config.json")
#: Persisted expiry alerting configuration (survives restarts; editable from CONTROL page).
ALERTING_CONFIG_FILE: str = os.path.join(AGENT_DATA_DIR, "alerting_config.json")

#: Explicit lifecycle flag. APScheduler's `shutdown(wait=False)` clears `running`
#: asynchronously (next loop iteration), so `scheduler.running` alone is not a
#: reliable idempotency guard for rapid start/stop sequences. This flag is.
_scheduler_started: bool = False


def _load_scheduler_config() -> Dict[str, Any]:
    """Load persisted scheduler config; falls back to env/config defaults."""
    try:
        with open(SCHEDULER_CONFIG_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            "enabled": bool(data.get("enabled", True)),
            "require_approval": bool(data.get("require_approval", True)),
            "interval_seconds": int(data.get("interval_seconds", DEFAULT_INGESTION_INTERVAL_SECONDS)),
            "target_url": str(data.get("target_url", "")),
        }
    except Exception:
        from server.config import AUTONOMOUS_INGESTION_INTERVAL_SECONDS, AUTONOMOUS_INGESTION_TARGET_URL

        return {
            "enabled": True,
            "require_approval": True,
            "interval_seconds": int(AUTONOMOUS_INGESTION_INTERVAL_SECONDS),
            "target_url": str(AUTONOMOUS_INGESTION_TARGET_URL or ""),
        }


def _save_scheduler_config(cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(SCHEDULER_CONFIG_FILE), exist_ok=True)
    with open(SCHEDULER_CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def _load_alerting_config() -> Dict[str, Any]:
    """Load persisted expiry alerting config; falls back to server.config defaults."""
    from server.config import (
        EXPIRY_ALERT_DEFAULT_DAYS,
        EXPIRY_ALERT_FALLBACK_EMAIL,
        EXPIRY_ALERT_INTERVAL_SECONDS,
    )

    try:
        with open(ALERTING_CONFIG_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            "enabled": bool(data.get("enabled", True)),
            "require_approval": bool(data.get("require_approval", True)),
            "interval_seconds": max(60, int(data.get("interval_seconds", EXPIRY_ALERT_INTERVAL_SECONDS))),
            "alert_days_before": max(1, min(365, int(data.get("alert_days_before", EXPIRY_ALERT_DEFAULT_DAYS)))),
            "fallback_email": str(data.get("fallback_email", EXPIRY_ALERT_FALLBACK_EMAIL) or "").strip(),
            "sender_email": str(data.get("sender_email", "") or "").strip(),
            "smtp_password": str(data.get("smtp_password", "") or "").strip(),
            "google_client_id": str(data.get("google_client_id", "") or "").strip(),
            "google_client_secret": str(data.get("google_client_secret", "") or "").strip(),
            "google_refresh_token": str(data.get("google_refresh_token", "") or "").strip(),
            "google_access_token": str(data.get("google_access_token", "") or "").strip(),
            "google_token_expires_at": float(data.get("google_token_expires_at", 0) or 0),
            "google_connected_email": str(data.get("google_connected_email", "") or "").strip(),
        }
    except Exception:
        return {
            "enabled": True,
            "require_approval": True,
            "interval_seconds": int(EXPIRY_ALERT_INTERVAL_SECONDS),
            "alert_days_before": int(EXPIRY_ALERT_DEFAULT_DAYS),
            "fallback_email": str(EXPIRY_ALERT_FALLBACK_EMAIL or "").strip(),
            "sender_email": "",
            "smtp_password": "",
            "google_client_id": "",
            "google_client_secret": "",
            "google_refresh_token": "",
            "google_access_token": "",
            "google_token_expires_at": 0.0,
            "google_connected_email": "",
        }


def _save_alerting_config(cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(ALERTING_CONFIG_FILE), exist_ok=True)
    with open(ALERTING_CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def _add_alert_interval_job(interval_seconds: int) -> None:
    """Register/reschedule the expiry alert scanner job on a fixed interval."""
    scheduler.add_job(
        expiry_alert_job,
        trigger="interval",
        seconds=max(int(interval_seconds), 60),
        id=ALERT_JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )


async def expiry_alert_job(force_rescan: bool = False) -> Optional[Dict[str, Any]]:
    """
    Autonomous background job: scans `certificates` for upcoming expirations
    within the configured `alert_days_before` window, emits in-app notifications,
    and drafts/stages renewal alert emails for responsible engineers.
    """
    from core.agent.alerting import run_expiry_alert_scan

    cfg = _load_alerting_config()
    try:
        return await asyncio.to_thread(
            run_expiry_alert_scan,
            alert_days_before=cfg["alert_days_before"],
            require_approval=cfg["require_approval"],
            fallback_email=cfg["fallback_email"] or None,
            force_rescan=force_rescan,
        )
    except Exception as exc:
        logger.error(f"Expiry alert job failed: {exc}")
        return None


def _add_interval_job(interval_seconds: int, target_url: Optional[str]) -> None:
    """Register/reschedule the autonomous ingestion job on a fixed interval.

    Scheduled runs honor the same fetcher/cookie config as manual runs so both
    paths behave identically.
    """
    from server.config import SCRAPER_FETCHER, SCRAPER_COOKIE_HEADER

    scheduler.add_job(
        autonomous_ingestion_job,
        trigger="interval",
        seconds=max(int(interval_seconds), 60),  # floor of 60s avoids tight loops
        args=[target_url, SCRAPER_COOKIE_HEADER or None, str(SCRAPER_FETCHER or "html").lower()],
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )


def _load_active_sources() -> List[Dict[str, Any]]:
    """Load active sources (url + optional cookie_header) from the `sources` table."""
    try:
        from storage.database import get_db_session
        from schemas.extraction import Source

        with get_db_session() as session:
            rows = session.query(Source).filter(Source.active.is_(True)).all()
            return [
                {
                    "url": str(r.url).strip(),
                    "cookie_header": str(r.cookie_header).strip() if getattr(r, "cookie_header", None) else None,
                }
                for r in rows
                if r.url and str(r.url).strip()
            ]
    except Exception as exc:
        logger.warning(f" Could not load sources from database: {exc}")
        return []


def _load_active_source_urls() -> List[str]:
    """Backwards compatible helper returning raw URL strings."""
    return [s["url"] for s in _load_active_sources()]


def _stage_ingest_proposal(file_path: str, source_url: str) -> Optional[Dict[str, Any]]:
    """
    Stage an INGEST_DOCUMENT proposal for a downloaded RF PDF (idempotent: skips
    if a PENDING proposal for the same file already exists).
    """
    from core.agent.proposals import proposal_manager

    for p in proposal_manager.list_pending_proposals():
        if p.get("type") == "INGEST_DOCUMENT" and (p.get("payload") or {}).get("file_path") == file_path:
            return None
    return proposal_manager.create_proposal(
        "INGEST_DOCUMENT",
        {"file_path": file_path, "source_url": source_url},
        sql_preview=source_url,
    )


async def autonomous_ingestion_job(
    target_url: Optional[str] = None,
    source_urls: Optional[List[str]] = None,
    cookie_header: Optional[str] = None,
    fetcher_type: str = "auto",
) -> Optional[Dict[str, Any]]:
    """
    Autonomous background job: for every source URL (from the `sources` table,
    or the passed lists), discovers the target certificate/compliance PDF URLs,
    verifies each is a real PDF, skips any already in the database/manifest,
    downloads the new ones with the WebDownloaderTool, and logs the local paths.
    """
    from core.agent.scraper import discover_pdf_urls, append_manifest, classify_pdf_relevance
    from core.agent.tools import WebDownloaderTool
    from server.config import (
        SCRAPER_MAX_PAGES, SCRAPER_MAX_DEPTH, SCRAPER_TIMEOUT_SECONDS,
        SCRAPER_USER_AGENT, SCRAPER_POLITE_DELAY_SECONDS,
        SCRAPER_BLOCKED_KEYWORDS, SCRAPER_ALLOWED_KEYWORDS,
        SCRAPER_USE_LLM_FILTER, SCRAPER_FETCHED_MANIFEST,
    )

    targets: List[Dict[str, Any]] = []
    if source_urls:
        for u in source_urls:
            u_str = str(u).strip()
            if u_str:
                targets.append({"url": u_str, "cookie_header": cookie_header})
    else:
        targets.extend(_load_active_sources())

    if target_url and str(target_url).strip():
        t_str = str(target_url).strip()
        if not any(t["url"] == t_str for t in targets):
            targets.append({"url": t_str, "cookie_header": cookie_header})

    if not targets:
        logger.warning("Autonomous ingestion job has no source URLs configured; skipping run.")
        return None

    try:
        from storage.database import SessionLocal

        downloaded: List[str] = []
        skipped_existing: List[str] = []
        failed_verification: List[str] = []
        irrelevant: List[str] = []
        staged_proposals: List[str] = []
        discovered_urls: List[str] = []
        tool = WebDownloaderTool()

        for item in targets:
            url = item["url"]
            effective_cookie = item.get("cookie_header") or cookie_header

            from storage.database import get_db_session
            with get_db_session() as session:
                result = await asyncio.to_thread(
                    discover_pdf_urls,
                    url,
                    db_session=session,
                    max_pages=SCRAPER_MAX_PAGES,
                    max_depth=SCRAPER_MAX_DEPTH,
                    timeout=SCRAPER_TIMEOUT_SECONDS,
                    user_agent=SCRAPER_USER_AGENT,
                    polite_delay=SCRAPER_POLITE_DELAY_SECONDS,
                    allowed_keywords=SCRAPER_ALLOWED_KEYWORDS,
                    blocked_keywords=SCRAPER_BLOCKED_KEYWORDS,
                    manifest_path=SCRAPER_FETCHED_MANIFEST,
                    use_llm=SCRAPER_USE_LLM_FILTER,
                    fetcher_type=fetcher_type,
                    cookie_header=effective_cookie,
                )

            logger.info(
                f" Autonomous discovery for '{url}' (fetcher={fetcher_type}, has_cookie={bool(effective_cookie)}): "
                f"{len(result.verified_urls)} new PDF(s), "
                f"{len(result.skipped_existing)} already fetched. ({result.reason})"
            )

            discovered_urls.extend(result.verified_urls)
            skipped_existing.extend(result.skipped_existing)
            failed_verification.extend(result.failed_verification)

            for pdf_url in result.verified_urls:
                try:
                    dl = await asyncio.to_thread(tool.execute, url=pdf_url, cookie_header=effective_cookie)
                    path = str(dl.get("path", ""))
                    if path:
                        # Content gate: keep only RF-certificate PDFs. "unclear"
                        # (scanned) PDFs are kept; the INGEST_DOCUMENT OCR is the
                        # authoritative verification.
                        relevance = await asyncio.to_thread(classify_pdf_relevance, path)
                        if relevance.get("status") == "irrelevant":
                            try:
                                os.remove(path)
                            except OSError:
                                pass
                            irrelevant.append(pdf_url)
                            await asyncio.to_thread(append_manifest, [pdf_url], SCRAPER_FETCHED_MANIFEST)
                            logger.info(f" Dropped irrelevant (non-RF) document: {pdf_url}")
                            continue
                        downloaded.append(path)
                        logger.info(f" Autonomous ingestion downloaded: {path}")

                        cfg = _load_scheduler_config()
                        require_approval = cfg.get("require_approval", True)

                        if require_approval:
                            # Stage an INGEST_DOCUMENT proposal (HITL gate): approving
                            # it runs OCR -> extraction -> persistence into the
                            # certificates table.
                            staged = await asyncio.to_thread(_stage_ingest_proposal, path, pdf_url)
                            if staged:
                                staged_proposals.append(staged["proposal_id"])
                        else:
                            # Direct ingestion mode (approval bypassed)
                            from core.agent.ingestor import ingest_document_file
                            await asyncio.to_thread(ingest_document_file, path, pdf_url)
                            logger.info(f" Direct autonomous ingestion completed (approval bypassed): {path}")

                        # Record the URL as fetched immediately so a crash mid-run
                        # does not cause re-download loops.
                        await asyncio.to_thread(append_manifest, [pdf_url], SCRAPER_FETCHED_MANIFEST)
                except Exception as exc:
                    logger.error(f" Autonomous download failed for '{pdf_url}': {exc}")

        summary = (
            f"Checked {len(urls)} source(s): {len(downloaded)} RF PDF(s) downloaded "
            f"({len(staged_proposals)} staged for approval, require_approval={require_approval}), {len(irrelevant)} dropped (non-RF), "
            f"{len(skipped_existing)} already in database, {len(failed_verification)} failed verification."
        )
        logger.info(f" Autonomous ingestion summary: {summary}")
        return {
            "summary": summary,
            "discovered_urls": discovered_urls,
            "downloaded_paths": downloaded,
            "staged_proposals": staged_proposals,
            "skipped_existing": skipped_existing,
            "failed_verification": failed_verification,
            "irrelevant": irrelevant,
        }
    except Exception as exc:
        # A background tool failure must never propagate up to the event loop.
        logger.error(f" Autonomous ingestion job failed for source(s) {urls}: {exc}")
        return None


def start_scheduler(
    interval_seconds: Optional[int] = None,
    target_url: Optional[str] = None,
) -> AsyncIOScheduler:
    """
    Start the autonomous ingestion scheduler from persisted config (or the given
    overrides). Idempotent: calling again while already running is a no-op.

    Args:
        interval_seconds (Optional[int]): override for the run interval.
        target_url (Optional[str]): override for the target URL (None = use config).

    Returns:
        AsyncIOScheduler: The (possibly running) scheduler instance.
    """
    global _scheduler_started

    if _scheduler_started or scheduler.running:
        logger.info("Scheduler already running; skipping duplicate start.")
        return scheduler

    cfg = _load_scheduler_config()
    interval = interval_seconds or cfg["interval_seconds"]
    url = cfg["target_url"] if target_url is None else target_url

    if cfg["enabled"]:
        _add_interval_job(interval, url or None)
        logger.info(f"Scheduler configured: autonomous ingestion every {interval}s.")
    else:
        logger.info("Autonomous ingestion is configured as disabled.")

    alert_cfg = _load_alerting_config()
    if alert_cfg["enabled"]:
        _add_alert_interval_job(alert_cfg["interval_seconds"])
        logger.info(
            f"Scheduler configured: expiry alert scanner every {alert_cfg['interval_seconds']}s "
            f"(lead_time={alert_cfg['alert_days_before']}d)."
        )
    else:
        logger.info("Expiry alert scanner is configured as disabled.")

    if cfg["enabled"] or alert_cfg["enabled"]:
        scheduler.start()
        _scheduler_started = True
    return scheduler


def _apply_scheduler_config(cfg: Dict[str, Any]) -> None:
    """Apply a scheduler config to the live scheduler (add/remove/reschedule)."""
    global _scheduler_started

    if cfg["enabled"]:
        _add_interval_job(cfg["interval_seconds"], cfg["target_url"] or None)
        if not scheduler.running:
            try:
                scheduler.start()
                _scheduler_started = True
            except Exception as exc:
                logger.warning(f" Could not start scheduler inline ({exc}); job configured for active loop.")
        logger.info(f"Scheduler reconfigured: every {cfg['interval_seconds']}s, target={cfg['target_url'] or '(none)'}")
    else:
        try:
            scheduler.remove_job(JOB_ID)
        except Exception:
            pass
        logger.info("Scheduler disabled: autonomous ingestion job removed.")


def get_scheduler_status() -> Dict[str, Any]:
    """
    Return the scheduler configuration + live job state for the CONTROL page.
    """
    from schemas.automation import AutomationConfig

    cfg = _load_scheduler_config()
    next_run = None
    try:
        job = scheduler.get_job(JOB_ID)
        if job is not None and job.next_run_time is not None:
            next_run = job.next_run_time.isoformat()
    except Exception:
        pass

    auto_config = AutomationConfig(
        id="autonomous_scheduler",
        title="Autonomous Scheduler Configuration",
        description="Configure background discovery intervals and staged approval behavior for the autonomous ingestion engine.",
        enabled=cfg.get("enabled", True),
        require_approval=cfg.get("require_approval", True),
        interval_hours=max(1, int((cfg.get("interval_seconds") or 86400) // 3600)),
        target_urls=[cfg.get("target_url")] if cfg.get("target_url") else [],
        running=bool(scheduler.running),
        next_run_time=next_run,
        custom_params={
            "interval_seconds": cfg.get("interval_seconds", 86400),
            "target_url": cfg.get("target_url", ""),
        },
    )

    result = auto_config.to_dict()
    # Retain interval_seconds and target_url keys for backwards compatibility
    result["interval_seconds"] = cfg.get("interval_seconds", 86400)
    result["target_url"] = cfg.get("target_url", "")
    return result


def update_scheduler_config(
    enabled: Optional[bool] = None,
    require_approval: Optional[bool] = None,
    interval_seconds: Optional[int] = None,
    target_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist and apply a scheduler configuration change (on/off, approval, interval, URL).

    Args:
        enabled: whether the scheduled job runs.
        require_approval: whether ingestion requires staged approval or runs directly.
        interval_seconds: seconds between runs (floored at 60).
        target_url: persistent URL the scheduled job checks.

    Returns:
        The updated scheduler status (see get_scheduler_status).
    """
    cfg = _load_scheduler_config()
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if require_approval is not None:
        cfg["require_approval"] = bool(require_approval)
    if interval_seconds is not None:
        cfg["interval_seconds"] = max(int(interval_seconds), 60)
    if target_url is not None:
        cfg["target_url"] = str(target_url or "").strip()
    _save_scheduler_config(cfg)
    _apply_scheduler_config(cfg)
    return get_scheduler_status()


def _apply_alerting_config(cfg: Dict[str, Any]) -> None:
    """Apply an alerting config to the live scheduler (add/remove/reschedule)."""
    global _scheduler_started

    if cfg["enabled"]:
        _add_alert_interval_job(cfg["interval_seconds"])
        if not scheduler.running:
            try:
                scheduler.start()
                _scheduler_started = True
            except Exception as exc:
                logger.warning(f"Could not start scheduler inline ({exc}); alert job configured for active loop.")
        logger.info(
            f"Expiry alert scheduler reconfigured: every {cfg['interval_seconds']}s, "
            f"lead_time={cfg['alert_days_before']}d"
        )
    else:
        try:
            scheduler.remove_job(ALERT_JOB_ID)
        except Exception:
            pass
        logger.info("Expiry alert scheduler disabled: scanner job removed.")


def get_alerting_status() -> Dict[str, Any]:
    """Return the expiry alerting configuration and live job state for the CONTROL page."""
    from schemas.automation import AutomationConfig

    cfg = _load_alerting_config()
    next_run = None
    try:
        job = scheduler.get_job(ALERT_JOB_ID)
        if job is not None and job.next_run_time is not None:
            next_run = job.next_run_time.isoformat()
    except Exception:
        pass

    from server.config import (
        GOOGLE_OAUTH_CLIENT_ID,
        GOOGLE_OAUTH_CLIENT_SECRET,
        SMTP_PASSWORD,
        SMTP_USER,
    )

    effective_client_id = cfg.get("google_client_id") or str(GOOGLE_OAUTH_CLIENT_ID or "").strip()
    has_client_secret = bool(cfg.get("google_client_secret") or str(GOOGLE_OAUTH_CLIENT_SECRET or "").strip())
    google_connected_email = cfg.get("google_connected_email") or ""
    google_connected = bool(cfg.get("google_refresh_token") and effective_client_id and has_client_secret)
    effective_sender = google_connected_email or cfg.get("sender_email") or str(SMTP_USER or "").strip()
    has_pwd = bool(cfg.get("smtp_password") or str(SMTP_PASSWORD or "").strip())

    auto_config = AutomationConfig(
        id="expiry_alert_scanner",
        title="Certificate Expiry Alerting System",
        description="Automatically monitors certificate expiration dates, notifies compliance teams, and drafts renewal alerts for responsible engineers.",
        enabled=cfg.get("enabled", True),
        require_approval=cfg.get("require_approval", True),
        interval_hours=max(1, int((cfg.get("interval_seconds") or 86400) // 3600)),
        action_type="certificate_expiry_alert",
        running=bool(scheduler.running),
        next_run_time=next_run,
        custom_params={
            "interval_seconds": cfg.get("interval_seconds", 86400),
            "alert_days_before": cfg.get("alert_days_before", 30),
            "fallback_email": cfg.get("fallback_email", ""),
            "sender_email": effective_sender,
            "has_smtp_password": has_pwd,
            "google_client_id": effective_client_id,
            "has_google_client_secret": has_client_secret,
            "google_connected": google_connected,
            "google_connected_email": google_connected_email,
        },
    )

    result = auto_config.to_dict()
    result["interval_seconds"] = cfg.get("interval_seconds", 86400)
    result["alert_days_before"] = cfg.get("alert_days_before", 30)
    result["fallback_email"] = cfg.get("fallback_email", "")
    result["sender_email"] = effective_sender
    result["has_smtp_password"] = has_pwd
    result["google_client_id"] = effective_client_id
    result["has_google_client_secret"] = has_client_secret
    result["google_connected"] = google_connected
    result["google_connected_email"] = google_connected_email
    return result


def update_alerting_config(
    enabled: Optional[bool] = None,
    require_approval: Optional[bool] = None,
    interval_seconds: Optional[int] = None,
    alert_days_before: Optional[int] = None,
    fallback_email: Optional[str] = None,
    sender_email: Optional[str] = None,
    smtp_password: Optional[str] = None,
    google_client_id: Optional[str] = None,
    google_client_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist and apply an expiry alerting configuration change."""
    cfg = _load_alerting_config()
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if require_approval is not None:
        cfg["require_approval"] = bool(require_approval)
    if interval_seconds is not None:
        cfg["interval_seconds"] = max(int(interval_seconds), 60)
    if alert_days_before is not None:
        cfg["alert_days_before"] = max(1, min(365, int(alert_days_before)))
    if fallback_email is not None:
        cfg["fallback_email"] = str(fallback_email or "").strip()
    if sender_email is not None:
        cfg["sender_email"] = str(sender_email or "").strip()
    if smtp_password is not None:
        cleaned_pwd = str(smtp_password).strip()
        if cleaned_pwd:
            cfg["smtp_password"] = cleaned_pwd
    if google_client_id is not None:
        cfg["google_client_id"] = str(google_client_id or "").strip()
    if google_client_secret is not None:
        cleaned_sec = str(google_client_secret).strip()
        if cleaned_sec:
            cfg["google_client_secret"] = cleaned_sec
    _save_alerting_config(cfg)
    _apply_alerting_config(cfg)
    return get_alerting_status()


def shutdown_scheduler() -> None:
    """
    Gracefully shut down the scheduler so Docker teardown leaves no zombie
    background processes. Idempotent and never raises.

    Only the explicit `_scheduler_started` flag triggers shutdown: APScheduler's
    `shutdown(wait=False)` clears `running` asynchronously, so consulting
    `scheduler.running` here can double-schedule `_shutdown`.
    """
    global _scheduler_started

    if not _scheduler_started:
        return

    try:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down.")
    except Exception as exc:
        logger.warning(f"Scheduler shutdown error (continuing teardown): {exc}")
    finally:
        _scheduler_started = False