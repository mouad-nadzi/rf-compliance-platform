import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from core.agent.alerting import (
    _clamp_alert_days,
    query_expiring_certificates,
    run_expiry_alert_scan,
)
from core.agent import worker
from schemas.extraction import CertificateMetadata, ExpiryAlertLog, NotificationItem
from storage.database import SessionLocal


class TestAlertingSystem(unittest.TestCase):
    def test_clamp_alert_days(self):
        self.assertEqual(_clamp_alert_days(0), 1)
        self.assertEqual(_clamp_alert_days(-10), 1)
        self.assertEqual(_clamp_alert_days(30), 30)
        self.assertEqual(_clamp_alert_days(60), 60)
        self.assertEqual(_clamp_alert_days(999), 365)

    def test_query_expiring_certificates_excludes_permanent(self):
        db = SessionLocal()
        try:
            res_30 = query_expiring_certificates(db, alert_days_before=30)
            res_60 = query_expiring_certificates(db, alert_days_before=60)
            self.assertLessEqual(len(res_30), len(res_60))
            for item in res_60:
                self.assertGreaterEqual(item["days_remaining"], 0)
                self.assertLessEqual(item["days_remaining"], 60)
                self.assertLess(item["exp_date"].year, 2900)
        finally:
            db.close()

    def test_scan_idempotency_and_window_transition(self):
        db = SessionLocal()
        test_cert_id_1 = "__test_alert_cert_15d__"
        test_cert_id_2 = "__test_alert_cert_45d__"
        try:
            # Clean any leftover test rows first
            db.query(ExpiryAlertLog).filter(
                ExpiryAlertLog.certificate_id.in_([test_cert_id_1, test_cert_id_2])
            ).delete(synchronize_session=False)
            db.query(CertificateMetadata).filter(
                CertificateMetadata.certificate_id.in_([test_cert_id_1, test_cert_id_2])
            ).delete(synchronize_session=False)
            db.commit()

            today = date.today()
            c1 = CertificateMetadata(
                certificate_id=test_cert_id_1,
                component="__TEST_COMP_UNASSIGNED__",
                supplier="TEST_SUPPLIER",
                country="France",
                certif_number="TEST-15D",
                authority="ANFR",
                exp_date=today + timedelta(days=15),
            )
            c2 = CertificateMetadata(
                certificate_id=test_cert_id_2,
                component="__TEST_COMP_UNASSIGNED__",
                supplier="TEST_SUPPLIER",
                country="Germany",
                certif_number="TEST-45D",
                authority="BNetzA",
                exp_date=today + timedelta(days=45),
            )
            db.add_all([c1, c2])
            db.commit()

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("core.agent.proposals.proposal_manager.dir", tmpdir), patch(
                    "core.agent.tools.DRAFTS_DIR", tmpdir
                ):
                    # First 30d scan: c1 should be alerted
                    res1 = run_expiry_alert_scan(
                        alert_days_before=30,
                        require_approval=True,
                        fallback_email="test-fallback@stellantis.com",
                        db_session=db,
                    )
                    log1 = (
                        db.query(ExpiryAlertLog)
                        .filter(ExpiryAlertLog.certificate_id == test_cert_id_1)
                        .all()
                    )
                    self.assertEqual(len(log1), 1)
                    self.assertEqual(log1[0].alert_window_days, 30)
                    self.assertGreaterEqual(res1["new_alerts_created"], 1)

                    # Second 30d scan: c1 must be skipped (idempotent)
                    res2 = run_expiry_alert_scan(
                        alert_days_before=30,
                        require_approval=True,
                        fallback_email="test-fallback@stellantis.com",
                        db_session=db,
                    )
                    log2 = (
                        db.query(ExpiryAlertLog)
                        .filter(ExpiryAlertLog.certificate_id == test_cert_id_1)
                        .all()
                    )
                    self.assertEqual(len(log2), 1)
                    self.assertEqual(res2["new_alerts_created"], 0)

                    # 60d scan: c2 (45d) is newly included
                    run_expiry_alert_scan(
                        alert_days_before=60,
                        require_approval=True,
                        fallback_email="test-fallback@stellantis.com",
                        db_session=db,
                    )
                    log_c2 = (
                        db.query(ExpiryAlertLog)
                        .filter(ExpiryAlertLog.certificate_id == test_cert_id_2)
                        .all()
                    )
                    self.assertEqual(len(log_c2), 1)
                    self.assertEqual(log_c2[0].alert_window_days, 60)
        finally:
            # Cleanup test records & generated notifications
            db.query(ExpiryAlertLog).filter(
                ExpiryAlertLog.certificate_id.in_([test_cert_id_1, test_cert_id_2])
            ).delete(synchronize_session=False)
            db.query(NotificationItem).filter(
                NotificationItem.message.like("%TEST-15D%")
                | NotificationItem.message.like("%TEST-45D%")
            ).delete(synchronize_session=False)
            db.query(CertificateMetadata).filter(
                CertificateMetadata.certificate_id.in_([test_cert_id_1, test_cert_id_2])
            ).delete(synchronize_session=False)
            db.commit()
            db.close()

    def test_alerting_config_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "alerting_config.json")
            with patch.object(worker, "ALERTING_CONFIG_FILE", cfg_path), patch.object(
                worker, "_apply_alerting_config"
            ):
                updated = worker.update_alerting_config(
                    enabled=False,
                    require_approval=False,
                    interval_seconds=43200,
                    alert_days_before=60,
                    fallback_email="alerts@stellantis.com",
                )
                self.assertFalse(updated["enabled"])
                self.assertFalse(updated["require_approval"])
                self.assertEqual(updated["interval_hours"], 12)
                self.assertEqual(updated["alert_days_before"], 60)
                self.assertEqual(updated["fallback_email"], "alerts@stellantis.com")

    def test_insert_record_tool_multi_table_association(self):
        from sqlalchemy import text
        from core.agent.tools import InsertRecordTool, get_tool_registry

        registry = get_tool_registry()
        self.assertIn("insert_record", registry)
        tool = InsertRecordTool()
        today = date.today()
        exp_30d = (today + timedelta(days=30)).isoformat()

        db = SessionLocal()
        try:
            # 1. Insert responsible
            tool.execute(
                table_name="responsibles",
                values={"firstname": "__TestMouad__", "email": "__test_mouad@example.com"},
            )
            # 2. Insert component association (using mixed-case keys to test case-insensitive column binding)
            tool.execute(
                table_name="components",
                values={"ComponentName": "__TEST_FICT_COMP_1__", "Id_Responsible": "__test_mouad@example.com"},
            )
            # 3. Insert certificate without explicit certificate_id
            tool.execute(
                table_name="certificates",
                records=[
                    {
                        "component": "__TEST_FICT_COMP_1__",
                        "supplier": "Valeo",
                        "country": "France",
                        "certif_number": "FICT-CERT-001",
                        "authority": "ANFR",
                        "exp_date": exp_30d,
                    }
                ],
            )

            matches = query_expiring_certificates(db, alert_days_before=30)
            matched = [m for m in matches if m["component"] == "__TEST_FICT_COMP_1__"]
            self.assertEqual(len(matched), 1)
            self.assertEqual(matched[0]["responsible_email"], "__test_mouad@example.com")
            self.assertTrue(bool(matched[0]["certificate_id"]))
        finally:
            db.execute(text("DELETE FROM certificates WHERE component = '__TEST_FICT_COMP_1__'"))
            db.execute(text("DELETE FROM components WHERE componentname = '__TEST_FICT_COMP_1__'"))
            db.execute(text("DELETE FROM responsibles WHERE email = '__test_mouad@example.com'"))
            db.commit()
            db.close()

    def test_smtp_mailer_dispatch_and_missing_creds(self):
        import json
        from unittest.mock import MagicMock
        from core.agent.mailer import dispatch_email_draft, send_email_via_smtp

        with patch("core.agent.worker._load_alerting_config", return_value={}), patch(
            "server.config.SMTP_USER", ""
        ), patch("server.config.SMTP_PASSWORD", ""):
            with self.assertRaises(RuntimeError):
                send_email_via_smtp("user@example.com", "Test", "Body")

        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = os.path.join(tmpdir, "draft_1.json")
            with open(draft_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "draft_id": "draft_1",
                        "recipient": "mouadnadzi3@gmail.com",
                        "subject": "Expiry Test",
                        "body_content": "Hello",
                        "status": "draft",
                    },
                    fh,
                )

            mock_smtp = MagicMock()
            with patch("core.agent.worker._load_alerting_config", return_value={}), patch(
                "server.config.SMTP_HOST", "smtp.gmail.com"
            ), patch("server.config.SMTP_PORT", 587), patch(
                "server.config.SMTP_USER", "sender@gmail.com"
            ), patch("server.config.SMTP_PASSWORD", "app-pass"), patch(
                "server.config.SMTP_FROM", "sender@gmail.com"
            ), patch("server.config.SMTP_USE_TLS", True), patch(
                "core.agent.mailer.smtplib.SMTP", return_value=mock_smtp
            ):
                res = dispatch_email_draft(draft_path)
                self.assertEqual(res["recipient"], "mouadnadzi3@gmail.com")
                srv = mock_smtp.__enter__.return_value
                srv.starttls.assert_called_once()
                srv.login.assert_called_once_with("sender@gmail.com", "app-pass")
                srv.send_message.assert_called_once()

            with open(draft_path, encoding="utf-8") as fh:
                updated = json.load(fh)
            self.assertEqual(updated["status"], "dispatched")
            self.assertIn("dispatched_at", updated)

    def test_google_oauth_gmail_api_flow(self):
        from unittest.mock import MagicMock
        from core.agent.mailer import (
            build_google_oauth_url,
            disconnect_google_oauth,
            exchange_google_oauth_code,
            send_email_via_smtp,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "alerting_config.json")
            with patch.object(worker, "ALERTING_CONFIG_FILE", cfg_path), patch.object(
                worker, "_apply_alerting_config"
            ):
                worker.update_alerting_config(
                    google_client_id="client-123.apps.googleusercontent.com",
                    google_client_secret="secret-xyz",
                )
                auth_url = build_google_oauth_url(
                    redirect_uri="http://localhost:8000/api/v1/auth/google/callback",
                    state="state-abc",
                )
                self.assertIn("accounts.google.com/o/oauth2/v2/auth", auth_url)
                self.assertIn("gmail.send", auth_url)

                token_resp = MagicMock(ok=True, status_code=200)
                token_resp.json.return_value = {
                    "access_token": "ya29.test_access",
                    "refresh_token": "1//test_refresh",
                    "expires_in": 3600,
                }
                userinfo_resp = MagicMock(ok=True, status_code=200)
                userinfo_resp.json.return_value = {"email": "mouad.n047@gmail.com"}
                gmail_send_resp = MagicMock(ok=True, status_code=200)

                with patch("core.agent.mailer.requests.post", return_value=token_resp), patch(
                    "core.agent.mailer.requests.get", return_value=userinfo_resp
                ):
                    ex = exchange_google_oauth_code(
                        "code-123", "http://localhost:8000/api/v1/auth/google/callback"
                    )
                    self.assertEqual(ex["email"], "mouad.n047@gmail.com")
                    self.assertTrue(ex["has_refresh_token"])

                with patch("core.agent.mailer.requests.post", return_value=gmail_send_resp) as post_mock:
                    send_res = send_email_via_smtp(
                        "mouadnadzi3@gmail.com", "OAuth Test", "Body via Gmail API"
                    )
                    self.assertEqual(send_res["transport"], "gmail_oauth_api")
                    self.assertEqual(send_res["from"], "mouad.n047@gmail.com")
                    post_mock.assert_called_once()
                    self.assertIn("gmail.googleapis.com", post_mock.call_args[0][0])

                disconnect_google_oauth()
                status = worker.get_alerting_status()
                self.assertFalse(status["google_connected"])

    def test_renewal_pdf_report_generation_and_attachment(self):
        import base64
        from unittest.mock import MagicMock
        from core.agent.mailer import send_email_via_gmail_api
        from core.agent.report_pdf import generate_renewal_alert_pdf

        sample_certs = [
            {
                "component": "Fictive Antenna A",
                "country": "France",
                "certif_number": "FICT-001",
                "authority": "ARCEP",
                "supplier": "Valeo",
                "exp_date": date.today() + timedelta(days=10),
                "days_remaining": 10,
            },
            {
                "component": "Fictive Antenna B",
                "country": "Brazil",
                "certif_number": "FICT-002",
                "authority": "ANATEL",
                "supplier": "Bosch",
                "exp_date": date.today() + timedelta(days=25),
                "days_remaining": 25,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "Renewal_Report_Test.pdf")
            out_path = generate_renewal_alert_pdf(
                certs=sample_certs,
                recipient_name="Mouad Nadzi",
                alert_days_before=30,
                output_path=pdf_path,
            )
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "rb") as fh:
                header = fh.read(5)
            self.assertEqual(header, b"%PDF-")

            gmail_resp = MagicMock(ok=True, status_code=200)
            with patch(
                "core.agent.mailer._get_valid_gmail_access_token",
                return_value=("ya29.token", "mouad.n047@gmail.com"),
            ), patch("core.agent.mailer.requests.post", return_value=gmail_resp) as post_mock:
                res = send_email_via_gmail_api(
                    recipient="mouadnadzi3@gmail.com",
                    subject="URGENT! RF Certification Renewal Alert",
                    body_content="Dear Mouad Nadzi,\n\nPlease find attached...",
                    attachment_path=out_path,
                )
                self.assertEqual(res["attachment"], "Renewal_Report_Test.pdf")
                raw_b64 = post_mock.call_args.kwargs["json"]["raw"]
                raw_mime = base64.urlsafe_b64decode(raw_b64.encode("ascii"))
                self.assertIn(b"application/pdf", raw_mime)
                self.assertIn(b"Renewal_Report_Test.pdf", raw_mime)


if __name__ == "__main__":
    unittest.main()
