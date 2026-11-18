import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from schemas.extraction import ChatSession, UserAccount
from server.auth import (
    create_access_token,
    decode_access_token,
    format_user_context_for_prompt,
    hash_password,
    verify_password,
)
from storage.database import SessionLocal


class TestAuthAndRBAC(unittest.TestCase):
    def test_password_hashing_and_token_verification(self):
        pw_hash = hash_password("SecurePass123")
        self.assertTrue(verify_password("SecurePass123", pw_hash))
        self.assertFalse(verify_password("WrongPass", pw_hash))

        token = create_access_token("usr_1", "user@example.com", "editor")
        claims = decode_access_token(token)
        self.assertIsNotNone(claims)
        self.assertEqual(claims["sub"], "usr_1")
        self.assertEqual(claims["role"], "editor")

        # Tampered token must fail verification
        tampered = token[:-4] + "XXXX"
        self.assertIsNone(decode_access_token(tampered))

    def test_format_user_context_for_prompt(self):
        admin_ctx = format_user_context_for_prompt(
            {"full_name": "Mouad", "email": "mouad@example.com", "role": "admin"}
        )
        self.assertIn("Administrator", admin_ctx)
        self.assertIn("mouad@example.com", admin_ctx)

        viewer_ctx = format_user_context_for_prompt(
            {"full_name": "Auditor", "email": "auditor@example.com", "role": "viewer"}
        )
        self.assertIn("Read-Only", viewer_ctx)
        self.assertIn("CANNOT", viewer_ctx)

    def test_rbac_invite_onboarding_and_session_isolation(self):
        from server.main import app

        db = SessionLocal()
        test_editor_email = "__test_editor_rbac@example.com"
        test_viewer_email = "__test_viewer_rbac@example.com"
        created_session_ids = []
        try:
            # Clean any leftover test users
            db.query(UserAccount).filter(
                UserAccount.email.in_([test_editor_email, test_viewer_email])
            ).delete(synchronize_session=False)
            db.commit()

            admin_user = db.query(UserAccount).filter(UserAccount.role == "admin").first()
            self.assertIsNotNone(admin_user)
            admin_token = create_access_token(admin_user.id, admin_user.email, "admin")

            with patch.object(app.router, "lifespan_context", None):
                client = TestClient(app)

                # Unauthenticated access to protected API must return 401
                unauth = client.get("/api/v1/chat/sessions")
                self.assertEqual(unauth.status_code, 401)

                # Admin cannot invite a second admin
                bad_invite = client.post(
                    "/api/v1/users/invite",
                    json={"email": test_editor_email, "full_name": "Bad Admin", "role": "admin"},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                self.assertEqual(bad_invite.status_code, 422)

                # Admin invites an Editor and a Viewer
                with patch("core.agent.mailer.send_email_via_smtp", side_effect=RuntimeError("no smtp in unit test")):
                    inv_ed = client.post(
                        "/api/v1/users/invite",
                        json={"email": test_editor_email, "full_name": "Test Editor", "role": "editor"},
                        headers={"Authorization": f"Bearer {admin_token}"},
                    )
                    self.assertEqual(inv_ed.status_code, 200)
                    ed_token_str = inv_ed.json()["user"]["invite_token"]

                    inv_vw = client.post(
                        "/api/v1/users/invite",
                        json={"email": test_viewer_email, "full_name": "Test Viewer", "role": "viewer"},
                        headers={"Authorization": f"Bearer {admin_token}"},
                    )
                    self.assertEqual(inv_vw.status_code, 200)
                    vw_token_str = inv_vw.json()["user"]["invite_token"]

                # Accept invites
                acc_ed = client.post(
                    "/api/v1/auth/accept-invite",
                    json={"token": ed_token_str, "password": "EditorPass123"},
                )
                self.assertEqual(acc_ed.status_code, 200)
                ed_jwt = acc_ed.json()["token"]

                acc_vw = client.post(
                    "/api/v1/auth/accept-invite",
                    json={"token": vw_token_str, "password": "ViewerPass123"},
                )
                self.assertEqual(acc_vw.status_code, 200)
                vw_jwt = acc_vw.json()["token"]

                # Viewer is blocked from /api/v1/users and DB/automation writes
                self.assertEqual(
                    client.get("/api/v1/users", headers={"Authorization": f"Bearer {vw_jwt}"}).status_code,
                    403,
                )
                self.assertEqual(
                    client.post(
                        "/api/v1/agent/alerts/run",
                        json={},
                        headers={"Authorization": f"Bearer {vw_jwt}"},
                    ).status_code,
                    403,
                )

                # Editor is blocked from /api/v1/users and from updating sender_email
                self.assertEqual(
                    client.get("/api/v1/users", headers={"Authorization": f"Bearer {ed_jwt}"}).status_code,
                    403,
                )
                self.assertEqual(
                    client.post(
                        "/api/v1/agent/alerts/config",
                        json={"sender_email": "hacker@example.com"},
                        headers={"Authorization": f"Bearer {ed_jwt}"},
                    ).status_code,
                    403,
                )

                # Per-user chat session isolation
                s_ed = client.post(
                    "/api/v1/chat/sessions",
                    headers={"Authorization": f"Bearer {ed_jwt}"},
                )
                self.assertEqual(s_ed.status_code, 200)
                ed_sess_id = s_ed.json()["id"]
                created_session_ids.append(ed_sess_id)

                vw_sessions = client.get(
                    "/api/v1/chat/sessions",
                    headers={"Authorization": f"Bearer {vw_jwt}"},
                ).json()
                self.assertFalse(any(s["id"] == ed_sess_id for s in vw_sessions))
                self.assertEqual(
                    client.get(
                        f"/api/v1/chat/sessions/{ed_sess_id}/messages",
                        headers={"Authorization": f"Bearer {vw_jwt}"},
                    ).status_code,
                    403,
                )
        finally:
            if created_session_ids:
                db.query(ChatSession).filter(ChatSession.id.in_(created_session_ids)).delete(
                    synchronize_session=False
                )
            db.query(UserAccount).filter(
                UserAccount.email.in_([test_editor_email, test_viewer_email])
            ).delete(synchronize_session=False)
            db.commit()
            db.close()


if __name__ == "__main__":
    unittest.main()
