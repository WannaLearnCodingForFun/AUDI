"""Critical-defect escalation (FR-19).

On a critical event the designated recipients — active users with the
maintenance role subscribed to the event's site — receive a stored
notification, a WebSocket push (when connected) and, if SMTP is configured
through environment variables, an email. Nobody else is notified.
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from memory.store import LocalStore
from ml.config import cfg
from ml.semantic_memory import AssetEvent

logger = logging.getLogger(__name__)

Publisher = Callable[[int, Dict[str, Any]], None]


class SmtpMailer:
    def __init__(self, config: Dict[str, Any]):
        e = cfg(config, "escalation")
        self.host = os.environ.get(e["smtp_host_env"])
        self.port = int(os.environ.get(e["smtp_port_env"]) or 0)
        self.user = os.environ.get(e["smtp_user_env"])
        self.password = os.environ.get(e["smtp_password_env"])
        self.sender = os.environ.get(e["smtp_from_env"])
        self.timeout = float(e["smtp_timeout_s"])

    @property
    def configured(self) -> bool:
        return bool(self.host and self.port and self.sender)

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
            smtp.starttls()
            if self.user and self.password:
                smtp.login(self.user, self.password)
            smtp.send_message(msg)


class EscalationService:
    def __init__(self, config: Dict[str, Any], store: LocalStore, publisher: Optional[Publisher] = None, mailer: Optional[SmtpMailer] = None):
        self.config = config
        self.store = store
        self.publisher = publisher
        self.mailer = mailer or SmtpMailer(config)
        self._sent: Set[Tuple[str, str, Optional[str]]] = set()
        self._lock = threading.Lock()

    def on_event(self, ev: AssetEvent) -> None:
        if ev.critical and ev.persisted:
            self.escalate(ev)

    def escalate(self, ev: AssetEvent) -> List[Dict[str, Any]]:
        session = self.store.get_session(ev.session_id)
        site_id = session["site_id"]
        defect_type = ev.condition.get("defect_type")
        key = (ev.session_id, ev.asset_id, defect_type)
        with self._lock:
            if key in self._sent:
                return []
            self._sent.add(key)
        defect_id = None
        if ev.defect:
            for d in ev.defect["defects"]:
                if d["defect_type"] == defect_type:
                    defect_id = d["defect_id"]
        message = (
            f"CRITICAL {defect_type or ev.condition.get('label')} on {ev.asset_type} {ev.asset_id} "
            f"at site {site_id}: severity {ev.severity_score:.2f}"
        )
        out = []
        for user in self.store.escalation_recipients(site_id):
            n = self.store.add_notification(
                user_id=user["user_id"],
                site_id=site_id,
                session_id=ev.session_id,
                asset_id=ev.asset_id,
                defect_id=defect_id,
                severity_score=ev.severity_score,
                message=message,
            )
            if self.publisher is not None:
                try:
                    self.publisher(user["user_id"], {"type": "notification", "notification": n})
                except Exception:  # noqa: BLE001 - push failure must not lose the stored notification
                    logger.exception("WebSocket push failed")
            email_status = "not_configured"
            if self.mailer.configured and user.get("email"):
                try:
                    self.mailer.send(user["email"], f"[AUDI] {message}", message)
                    email_status = "sent"
                except (OSError, smtplib.SMTPException) as exc:
                    email_status = f"failed: {exc}"
            elif self.mailer.configured:
                email_status = "no_address"
            self.store.set_notification_email_status(n["notification_id"], email_status)
            n["email_status"] = email_status
            out.append({"user_id": user["user_id"], "username": user["username"], **n})
        logger.warning("%s → %d recipient(s)", message, len(out))
        return out

    def for_session(self, session_id: str) -> List[Dict[str, Any]]:
        return self.store.session_notifications(session_id)
