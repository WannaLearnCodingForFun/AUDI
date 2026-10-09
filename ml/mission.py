"""Inspection session lifecycle.

start: FR-1 connectivity → FR-2 pull (online) / FR-3 offline → session row →
       FR-4 briefing (R recomputed from "now" for every site asset, D9).
end:   FR-20 recompute R for every site asset → FR-6 push / FR-7 queue →
       FR-18 report (includes FR-19 escalations raised during the session).
Interrupted sessions (crash, §6.2) can be resumed or closed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from memory.store import LocalStore
from memory.sync import SyncManager
from ml.escalation import EscalationService
from ml.reporting import build_report
from ml.schemas import utc_now
from ml.semantic_memory import SemanticMemoryEngine


class SessionStateError(Exception):
    """The requested session transition is not allowed in the session's current state."""


class MissionService:
    def __init__(self, store: LocalStore, engine: SemanticMemoryEngine, sync: SyncManager, escalation: EscalationService):
        self.store = store
        self.engine = engine
        self.sync = sync
        self.escalation = escalation

    def briefing(self, site_id: str) -> Dict[str, Any]:
        ranked = self.store.recompute_revisit(site_id, utc_now(), enqueue=False)
        unresolved = [d for d in self.store.list_defects(site_id=site_id) if d["status"] != "Resolved"]
        return {
            "site_id": site_id,
            "generated_ts": utc_now().isoformat(),
            "assets_due": [
                {
                    "asset_id": a["asset_id"],
                    "asset_type": a["asset_type"],
                    "current_condition": a["current_condition"],
                    "last_checked_ts": a["last_checked_ts"],
                    "revisit_priority": a["revisit_priority_score"],
                    "components": a["revisit_components"],
                    "location": a["location"],
                    "marker_id": a["marker_id"],
                    "simulated": a["simulated"],
                }
                for a in ranked
            ],
            "unresolved_defects": unresolved,
        }

    def start(self, site_id: str, operator_id: str, simulated: bool = False) -> Dict[str, Any]:
        if self.store.get_site(site_id) is None:
            raise KeyError(f"Unknown site: {site_id}")
        sync = self.sync.start_sync(site_id)
        if sync.get("pulled") and any(sync["pulled"].values()):
            self.engine.reload_index()
        session = self.store.create_session(site_id, operator_id, sync["mode"], sync, simulated)
        self.engine.start_prefetch(site_id)
        return {"session": session, "sync": sync, "briefing": self.briefing(site_id)}

    def end(self, session_id: str) -> Dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if session["status"] == "ended":
            raise SessionStateError(f"Session {session_id} already ended")
        self.engine.recompute_site(session["site_id"], utc_now())
        self.store.update_session(session_id, status="ended", ended_ts=utc_now().isoformat())
        self.engine.end_session_hook(session_id)
        sync = self.sync.end_sync()
        report = build_report(self.store, session_id, sync, self.escalation.for_session(session_id))
        self.store.update_session(session_id, end_sync=sync, report=report)
        return report

    def interrupted(self) -> list:
        return [s for s in self.store.list_sessions() if s["status"] in {"active", "interrupted"}]

    def resume(self, session_id: str) -> Dict[str, Any]:
        s = self.store.get_session(session_id)
        if s is None:
            raise KeyError(session_id)
        if s["status"] == "ended":
            raise SessionStateError("Cannot resume an ended session")
        self.engine.end_session_hook(session_id)
        return self.store.update_session(session_id, status="active")

    def report(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self.store.get_session(session_id)
        if s is None:
            raise KeyError(session_id)
        return s["report"] or build_report(self.store, session_id, s.get("end_sync") or {}, self.escalation.for_session(session_id))
