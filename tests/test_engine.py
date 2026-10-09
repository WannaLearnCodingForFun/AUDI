"""Engine + Local DB flows: identity, sessions, de-dup, defects, revisit, recovery."""

from datetime import timedelta

import cv2
import numpy as np
import pytest
from PIL import Image

from memory.store import LocalStore
from ml.detector import FullFrameDetector
from ml.schemas import utc_now
from ml.semantic_memory import SemanticMemoryEngine
from ml.tracker import SessionTracker
from tests.conftest import add_crack, add_dust, make_panel

SITE = "site_t"


@pytest.fixture
def engine(config, embedder, classifier):
    store = LocalStore(config)
    store.ensure_site(SITE, "Test site")
    return SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))


def _session(engine):
    return engine.store.create_session(SITE, "op1", "offline", {})["session_id"]


def _end(engine, sid):
    engine.store.update_session(sid, status="ended", ended_ts=utc_now().isoformat())


def test_new_then_matched_across_sessions(engine):
    s1 = _session(engine)
    ev1 = engine.observe(make_panel(1), "solar_panel", s1)
    assert ev1.type == "new_asset" and ev1.persisted
    _end(engine, s1)
    s2 = _session(engine)
    ev2 = engine.observe(make_panel(1).rotate(3), "solar_panel", s2)
    assert ev2.asset_id == ev1.asset_id
    assert ev2.identity["stage"] == "geometric"
    assert ev2.type == "confirmed"


def test_d2_damaged_asset_still_matches(engine):
    """Condition change must not break identity: geometry survives crack + dust."""
    s1 = _session(engine)
    base = engine.observe(make_panel(7), "solar_panel", s1)
    _end(engine, s1)
    s2 = _session(engine)
    damaged = add_dust(add_crack(make_panel(7)))
    ev = engine.observe(damaged, "solar_panel", s2)
    assert ev.asset_id == base.asset_id
    assert ev.identity["stage"] == "geometric"
    assert ev.type == "change"
    assert ev.change["significant"]
    # FR-12: the match carries what memory held before this observation
    assert ev.history["previous_condition"] == base.condition["label"]
    assert ev.history["previous_image"] == base.image_reference
    assert ev.history["open_defects"] == []
    s3 = _session(engine)
    again = engine.observe(add_crack(make_panel(7)), "solar_panel", s3)
    opened = engine.store.list_defects(asset_id=base.asset_id)
    assert opened and [d["defect_id"] for d in again.history["open_defects"]] == [d["defect_id"] for d in opened]


def test_distinct_assets_not_merged(engine):
    s1 = _session(engine)
    a = engine.observe(make_panel(10), "solar_panel", s1)
    b = engine.observe(make_panel(11), "solar_panel", s1)
    assert a.asset_id != b.asset_id


def test_d4_fr16_unchanged_asset_confirmed_with_zero_severity(engine):
    s1 = _session(engine)
    first = engine.observe(make_panel(2), "solar_panel", s1)
    _end(engine, s1)
    before = engine.store.get_asset(first.asset_id)["last_checked_ts"]
    s2 = _session(engine)
    ev = engine.observe(make_panel(2), "solar_panel", s2)
    assert ev.type == "confirmed" and ev.severity_score == 0.0
    assert engine.store.get_asset(first.asset_id)["last_checked_ts"] > before


def test_d7_observation_requires_session(engine):
    ev = engine.observe(make_panel(3), "solar_panel", None)
    assert ev.type == "error" and "session" in ev.error
    s = _session(engine)
    ok = engine.observe(make_panel(3), "solar_panel", s)
    assert engine.store.observations(ok.asset_id)[0]["session_id"] == s


def test_d6_no_persist_writes_nothing(engine):
    s = _session(engine)
    ev = engine.observe(make_panel(4), "solar_panel", s, persist=False)
    assert ev.type == "new_asset" and not ev.persisted and ev.asset_id is None
    assert engine.store.list_assets() == []


def test_d8_one_observation_per_asset_per_session(engine):
    s = _session(engine)
    a = engine.observe(make_panel(5), "solar_panel", s)
    dup = engine.observe(make_panel(5), "solar_panel", s)
    assert dup.type == "duplicate" and dup.asset_id == a.asset_id
    assert len(engine.store.observations(a.asset_id)) == 1
    changed = engine.observe(add_crack(make_panel(5)), "solar_panel", s)
    assert changed.type == "change"
    assert len(engine.store.observations(a.asset_id)) == 2


def test_d8_tracker_processes_track_once(engine):
    s = _session(engine)
    tracker = SessionTracker.from_config(engine.config)
    frame = make_panel(6)
    evs = [engine.process_frame(frame, s, tracker)[0] for _ in range(4)]
    assert evs[0].type == "new_asset"
    assert [e.type for e in evs[1:]] == ["tracked"] * 3
    assert len(engine.store.observations(evs[0].asset_id)) == 1


def test_d8_tracker_does_not_merge_different_assets_in_same_box(engine):
    """Full-frame detection gives every asset the same box; appearance must split the tracks."""
    s = _session(engine)
    tracker = SessionTracker.from_config(engine.config)
    a, b = make_panel(21), make_panel(22)
    emb = engine.embedder.encode_batch([a, b])
    assert float(np.dot(emb[0], emb[1])) >= engine.config["tracker"]["embedding_similarity"]
    ev_a = engine.process_frame(a, s, tracker)[0]
    ev_b = engine.process_frame(b, s, tracker)[0]
    assert ev_a.type == "new_asset" and ev_b.type == "new_asset"
    assert ev_a.asset_id != ev_b.asset_id
    jittered = Image.fromarray(np.clip(np.asarray(b).astype(np.int16) + 6, 0, 255).astype(np.uint8))
    assert engine.process_frame(jittered, s, tracker)[0].type == "tracked"
    # Returning to the first asset after another one took over the region starts a new
    # track, so identity runs again (and per-session de-dup reports the duplicate).
    again = engine.process_frame(a, s, tracker)[0]
    assert again.type == "duplicate" and again.asset_id == ev_a.asset_id


def test_fr17_defect_lifecycle_and_resolution_suggestion(engine):
    n = engine.config["defects"]["resolution_suggestion_missions"]
    s = _session(engine)
    base = engine.observe(make_panel(8), "solar_panel", s)
    _end(engine, s)
    s = _session(engine)
    ev = engine.observe(add_crack(make_panel(8)), "solar_panel", s)
    assert ev.defect and ev.defect["opened"]["status"] == "Open"
    defect_id = ev.defect["opened"]["defect_id"]
    _end(engine, s)
    s = _session(engine)
    engine.observe(add_crack(make_panel(8)), "solar_panel", s)
    _end(engine, s)
    assert len(engine.store.get_defect(defect_id)["trend"]) == 2
    for _ in range(n):
        s = _session(engine)
        engine.observe(make_panel(8), "solar_panel", s)
        _end(engine, s)
    d = engine.store.get_defect(defect_id)
    assert d["resolution_suggested"] and d["status"] == "Open"  # suggested, never auto-resolved
    engine.store.update_defect_status(defect_id, "In Progress")
    assert engine.store.update_defect_status(defect_id, "Resolved")["resolved_ts"]
    with pytest.raises(ValueError):
        engine.store.update_defect_status(defect_id, "Closed")
    assert base.asset_id == d["asset_id"]


def test_fr20_recompute_includes_unseen_assets(engine):
    s = _session(engine)
    seen = engine.observe(make_panel(20), "solar_panel", s)
    unseen = engine.observe(make_panel(21), "solar_panel", s)
    later = utc_now() + timedelta(days=15)
    ranked = engine.store.recompute_revisit(SITE, later)
    by_id = {a["asset_id"]: a for a in ranked}
    assert by_id[unseen.asset_id]["revisit_components"]["days_since_inspection"] == pytest.approx(0.5, abs=0.01)
    assert by_id[seen.asset_id]["revisit_priority_score"] > 0


def test_aruco_marker_identity(engine):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, engine.config["aruco"]["dictionary"]))
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 120)
    def tagged(seed):
        img = np.asarray(make_panel(seed, (400, 300))).copy()
        img[20:160, 20:160] = 255
        img[30:150, 30:150] = np.stack([marker] * 3, axis=-1)
        return Image.fromarray(img)
    s = _session(engine)
    first = engine.observe(tagged(30), "solar_panel", s)
    assert engine.store.get_asset(first.asset_id)["marker_id"] == 7
    _end(engine, s)
    s = _session(engine)
    # Same marker on a visually different body → identity from the fiducial alone.
    ev = engine.observe(tagged(31), "solar_panel", s)
    assert ev.identity["stage"] == "aruco" and ev.asset_id == first.asset_id


def test_d3_location_gating_separates_identical_panels(engine):
    s = _session(engine)
    a = engine.observe(make_panel(40), "solar_panel", s, location={"latitude": 12.0, "longitude": 77.0})
    far = {"latitude": 12.01, "longitude": 77.0}  # ~1.1 km away
    b = engine.observe(make_panel(40), "solar_panel", s, location=far)
    assert b.type == "new_asset" and b.asset_id != a.asset_id
    near = {"latitude": 12.00005, "longitude": 77.0}
    _end(engine, s)
    s = _session(engine)
    c = engine.observe(make_panel(40), "solar_panel", s, location=near)
    assert c.asset_id == a.asset_id
    loc = engine.store.get_asset(a.asset_id)["location"]
    assert loc["latitude"] == pytest.approx((12.0 + 12.00005) / 2)  # running mean, not overwrite


def test_crash_mid_commit_rolls_back(engine, monkeypatch):
    s = _session(engine)
    engine.observe(make_panel(50), "solar_panel", s)
    counts_before = engine.store.outbox_counts()["pending"]
    def boom(*a, **k):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(engine.store, "_apply_revisit", boom)
    ev = engine.observe(make_panel(51), "solar_panel", s)
    assert ev.type == "error"
    assert len(engine.store.list_assets()) == 1
    assert engine.store.outbox_counts()["pending"] == counts_before
    assert engine.store.integrity_check()["ok"]


def test_restart_recovers_state_and_index(engine, config, embedder, classifier):
    s = _session(engine)
    a = engine.observe(make_panel(60), "solar_panel", s)
    # Simulate a process restart: new store + engine over the same DB file.
    store2 = LocalStore(config)
    assert s in store2.integrity["active_sessions"]
    eng2 = SemanticMemoryEngine(config, store2, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    assert len(eng2.index) >= 1
    store2.update_session(s, status="interrupted")
    s2 = store2.create_session(SITE, "op1", "offline", {})["session_id"]
    ev = eng2.observe(make_panel(60).rotate(2), "solar_panel", s2)
    assert ev.asset_id == a.asset_id
