"""1.4 merge and split: no observation is lost or duplicated; every action is audited."""

import pytest

from memory.store import LocalStore
from ml.detector import FullFrameDetector
from ml.schemas import utc_now
from ml.semantic_memory import SemanticMemoryEngine
from tests.conftest import add_crack, make_panel

SITE = "site_ms"


@pytest.fixture
def engine(config, embedder, classifier):
    config["geo"]["identity_mode"] = "appearance"
    store = LocalStore(config)
    store.ensure_site(SITE, "Merge site")
    return SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))


def _obs_ids(store, asset_id):
    return [o["observation_id"] for o in store.observations(asset_id)]


def _all_obs(store):
    out = []
    for a in store.list_assets(SITE):
        out += _obs_ids(store, a["asset_id"])
    return out


def _two_assets(engine):
    """The same physical panel registered twice (the failure mode merge repairs)."""
    store = engine.store
    s1 = store.create_session(SITE, "op", "offline", {})["session_id"]
    a = engine.observe(make_panel(31), "solar_panel", s1)
    engine.observe(add_crack(make_panel(31)), "solar_panel", s1, force=True)
    store.update_session(s1, status="ended", ended_ts=utc_now().isoformat())
    s2 = store.create_session(SITE, "op", "offline", {})["session_id"]
    b = engine.observe(make_panel(32), "solar_panel", s2)
    engine.observe(add_crack(make_panel(32)), "solar_panel", s2, force=True)
    assert a.asset_id != b.asset_id
    return a.asset_id, b.asset_id


def test_merge_moves_everything_and_deletes_source(engine):
    store = engine.store
    keep, gone = _two_assets(engine)
    before = sorted(_all_obs(store))
    obs_gone = set(_obs_ids(store, gone))
    defects_before = len(store.list_defects(site_id=SITE))
    res = store.merge_assets(keep, gone, "admin")
    after = sorted(_all_obs(store))
    assert after == before, "merge must not lose or duplicate observations"
    assert len(set(after)) == len(after)
    assert obs_gone <= set(_obs_ids(store, keep))
    assert store.get_asset(gone) is None
    assert res["counts"]["observations"] == len(obs_gone)
    # same-type open defects folded into one, none lost
    d = store.list_defects(asset_id=keep)
    assert len(d) <= defects_before and all(x["asset_id"] == keep for x in d)
    assert sum(len(store.get_defect(x["defect_id"])["trend"]) for x in d) >= 2
    log = store.list_audit(SITE)
    assert log[0]["action"] == "asset.merge" and log[0]["actor"] == "admin"
    pending = [r for r in store.pending_outbox(10_000) if r["entity"] == "asset_op"]
    assert pending and pending[-1]["payload"]["op"] == "merge"
    engine.reload_index()
    assert gone not in engine.index.asset_ids(SITE)


def test_split_then_merge_roundtrip(engine):
    store = engine.store
    keep, gone = _two_assets(engine)
    store.merge_assets(keep, gone, "admin")
    obs_all = _obs_ids(store, keep)
    moved = obs_all[2:]
    res = store.split_asset(keep, moved, "admin")
    new_id = res["new_asset"]["asset_id"]
    assert sorted(_obs_ids(store, new_id)) == sorted(moved)
    assert sorted(_obs_ids(store, keep)) == sorted(obs_all[:2])
    assert sorted(_all_obs(store)) == sorted(obs_all)
    assert store.list_audit(SITE)[0]["action"] == "asset.split"
    with pytest.raises(ValueError):
        store.split_asset(keep, _obs_ids(store, keep), "admin")    # would empty the source
    with pytest.raises(ValueError):
        store.split_asset(keep, ["obs_missing"], "admin")


def test_merge_validation(engine):
    store = engine.store
    keep, gone = _two_assets(engine)
    with pytest.raises(ValueError):
        store.merge_assets(keep, keep, "admin")
    with pytest.raises(KeyError):
        store.merge_assets(keep, "asset_missing", "admin")


def test_asset_ops_replay_on_second_device(engine, config, tmp_path):
    """A merge done on one device is replayed from the pulled op on another."""
    store = engine.store
    keep, gone = _two_assets(engine)
    other_cfg = {**config, "storage": {**config["storage"], "db_url": f"sqlite:///{tmp_path}/other.db"}}
    other = LocalStore(other_cfg, images_dir=tmp_path / "other_images")
    records = {}
    for r in store.pending_outbox(100_000):
        records.setdefault(r["entity"], []).append(r["payload"])
    other.apply_remote(records)
    assert other.get_asset(gone) is not None
    res = store.merge_assets(keep, gone, "admin")
    other.apply_remote({"asset_op": [res["op"]]})
    assert other.get_asset(gone) is None
    assert sorted(_obs_ids(other, keep)) == sorted(_obs_ids(store, keep))
    assert other.apply_asset_ops([res["op"]]) == 0      # idempotent
