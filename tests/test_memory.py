"""Tests for semantic memory store."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from memory.local_store import LocalSemanticMemoryStore
from ml.schemas import AssetRecord, Location, Observation, generate_asset_id, utc_now_iso


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "test_memory.json"
    return LocalSemanticMemoryStore(store_path=path)


def _make_asset(asset_id=None):
    emb = np.random.randn(32).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    ts = utc_now_iso()
    obs = Observation(
        observation_id="obs_test",
        observation_ts=ts,
        condition_description="normal",
        image_reference="test.png",
        embedding=emb.tolist(),
    )
    return AssetRecord(
        asset_id=asset_id or generate_asset_id(),
        asset_type="solar_panel",
        location=Location(37.0, -122.0, 50.0),
        first_observed_ts=ts,
        current_condition="normal",
        current_embedding=emb.tolist(),
        observations=[obs],
    )


def test_save_and_get_asset(store):
    asset = _make_asset()
    store.save_asset(asset.to_dict())
    retrieved = store.get_asset(asset.asset_id)
    assert retrieved is not None
    assert retrieved["asset_id"] == asset.asset_id


def test_add_observation_preserves_history(store):
    asset = _make_asset()
    store.save_asset(asset.to_dict())
    n_initial = len(store.get_history(asset.asset_id))

    new_obs = {
        "observation_id": "obs_new",
        "observation_ts": utc_now_iso(),
        "condition_description": "minor_damage",
        "image_reference": "test2.png",
        "embedding": asset.current_embedding,
        "severity_score": 0.3,
        "change_magnitude": 0.2,
        "confidence": 0.9,
    }
    store.add_observation(asset.asset_id, new_obs)
    history = store.get_history(asset.asset_id)
    assert len(history) == n_initial + 1
    assert history[0]["condition_description"] == "normal"
    assert history[-1]["condition_description"] == "minor_damage"


def test_search_assets(store):
    asset = _make_asset()
    store.save_asset(asset.to_dict())
    results = store.search_assets(asset.current_embedding, top_k=1)
    assert len(results) >= 1
    assert results[0]["asset_id"] == asset.asset_id


def test_offline_operation_no_cloud(store):
    """Local store works without cloud DB."""
    asset = _make_asset()
    store.save_asset(asset.to_dict())
    assert store.get_asset(asset.asset_id) is not None
