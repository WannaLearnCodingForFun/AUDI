"""End-to-end semantic memory pipeline tests."""

import numpy as np
import pytest
from PIL import Image

from memory.local_store import LocalSemanticMemoryStore
from ml.asset_encoder import AssetEncoder
from ml.semantic_memory import SemanticMemoryEngine


def _colored_image(r, g, b, size=224):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = [r, g, b]
    return Image.fromarray(arr)


def _pattern_image(seed: int, size=224) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    arr[30:190, 30:190] = [(seed * 41) % 256, (seed * 67) % 256, (seed * 89) % 256]
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def encoder():
    return AssetEncoder(model_name="projection", device="cpu")


@pytest.fixture
def engine(tmp_path, encoder):
    from ml.config import load_config

    config = load_config(reload=True)
    store = LocalSemanticMemoryStore(store_path=tmp_path / "e2e_memory.json", config=config)
    return SemanticMemoryEngine(store=store, encoder=encoder, config=config)


def test_new_asset_registration(engine):
    img = _colored_image(30, 144, 255)
    result = engine.process_observation(
        img, asset_type="solar_panel", condition="normal", force=True
    )
    assert result.get("is_new_asset") is True
    assert "asset_id" in result
    assert result["change_magnitude"] == 0.0


def test_existing_asset_match(engine):
    img_a = _colored_image(30, 144, 255)
    r1 = engine.process_observation(img_a, asset_type="solar_panel", force=True)
    asset_id = r1["asset_id"]

    r2 = engine.process_observation(img_a, asset_type="solar_panel", force=True)
    assert r2.get("is_new_asset") is False
    assert r2["asset_id"] == asset_id


def test_history_retrieved(engine):
    img = _colored_image(65, 105, 225)
    r1 = engine.process_observation(img, asset_type="solar_panel", force=True)
    engine.process_observation(img, asset_type="solar_panel", condition="normal", force=True)

    history = engine.get_asset_history(r1["asset_id"])
    assert history["found"] is True
    assert len(history["observations"]) >= 2


def test_historical_observations_never_deleted(engine):
    img = _colored_image(139, 69, 19)
    r1 = engine.process_observation(img, asset_type="utility_pole", force=True)
    aid = r1["asset_id"]

    for i in range(3):
        engine.process_observation(
            img, asset_type="utility_pole", condition=f"obs_{i}", force=True
        )

    history = engine.get_asset_history(aid)
    conditions = [o["condition_description"] for o in history["observations"]]
    assert "normal" in conditions or "obs_0" in conditions
    assert len(history["observations"]) >= 4


def test_different_asset_registered(tmp_path):
    """Verify distinct embeddings register as separate assets (model-independent)."""
    from unittest.mock import MagicMock

    from ml.config import load_config

    dim = 64
    emb_a = np.zeros(dim, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(dim, dtype=np.float32)
    emb_b[1] = 1.0

    call_count = [0]

    def fake_encode(image, bbox=None):
        call_count[0] += 1
        return emb_a if call_count[0] == 1 else emb_b

    encoder = MagicMock()
    encoder.encode_asset = fake_encode

    config = load_config(reload=True)
    config["matching"]["similarity_threshold"] = 0.75
    store = LocalSemanticMemoryStore(store_path=tmp_path / "distinct_assets.json", config=config)
    engine = SemanticMemoryEngine(store=store, encoder=encoder, config=config)

    img = _pattern_image(seed=1)
    r1 = engine.process_observation(img, asset_type="solar_panel", force=True)
    r2 = engine.process_observation(img, asset_type="building", force=True)
    assert r1["asset_id"] != r2["asset_id"]


def test_change_magnitude_in_result(engine):
    img = _colored_image(100, 100, 100)
    engine.process_observation(img, asset_type="building", force=True)
    img2 = _colored_image(200, 50, 50)
    r = engine.process_observation(img2, asset_type="building", condition="damage", force=True)
    if not r.get("is_new_asset"):
        assert 0.0 <= r.get("change_magnitude", 0) <= 1.0
        assert 0.0 <= r.get("severity_score", 0) <= 1.0
