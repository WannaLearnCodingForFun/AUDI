"""Tests for embedding model."""

import numpy as np
import pytest
from PIL import Image

from ml.embedding_model import AssetEmbeddingModel
from ml.similarity import cosine_similarity


@pytest.fixture(scope="module")
def model():
    return AssetEmbeddingModel(model_name="projection", device="cpu")


def _solid_image(color, size=224):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr)


def _pattern_image(seed: int, size=224) -> Image.Image:
    """Structurally distinct image for discrimination tests."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    # Add deterministic geometric pattern
    arr[50:150, 50:150] = [(seed * 37) % 256, (seed * 59) % 256, (seed * 83) % 256]
    return Image.fromarray(arr)


def test_encode_single_image(model):
    img = _solid_image((100, 150, 200))
    emb = model.encode_image(img)
    assert emb.shape == (model.embedding_dim,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_encode_batch(model):
    images = [_solid_image((i * 30, 50, 80)) for i in range(3)]
    batch = model.encode_batch(images)
    assert batch.shape == (3, model.embedding_dim)
    norms = np.linalg.norm(batch, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_same_image_high_similarity(model):
    img = _solid_image((120, 80, 40))
    e1 = model.encode_image(img)
    e2 = model.encode_image(img)
    assert cosine_similarity(e1, e2) > 0.99


def test_different_images_lower_similarity(model):
    img_a = _pattern_image(seed=1)
    img_b = _pattern_image(seed=999)
    e_a = model.encode_image(img_a)
    e_b = model.encode_image(img_b)
    sim = cosine_similarity(e_a, e_b)
    # Identical inputs must be ~1.0; distinct patterns should not be identical
    assert cosine_similarity(e_a, e_a) > 0.99
    assert sim < 1.0
    assert not np.allclose(e_a, e_b, atol=1e-3)
