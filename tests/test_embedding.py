"""Backbones — normalisation and D12 (dimension derived from the model)."""

import numpy as np

from ml.device import resolve_device
from tests.conftest import make_panel


def test_embeddings_unit_norm(embedder):
    e = embedder.encode_batch([make_panel(1), make_panel(2)])
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_d12_dim_derived_from_model(embedder):
    e = embedder.encode_image(make_panel(3))
    assert e.shape == (embedder.embedding_dim,)
    g, p = embedder.encode_with_patches([make_panel(3)])
    assert p.shape[1] == p.shape[2] == embedder.patch_grid


def test_deterministic(embedder):
    a = embedder.encode_image(make_panel(4))
    b = embedder.encode_image(make_panel(4))
    assert np.allclose(a, b)


def test_device_explicit():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}
