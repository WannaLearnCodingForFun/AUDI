"""Tests for similarity utilities."""

import numpy as np

from ml.similarity import batch_cosine_similarity, cosine_similarity, top_k_similar


def test_cosine_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_batch_cosine():
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = batch_cosine_similarity(query, candidates)
    assert sims[0] > sims[1]


def test_top_k():
    query = np.array([1.0, 0.0])
    candidates = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    ids = ["a", "b", "c"]
    results = top_k_similar(query, candidates, ids, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"
