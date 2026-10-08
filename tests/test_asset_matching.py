"""Tests for asset matching."""

import numpy as np

from ml.memory_matcher import AssetMatcher


def test_match_above_threshold():
    matcher = AssetMatcher()
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([[0.99, 0.01, 0.0], [0.0, 1.0, 0.0]])
    ids = ["asset_a", "asset_b"]
    result = matcher.match(query, candidates, ids, threshold=0.75)
    assert result.matched is True
    assert result.asset_id == "asset_a"
    assert result.similarity > 0.75


def test_no_match_below_threshold():
    matcher = AssetMatcher()
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ids = ["asset_x", "asset_y"]
    result = matcher.match(query, candidates, ids, threshold=0.75)
    assert result.matched is False
    assert result.asset_id is None


def test_empty_candidates():
    matcher = AssetMatcher()
    result = matcher.match(np.array([1.0, 0.0]), np.zeros((0, 2)), [], threshold=0.75)
    assert result.matched is False
