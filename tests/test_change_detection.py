"""Tests for change detection."""

import numpy as np

from ml.change_detector import ChangeDetector


def test_no_history_zero_change():
    detector = ChangeDetector(threshold=0.30)
    current = np.array([1.0, 0.0, 0.0])
    result = detector.compare(current, [])
    assert result.change_magnitude == 0.0
    assert result.changed is False


def test_identical_low_change():
    detector = ChangeDetector(threshold=0.30)
    emb = np.array([0.6, 0.8, 0.0])
    emb = emb / np.linalg.norm(emb)
    result = detector.compare(emb, [emb.tolist()])
    assert result.change_magnitude < 0.1
    assert result.changed is False


def test_different_high_change():
    detector = ChangeDetector(threshold=0.30)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    result = detector.compare(a, [b.tolist()])
    assert result.change_magnitude > 0.5
    assert 0.0 <= result.change_magnitude <= 1.0


def test_change_magnitude_bounded():
    detector = ChangeDetector()
    for _ in range(10):
        a = np.random.randn(16).astype(np.float32)
        b = np.random.randn(16).astype(np.float32)
        result = detector.compare(a, [b.tolist()])
        assert 0.0 <= result.change_magnitude <= 1.0
