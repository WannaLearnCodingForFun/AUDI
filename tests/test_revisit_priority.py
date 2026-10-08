"""Tests for revisit priority."""

from ml.revisit_priority import calculate_revisit_priority, calculate_trend


def test_revisit_priority_bounded():
    r = calculate_revisit_priority(0.8, 0.5, 0.3)
    assert 0.0 <= r <= 1.0


def test_revisit_priority_formula():
    r = calculate_revisit_priority(0.8, 0.5, 0.3, w3=0.5, w4=0.3, w5=0.2)
    expected = 0.5 * 0.8 + 0.3 * 0.5 + 0.2 * 0.3
    assert abs(r - expected) < 1e-6


def test_trend_increasing():
    trend = calculate_trend([0.1, 0.2, 0.3, 0.4])
    assert trend > 0.0


def test_trend_flat():
    trend = calculate_trend([0.5, 0.5, 0.5])
    assert trend == 0.0
