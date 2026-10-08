"""Tests for severity scoring."""

from ml.severity import calculate_severity, is_priority


def test_severity_formula():
    s = calculate_severity(change_magnitude=0.5, asset_criticality=0.8, observation_confidence=0.9)
    expected = (0.6 * 0.5 + 0.4 * 0.8) * 0.9
    assert abs(s - expected) < 1e-6


def test_severity_bounded():
    for m in [0.0, 0.5, 1.0]:
        for c in [0.0, 0.5, 1.0]:
            for cf in [0.0, 0.5, 1.0]:
                s = calculate_severity(m, c, cf)
                assert 0.0 <= s <= 1.0


def test_priority_threshold():
    assert is_priority(0.6) is True
    assert is_priority(0.3) is False
