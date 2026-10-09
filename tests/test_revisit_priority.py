"""Revisit priority §8.2 — D15 windowed trend, D9 time decay at query time."""

from datetime import datetime, timedelta, timezone

from ml.revisit_priority import calculate_trend, compute_revisit, time_decay


def test_d15_trend_uses_only_last_window():
    # A long rise followed by a flat window → trend 0 with window 5.
    history = [0.0, 0.2, 0.4, 0.6, 0.8] + [0.8] * 5
    assert calculate_trend(history, 5) == 0.0
    assert calculate_trend(history, 10) > 0.0


def test_d15_trend_clipped():
    assert calculate_trend([0.0, 1.0], 5) == 1.0
    assert calculate_trend([1.0, 0.0], 5) == 0.0
    assert calculate_trend([0.5], 5) == 0.0
    assert 0.0 < calculate_trend([0.1, 0.2, 0.3], 5) <= 1.0


def test_d9_time_decay_grows_with_now():
    last = datetime(2026, 1, 1, tzinfo=timezone.utc)
    d1 = time_decay(last.isoformat(), last + timedelta(days=3), 30.0)
    d2 = time_decay(last.isoformat(), last + timedelta(days=15), 30.0)
    assert d1 < d2 == 0.5
    assert time_decay(last.isoformat(), last + timedelta(days=90), 30.0) == 1.0


def test_formula(config):
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    r = compute_revisit([0.2, 0.4], "2026-01-16T00:00:00+00:00", config, now)
    assert abs(r["revisit_priority"] - (0.5 * 0.4 + 0.3 * r["trend"] + 0.2 * 0.5)) < 1e-9
