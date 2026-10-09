"""Prototype index — D10 bounded prototypes and site/type partitioning."""

import numpy as np
import pytest

from memory.vector_memory import PrototypeIndex, faiss_available


def _unit(rng, dim):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_partition_filters_by_site_and_type():
    rng = np.random.default_rng(0)
    idx = PrototypeIndex(16, "numpy", 8, 0.05)
    v = _unit(rng, 16)
    idx.add(1, "a1", "site1", "solar_panel", v)
    idx.add(2, "a2", "site2", "solar_panel", v)
    idx.add(3, "a3", "site1", "building", v)
    hits = idx.search(v, "site1", "solar_panel", 5)
    assert [h.asset_id for h in hits] == ["a1"]


def test_prototypes_bounded_and_novel():
    rng = np.random.default_rng(1)
    idx = PrototypeIndex(16, "numpy", 3, 0.05)
    base = _unit(rng, 16)
    idx.add(1, "a", "s", "t", base)
    assert idx.plan_update("a", base) == (False, None)
    pid = 2
    for _ in range(10):
        add, evict = idx.plan_update("a", _unit(rng, 16))
        if add:
            if evict is not None:
                idx.remove(evict)
            idx.add(pid, "a", "s", "t", _unit(rng, 16))
            pid += 1
    assert len(idx.prototypes("a")) <= 3


def test_location_allowed_subset():
    rng = np.random.default_rng(2)
    idx = PrototypeIndex(16, "numpy", 8, 0.05)
    v = _unit(rng, 16)
    idx.add(1, "near", "s", "t", v)
    idx.add(2, "far", "s", "t", v)
    assert [c.asset_id for c in idx.search(v, "s", "t", 5, allowed_assets={"near"})] == ["near"]


def test_faiss_is_opt_in():
    idx = PrototypeIndex(16, "numpy", 8, 0.05)
    assert idx.backend_name == "numpy"
    with pytest.raises(ValueError):
        PrototypeIndex(16, "auto", 8, 0.05)
    if not faiss_available():
        with pytest.raises(ImportError):
            PrototypeIndex(16, "faiss", 8, 0.05)


def test_nfr2_index_search_with_5000_assets(config):
    """NFR-2 guard for the shortlist stage (full cascade incl. SIFT is measured by scripts/bench.py)."""
    import time

    rng = np.random.default_rng(3)
    dim, n = 384, 5000
    idx = PrototypeIndex(dim, "numpy", 8, 0.05)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(i, f"a{i}", "s", "t", v)
    times, hits = [], 0
    for i in rng.choice(n, 200, replace=False):
        q = vecs[i] + rng.standard_normal(dim).astype(np.float32) * 0.1
        t = time.perf_counter()
        top = idx.search(q, "s", "t", 5)
        times.append((time.perf_counter() - t) * 1000)
        hits += top[0].asset_id == f"a{i}"
    assert hits == 200
    assert np.percentile(times, 95) <= config["bench"]["targets"]["matching_p95_ms"]
