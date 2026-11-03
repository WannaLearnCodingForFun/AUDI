"""2.1 pixel change, 3.3 segmentation, 2.2 defect-level tracking, 2.3 forecast, revisit integration."""

import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from memory.store import LocalStore
from ml.damage_synth import DamageSpec, change_pair, copy_paste, render_damage
from ml.defect_tracking import MeasuredRegion, match_regions
from ml.detector import FullFrameDetector
from ml.forecast import family, forecast_series, holdout_error_days
from ml.pixel_change import PixelChangeDetector
from ml.revisit_priority import compute_revisit
from ml.segmentation import ClassicalSegmenter, regions_from_mask, skeleton_length
from ml.semantic_memory import SemanticMemoryEngine
from ml.siamese_change import mask_metrics
from tests.conftest import add_crack, make_panel

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def flat_panel(seed=0, size=(320, 240)):
    r = np.random.default_rng(seed)
    arr = np.full((size[1], size[0], 3), (70, 90, 140), np.float32) + r.normal(0, 4, (size[1], size[0], 3))
    arr[::40] *= 0.8
    arr[:, ::40] *= 0.8
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── 2.1 pixel change ─────────────────────────────────────────────────────
def test_pixel_change_finds_new_crack(config):
    base = flat_panel(1)
    before, after, gt = change_pair(base, DamageSpec("crack", 3, (0.3, 0.4)), 0.0, 0.5)
    cm = PixelChangeDetector(config).detect(after, before, None, cm2_per_current_px=0.01)
    m = mask_metrics(cm.mask, (np.asarray(Image.fromarray(gt * 255).resize(cm.mask.shape[::-1])) > 127).astype(np.uint8))
    assert m["recall"] > 0.6 and m["precision"] > 0.3, m
    assert cm.change_type == "crack"
    assert cm.changed_area_cm2 == pytest.approx(cm.changed_area_px * 0.01, rel=1e-6)


def test_lighting_change_is_not_change(config):
    base = flat_panel(2)
    brighter = Image.fromarray(np.clip(np.asarray(base).astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8))
    cm = PixelChangeDetector(config).detect(brighter, base, None)
    assert cm.changed_fraction < 0.002


def test_rust_and_soiling_change_types(config):
    det = PixelChangeDetector(config)
    base = flat_panel(3)
    rust, _ = render_damage(base, DamageSpec("rust", 4, (0.5, 0.5)), 0.4)
    assert det.detect(rust, base, None).change_type == "rust"
    soil, _ = render_damage(base, DamageSpec("soiling", 5), 0.3)
    assert det.detect(soil, base, None).change_type == "soiling"


# ── 3.3 segmentation ─────────────────────────────────────────────────────
def test_classical_crack_segmentation_length(config):
    base = flat_panel(4)
    img, gt = render_damage(base, DamageSpec("crack", 9, (0.2, 0.5)), 0.6)
    seg = ClassicalSegmenter(config).segment(img, "crack")
    import cv2

    # crack masks are scored with a 1-px tolerance band (standard for thin-structure masks)
    tol = cv2.dilate(gt, np.ones((3, 3), np.uint8), iterations=2)
    recall = mask_metrics(seg, gt)["recall"]
    precision = mask_metrics(seg, tol)["precision"]
    assert recall > 0.6 and precision > 0.8, (recall, precision)
    gl, sl = skeleton_length(gt), skeleton_length(seg)
    assert abs(sl - gl) / gl < 0.25
    regs = regions_from_mask(seg, 0.0005, "crack")
    assert regs and all(0 <= r.centroid[0] <= 1 for r in regs)


def test_copy_paste_augmentation_keeps_mask_consistent():
    donor, dmask = render_damage(flat_panel(6), DamageSpec("crack", 2), 0.5)
    out, mask = copy_paste(flat_panel(7), donor, dmask, random.Random(1))
    assert mask.sum() > 0 and out.size == (320, 240)
    assert np.asarray(out)[mask > 0].mean() < np.asarray(flat_panel(7))[mask > 0].mean()


# ── 2.2 matching on simulated growth sequences ───────────────────────────
def _measure(img, spec_kind, config):
    seg = ClassicalSegmenter(config).segment(img, spec_kind)
    return [MeasuredRegion(r.polygon, r.bbox, r.centroid, r.area_norm, r.area_px, r.length_px, spec_kind)
            for r in regions_from_mask(seg, 0.0005, spec_kind)]


def test_defect_matching_accuracy_and_size_error_on_growth_sequences(config):
    """≥95% of visits matched to the right instance, mean size error ≤ 10% (plan 2.2)."""
    correct = total = 0
    size_err = []
    for seq in range(12):
        base = flat_panel(100 + seq)
        specs = [DamageSpec("crack", 1000 + seq, (0.15, 0.25)), DamageSpec("crack", 2000 + seq, (0.7, 0.7))]
        instances = []
        for visit, ext in enumerate([0.15, 0.22, 0.3, 0.38]):
            img, gts = base, []
            for sp in specs:
                img, g = render_damage(img, sp, ext)
                gts.append(g)
            regions = _measure(img, "crack", config)
            pairs, new, _ = match_regions(instances, regions)
            if visit == 0:
                instances = [{"defect_type": "crack", "polygon": r.polygon, "centroid_x": r.centroid[0], "centroid_y": r.centroid[1],
                              "truth": int(np.argmax([g[int(r.centroid[1] * 240), int(r.centroid[0] * 320)] for g in gts]))
                              if any(g[int(r.centroid[1] * 240), int(r.centroid[0] * 320)] for g in gts) else
                              int(np.argmin([np.hypot(*(np.argwhere(g).mean(0)[::-1] / [320, 240] - r.centroid)) for g in gts]))}
                             for r in regions]
                continue
            for i, j in pairs:
                r = regions[j]
                truth = int(np.argmin([np.hypot(*(np.argwhere(g).mean(0)[::-1] / [320, 240] - r.centroid)) for g in gts]))
                total += 1
                correct += int(instances[i]["truth"] == truth)
                gt_len = skeleton_length(gts[truth])
                size_err.append(abs(r.length_px - gt_len) / gt_len)
                instances[i].update(polygon=r.polygon, centroid_x=r.centroid[0], centroid_y=r.centroid[1])
            total += len(new)   # a split / unmatched growth counts as a miss
    assert total and correct / total >= 0.95, (correct, total)
    assert float(np.mean(size_err)) <= 0.10, np.mean(size_err)


# ── 2.3 forecast ─────────────────────────────────────────────────────────
def _series(values, days, key="length_cm"):
    return [{"ts": (T0 + timedelta(days=d)).isoformat(), key: v, "area_norm": v / 1000.0} for v, d in zip(values, days)]


def test_crack_power_law_fit_and_critical_date(config):
    a0, k, m = 5.0, 0.02, 1.8
    days = [0, 14, 28, 42, 56]
    vals = [a0 * (1 + k * d) ** m for d in days]
    f = forecast_series(_series(vals, days), "crack", config)
    assert f.model == "power" and f.metric == "length_cm"
    assert f.params["m"] == pytest.approx(m, rel=0.05) and f.rmse < 0.01
    true_t = ((30.0 / a0) ** (1 / m) - 1) / k
    assert f.days_to_critical == pytest.approx(true_t - 56, abs=0.5)
    assert f.growth_since_last_pct > 0


def test_corrosion_linear_and_soiling_reset(config):
    days = [0, 10, 20, 30]
    f = forecast_series(_series([100, 150, 200, 250], days, "area_cm2"), "rust", config)
    assert f.model == "linear" and f.params["rate_per_day"] == pytest.approx(5.0)
    assert f.days_to_critical == pytest.approx((500 - 250) / 5.0)
    soil = [{"ts": (T0 + timedelta(days=d)).isoformat(), "area_norm": v} for d, v in [(0, .05), (10, .15), (20, .25), (25, .02), (35, .12)]]
    s = forecast_series(soil, "dust", config)
    assert s.model == "linear_reset" and s.params["segment_start_day"] == 25
    assert s.params["rate_per_day"] == pytest.approx(0.01)
    assert s.days_to_critical == pytest.approx((0.30 - 0.12) / 0.01, abs=1e-6)


def test_forecast_deterministic_and_holdout_error(config):
    days = [0, 7, 14, 21, 28]
    vals = [4 * (1 + 0.03 * d) ** 1.5 for d in days]
    s = _series(vals, days)
    assert forecast_series(s, "crack", config).to_dict() == forecast_series(s, "crack", config).to_dict()
    err = holdout_error_days(s, "crack", config)
    assert err is not None and abs(err) < 1.0
    assert family("bird_drop") == "soiling" and family("corrosion") == "corrosion"


def test_revisit_uses_forecast_date_instead_of_slope(config):
    flat = compute_revisit([0.6, 0.6, 0.6], T0.isoformat(), config, T0)
    soon = compute_revisit([0.6, 0.6, 0.6], T0.isoformat(), config, T0, days_to_critical=9.0)
    assert flat["trend_source"] == "severity_slope" and flat["trend"] == 0.0
    assert soon["trend_source"] == "forecast" and soon["trend"] == pytest.approx(0.9)
    assert soon["revisit_priority"] > flat["revisit_priority"]


# ── engine integration: one crack, measured across visits ────────────────
def test_engine_tracks_one_growing_crack_and_forecasts(config, embedder, classifier):
    config["geo"]["identity_mode"] = "appearance"
    store = LocalStore(config)
    store.ensure_site("site_d", "Defects")
    e = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    sids = []
    first = None
    for visit, width in enumerate([4, 6, 8, 10]):
        sid = store.create_session("site_d", "op", "offline", {})["session_id"]
        sids.append(sid)
        img = make_panel(61) if visit == 0 else add_crack(make_panel(61), width)
        ts = (T0 + timedelta(days=14 * visit)).isoformat()
        ev = e.observe(img, "solar_panel", sid, timestamp=ts)
        first = first or ev.asset_id
        assert ev.asset_id == first
        assert ev.explanation and ev.explanation["terms"]["S"] == ev.severity_score
        store.update_session(sid, status="ended", ended_ts=ts)
    inst = store.list_defect_instances(asset_id=first)
    active = [i for i in inst if i["status"] == "active"]
    assert active, "the crack must become a tracked defect instance"
    biggest = max(active, key=lambda i: len(store.instance_measurements(i["instance_id"])))
    ms = store.instance_measurements(biggest["instance_id"])
    assert len(ms) >= 2
    assert ms[-1]["area_norm"] >= ms[0]["area_norm"]
    assert biggest["forecast"] and biggest["forecast"]["model"] == "power"
    a = store.get_asset(first)
    assert a["health_index"] is not None and 0 <= a["health_index"] < 100
