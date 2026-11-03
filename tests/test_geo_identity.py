"""1.1 geo-anchored identity and 1.2 site-map footprints."""

import math

import numpy as np
import pytest
from PIL import Image

from memory.store import LocalStore
from ml.asset_identity import haversine_m
from ml.camera_model import CameraIntrinsics, distance_from_bbox, scale_for_bbox
from ml.detector import Detection, Detector, FullFrameDetector
from ml.geo_projection import CameraPose, camera_axes, latlon_to_offset, project_pixel
from ml.semantic_memory import SemanticMemoryEngine
from ml.site_map import MapRegistrar, SiteMap, geo_to_pixel, pixel_to_geo, point_in_polygon
from ml.schemas import utc_now
from tests.conftest import make_panel

LAT, LON = 47.0, 8.0


def intr(w=640, h=480, hfov=90.0):
    return CameraIntrinsics.from_hfov(w, h, hfov)


def test_camera_axes_are_orthonormal():
    for h, p, r in [(0, 90, 0), (37, 30, 5), (270, 10, -12)]:
        R = np.stack(camera_axes(h, p, r))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_nadir_centre_pixel_projects_below_the_drone():
    k = intr()
    fix = project_pixel(k.cx, k.cy, k, CameraPose(LAT, LON, 50.0, heading_deg=0, pitch_deg=90))
    assert abs(fix.east_m) < 1e-6 and abs(fix.north_m) < 1e-6
    assert fix.slant_range_m == pytest.approx(50.0)


def test_nadir_offsets_follow_heading():
    k = intr()  # hfov 90 → fx = 320: a pixel 320 right of centre is 1 rad-tan = altitude metres away
    north = project_pixel(k.cx + 320, k.cy, k, CameraPose(LAT, LON, 20.0, heading_deg=0, pitch_deg=90))
    assert north.east_m == pytest.approx(20.0, rel=1e-6) and abs(north.north_m) < 1e-6
    east = project_pixel(k.cx + 320, k.cy, k, CameraPose(LAT, LON, 20.0, heading_deg=90, pitch_deg=90))
    assert east.north_m == pytest.approx(-20.0, rel=1e-6)   # facing east, image-right points south


def test_oblique_projection_and_latlon_roundtrip():
    k = intr()
    fix = project_pixel(k.cx, k.cy, k, CameraPose(LAT, LON, 30.0, heading_deg=45, pitch_deg=45))
    assert fix.slant_range_m == pytest.approx(30.0 * math.sqrt(2))
    assert math.hypot(fix.east_m, fix.north_m) == pytest.approx(30.0, rel=1e-6)
    e, n = latlon_to_offset(LAT, LON, fix.latitude, fix.longitude)
    assert e == pytest.approx(fix.east_m, abs=0.01) and n == pytest.approx(fix.north_m, abs=0.01)
    assert haversine_m(LAT, LON, fix.latitude, fix.longitude) == pytest.approx(30.0, rel=1e-3)


def test_rays_above_horizon_do_not_hit_ground_and_grazing_is_uncertain():
    k = intr()
    assert project_pixel(k.cx, 0, k, CameraPose(LAT, LON, 30.0, pitch_deg=10)) is None
    steep = project_pixel(k.cx, k.cy, k, CameraPose(LAT, LON, 30.0, pitch_deg=80))
    grazing = project_pixel(k.cx, k.cy, k, CameraPose(LAT, LON, 30.0, pitch_deg=12))
    assert grazing.uncertainty_m > 3 * steep.uncertainty_m


def test_metric_scale_sources(config):
    k = intr()
    assert scale_for_bbox(config, k, (0, 0, 100, 100), "solar_panel", range_m=8.0).source == "range"
    assert scale_for_bbox(config, k, (0, 0, 100, 100), "solar_panel", altitude_m=10, pitch_deg=90).distance_m == pytest.approx(10)
    s = scale_for_bbox(config, k, (0, 0, 320, 100), "solar_panel")
    assert s.source == "size" and s.distance_m == pytest.approx(distance_from_bbox(k, (0, 0, 320, 100), 1.0))
    assert s.m_per_px * 320 == pytest.approx(1.0, rel=1e-6)     # 1 m wide panel spans its bbox


# ── engine: projected location separates identical-looking assets ────────
class SubBoxDetector(Detector):
    """One detection in the centre of the frame (a fixed sub-box so projection is meaningful)."""

    def detect(self, frame):
        w, h = frame.size
        return [Detection((w // 4, h // 4, 3 * w // 4, 3 * h // 4), "solar_panel", 0.95, "test")]


def _frame(seed):
    big = Image.new("RGB", (640, 480), (90, 90, 90))
    big.paste(make_panel(seed), (160, 120))
    return big


def _engine(config):
    store = LocalStore(config)
    store.ensure_site("site_g", "Geo site")
    return SemanticMemoryEngine(config, store, _engine.embedder, _engine.classifier, SubBoxDetector())


def _session(e):
    return e.store.create_session("site_g", "op", "offline", {})["session_id"]


@pytest.mark.parametrize("mode,expected_assets", [("location_appearance", 2), ("appearance", 1)])
def test_geo_gate_vs_appearance_only(config, embedder, classifier, mode, expected_assets):
    """Two identical panels 15 m apart: the projected fix keeps them apart; appearance alone merges them."""
    config["geo"]["identity_mode"] = mode
    _engine.embedder, _engine.classifier = embedder, classifier
    e = _engine(config)
    s1 = _session(e)
    pose_a = {"latitude": LAT, "longitude": LON, "altitude_agl": 20.0, "heading_deg": 0, "pitch_deg": 90, "gps_accuracy_m": 0.5}
    north_15m = LAT + 15.0 / 111_195.0
    pose_b = {**pose_a, "latitude": north_15m}
    ev_a = e.process_frame(_frame(3), s1, pose=pose_a)[0]
    ev_b = e.process_frame(_frame(3), s1, pose=pose_b)[0]
    assert ev_a.geo and ev_a.geo["ground_fix"]["uncertainty_m"] < 3.0
    ids = {ev_a.asset_id, ev_b.asset_id}
    assert len(ids) == expected_assets
    # revisit A: geo-gated mode re-identifies A specifically
    e.store.update_session(s1, status="ended", ended_ts=utc_now().isoformat())
    s2 = _session(e)
    again = e.process_frame(_frame(3), s2, pose={**pose_a, "longitude": LON + 1.0 / 76_000})[0]   # ~1 m east
    assert again.asset_id == ev_a.asset_id
    if mode == "location_appearance":
        a = e.store.get_asset(ev_a.asset_id)
        assert haversine_m(LAT, LON, a["location"]["latitude"], a["location"]["longitude"]) < 1.5


def test_no_gps_falls_back_to_appearance(config, embedder, classifier):
    _engine.embedder, _engine.classifier = embedder, classifier
    e = _engine(config)
    s1 = _session(e)
    first = e.process_frame(_frame(8), s1)[0]
    e.store.update_session(s1, status="ended", ended_ts=utc_now().isoformat())
    s2 = _session(e)
    assert e.process_frame(_frame(8), s2)[0].asset_id == first.asset_id


# ── site map ─────────────────────────────────────────────────────────────
def _ortho(seed=11, size=(1200, 900)):
    r = np.random.default_rng(seed)
    w, h = size
    arr = (r.random((h // 8, w // 8)) * 255).astype(np.uint8)
    import cv2

    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
    rgb = np.stack([arr, np.roll(arr, 7, 0), np.roll(arr, 13, 1)], -1)
    return Image.fromarray(rgb)


GT = [LON, 1e-6, 0.0, LAT, 0.0, -1e-6]   # ~0.08 m (lon) x 0.11 m (lat) per pixel


def test_geotransform_roundtrip_and_polygon():
    lat, lon = pixel_to_geo(GT, 100.5, 200.25)
    c, r = geo_to_pixel(GT, lat, lon)
    assert c == pytest.approx(100.5) and r == pytest.approx(200.25)
    assert point_in_polygon(5, 5, [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert not point_in_polygon(15, 5, [(0, 0), (10, 0), (10, 10), (0, 10)])


def _poly(c0, r0, c1, r1):
    return [list(pixel_to_geo(GT, c, r)) for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1))]


def test_frame_registers_to_map_and_hits_footprint(config, tmp_path):
    ortho = _ortho()
    smap = SiteMap("map_t", "s", np.asarray(ortho.convert("L")), GT)
    smap.set_footprints([
        {"asset_id": "A", "asset_type": "solar_panel", "footprint": _poly(300, 250, 500, 400)},
        {"asset_id": "B", "asset_type": "solar_panel", "footprint": _poly(700, 250, 900, 400)},
    ])
    reg = MapRegistrar(config)
    # a "drone frame": a slightly rotated, scaled view of the map around asset B
    view = ortho.crop((600, 150, 1000, 500)).rotate(4, resample=Image.BICUBIC).resize((480, 420))
    r = reg.register(view, smap, near=pixel_to_geo(GT, 800, 325))
    assert r is not None and r.inliers >= 25
    col, row = reg.frame_to_map(r, 240, 210)       # frame centre → map
    hit = smap.lookup_px(col, row, "solar_panel")
    assert hit is not None and hit.asset_id == "B"
    assert smap.lookup_px(col, row, "utility_pole") is None


def test_engine_uses_site_map_footprint(config, embedder, classifier, tmp_path):
    ortho = _ortho(5)
    path = tmp_path / "ortho.png"
    ortho.save(path)
    store = LocalStore(config)
    store.ensure_site("site_m", "Map site")
    store.add_site_map("site_m", str(path), GT, ortho.width, ortho.height)
    a = store.register_mapped_asset("site_m", "solar_panel", _poly(350, 300, 650, 600), 0.7)
    b = store.register_mapped_asset("site_m", "solar_panel", _poly(700, 300, 1000, 600), 0.7)

    class CentreBox(Detector):
        def detect(self, frame):
            w, h = frame.size
            return [Detection((w // 2 - 60, h // 2 - 60, w // 2 + 60, h // 2 + 60), "solar_panel", 0.95, "t")]

    e = SemanticMemoryEngine(config, store, embedder, classifier, CentreBox())
    sid = store.create_session("site_m", "op", "offline", {})["session_id"]
    view = ortho.crop((700, 300, 1000, 600))       # centred on asset B
    ev = e.process_frame(view, sid)[0]
    assert ev.geo["site_map"]["asset_id"] == b["asset_id"]
    assert ev.asset_id == b["asset_id"] and ev.identity["stage"] in {"site_map", "geometric"}
    assert ev.asset_id != a["asset_id"]
