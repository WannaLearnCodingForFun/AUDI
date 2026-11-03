"""Phase 3 sensing / perception: calibration (3.6), thermal fusion (3.1), quality gate (3.4),
focal loss + copy-paste (3.5), Siamese change net training (2.1), edge export (4.1)."""


import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from ml.calibration import calibrate, measure_length
from ml.camera_model import CameraIntrinsics, load_intrinsics
from ml.quality_gate import QualityGate
from ml.thermal import ThermalFrame, fuse, hotspots, register, thermal_features, warp_to_rgb
from tests.conftest import make_panel


def chessboard(cols=9, rows=6, sq=40, margin=60):
    w, h = (cols + 1) * sq + 2 * margin, (rows + 1) * sq + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[margin + r * sq:margin + (r + 1) * sq, margin + c * sq:margin + (c + 1) * sq] = 0
    return img


def test_calibration_recovers_focal_length_from_synthetic_views():
    board = chessboard()
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    bh, bw = board.shape
    sq_m = 0.025
    views = []
    rng = np.random.default_rng(0)
    for i in range(8):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.45, 0.6)])
        R, _ = cv2.Rodrigues(rvec)
        # plane points in metres (board pixel → metres), centred
        s = sq_m / 40.0
        corners = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        world = np.c_[(corners - [bw / 2, bh / 2]) * s, np.zeros(4)]
        proj = (K @ (R @ world.T + tvec[:, None])).T
        img_pts = (proj[:, :2] / proj[:, 2:]).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(corners, img_pts)
        views.append(cv2.warpPerspective(board, Hm, (640, 480), borderValue=255))
    intr, stats = calibrate(views, (9, 6), sq_m)
    assert stats["views_used"] >= 6 and stats["rms_reprojection_px"] < 1.0
    assert intr.fx == pytest.approx(700, rel=0.03) and intr.cx == pytest.approx(320, abs=15)
    # tape-measure check: 1 m seen at 2 m with fx = 700 spans 350 px
    k = CameraIntrinsics(700, 700, 320, 240, 640, 480)
    assert measure_length((145, 240), (495, 240), k, 2.0) == pytest.approx(1.0, rel=1e-6)


def test_intrinsics_file_is_loaded_and_rescaled(config, tmp_path):
    k = CameraIntrinsics(1000, 1000, 960, 540, 1920, 1080, [], "calibration")
    p = tmp_path / "cal.json"
    k.save(p)
    config["camera_model"]["calibration_file"] = str(p)
    s = load_intrinsics(config, 960, 540)
    assert s.fx == 500 and s.cx == 480 and s.source == "calibration"


def test_quality_gate_rejects_blur_and_exposure(config):
    g = QualityGate(config)
    sharp = make_panel(1, (640, 480))
    assert g.check_frame(sharp).ok
    blurred = Image.fromarray(cv2.GaussianBlur(np.asarray(sharp), (0, 0), 6))
    assert "blur" in g.check_frame(blurred).reasons
    assert "underexposed" in g.check_frame(Image.fromarray((np.asarray(sharp) * 0.05).astype(np.uint8))).reasons
    assert "too_far" in g.check_detection(sharp, "solar_panel", distance_m=200).reasons
    assert "steep_angle" in g.check_detection(sharp, "solar_panel", view_ray_pitch_deg=10).reasons
    assert g.check_detection(sharp, "utility_pole", view_ray_pitch_deg=20).ok
    assert g.counters["rejected"] >= 4


def test_thermal_hotspot_and_registration(config):
    rgb = np.asarray(make_panel(4, (320, 240)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # thermal camera: half resolution, shifted by 3 px, temperature follows the scene + a hot cell
    th = cv2.resize(gray, (160, 120)) / 255.0 * 15 + 25
    M = np.float32([[1, 0, 3], [0, 1, 2]])
    th = cv2.warpAffine(th, M, (160, 120), borderMode=cv2.BORDER_REFLECT)
    th[50:56, 80:86] += 18.0
    tf = ThermalFrame(th)
    H = register(tf, rgb)
    assert H is not None
    true_H = np.diag([2.0, 2.0, 1.0]) @ np.array([[1, 0, -3], [0, 1, -2], [0, 0, 1]])
    p = np.array([80.0, 60.0, 1.0])
    q, qt = H @ p, true_H @ p
    assert np.hypot(*(q[:2] / q[2] - qt[:2] / qt[2])) < 2.0
    aligned = warp_to_rgb(tf, H, (320, 240))
    rep = hotspots(aligned[80:150, 120:220], config)
    assert rep.severity in ("serious", "critical") and rep.hotspots
    f = thermal_features(aligned[80:150, 120:220])
    assert f.shape == (12,) and np.isfinite(f).all()
    v = fuse(np.ones(8, np.float32), f)
    assert v.shape == (20,) and abs(np.linalg.norm(v) - 1) < 1e-5


def test_engine_reports_thermal_hotspot(config, embedder, classifier):
    from memory.store import LocalStore
    from ml.detector import FullFrameDetector
    from ml.semantic_memory import SemanticMemoryEngine

    store = LocalStore(config)
    store.ensure_site("site_t", "T")
    e = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    sid = store.create_session("site_t", "op", "offline", {})["session_id"]
    frame = make_panel(5, (320, 240))
    th = np.full((240, 320), 30.0, np.float32)
    th[100:120, 150:170] = 55.0
    ev = e.process_frame(frame, sid, thermal=ThermalFrame(th), thermal_homography=np.eye(3))[0]
    assert ev.thermal["severity"] == "critical"
    assert any("thermal hotspot" in r for r in ev.explanation["reasons"])


def test_focal_loss_downweights_easy_examples():
    from training.train_condition_classifier import focal_loss

    logits = torch.tensor([[5.0, -5.0], [0.1, 0.0]])
    y = torch.tensor([0, 1])
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert focal_loss(logits, y, 0.0) == pytest.approx(float(ce))
    per = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    fl = focal_loss(logits, y, 2.0)
    # the easy example (p≈1) is suppressed far more than the hard one, so the hard one dominates
    easy_share_ce = float(per[0] / per.sum())
    pt = torch.exp(-per)
    w = (1 - pt) ** 2 * per
    assert float(fl) < float(ce) and float(w[0] / w.sum()) < easy_share_ce


def test_siamese_change_net_learns_synthetic_changes(tmp_path):
    from training.train_change_detector import main

    m = main(["--pairs", "96", "--val-pairs", "24", "--size", "64", "--epochs", "6", "--batch", "16", "--width", "8",
              "--out", str(tmp_path / "cd.pt"), "--metrics", str(tmp_path / "m.json")])
    assert (tmp_path / "cd.pt").exists()
    assert m["mask_f1"] is not None and m["mask_f1"] > 0.15
    from ml.siamese_change import SiameseChangeNet

    net = SiameseChangeNet.load(tmp_path / "cd.pt")
    out = net.predict_mask(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
    assert out.shape == (64, 64)


def test_reid_head_training_beats_or_matches_baseline_on_synthetic(embedder, monkeypatch):
    import training.train_reid_head as tr
    from types import SimpleNamespace

    imgs, ids = tr.build_views(40, 4, seed=3, size=96)
    emb = embedder.encode_batch(imgs)
    args = SimpleNamespace(val_frac=0.25, seed=3, dim=64, hidden=128, loss="arcface", lr=1e-3, epochs=30, batch=64, margin=0.3)
    head, m = tr.train(emb, ids, args)
    assert m["reid_head"]["rank1"] >= m["baseline_global"]["rank1"] - 0.05
    assert 0 <= m["reid_head"]["mAP"] <= 1


def test_edge_export_onnx(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from ml.reid_head import IdentityHead
    from ml.siamese_change import SiameseChangeNet
    from scripts.export_edge import export_module

    h = IdentityHead(32, 16, 32)
    p = export_module(h, (torch.randn(2, 32),), tmp_path / "reid.onnx", ["embedding"], ["identity"], {"embedding": {0: "b"}, "identity": {0: "b"}})
    x = np.random.rand(3, 32).astype(np.float32)
    h.eval()
    got = ort.InferenceSession(str(p)).run(None, {"embedding": x})[0]
    np.testing.assert_allclose(got, h(torch.from_numpy(x)).detach().numpy(), atol=1e-5)
    net = SiameseChangeNet(8)
    q = export_module(net, (torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)), tmp_path / "cd.onnx", ["reference", "current"], ["logits"],
                      {"reference": {0: "b"}, "current": {0: "b"}})
    r = ort.InferenceSession(str(q)).run(None, {"reference": np.zeros((1, 3, 64, 64), np.float32), "current": np.zeros((1, 3, 64, 64), np.float32)})[0]
    assert r.shape == (1, 1, 64, 64)
