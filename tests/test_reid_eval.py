"""Re-ID evaluation protocol (D5): gallery session vs later sessions, global-only vs cascade."""

import csv

import cv2
import numpy as np
from PIL import Image

from tests.conftest import add_crack, make_panel
from training.evaluate_reid import evaluate, load_records, write_report


def rephotograph(im: Image.Image, seed: int) -> Image.Image:
    """New viewpoint: small perspective warp plus a brightness change."""
    r = np.random.default_rng(seed)
    a = np.asarray(im)
    h, w = a.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + r.uniform(-0.05, 0.05, src.shape) * [w, h]).astype(np.float32)
    out = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(np.clip(out.astype(np.float32) * r.uniform(0.85, 1.15), 0, 255).astype(np.uint8))


def with_marker(im: Image.Image, config, marker_id: int) -> Image.Image:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["aruco"]["dictionary"]))
    m = cv2.aruco.generateImageMarker(d, marker_id, 90)
    tag = np.full((130, 130), 255, np.uint8)
    tag[20:110, 20:110] = m
    out = im.copy()
    out.paste(Image.fromarray(tag).convert("RGB"), (40, 40))
    return out


def test_reid_protocol_cascade_vs_global(config, embedder, tmp_path):
    from ml.asset_identity import ArucoReader

    reader = ArucoReader(config["aruco"]["dictionary"], 0.0)
    objects = {f"obj_{i}": make_panel(500 + i) for i in range(4)}
    objects["obj_0"] = with_marker(objects["obj_0"], config, 7)
    rows = []

    def record(session, oid, cond, img, k):
        rel = f"{session}/{oid}/{cond}_{k:03d}.jpg"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(tmp_path / rel, quality=95)
        mk = reader.read(img, None)
        rows.append({"session": session, "object_id": oid, "condition": cond, "path": rel,
                     "ts": f"2026-10-0{1 if session == 'day1' else 2}T00:00:0{k}", "marker_id": mk[0] if mk else ""})

    for oid, img in objects.items():
        record("day1", oid, "normal", img, 0)
        record("day2", oid, "normal", rephotograph(img, 1), 0)
        record("day2", oid, "damaged", add_crack(rephotograph(img, 2)), 1)
    record("day2", "obj_unknown", "normal", make_panel(999), 0)
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    records = load_records(tmp_path)
    m = evaluate(config, records, tmp_path, embedder)
    assert m["gallery_session"] == "day1" and m["markers"] == 1
    cas, glo = m["methods"]["cascade"], m["methods"]["global"]
    assert cas["overall"]["known_queries"] == 8 and cas["overall"]["unknown_queries"] == 1
    assert cas["overall"]["top1_accuracy"] == 1.0
    assert cas["damaged"]["top1_accuracy"] == 1.0, "a damaged object keeps its identity via geometry"
    assert cas["overall"]["false_match_rate"] == 0.0
    assert cas["overall"]["top1_accuracy"] >= glo["overall"]["top1_accuracy"]
    assert m["cascade_stages"].get("aruco", 0) == 2
    text = write_report(m, tmp_path / "report")
    assert "cascade" in text and (tmp_path / "report" / "metrics.json").exists()
