"""Detection (FR-9): YOLO-World post-processing with a fake model, multi-asset frames, real weights when present."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from memory.store import LocalStore
from ml.config import resolve_path
from ml.detector import Detection, Detector, FullFrameDetector, YoloWorldDetector, build_detector
from ml.semantic_memory import SemanticMemoryEngine
from ml.tracker import SessionTracker
from tests.conftest import make_panel


class FakeYolo:
    def __init__(self, boxes, confs, classes):
        self.out = SimpleNamespace(boxes=SimpleNamespace(
            xyxy=torch.tensor(boxes, dtype=torch.float32), conf=torch.tensor(confs), cls=torch.tensor(classes, dtype=torch.float32)))

    def predict(self, *a, **kw):
        return [self.out]


def fake_world(config, model):
    d = YoloWorldDetector.__new__(YoloWorldDetector)
    dc = config["detection"]
    d.conf, d.iou, d.imgsz, d.max_det = dc["confidence"], dc["iou"], dc["image_size"], dc["max_detections"]
    d.min_area, d.device, d.model, d.backend = float(dc["min_box_area_fraction"]), "cpu", model, "torch"
    import threading

    d._lock = threading.Lock()
    d.labels = [(t, p) for t, ps in dc["prompts"].items() for p in ps]
    return d


def test_yolo_world_postprocessing_maps_prompts_to_asset_types(config):
    labels = [(t, p) for t, ps in config["detection"]["prompts"].items() for p in ps]
    pole = next(i for i, (t, p) in enumerate(labels) if t == "utility_pole")
    panel = next(i for i, (t, p) in enumerate(labels) if t == "solar_panel")
    model = FakeYolo(
        boxes=[[10, 10, 200, 150], [-30, 50, 700, 900], [5, 5, 8, 8]],
        confs=[0.9, 0.6, 0.8], classes=[panel, pole, panel],
    )
    dets = fake_world(config, model).detect(Image.new("RGB", (640, 480)))
    assert [d.asset_type for d in dets] == ["solar_panel", "utility_pole"], "tiny box filtered by min area"
    assert dets[1].bbox == (0, 50, 640, 480), "boxes clipped to the frame"
    assert dets[0].label == labels[panel][1] and dets[0].confidence == pytest.approx(0.9)


def test_build_detector_modes(config):
    assert isinstance(build_detector(config), FullFrameDetector)
    with pytest.raises(ValueError):
        build_detector({**config, "detection": {**config["detection"], "mode": "nope"}})


class TwoAssets(Detector):
    def detect(self, frame):
        w, h = frame.size
        return [Detection((0, 0, w // 2, h), "solar_panel", 0.9, "solar panel"),
                Detection((w // 2, 0, w, h), "solar_panel", 0.8, "solar panel")]


def test_every_detection_in_a_frame_becomes_an_asset(config, embedder, classifier):
    store = LocalStore(config)
    store.ensure_site("s", "S")
    engine = SemanticMemoryEngine(config, store, embedder, classifier, TwoAssets())
    sid = store.create_session("s", "op", "offline", {})["session_id"]
    frame = Image.new("RGB", (640, 240))
    frame.paste(make_panel(1), (0, 0))
    frame.paste(make_panel(2), (320, 0))
    events = engine.process_frame(frame, sid, SessionTracker.from_config(config))
    assert [e.type for e in events] == ["new_asset", "new_asset"]
    assert len({e.asset_id for e in events}) == 2


@pytest.mark.skipif(not resolve_path("models/yolov8s-worldv2.pt").exists(), reason="YOLO-World weights not downloaded")
def test_real_yolo_world_loads_and_runs(config):
    det = YoloWorldDetector({**config, "detection": {**config["detection"], "mode": "yolo_world"}})
    out = det.detect(Image.fromarray(np.zeros((480, 640, 3), np.uint8)))
    assert isinstance(out, list)
