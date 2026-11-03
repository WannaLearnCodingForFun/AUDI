"""Asset detection (FR-9).

`Detector` is the swappable interface. `YoloWorldDetector` runs an
open-vocabulary YOLO-World model whose text prompts come from
config `detection.prompts`; `FullFrameDetector` treats the whole frame as one
asset for close-up desk demos.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ml.config import resolve_path
from ml.device import resolve_device
from ml.imaging import clip_bbox, pil_to_bgr
from ml.net import ensure_ca_bundle

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    asset_type: str
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Image.Image) -> List[Detection]:
        """Return candidate assets in an RGB frame."""


class FullFrameDetector(Detector):
    def __init__(self, asset_type: str, confidence: float):
        self.asset_type = asset_type
        self.confidence = confidence

    def detect(self, frame: Image.Image) -> List[Detection]:
        return [Detection((0, 0, frame.width, frame.height), self.asset_type, self.confidence, "full_frame")]


class YoloWorldDetector(Detector):
    def __init__(self, config: Dict[str, Any]):
        ensure_ca_bundle()
        from ultralytics import YOLO, YOLOWorld

        dcfg = config["detection"]
        self.conf = float(dcfg["confidence"])
        self.iou = float(dcfg["iou"])
        self.imgsz = int(dcfg["image_size"])
        self.max_det = int(dcfg["max_detections"])
        self.min_area = float(dcfg["min_box_area_fraction"])
        device = resolve_device(config["model"]["device"])
        self.device = device.type if device.type != "cuda" else 0
        self._lock = threading.Lock()

        self.labels: List[Tuple[str, str]] = []
        for asset_type, prompts in dcfg["prompts"].items():
            for prompt in prompts:
                self.labels.append((asset_type, prompt))
        if not self.labels:
            raise ValueError("detection.prompts is empty")

        import ultralytics.nn.text_model as text_model

        # The CLIP text encoder otherwise downloads relative to the working directory.
        text_model.WEIGHTS_DIR = resolve_path(dcfg["text_encoder_dir"])
        weights = resolve_path(dcfg["weights"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        world = YOLOWorld(str(weights))
        world.set_classes([p for _, p in self.labels])

        if dcfg["onnx_export"]:
            onnx_path = weights.with_name(f"{weights.stem}_{len(self.labels)}cls.onnx")
            if not onnx_path.exists():
                exported = world.export(format="onnx", imgsz=self.imgsz)
                Path(exported).replace(onnx_path)
            self.model = YOLO(str(onnx_path), task="detect")
            self.device = "cpu"
            self.backend = "onnx"
        else:
            self.model = world
            self.backend = "torch"
        logger.info("YOLO-World detector ready (%s, %d prompts)", self.backend, len(self.labels))

    def detect(self, frame: Image.Image) -> List[Detection]:
        bgr = pil_to_bgr(frame)
        with self._lock:
            results = self.model.predict(
                bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )
        out: List[Detection] = []
        frame_area = float(frame.width * frame.height)
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls in zip(xyxy, confs, classes):
            clipped = clip_bbox(box, frame.width, frame.height)
            if clipped is None:
                continue
            area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
            if area / frame_area < self.min_area:
                continue
            asset_type, label = self.labels[int(cls)]
            out.append(Detection(clipped, asset_type, float(conf), label))
        return out


class FineTunedDetector(Detector):
    """Closed-set YOLOv8 / YOLO11 trained on the known asset types (3.2, training/train_detector.py).

    Class names in the checkpoint are the asset types (detection.prompts keys)."""

    def __init__(self, config: Dict[str, Any], weights: Path):
        ensure_ca_bundle()
        from ultralytics import YOLO

        dcfg = config["detection"]
        self.conf = float(dcfg.get("finetuned_confidence", dcfg["confidence"]))
        self.iou = float(dcfg["iou"])
        self.imgsz = int(dcfg["image_size"])
        self.max_det = int(dcfg["max_detections"])
        self.min_area = float(dcfg["min_box_area_fraction"])
        device = resolve_device(config["model"]["device"])
        self.device = device.type if device.type != "cuda" else 0
        self.model = YOLO(str(weights), task="detect")
        self.names = dict(self.model.names)
        self._lock = threading.Lock()

    def detect(self, frame: Image.Image) -> List[Detection]:
        with self._lock:
            results = self.model.predict(pil_to_bgr(frame), conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                         max_det=self.max_det, device=self.device, verbose=False)
        return postprocess(results[0].boxes, frame, self.min_area, lambda c: (self.names[c], self.names[c]))


class HybridDetector(Detector):
    """Fine-tuned detector for known types; YOLO-World only when it finds nothing (3.2)."""

    def __init__(self, primary: Detector, fallback: Optional[Detector]):
        self.primary = primary
        self.fallback = fallback

    def detect(self, frame: Image.Image) -> List[Detection]:
        dets = self.primary.detect(frame)
        if dets or self.fallback is None:
            return dets
        return self.fallback.detect(frame)


def postprocess(boxes, frame: Image.Image, min_area: float, label_of) -> List[Detection]:
    out: List[Detection] = []
    if boxes is None:
        return out
    frame_area = float(frame.width * frame.height)
    for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
        clipped = clip_bbox(box, frame.width, frame.height)
        if clipped is None:
            continue
        if (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) / frame_area < min_area:
            continue
        asset_type, label = label_of(int(cls))
        out.append(Detection(clipped, asset_type, float(conf), label))
    return out


def build_detector(config: Dict[str, Any]) -> Detector:
    dcfg = config["detection"]
    if dcfg["mode"] == "full_frame":
        return FullFrameDetector(dcfg["full_frame_asset_type"], float(dcfg["full_frame_confidence"]))
    if dcfg["mode"] == "yolo_world":
        return YoloWorldDetector(config)
    if dcfg["mode"] == "hybrid":
        weights = resolve_path(dcfg.get("finetuned_weights", "models/detector_finetuned.pt"))
        world = YoloWorldDetector(config) if dcfg.get("hybrid_open_vocabulary", True) else None
        if not weights.is_file():
            logger.warning("detection.mode=hybrid but %s is missing; using YOLO-World only", weights)
            return world or FullFrameDetector(dcfg["full_frame_asset_type"], float(dcfg["full_frame_confidence"]))
        return HybridDetector(FineTunedDetector(config, weights), world)
    raise ValueError(f"Unknown detection.mode: {dcfg['mode']}")
