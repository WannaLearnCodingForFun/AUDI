"""Image loading, cropping and colour-space helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

BBox = Tuple[int, int, int, int]
ImageLike = Union[Image.Image, np.ndarray, str, Path, bytes]


def load_image(image: ImageLike) -> Image.Image:
    """Load any supported input as an RGB PIL image (ndarrays are assumed RGB)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return im.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as im:
            return im.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[BBox]:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop(pil: Image.Image, bbox: Optional[Sequence[float]]) -> Image.Image:
    if bbox is None:
        return pil
    box = clip_bbox(bbox, pil.width, pil.height)
    return pil.crop(box) if box else pil


def expand_bbox(bbox: Sequence[float], margin: float, width: int, height: int) -> Optional[BBox]:
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return clip_bbox((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def to_gray(pil: Image.Image, max_side: int) -> np.ndarray:
    """Grayscale uint8 array, downscaled so the longest side ≤ max_side."""
    arr = np.asarray(pil.convert("L"))
    h, w = arr.shape
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return arr


def encode_jpeg(pil: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
