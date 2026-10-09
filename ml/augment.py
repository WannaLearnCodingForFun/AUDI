"""Webcam-style image degradations used for training and robustness evaluation.

Simulates what a consumer USB webcam adds to clean dataset images: defocus
blur, heavy JPEG compression, exposure / contrast / saturation drift, white
balance cast, off-axis perspective, mirroring and loose framing. All strengths
come from `condition_classifier.augment` in config.yaml.
"""

from __future__ import annotations

import io
import random
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomPerspective


class WebcamAugment:
    def __init__(self, params: Dict[str, Any]):
        self.p = dict(params)

    def __call__(self, img: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
        r = rng or random.Random()
        p = self.p
        img = img.convert("RGB")

        w, h = img.size
        scale = r.uniform(float(p["crop_scale_min"]), 1.0)
        cw, ch = max(1, int(w * np.sqrt(scale))), max(1, int(h * np.sqrt(scale)))
        x0, y0 = r.randint(0, w - cw), r.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))

        if r.random() < float(p["hflip_prob"]):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if r.random() < float(p["perspective_prob"]):
            start, end = RandomPerspective.get_params(img.width, img.height, float(p["perspective_distortion"]))
            img = TF.perspective(img, start, end, fill=[0, 0, 0])

        for key, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast),
                              ("saturation", ImageEnhance.Color)):
            amount = float(p[key])
            img = enhancer(img).enhance(r.uniform(1.0 - amount, 1.0 + amount))

        shift = float(p["white_balance_shift"])
        if shift > 0:
            gains = np.array([r.uniform(1.0 - shift, 1.0 + shift) for _ in range(3)], dtype=np.float32)
            arr = np.asarray(img, dtype=np.float32) * gains
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        if r.random() < float(p["blur_prob"]):
            img = img.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.1, float(p["blur_sigma_max"]))))

        if r.random() < float(p["jpeg_prob"]):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=r.randint(int(p["jpeg_quality_min"]), 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img
