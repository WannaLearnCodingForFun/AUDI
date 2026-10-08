"""Asset crop encoding utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ml.embedding_model import AssetEmbeddingModel, ImageInput

BBox = Tuple[int, int, int, int]


class AssetEncoder:
    """Encode full frames or cropped asset regions into embeddings."""

    def __init__(self, embedding_model: Optional[AssetEmbeddingModel] = None, model_name: str = "dinov2", device: str = "auto"):
        self.embedding_model = embedding_model or AssetEmbeddingModel(model_name=model_name, device=device)

    @staticmethod
    def crop_asset(image: ImageInput, bbox: Optional[BBox] = None) -> Image.Image:
        """Crop asset region from image using bounding box (x1, y1, x2, y2)."""
        if isinstance(image, (str, Path)):
            from ml.semantic_memory import _load_image
            pil = _load_image(image)
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for crop: {type(image)}")

        if bbox is None:
            return pil

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pil.width, x2), min(pil.height, y2)
        if x2 <= x1 or y2 <= y1:
            return pil
        return pil.crop((x1, y1, x2, y2))

    def encode_asset(
        self,
        image: ImageInput,
        bbox: Optional[BBox] = None,
    ) -> np.ndarray:
        """Crop (optional) and encode a single asset."""
        cropped = self.crop_asset(image, bbox)
        return self.embedding_model.encode_image(cropped)

    def encode_assets(
        self,
        images: Sequence[ImageInput],
        bboxes: Optional[Sequence[Optional[BBox]]] = None,
    ) -> np.ndarray:
        """Encode multiple assets, optionally with per-image bounding boxes."""
        if bboxes is None:
            bboxes = [None] * len(images)
        crops = [self.crop_asset(img, bbox) for img, bbox in zip(images, bboxes)]
        return self.embedding_model.encode_batch(crops)
