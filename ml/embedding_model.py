"""Pretrained visual embedding model for asset representation."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from ml.similarity import to_unit_vector

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor]

PROJECTION_DIM = 512
PROJECTION_SEED = 42


class FixedProjectionEmbedder:
    """
    Deterministic visual embedder for offline/dev use when pretrained weights
    are unavailable. Combines spatial color/texture statistics with a fixed
    random projection — not a perceptual hash.
    """

    def __init__(self, dim: int = PROJECTION_DIM, seed: int = PROJECTION_SEED, grid: int = 8):
        self.dim = dim
        self.grid = grid
        feat_dim = grid * grid * 6  # mean+std per RGB channel per cell
        rng = np.random.default_rng(seed)
        self._matrix = rng.standard_normal((feat_dim, dim)).astype(np.float32)
        self._matrix /= np.sqrt(feat_dim)

    def _spatial_features(self, pil: Image.Image) -> np.ndarray:
        arr = np.array(pil.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        parts: list[np.ndarray] = []
        for gi in range(self.grid):
            for gj in range(self.grid):
                patch = arr[gi * h // self.grid : (gi + 1) * h // self.grid, gj * w // self.grid : (gj + 1) * w // self.grid]
                parts.append(patch.mean(axis=(0, 1)))
                parts.append(patch.std(axis=(0, 1)))
        return np.concatenate(parts).astype(np.float32)

    def encode_pil(self, pil: Image.Image) -> np.ndarray:
        features = self._spatial_features(pil) @ self._matrix
        return to_unit_vector(features)

    def encode_batch_pil(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.stack([self.encode_pil(img) for img in images])


class AssetEmbeddingModel:
    """Load a pretrained vision model once and produce normalized embeddings."""

    def __init__(self, model_name: str = "dinov2", device: str = "auto", fallback: str = "resnet50"):
        self.model_name = model_name
        self.fallback_name = fallback
        self.device = self._resolve_device(device)
        self._projection: Optional[FixedProjectionEmbedder] = None
        self.model: Optional[nn.Module] = None
        self.preprocess: Optional[transforms.Compose] = None
        self.embedding_dim = PROJECTION_DIM
        self.active_model_name = "fixed_projection"

        if model_name.lower() == "projection":
            self._projection = FixedProjectionEmbedder()
        else:
            self.model, self.preprocess, self.embedding_dim = self._load_model()
            if self.model is not None:
                self.model.eval()

        logger.info(
            "Loading embedding model: %s on %s (dim=%d)",
            self.active_model_name,
            self.device,
            self.embedding_dim,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self) -> tuple[Optional[nn.Module], Optional[transforms.Compose], int]:
        if self.model_name.lower() in {"dinov2", "dino_v2", "dinov2_vits14"}:
            try:
                return self._load_dinov2()
            except Exception as exc:
                logger.warning("DINOv2 unavailable (%s); falling back to %s", exc, self.fallback_name)

        if self.model_name.lower() in {"resnet50", "resnet18", "mobilenet_v3_large"}:
            try:
                return self._load_torchvision_fallback(self.model_name, pretrained=True)
            except Exception as exc:
                logger.warning("Pretrained %s unavailable (%s)", self.model_name, exc)

        try:
            return self._load_torchvision_fallback(self.fallback_name, pretrained=True)
        except Exception as exc:
            logger.warning(
                "Pretrained %s weights unavailable (%s); using fixed projection embedder",
                self.fallback_name,
                exc,
            )
            self._projection = FixedProjectionEmbedder()
            self.active_model_name = "fixed_projection"
            return None, None, PROJECTION_DIM

    def _load_dinov2(self) -> tuple[nn.Module, transforms.Compose, int]:
        variant = "dinov2_vits14"
        model = torch.hub.load("facebookresearch/dinov2", variant, pretrained=True)
        model = model.to(self.device)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.active_model_name = variant
        return model, preprocess, 384

    def _load_torchvision_fallback(
        self, name: str, pretrained: bool = True
    ) -> tuple[nn.Module, transforms.Compose, int]:
        weights_map = {
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        }
        model_key = name if name in weights_map else "resnet50"
        if not pretrained:
            raise RuntimeError("Non-pretrained torchvision backbone is not used")

        weights = weights_map[model_key]
        backbone = getattr(models, model_key)(weights=weights)
        preprocess = weights.transforms()
        self.active_model_name = model_key
        backbone.fc = nn.Identity()
        model = backbone.to(self.device)
        dim_map = {"resnet50": 2048, "resnet18": 512, "mobilenet_v3_large": 960}
        dim = dim_map.get(model_key, 2048)
        return model, preprocess, dim

    def _to_pil(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in {1, 3}:
                arr = np.transpose(arr, (1, 2, 0))
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def encode_image(self, image: ImageInput) -> np.ndarray:
        """Encode a single image into a normalized embedding vector."""
        pil = self._to_pil(image)
        if self._projection is not None:
            return self._projection.encode_pil(pil)

        assert self.model is not None and self.preprocess is not None
        with torch.inference_mode():
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
            features = self.model(tensor)
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken", next(iter(features.values())))
            embedding = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return to_unit_vector(embedding)

    def encode_batch(self, images: Sequence[ImageInput]) -> np.ndarray:
        """Encode a batch of images; returns shape (N, D) normalized embeddings."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._projection is not None:
            pils = [self._to_pil(img) for img in images]
            return self._projection.encode_batch_pil(pils)

        assert self.model is not None and self.preprocess is not None
        with torch.inference_mode():
            tensors = [self.preprocess(self._to_pil(img)) for img in images]
            batch = torch.stack(tensors).to(self.device)
            features = self.model(batch)
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken", next(iter(features.values())))
            embeddings = features.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return embeddings / norms
