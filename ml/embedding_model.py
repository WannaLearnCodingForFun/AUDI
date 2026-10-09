"""Visual backbones producing global embeddings and patch tokens.

Backbones are swappable behind :class:`Backbone`. The embedding dimension is
measured from the loaded model (never read from config) so the vector index,
classifier heads and stored prototypes always agree (D12).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from ml.device import resolve_device
from ml.net import ensure_ca_bundle

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_pil(image: ImageInput) -> Image.Image:
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


def _l2_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.where(norms < 1e-12, 1.0, norms)).astype(np.float32)


class Backbone(ABC):
    """Image → (global embedding, patch-token grid)."""

    name: str = "backbone"
    device: torch.device = torch.device("cpu")

    @abstractmethod
    def _forward(self, pils: Sequence[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (N, D) global features and (N, G, G, Dp) patch features (unnormalised)."""

    def embed_with_patches(self, images: Sequence[ImageInput]) -> Tuple[np.ndarray, np.ndarray]:
        pils = [to_pil(img) for img in images]
        glob, patches = self._forward(pils)
        return _l2_rows(glob), _l2_rows(patches)

    def embed(self, images: Sequence[ImageInput]) -> np.ndarray:
        return self.embed_with_patches(images)[0]


class DinoV2Backbone(Backbone):
    def __init__(self, device: torch.device, repo: str, variant: str, image_size: int):
        ensure_ca_bundle()
        self.device = device
        self.name = variant
        self.image_size = image_size
        self.model = torch.hub.load(repo, variant, trust_repo=True).to(device).eval()
        self.patch_size = int(self.model.patch_size)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def forward_tensor(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model.forward_features(batch)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    def _forward(self, pils):
        grid = self.image_size // self.patch_size
        with torch.inference_mode():
            batch = torch.stack([self.preprocess(p) for p in pils]).to(self.device)
            cls, patches = self.forward_tensor(batch)
        cls_np = cls.float().cpu().numpy()
        patch_np = patches.float().cpu().numpy().reshape(len(pils), grid, grid, -1)
        return cls_np, patch_np


class ResNetBackbone(Backbone):
    def __init__(self, device: torch.device, name: str, image_size: int):
        ensure_ca_bundle()
        weights_map = {
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        }
        if name not in weights_map:
            raise ValueError(f"Unsupported ResNet backbone: {name}")
        self.device = device
        self.name = name
        net = getattr(models, name)(weights=weights_map[name])
        self.trunk = nn.Sequential(*list(net.children())[:-2]).to(device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def forward_tensor(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.trunk(batch)
        glob = fmap.mean(dim=(2, 3))
        patches = fmap.flatten(2).transpose(1, 2)
        return glob, patches

    def _forward(self, pils):
        with torch.inference_mode():
            batch = torch.stack([self.preprocess(p) for p in pils]).to(self.device)
            fmap = self.trunk(batch)
        glob = fmap.mean(dim=(2, 3)).float().cpu().numpy()
        patches = fmap.permute(0, 2, 3, 1).float().cpu().numpy()
        return glob, patches


class ProjectionBackbone(Backbone):
    """Deterministic, weight-free embedder for offline development and tests.

    Global vector: per-cell RGB mean/std on a grid, projected by a fixed random
    matrix. Patch tokens: per-cell sub-grid statistics plus gradient energy,
    projected the same way. Not a learned model; accuracy numbers are never
    reported for it.
    """

    SUB = 4

    def __init__(self, dim: int, grid: int, seed: int, image_size: int):
        self.name = "fixed_projection"
        self.dim = dim
        self.grid = grid
        self.image_size = image_size
        rng = np.random.default_rng(seed)
        g_in = grid * grid * 6
        self._g_matrix = (rng.standard_normal((g_in, dim)) / np.sqrt(g_in)).astype(np.float32)
        p_in = self.SUB * self.SUB * 6 + 2
        self._p_matrix = (rng.standard_normal((p_in, dim)) / np.sqrt(p_in)).astype(np.float32)

    def _cells(self, arr: np.ndarray, n: int) -> list[np.ndarray]:
        h, w = arr.shape[:2]
        return [
            arr[i * h // n : (i + 1) * h // n, j * w // n : (j + 1) * w // n]
            for i in range(n)
            for j in range(n)
        ]

    def _forward(self, pils):
        globs, patch_sets = [], []
        for pil in pils:
            arr = np.asarray(
                pil.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR), dtype=np.float32
            ) / 255.0
            gray = arr.mean(axis=2)
            gy, gx = np.gradient(gray)
            grad = np.hypot(gx, gy)
            g_feats = []
            p_feats = []
            for cell, gcell in zip(self._cells(arr, self.grid), self._cells(grad, self.grid)):
                g_feats.extend([cell.mean(axis=(0, 1)), cell.std(axis=(0, 1))])
                sub = []
                for sc in self._cells(cell, self.SUB):
                    sub.extend([sc.mean(axis=(0, 1)), sc.std(axis=(0, 1))])
                sub.append(np.array([gcell.mean(), gcell.std()], dtype=np.float32))
                p_feats.append(np.concatenate(sub))
            globs.append(np.concatenate(g_feats) @ self._g_matrix)
            patch_sets.append((np.stack(p_feats) @ self._p_matrix).reshape(self.grid, self.grid, -1))
        return np.stack(globs), np.stack(patch_sets)


def load_backbone(config: Dict[str, Any], name: Optional[str] = None) -> Backbone:
    """Build the configured backbone with fallback DINOv2 → ResNet → projection."""
    mcfg = config["model"]
    device = resolve_device(mcfg["device"])
    requested = (name or mcfg["name"]).lower()
    image_size = int(mcfg["image_size"])

    def projection() -> Backbone:
        return ProjectionBackbone(
            dim=int(mcfg["projection_dim"]),
            grid=int(mcfg["projection_grid"]),
            seed=int(mcfg["projection_seed"]),
            image_size=image_size,
        )

    if requested == "projection":
        return projection()

    chain = [requested]
    if mcfg["fallback"].lower() not in chain:
        chain.append(mcfg["fallback"].lower())
    for candidate in chain:
        try:
            if candidate.startswith("dinov2"):
                return DinoV2Backbone(device, mcfg["hub_repo"], mcfg["hub_variant"], image_size)
            return ResNetBackbone(device, candidate, image_size)
        except Exception as exc:  # noqa: BLE001 - any load failure moves to the next backbone
            logger.warning("Backbone %s unavailable (%s)", candidate, exc)
    logger.warning("No pretrained backbone available; using the projection embedder")
    return projection()


class AssetEmbeddingModel:
    """Thread-safe facade over a :class:`Backbone`."""

    def __init__(self, backbone: Backbone):
        self.backbone = backbone
        self._lock = threading.Lock()
        probe_g, probe_p = backbone.embed_with_patches([Image.new("RGB", (64, 64))])
        self.embedding_dim = int(probe_g.shape[1])
        self.patch_grid = int(probe_p.shape[1])
        self.patch_dim = int(probe_p.shape[-1])
        self.active_model_name = backbone.name
        self.device = backbone.device
        logger.info(
            "Embedding model %s on %s (dim=%d, patch grid=%d)",
            self.active_model_name,
            self.device,
            self.embedding_dim,
            self.patch_grid,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], name: Optional[str] = None) -> "AssetEmbeddingModel":
        return cls(load_backbone(config, name))

    def encode_image(self, image: ImageInput) -> np.ndarray:
        return self.encode_batch([image])[0]

    def encode_batch(self, images: Sequence[ImageInput]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        with self._lock:
            return self.backbone.embed(images)

    def encode_with_patches(self, images: Sequence[ImageInput]) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self.backbone.embed_with_patches(images)
