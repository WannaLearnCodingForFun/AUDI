"""Shared helpers for condition-classifier training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.augment import WebcamAugment
from ml.condition_mapping import UNIFIED_CONDITIONS
from ml.config import ROOT, cfg, resolve_path
from ml.embedding_model import AssetEmbeddingModel

logger = logging.getLogger(__name__)

MANIFEST = "manifest.csv"


@dataclass
class Row:
    path: str
    asset_type: str
    condition: str
    split: str
    source: str


def load_manifest(config: Dict[str, Any]) -> List[Row]:
    path = resolve_path(cfg(config, "datasets", "prepare", "output_dir")) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/prepare_unified_dataset.py first")
    with open(path, encoding="utf-8") as f:
        return [Row(r["path"], r["asset_type"], r["condition"], r["split"], r["source"]) for r in csv.DictReader(f)]


def manifest_digest(rows: Sequence[Row]) -> str:
    h = hashlib.sha1()
    for r in rows:
        h.update(f"{r.path}|{r.asset_type}|{r.condition}|{r.split}\n".encode())
    return h.hexdigest()[:12]


def label_maps_from(rows: Sequence[Row]) -> Dict[str, List[str]]:
    """asset_type → conditions present in the training split, in unified (normal-first) order."""
    present: Dict[str, set] = {}
    for r in rows:
        if r.split == "train":
            present.setdefault(r.asset_type, set()).add(r.condition)
    maps = {}
    for asset_type, conds in UNIFIED_CONDITIONS.items():
        if asset_type in present and len(present[asset_type]) >= 2:
            maps[asset_type] = [c for c in conds if c in present[asset_type]]
    return maps


class ImageListDataset(Dataset):
    """Loads images (optionally webcam-augmented with a per-item deterministic seed)."""

    def __init__(self, paths: Sequence[str], augment: Optional[WebcamAugment] = None, seed: int = 0):
        self.paths = list(paths)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Image.Image:
        with Image.open(ROOT / self.paths[i]) as im:
            img = im.convert("RGB")
        if self.augment is not None:
            img = self.augment(img, random.Random(self.seed * 1_000_003 + i))
        return img


def _collate(batch):
    return batch


def extract_features(
    embedder: AssetEmbeddingModel,
    paths: Sequence[str],
    config: Dict[str, Any],
    view: int = 0,
    tag: str = "",
) -> np.ndarray:
    """L2-normalised global embeddings; view 0 = clean, view k>0 = k-th augmented view. Cached on disk."""
    ccfg = cfg(config, "condition_classifier")
    aug_params = ccfg["augment"] if view > 0 else None
    key_src = json.dumps(
        {
            "backbone": embedder.active_model_name,
            "image_size": cfg(config, "model", "image_size"),
            "paths": hashlib.sha1("\n".join(paths).encode()).hexdigest(),
            "view": view,
            "augment": aug_params,
            "tag": tag,
        },
        sort_keys=True,
    )
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    cache_dir = resolve_path(ccfg["feature_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{key}.npy"
    if cache.exists():
        feats = np.load(cache)
        if feats.shape == (len(paths), embedder.embedding_dim):
            return feats

    augment = WebcamAugment(aug_params) if aug_params else None
    loader = DataLoader(
        ImageListDataset(paths, augment, seed=view),
        batch_size=int(ccfg["feature_batch_size"]),
        num_workers=int(ccfg["loader_workers"]),
        collate_fn=_collate,
        persistent_workers=False,
    )
    out = []
    done = 0
    for batch in loader:
        out.append(embedder.encode_batch(batch))
        done += len(batch)
        if done % (20 * len(batch)) < len(batch):
            logger.info("features view %d: %d/%d", view, done, len(paths))
    feats = np.concatenate(out).astype(np.float32) if out else np.zeros((0, embedder.embedding_dim), np.float32)
    np.save(cache, feats)
    return feats


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
