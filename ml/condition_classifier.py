"""Condition classification of asset crops.

One shared backbone embedding feeds one head per asset_type. Confidence is the
top-1 softmax probability (deterministic). `PrototypeConditionClassifier` is
the nearest-neighbour label-copy baseline the repository used before; it is
kept for the before/after evaluation and for weight-free tests.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ml.embedding_model import AssetEmbeddingModel, ImageInput

logger = logging.getLogger(__name__)

NORMAL_LABEL = "normal"


@dataclass
class ConditionPrediction:
    asset_type: str
    label: str
    scores: Dict[str, float]
    is_defect: bool
    defect_type: Optional[str]
    confidence: float
    labels: List[str] = field(default_factory=list)

    def score_vector(self, labels: Optional[Sequence[str]] = None) -> np.ndarray:
        order = list(labels) if labels is not None else self.labels
        return np.array([self.scores.get(lbl, 0.0) for lbl in order], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("labels", None)
        return d


def prediction_from_scores(asset_type: str, labels: Sequence[str], probs: np.ndarray) -> ConditionPrediction:
    idx = int(np.argmax(probs))
    label = labels[idx]
    is_defect = label != NORMAL_LABEL
    return ConditionPrediction(
        asset_type=asset_type,
        label=label,
        scores={lbl: float(p) for lbl, p in zip(labels, probs)},
        is_defect=is_defect,
        defect_type=label if is_defect else None,
        confidence=float(probs[idx]),
        labels=list(labels),
    )


class ConditionClassifier(ABC):
    """Maps a backbone embedding of an asset crop to a condition prediction."""

    @property
    @abstractmethod
    def label_maps(self) -> Dict[str, List[str]]:
        """asset_type → ordered condition labels."""

    @abstractmethod
    def predict_embeddings(self, embeddings: np.ndarray, asset_type: str) -> List[ConditionPrediction]:
        """Classify (N, D) embeddings that all belong to `asset_type`."""

    def asset_types(self) -> List[str]:
        return list(self.label_maps)

    def supports(self, asset_type: str) -> bool:
        return asset_type in self.label_maps

    def predict(
        self, embedder: AssetEmbeddingModel, images: Sequence[ImageInput], asset_type: str
    ) -> List[ConditionPrediction]:
        return self.predict_embeddings(embedder.encode_batch(images), asset_type)


class ConditionHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, kind: str, hidden_dim: int, dropout: float):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadNet(nn.Module):
    def __init__(self, in_dim: int, label_maps: Dict[str, List[str]], kind: str, hidden_dim: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleDict(
            {t: ConditionHead(in_dim, len(lbls), kind, hidden_dim, dropout) for t, lbls in label_maps.items()}
        )

    def forward(self, x: torch.Tensor, asset_type: str) -> torch.Tensor:
        return self.heads[asset_type](x)


class MultiHeadConditionClassifier(ConditionClassifier):
    def __init__(
        self,
        net: MultiHeadNet,
        label_maps: Dict[str, List[str]],
        meta: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ):
        self.net = net.to(device).eval()
        self._label_maps = {k: list(v) for k, v in label_maps.items()}
        self.meta = meta
        self.device = device

    @property
    def label_maps(self) -> Dict[str, List[str]]:
        return self._label_maps

    def predict_embeddings(self, embeddings: np.ndarray, asset_type: str) -> List[ConditionPrediction]:
        if asset_type not in self._label_maps:
            raise KeyError(f"No condition head for asset_type '{asset_type}'")
        labels = self._label_maps[asset_type]
        with torch.inference_mode():
            x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=self.device)
            probs = torch.softmax(self.net(x, asset_type), dim=1).cpu().numpy()
        return [prediction_from_scores(asset_type, labels, p) for p in probs]

    def save(self, path: Path, backbone_state: Optional[Dict[str, torch.Tensor]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "label_maps": self._label_maps,
            "meta": self.meta,
            "backbone_state": backbone_state,
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: Path,
        embedder: Optional[AssetEmbeddingModel] = None,
        device: torch.device = torch.device("cpu"),
    ) -> "MultiHeadConditionClassifier":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        if embedder is not None:
            base_name = embedder.active_model_name.split("+ft", 1)[0]
            if meta["backbone"] != base_name or int(meta["embedding_dim"]) != embedder.embedding_dim:
                raise ValueError(
                    f"Classifier was trained on {meta['backbone']} (dim {meta['embedding_dim']}) but the "
                    f"loaded backbone is {embedder.active_model_name} (dim {embedder.embedding_dim})"
                )
            if payload.get("backbone_state"):
                module = getattr(embedder.backbone, "model", None)
                if module is None:
                    raise ValueError("Checkpoint contains fine-tuned backbone weights for a non-DINOv2 backbone")
                missing = module.load_state_dict(payload["backbone_state"], strict=False)
                # distinct name so stored prototypes are re-embedded with the fine-tuned weights (D12)
                embedder.active_model_name = f"{meta['backbone']}+ft{meta.get('fine_tune_blocks')}@{meta.get('trained_at')}"
                logger.info("Applied fine-tuned backbone blocks (%d unexpected keys) → %s",
                            len(missing.unexpected_keys), embedder.active_model_name)
        net = MultiHeadNet(
            int(meta["embedding_dim"]),
            payload["label_maps"],
            meta["head"],
            int(meta["hidden_dim"]),
            float(meta["dropout"]),
        )
        net.load_state_dict(payload["state_dict"])
        return cls(net, payload["label_maps"], meta, device)


class PrototypeConditionClassifier(ConditionClassifier):
    """Nearest-neighbour label copy over labelled reference embeddings (baseline)."""

    def __init__(self, embeddings: Dict[str, np.ndarray], labels: Dict[str, List[str]], temperature: float):
        self.temperature = temperature
        self._emb: Dict[str, np.ndarray] = {}
        self._lab: Dict[str, np.ndarray] = {}
        self._label_maps: Dict[str, List[str]] = {}
        for asset_type, mat in embeddings.items():
            mat = np.asarray(mat, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self._emb[asset_type] = mat / np.where(norms < 1e-12, 1.0, norms)
            self._lab[asset_type] = np.asarray(labels[asset_type])
            ordered = sorted(set(labels[asset_type]), key=lambda s: (s != NORMAL_LABEL, s))
            self._label_maps[asset_type] = ordered

    @property
    def label_maps(self) -> Dict[str, List[str]]:
        return self._label_maps

    def predict_embeddings(self, embeddings: np.ndarray, asset_type: str) -> List[ConditionPrediction]:
        ref = self._emb[asset_type]
        ref_labels = self._lab[asset_type]
        labels = self._label_maps[asset_type]
        q = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        sims = (q / np.where(norms < 1e-12, 1.0, norms)) @ ref.T
        out = []
        for row in sims:
            # argmax of the per-class maximum similarity == label of the nearest neighbour;
            # scores are a temperature softmax over those maxima.
            per_class = np.array([float(row[ref_labels == lbl].max()) for lbl in labels])
            logits = (per_class - per_class.max()) / self.temperature
            probs = np.exp(logits) / np.exp(logits).sum()
            out.append(prediction_from_scores(asset_type, labels, probs))
        return out


def load_condition_classifier(
    config: Dict[str, Any], embedder: AssetEmbeddingModel
) -> MultiHeadConditionClassifier:
    from ml.config import resolve_path

    path = resolve_path(config["condition_classifier"]["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(
            f"Condition classifier checkpoint not found at {path}. Train it with `make train`."
        )
    return MultiHeadConditionClassifier.load(path, embedder, device=torch.device("cpu"))
