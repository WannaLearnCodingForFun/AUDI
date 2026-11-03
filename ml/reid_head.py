"""Purpose-trained re-identification head (upgrade 1.3).

A metric-learning MLP on top of the frozen backbone (DINOv2) embedding produces a 256-d
IDENTITY embedding, kept separate from the condition features the classifier uses. It is
trained (training/train_reid_head.py) with ArcFace or batch-hard triplet loss on pairs of the
same asset across visits plus synthetic changes — lighting, viewpoint, added cracks, rust,
dust and snow — so that damage moves a view along the condition axis but not the identity axis.

`IdentityEncoder` is what the engine uses: with a checkpoint it maps backbone embeddings
through the head; without one it is the identity map (the global-embedding baseline).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import get_nested, resolve_path

logger = logging.getLogger(__name__)


class IdentityHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_dim, self.out_dim, self.hidden = in_dim, out_dim, hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ArcFace(nn.Module):
    """Additive angular margin softmax (Deng et al. 2019)."""

    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(z, F.normalize(self.W, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y)


def batch_hard_triplet(z: torch.Tensor, y: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    d = torch.cdist(z, z)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    pos = d.masked_fill(~same | eye, -1.0).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values
    valid = (pos >= 0) & torch.isfinite(neg)
    if not valid.any():
        return z.sum() * 0.0
    return F.relu(pos[valid] - neg[valid] + margin).mean()


class IdentityEncoder:
    """Backbone embedding → identity embedding (passthrough without a trained head)."""

    def __init__(self, head: Optional[IdentityHead] = None, name: str = "global"):
        self.head = head.eval() if head is not None else None
        self.name = name

    @property
    def active(self) -> bool:
        return self.head is not None

    def dim(self, backbone_dim: int) -> int:
        return self.head.out_dim if self.head is not None else backbone_dim

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if self.head is None:
            out = x
        else:
            with torch.no_grad():
                out = self.head(torch.from_numpy(x)).numpy().astype(np.float32)
        return out[0] if single else out

    @classmethod
    def from_config(cls, config: Dict[str, Any], backbone_dim: int) -> "IdentityEncoder":
        c = get_nested(config, "reid_head", default={}) or {}
        path = c.get("checkpoint")
        if not c.get("enabled", True) or not path or not resolve_path(path).is_file():
            return cls(None)
        try:
            ck = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
            if int(ck["in_dim"]) != backbone_dim:
                logger.warning("re-ID head expects %d-d input, backbone gives %d; using global embeddings", ck["in_dim"], backbone_dim)
                return cls(None)
            head = IdentityHead(int(ck["in_dim"]), int(ck["out_dim"]), int(ck.get("hidden", 512)))
            head.load_state_dict(ck["state"])
            return cls(head, f"reid_head:{ck.get('tag', Path(path).stem)}")
        except Exception:  # noqa: BLE001
            logger.exception("Could not load re-ID head %s; using global embeddings", path)
            return cls(None)


def save_head(head: IdentityHead, path: Path, tag: str, meta: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "in_dim": head.in_dim, "out_dim": head.out_dim, "hidden": head.hidden,
                "tag": tag, "meta": meta or {}}, path)


def retrieval_metrics(query: np.ndarray, q_ids: Sequence[str], gallery: np.ndarray, g_ids: Sequence[str], ks=(1, 5)) -> Dict[str, float]:
    """Rank-k accuracy and mAP of cosine retrieval (queries whose id is in the gallery)."""
    q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    g = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    sims = q @ g.T
    g_ids = np.asarray(g_ids)
    ranks = {k: 0 for k in ks}
    aps: List[float] = []
    n = 0
    for i, qid in enumerate(q_ids):
        rel = g_ids == qid
        if not rel.any():
            continue
        n += 1
        order = np.argsort(-sims[i])
        hits = rel[order]
        # rank-k over unique identities in ranked order
        seen: List[str] = []
        for j in order:
            if g_ids[j] not in seen:
                seen.append(g_ids[j])
        for k in ks:
            ranks[k] += int(qid in seen[:k])
        cum = np.cumsum(hits)
        prec = cum / np.arange(1, len(hits) + 1)
        aps.append(float((prec * hits).sum() / max(1, hits.sum())))
    out = {f"rank{k}": (ranks[k] / n if n else math.nan) for k in ks}
    out["mAP"] = float(np.mean(aps)) if aps else math.nan
    out["queries"] = n
    return out
