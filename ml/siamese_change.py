"""Siamese change-detection network (upgrade 2.1), BIT-style.

A shared CNN encoder embeds the aligned reference and current views; a small transformer over
semantic tokens of both feature maps (Bitemporal Image Transformer, Chen et al. 2021) refines
them; the absolute difference of the refined maps is decoded to a per-pixel change logit.
Kept compact (≈1.3 M parameters) so it runs on an edge device next to the backbone.

Training: `training/train_change_detector.py` (synthetic before/after pairs from
ml/damage_synth.py, optionally pre-trained on a LEVIR-CD style folder A/ B/ label/).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.s1 = _block(3, width, 2)            # 1/2
        self.s2 = _block(width, width * 2, 2)    # 1/4
        self.s3 = _block(width * 2, width * 4, 2)  # 1/8

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return f1, f2, f3


class TokenTransformer(nn.Module):
    """Pool each map into L semantic tokens, encode both sets jointly, project back."""

    def __init__(self, dim: int, tokens: int = 4, depth: int = 1, heads: int = 4):
        super().__init__()
        self.att = nn.Conv2d(dim, tokens, 1)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        dec = nn.TransformerDecoderLayer(dim, heads, dim * 2, dropout=0.0, batch_first=True)
        self.dec = nn.TransformerDecoder(dec, 1)

    def tokens(self, f):
        a = self.att(f).flatten(2).softmax(-1)              # B, L, HW
        return torch.einsum("blk,bck->blc", a, f.flatten(2))  # B, L, C

    def forward(self, fa, fb):
        B, C, H, W = fa.shape
        ta, tb = self.tokens(fa), self.tokens(fb)
        t = self.enc(torch.cat([ta, tb], 1))
        ta, tb = t.chunk(2, dim=1)
        qa = fa.flatten(2).transpose(1, 2)
        qb = fb.flatten(2).transpose(1, 2)
        ra = self.dec(qa, ta).transpose(1, 2).reshape(B, C, H, W)
        rb = self.dec(qb, tb).transpose(1, 2).reshape(B, C, H, W)
        return ra, rb


class SiameseChangeNet(nn.Module):
    def __init__(self, width: int = 32, tokens: int = 4):
        super().__init__()
        self.width = width
        self.tokens_n = tokens
        self.encoder = Encoder(width)
        self.bit = TokenTransformer(width * 4, tokens)
        self.up3 = _block(width * 4, width * 2)
        self.up2 = _block(width * 2 + width * 2, width)
        self.up1 = _block(width + width, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, ref: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        a1, a2, a3 = self.encoder(ref)
        b1, b2, b3 = self.encoder(cur)
        r3, c3 = self.bit(a3, b3)
        d3 = self.up3(torch.abs(r3 - c3))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=a2.shape[-2:], mode="bilinear", align_corners=False), torch.abs(a2 - b2)], 1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=a1.shape[-2:], mode="bilinear", align_corners=False), torch.abs(a1 - b1)], 1))
        return F.interpolate(self.head(d1), size=ref.shape[-2:], mode="bilinear", align_corners=False)

    # ── I/O ───────────────────────────────────────────────────────────────
    @staticmethod
    def to_tensor(rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return (t - MEAN) / STD

    @torch.no_grad()
    def predict_mask(self, cur_rgb: np.ndarray, ref_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        self.eval()
        dev = next(self.parameters()).device
        h, w = ref_rgb.shape[:2]
        H, W = max(32, (h // 8) * 8), max(32, (w // 8) * 8)
        import cv2

        r = self.to_tensor(cv2.resize(ref_rgb, (W, H))).to(dev)
        c = self.to_tensor(cv2.resize(cur_rgb, (W, H))).to(dev)
        prob = torch.sigmoid(self(r, c))[0, 0].cpu().numpy()
        return (cv2.resize(prob, (w, h)) >= threshold).astype(np.uint8)

    def save(self, path: Path, meta: Optional[Dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "width": self.width, "tokens": self.tokens_n, "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SiameseChangeNet":
        ck = torch.load(path, map_location=device, weights_only=False)
        net = cls(int(ck.get("width", 32)), int(ck.get("tokens", 4)))
        net.load_state_dict(ck["state"])
        return net.to(device).eval()


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 5.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p, g = pred > 0, gt > 0
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else (1.0 if not g.any() else 0.0)
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}
