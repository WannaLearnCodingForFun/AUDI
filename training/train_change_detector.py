"""Train the Siamese change-detection network (upgrade 2.1).

Data
  synthetic  before/after pairs rendered on clean asset images from dataset/processed
             (or procedural textures when no dataset is present): pasted cracks, rust and
             soiling at random extents, with photometric jitter on the after view so lighting
             alone is never labelled as change. Half the pairs are unchanged (negatives).
  --levir    optional LEVIR-CD style folder (A/, B/, label/) used for pre-training first.

Outputs models/change_detector.pt (set change_mask.checkpoint) and
reports/train/change_detector_metrics.json with mask F1 / IoU and false changes per 100
unchanged pairs on a held-out synthetic set.

  python training/train_change_detector.py --pairs 4000 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import load_config, resolve_path  # noqa: E402
from ml.damage_synth import DamageSpec, change_pair  # noqa: E402
from ml.siamese_change import SiameseChangeNet, dice_bce_loss, mask_metrics  # noqa: E402


def procedural_texture(seed: int, size: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.uniform(60, 180, 3)
    arr = np.ones((size, size, 3)) * base + rng.normal(0, 10, (size, size, 3))
    # cell grid like a PV module, or panel seams
    step = int(rng.integers(16, 48))
    arr[::step, :, :] *= 0.6
    arr[:, ::step, :] *= 0.6
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def clean_images(limit: int) -> List[Path]:
    root = ROOT / "dataset" / "processed"
    out: List[Path] = []
    for p in root.rglob("*.jpg"):
        if "normal" in p.parts or "clean" in p.parts:
            out.append(p)
            if len(out) >= limit:
                break
    return out


class SyntheticPairs(Dataset):
    def __init__(self, n: int, size: int, seed: int, sources: List[Path]):
        self.n, self.size, self.seed, self.sources = n, size, seed, sources

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        rng = random.Random(self.seed * 100_003 + i)
        if self.sources:
            with Image.open(self.sources[i % len(self.sources)]) as im:
                img = im.convert("RGB").resize((self.size, self.size))
        else:
            img = procedural_texture(self.seed * 7 + i, self.size)
        kind = rng.choice(["crack", "rust", "soiling"])
        spec = DamageSpec(kind, rng.randrange(1 << 30), (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)))
        if rng.random() < 0.5:
            before = after = rng.uniform(0.0, 0.6)          # unchanged pair (possibly already damaged)
        else:
            before = rng.uniform(0.0, 0.4)
            after = before + rng.uniform(0.15, 0.5)
        b, a, m = change_pair(img, spec, before, after, jitter=rng)
        t = SiameseChangeNet.to_tensor
        return t(np.asarray(b))[0], t(np.asarray(a))[0], torch.from_numpy(m.astype(np.float32))[None]


class LevirPairs(Dataset):
    def __init__(self, root: Path, size: int):
        self.a = sorted((root / "A").glob("*"))
        self.root, self.size = root, size

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        name = self.a[i].name
        load = lambda d, mode: Image.open(self.root / d / name).convert(mode).resize((self.size, self.size))
        t = SiameseChangeNet.to_tensor
        m = (np.asarray(load("label", "L")) > 127).astype(np.float32)
        return t(np.asarray(load("A", "RGB")))[0], t(np.asarray(load("B", "RGB")))[0], torch.from_numpy(m)[None]


def run_epochs(net, loader, epochs: int, lr: float, device) -> None:
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        net.train()
        total = 0.0
        for ref, cur, m in loader:
            ref, cur, m = ref.to(device), cur.to(device), m.to(device)
            loss = dice_bce_loss(net(ref, cur), m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(ref)
        print(f"epoch {ep + 1}/{epochs} loss {total / len(loader.dataset):.4f}", flush=True)


@torch.no_grad()
def evaluate(net, ds: Dataset, device) -> dict:
    net.eval()
    f1s, ious, false_changes, unchanged = [], [], 0, 0
    for i in range(len(ds)):
        ref, cur, m = ds[i]
        prob = torch.sigmoid(net(ref[None].to(device), cur[None].to(device)))[0, 0].cpu().numpy()
        pred, gt = (prob >= 0.5).astype(np.uint8), m[0].numpy().astype(np.uint8)
        if gt.sum() == 0:
            unchanged += 1
            false_changes += int(pred.sum() > 0.001 * pred.size)
        else:
            mm = mask_metrics(pred, gt)
            f1s.append(mm["f1"])
            ious.append(mm["iou"])
    return {
        "mask_f1": float(np.mean(f1s)) if f1s else None,
        "mask_iou": float(np.mean(ious)) if ious else None,
        "false_changes_per_100_unchanged": 100.0 * false_changes / max(1, unchanged),
        "changed_pairs": len(f1s),
        "unchanged_pairs": unchanged,
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--val-pairs", type=int, default=400)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--levir", type=Path, default=None)
    ap.add_argument("--levir-epochs", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    config = load_config()
    out = args.out or resolve_path((config.get("change_mask") or {}).get("checkpoint") or "models/change_detector.pt")
    metrics_path = args.metrics or resolve_path("reports/train/change_detector_metrics.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SiameseChangeNet(args.width).to(device)
    if args.levir is not None:
        run_epochs(net, DataLoader(LevirPairs(args.levir, args.size), args.batch, shuffle=True), args.levir_epochs, args.lr, device)
    sources = clean_images(5000)
    train = SyntheticPairs(args.pairs, args.size, args.seed, sources)
    run_epochs(net, DataLoader(train, args.batch, shuffle=True), args.epochs, args.lr, device)
    metrics = evaluate(net, SyntheticPairs(args.val_pairs, args.size, args.seed + 1, sources), device)
    metrics.update({"pairs": args.pairs, "epochs": args.epochs, "size": args.size, "real_sources": len(sources)})
    net.save(out, metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
