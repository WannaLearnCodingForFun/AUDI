"""Train the identity (re-ID) head on frozen backbone embeddings (upgrade 1.3).

Identities and their views:
  * real repeated views from the re-ID set (`reid.dir`, scripts/capture_reid_set.py): one
    identity per object, every session is a different visit;
  * synthetic identities: each clean dataset image (or procedural texture) is an asset; its
    views are re-photographed with lighting change, viewpoint (perspective + scale + small
    rotation), and condition changes that MUST NOT change identity — added cracks, rust, dust,
    snow (ml/damage_synth.py).

Training: backbone frozen, embeddings computed once per view; head trained with ArcFace
(default) or batch-hard triplet loss. Validation on held-out identities reports Rank-1,
Rank-5 and mAP for the head against the global-embedding baseline, written to
reports/train/reid_head_metrics.json; the head goes to `reid_head.checkpoint`.

  python training/train_reid_head.py --identities 600 --views 6 --loss arcface
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
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import load_config, resolve_path  # noqa: E402
from ml.damage_synth import DamageSpec, render_damage  # noqa: E402
from ml.reid_head import ArcFace, IdentityHead, batch_hard_triplet, retrieval_metrics, save_head  # noqa: E402


def synthetic_view(img: Image.Image, rng: random.Random, damage_prob: float = 0.5) -> Image.Image:
    import cv2

    w, h = img.size
    arr = np.asarray(img.convert("RGB"))
    d = 0.12 * min(w, h)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
    arr = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)
    v = Image.fromarray(arr).rotate(rng.uniform(-8, 8), resample=Image.BILINEAR)
    v = ImageEnhance.Brightness(v).enhance(rng.uniform(0.6, 1.4))
    v = ImageEnhance.Contrast(v).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < damage_prob:
        kind = rng.choice(["crack", "rust", "soiling", "snow"])
        if kind == "snow":
            a = np.asarray(v).astype(np.float32)
            m = np.random.default_rng(rng.randrange(1 << 30)).random(a.shape[:2]) < rng.uniform(0.05, 0.3)
            a[m] = 0.3 * a[m] + 0.7 * 245
            v = Image.fromarray(a.astype(np.uint8))
        else:
            v, _ = render_damage(v, DamageSpec(kind, rng.randrange(1 << 30), (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))),
                                 rng.uniform(0.2, 0.7))
    return v


def build_views(n_ids: int, n_views: int, seed: int, size: int) -> Tuple[List[Image.Image], List[str]]:
    from training.train_change_detector import clean_images, procedural_texture

    rng = random.Random(seed)
    sources = clean_images(n_ids)
    imgs, ids = [], []
    for i in range(n_ids):
        if i < len(sources):
            with Image.open(sources[i]) as im:
                base = im.convert("RGB").resize((size, size))
        else:
            base = procedural_texture(seed * 1000 + i, size)
        for _ in range(n_views):
            imgs.append(synthetic_view(base, rng))
            ids.append(f"syn_{i}")
    return imgs, ids


def real_views(config) -> Tuple[List[Image.Image], List[str]]:
    from training.evaluate_reid import load_records

    root = resolve_path(config["reid"]["dir"])
    imgs, ids = [], []
    for r in load_records(root):
        with Image.open(root / r["path"]) as im:
            imgs.append(im.convert("RGB"))
        ids.append(f"real_{r['object_id']}")
    return imgs, ids


def split_by_identity(ids: List[str], val_frac: float, seed: int):
    uniq = sorted(set(ids))
    random.Random(seed).shuffle(uniq)
    val = set(uniq[: max(1, int(len(uniq) * val_frac))])
    return [i for i, x in enumerate(ids) if x not in val], [i for i, x in enumerate(ids) if x in val]


def gallery_query(idx: List[int], ids: List[str]):
    first: dict = {}
    g, q = [], []
    for i in idx:
        if ids[i] not in first:
            first[ids[i]] = i
            g.append(i)
        else:
            q.append(i)
    return g, q


def train(emb: np.ndarray, ids: List[str], args) -> Tuple[IdentityHead, dict]:
    tr, va = split_by_identity(ids, args.val_frac, args.seed)
    classes = {c: k for k, c in enumerate(sorted({ids[i] for i in tr}))}
    X = torch.from_numpy(emb[tr]).float()
    Y = torch.tensor([classes[ids[i]] for i in tr])
    head = IdentityHead(emb.shape[1], args.dim, args.hidden)
    arc = ArcFace(args.dim, len(classes)) if args.loss == "arcface" else None
    params = list(head.parameters()) + (list(arc.parameters()) if arc else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(args.seed)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for b in range(0, len(X), args.batch):
            sel = perm[b:b + args.batch]
            z = head(X[sel])
            loss = arc(z, Y[sel]) if arc else batch_hard_triplet(z, Y[sel], args.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(sel)
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"epoch {ep + 1}/{args.epochs} loss {total / len(X):.4f}", flush=True)
    head.eval()
    g, q = gallery_query(va, ids)
    with torch.no_grad():
        z = head(torch.from_numpy(emb).float()).numpy()
    gi, qi = [ids[i] for i in g], [ids[i] for i in q]
    metrics = {
        "baseline_global": retrieval_metrics(emb[q], qi, emb[g], gi),
        "reid_head": retrieval_metrics(z[q], qi, z[g], gi),
        "train_identities": len(classes), "val_identities": len(set(gi)), "loss": args.loss,
    }
    return head, metrics


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", type=int, default=600)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--loss", choices=["arcface", "triplet"], default="arcface")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--model", default=None, help="backbone override (dinov2 | resnet50 | projection)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metrics", type=Path, default=None)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    config = load_config()
    from ml.embedding_model import AssetEmbeddingModel

    embedder = AssetEmbeddingModel.from_config(config, args.model)
    imgs, ids = build_views(args.identities, args.views, args.seed, args.size)
    r_imgs, r_ids = real_views(config)
    imgs, ids = imgs + r_imgs, ids + r_ids
    emb = np.concatenate([embedder.encode_batch(imgs[i:i + 64]) for i in range(0, len(imgs), 64)])
    head, metrics = train(emb, ids, args)
    metrics.update({"backbone": embedder.active_model_name, "views": len(imgs), "real_views": len(r_imgs)})
    out = args.out or resolve_path((config.get("reid_head") or {}).get("checkpoint") or "models/reid_head.pt")
    save_head(head, out, f"{embedder.active_model_name}_{args.loss}", metrics)
    mp = args.metrics or resolve_path("reports/train/reid_head_metrics.json")
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    main()
