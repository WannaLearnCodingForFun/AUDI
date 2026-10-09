#!/usr/bin/env python3
"""Train the multi-head condition classifier on dataset/processed/manifest.csv.

Stage 1 (always): frozen backbone. Clean + `augment_views` webcam-augmented
embeddings of every training image are cached, then one head per asset type is
trained with class-balanced sampling and early stopping on validation macro-F1.

Stage 2 (optional, `fine_tune_blocks > 0`, DINOv2 only): the last N transformer
blocks are unfrozen and trained end-to-end on augmented images together with
the heads, again selected on validation macro-F1.

Outputs models/condition_classifier.pt and a metrics JSON
(`condition_classifier.metrics_path`).

Usage: python training/train_condition_classifier.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402

from ml.augment import WebcamAugment  # noqa: E402
from ml.condition_classifier import MultiHeadConditionClassifier, MultiHeadNet  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.embedding_model import AssetEmbeddingModel, DinoV2Backbone  # noqa: E402
from ml.schemas import utc_now_iso  # noqa: E402
from training.common import (  # noqa: E402
    Row, extract_features, label_maps_from, load_manifest, macro_f1, manifest_digest, write_json,
)

logger = logging.getLogger("train_condition_classifier")


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    return (1.0 / counts[y]).astype(np.float64)


def evaluate_heads(net: MultiHeadNet, data: Dict[str, Tuple[np.ndarray, np.ndarray]], label_maps, device) -> Dict[str, float]:
    net.eval()
    out = {}
    with torch.no_grad():
        for t, (x, y) in data.items():
            logits = net(torch.as_tensor(x, device=device), t)
            pred = logits.argmax(1).cpu().numpy()
            out[t] = macro_f1(y, pred, len(label_maps[t]))
    return out


def train_heads(
    train: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val: Dict[str, Tuple[np.ndarray, np.ndarray]],
    label_maps: Dict[str, List[str]],
    ccfg: dict,
    in_dim: int,
    device: torch.device,
    seed: int,
) -> Tuple[MultiHeadNet, List[dict], int, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = MultiHeadNet(in_dim, label_maps, ccfg["head"], int(ccfg["hidden_dim"]), float(ccfg["dropout"])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(ccfg["lr"]), weight_decay=float(ccfg["weight_decay"]))
    bs = int(ccfg["batch_size"])
    tensors = {t: (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)) for t, (x, y) in train.items()}
    weights = {t: balanced_weights(y) for t, (_, y) in train.items()}

    best_state, best_f1, best_epoch, best_per_type = None, -1.0, 0, {}
    history, stale = [], 0
    for epoch in range(1, int(ccfg["epochs"]) + 1):
        net.train()
        batches = []
        for t, (_, y) in train.items():
            idx = rng.choice(len(y), size=len(y), replace=True, p=weights[t] / weights[t].sum())
            batches += [(t, idx[i:i + bs]) for i in range(0, len(idx), bs)]
        rng.shuffle(batches)
        total, n = 0.0, 0
        for t, idx in batches:
            x, y = tensors[t]
            ii = torch.as_tensor(idx, device=device)
            loss = F.cross_entropy(net(x[ii], t), y[ii])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            n += len(idx)
        per_type = evaluate_heads(net, val, label_maps, device)
        score = float(np.mean(list(per_type.values())))
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_macro_f1": score, "val_per_type": per_type})
        logger.info("epoch %d loss %.4f val macro-F1 %.4f %s", epoch, total / max(n, 1), score,
                    {k: round(v, 3) for k, v in per_type.items()})
        if score > best_f1:
            best_f1, best_epoch, best_per_type = score, epoch, per_type
            best_state = copy.deepcopy(net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(ccfg["patience"]):
                logger.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break
    net.load_state_dict(best_state)
    return net, history, best_epoch, best_per_type


class TensorImageDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, preprocess, augment: WebcamAugment, seed: int):
        self.paths, self.labels, self.preprocess, self.augment, self.seed = paths, labels, preprocess, augment, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(ROOT / self.paths[i]) as im:
            img = self.augment(im.convert("RGB"), random.Random(hash((self.seed, self.epoch, i))))
        return self.preprocess(img), int(self.labels[i])


def fine_tune(
    embedder: AssetEmbeddingModel,
    net: MultiHeadNet,
    rows_by_type: Dict[str, Tuple[List[str], np.ndarray]],
    val_by_type: Dict[str, Tuple[List[str], np.ndarray]],
    label_maps: Dict[str, List[str]],
    ccfg: dict,
    device: torch.device,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], List[dict], float]:
    backbone = embedder.backbone
    if not isinstance(backbone, DinoV2Backbone):
        raise ValueError("fine_tune_blocks > 0 requires the DINOv2 backbone")
    model = backbone.model
    n_blocks = int(ccfg["fine_tune_blocks"])
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = list(model.blocks[-n_blocks:]) + [model.norm]
    for m in trainable:
        for p in m.parameters():
            p.requires_grad_(True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": params, "lr": float(ccfg["fine_tune_lr"])}, {"params": net.parameters(), "lr": float(ccfg["fine_tune_head_lr"])}],
        weight_decay=float(ccfg["weight_decay"]),
    )
    augment = WebcamAugment(ccfg["augment"])
    bs = int(ccfg["fine_tune_batch_size"])
    workers = int(ccfg["loader_workers"])
    datasets = {t: TensorImageDataset(p, y, backbone.preprocess, augment, seed) for t, (p, y) in rows_by_type.items()}

    def encode_paths(paths: List[str]) -> np.ndarray:
        from PIL import Image

        out, step = [], int(ccfg["feature_batch_size"])
        for i in range(0, len(paths), step):
            imgs = []
            for p in paths[i:i + step]:
                with Image.open(ROOT / p) as im:
                    imgs.append(im.convert("RGB"))
            out.append(embedder.encode_batch(imgs))
        return np.concatenate(out)

    def val_score() -> Tuple[float, Dict[str, float]]:
        net.eval()
        data = {t: (encode_paths(p), y) for t, (p, y) in val_by_type.items()}
        per = evaluate_heads(net, data, label_maps, device)
        return float(np.mean(list(per.values()))), per

    best_f1, _ = val_score()
    best = ({n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in names}, copy.deepcopy(net.state_dict()))
    best_epoch = 0
    history = [{"epoch": 0, "val_macro_f1": best_f1}]
    for epoch in range(1, int(ccfg["fine_tune_epochs"]) + 1):
        loaders = {}
        for t, ds in datasets.items():
            ds.epoch = epoch
            sampler = WeightedRandomSampler(balanced_weights(ds.labels), num_samples=len(ds), replacement=True,
                                            generator=torch.Generator().manual_seed(seed + epoch))
            loaders[t] = iter(DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers))
        order = [t for t, ds in datasets.items() for _ in range((len(ds) + bs - 1) // bs)]
        random.Random(seed + epoch).shuffle(order)
        net.train()
        model.train()
        total, n = 0.0, 0
        for t in order:
            x, y = next(loaders[t])
            x, y = x.to(device), y.to(device)
            cls, _ = backbone.forward_tensor(x)
            loss = F.cross_entropy(net(F.normalize(cls, dim=1), t), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        model.eval()
        score, per = val_score()
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_macro_f1": score, "val_per_type": per})
        logger.info("fine-tune epoch %d loss %.4f val macro-F1 %.4f", epoch, total / max(n, 1), score)
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            best = ({n_: v.detach().cpu().clone() for n_, v in model.state_dict().items() if n_ in names}, copy.deepcopy(net.state_dict()))
    model.load_state_dict(best[0], strict=False)
    net.load_state_dict(best[1])
    for p in model.parameters():
        p.requires_grad_(False)
    if best_epoch == 0:
        logger.info("fine-tuning did not improve val macro-F1; keeping the frozen backbone")
        return None, history, best_f1
    return best[0], history, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    ccfg = cfg(config, "condition_classifier")
    seed = int(cfg(config, "datasets", "split", "seed"))
    t0 = time.time()

    rows = load_manifest(config)
    label_maps = label_maps_from(rows)
    rows = [r for r in rows if r.asset_type in label_maps and r.condition in label_maps[r.asset_type]]
    embedder = AssetEmbeddingModel.from_config(config)
    device = embedder.device
    logger.info("backbone %s dim %d on %s; heads %s", embedder.active_model_name, embedder.embedding_dim, device, label_maps)

    def split_rows(split: str, t: str) -> List[Row]:
        return [r for r in rows if r.split == split and r.asset_type == t]

    train, val = {}, {}
    train_paths, val_paths = {}, {}
    for t, labels in label_maps.items():
        idx = {c: i for i, c in enumerate(labels)}
        tr, va = split_rows("train", t), split_rows("val", t)
        p_tr, y_tr = [r.path for r in tr], np.array([idx[r.condition] for r in tr])
        p_va, y_va = [r.path for r in va], np.array([idx[r.condition] for r in va])
        views = [extract_features(embedder, p_tr, config, view=v) for v in range(int(ccfg["augment_views"]) + 1)]
        train[t] = (np.concatenate(views), np.tile(y_tr, len(views)))
        val[t] = (extract_features(embedder, p_va, config, view=0), y_va)
        train_paths[t], val_paths[t] = (p_tr, y_tr), (p_va, y_va)
    t_feat = time.time() - t0

    net, history, best_epoch, best_per_type = train_heads(train, val, label_maps, ccfg, embedder.embedding_dim, device, seed)
    best_f1 = float(np.mean(list(best_per_type.values())))

    backbone_state, ft_history = None, []
    if int(ccfg["fine_tune_blocks"]) > 0:
        backbone_state, ft_history, best_f1 = fine_tune(embedder, net, train_paths, val_paths, label_maps, ccfg, device, seed)

    meta = {
        "backbone": embedder.active_model_name,
        "embedding_dim": embedder.embedding_dim,
        "head": ccfg["head"],
        "hidden_dim": int(ccfg["hidden_dim"]),
        "dropout": float(ccfg["dropout"]),
        "fine_tune_blocks": int(ccfg["fine_tune_blocks"]) if backbone_state is not None else 0,
        "manifest": manifest_digest(rows),
        "trained_at": utc_now_iso(),
        "val_macro_f1": best_f1,
    }
    clf = MultiHeadConditionClassifier(net.cpu(), label_maps, meta)
    ckpt = resolve_path(ccfg["checkpoint"])
    clf.save(ckpt, backbone_state)

    counts = {t: dict(Counter(f"{r.split}:{r.condition}" for r in rows if r.asset_type == t)) for t in label_maps}
    metrics = {
        **meta,
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "label_maps": label_maps,
        "counts": counts,
        "best_epoch": best_epoch,
        "val_per_type_macro_f1": best_per_type,
        "history": history,
        "fine_tune_history": ft_history,
        "feature_seconds": round(t_feat, 1),
        "total_seconds": round(time.time() - t0, 1),
    }
    write_json(resolve_path(ccfg["metrics_path"]), metrics)
    logger.info("saved %s (val macro-F1 %.4f, best epoch %d) in %.0f s", ckpt, best_f1, best_epoch, time.time() - t0)


if __name__ == "__main__":
    main()
