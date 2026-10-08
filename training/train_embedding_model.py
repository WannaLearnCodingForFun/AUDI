#!/usr/bin/env python3
"""
Optional fine-tuning script for the asset embedding model.

The pretrained model works out-of-the-box; fine-tuning requires REAL labeled data.
Do not fine-tune on synthetic demo data for production deployment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.embedding_model import AssetEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetImageDataset(Dataset):
    """Simple folder-based dataset: split/asset_type/*.png"""

    def __init__(self, root: Path, split: str = "train"):
        self.samples: list[tuple[Path, str]] = []
        split_dir = root / split
        if not split_dir.exists():
            logger.warning("Split directory not found: %s", split_dir)
            return
        for type_dir in sorted(split_dir.iterdir()):
            if type_dir.is_dir():
                for img_path in sorted(type_dir.glob("*.*")):
                    if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        self.samples.append((img_path, type_dir.name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return Image.open(path).convert("RGB"), label


def fine_tune(args):
    data_root = Path(args.data)
    if not (data_root / "train").exists():
        logger.error("No training data at %s. Run generate_demo_dataset.py first for testing.", data_root)
        return

    logger.info("Loading pretrained embedding model")
    model_wrapper = AssetEmbeddingModel(model_name=args.model, device=args.device)
    model = model_wrapper.model
    model.train()

    # Simple linear probe on top of frozen backbone for asset type classification
    dataset = AssetImageDataset(data_root, "train")
    if len(dataset) == 0:
        logger.error("Empty training set")
        return

    labels = sorted({lbl for _, lbl in dataset.samples})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    classifier = nn.Linear(model_wrapper.embedding_dim, len(labels)).to(model_wrapper.device)
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + list(model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, lbls in loader:
            tensors = torch.stack([model_wrapper.preprocess(img) for img in images]).to(model_wrapper.device)
            targets = torch.tensor([label_to_idx[l] for l in lbls]).to(model_wrapper.device)

            with torch.set_grad_enabled(True):
                features = model(tensors)
                if isinstance(features, dict):
                    features = features.get("x_norm_clstoken", next(iter(features.values())))
                logits = classifier(features)
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, args.epochs, total_loss / max(len(loader), 1))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "embedding_model.pt")
    logger.info("Saved fine-tuned weights to %s", out_dir / "embedding_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Optional AUDI embedding model fine-tuning")
    parser.add_argument("--data", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="dinov2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="checkpoints")
    args = parser.parse_args()
    fine_tune(args)


if __name__ == "__main__":
    main()
