#!/usr/bin/env python3
"""Defect segmentation model (upgrade 3.3): YOLO-seg fine-tuned on the crack / corrosion
masks in dataset/raw (crack_concrete, uav_crack, corrosion_detect polygons), converted to
YOLO-seg polygons by scripts/prepare_unified_dataset.py into dataset/processed/segmentation/.

  python training/train_defect_seg.py --model yolo11s-seg.pt --epochs 60
Sets segmentation.backend = yolo_seg with models/defect_seg.pt. Reports mask IoU and the
crack-length error in mm on images whose metric scale is known (reports/train/defect_seg_metrics.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.segmentation import YoloSegSegmenter, skeleton_length  # noqa: E402
from ml.siamese_change import mask_metrics  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mm-per-px", type=float, default=None, help="scale of the val images for the length error")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    config = load_config()
    root = resolve_path(cfg(config, "datasets", "prepare", "output_dir")) / "segmentation"
    model = YOLO(args.model)
    model.train(data=str(root / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                project=str(resolve_path("reports/train/seg_runs")), name="defect_seg", exist_ok=True)
    out = resolve_path(cfg(config, "segmentation")["yolo_seg_weights"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(Path(model.trainer.best).read_bytes())
    seg = YoloSegSegmenter(str(out))
    ious, len_err = [], []
    for img_p in sorted((root / "images" / "val").glob("*.jpg"))[:300]:
        mp = root / "masks" / "val" / (img_p.stem + ".png")
        if not mp.exists():
            continue
        gt = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.uint8)
        with Image.open(img_p) as im:
            pred = seg.segment(im.convert("RGB"), "crack")
        ious.append(mask_metrics(pred, gt)["iou"])
        if args.mm_per_px and gt.any():
            len_err.append(abs(skeleton_length(pred) - skeleton_length(gt)) * args.mm_per_px)
    metrics = {"weights": str(out), "mask_iou": float(np.mean(ious)) if ious else None,
               "length_error_mm_mean": float(np.mean(len_err)) if len_err else None, "n": len(ious)}
    p = resolve_path("reports/train/defect_seg_metrics.json")
    p.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
