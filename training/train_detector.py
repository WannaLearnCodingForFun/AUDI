#!/usr/bin/env python3
"""Fine-tune a closed-set detector for the known asset types (upgrade 3.2).

YOLOv8 / YOLO11 (Ultralytics) on dataset/processed/detection/ (written by
scripts/prepare_unified_dataset.py in YOLO format). YOLO-World stays for unknown types:
detection.mode = "hybrid" runs the fine-tuned model first and YOLO-World only on frames
where it finds nothing (ml/detector.HybridDetector).

  python training/train_detector.py --model yolo11s.pt --epochs 80 --imgsz 640
Writes models/detector_finetuned.pt and reports/train/detector_metrics.json
(mAP@0.5 per class, mAP@0.5:0.95, and detection latency p50/p95 on the val images).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.net import ensure_ca_bundle  # noqa: E402


def data_yaml(config) -> Path:
    root = resolve_path(cfg(config, "datasets", "prepare", "output_dir")) / "detection"
    y = root / "data.yaml"
    if not y.exists():
        names = list(cfg(config, "detection", "prompts").keys())
        y.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: {json.dumps(dict(enumerate(names)))}\n")
    return y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    ensure_ca_bundle()
    from ultralytics import YOLO

    config = load_config()
    out = resolve_path(cfg(config, "detection").get("finetuned_weights", "models/detector_finetuned.pt"))
    model = YOLO(args.model)
    model.train(data=str(data_yaml(config)), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=str(resolve_path("reports/train/detector_runs")), name="finetune", exist_ok=True, patience=20)
    best = Path(model.trainer.best)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(best.read_bytes())
    tuned = YOLO(str(out))
    val = tuned.val(data=str(data_yaml(config)), imgsz=args.imgsz, device=args.device)
    names = tuned.names
    per_class = {names[int(c)]: float(ap50) for c, ap50 in zip(val.box.ap_class_index, val.box.ap50)}
    imgs = sorted((data_yaml(config).parent / "images" / "val").glob("*.jpg"))[:100]
    lat = []
    for p in imgs:
        t = time.perf_counter()
        tuned.predict(str(p), imgsz=args.imgsz, verbose=False, device=args.device)
        lat.append((time.perf_counter() - t) * 1000)
    metrics = {"weights": str(out), "map50": float(val.box.map50), "map50_95": float(val.box.map), "per_class_ap50": per_class,
               "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None, "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None}
    p = resolve_path("reports/train/detector_metrics.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
