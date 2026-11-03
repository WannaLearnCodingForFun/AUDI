#!/usr/bin/env python3
"""Thermal / fused condition head for solar modules (upgrade 3.1).

Data: Kaggle `marcosgabriel/infrared-solar-modules` (enable the `pv_infrared` source in
config.yaml and run scripts/download_datasets.py). Its 24×40 px IR module crops carry 12
anomaly classes (Cell, Cell-Multi, Hot-Spot, Hot-Spot-Multi, Diode, Diode-Multi, Cracking,
Offline-Module, Shadowing, Soiling, Vegetation, No-Anomaly).

Trains a logistic-regression head on `ml.thermal.thermal_features` (+ the RGB embedding when
paired RGB crops exist) and reports hotspot-detection recall — the 3.1 verification metric.

  python training/train_thermal.py --root dataset/raw/pv_infrared
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

from ml.config import resolve_path  # noqa: E402
from ml.thermal import ThermalFrame, thermal_features  # noqa: E402

HOT = {"Hot-Spot", "Hot-Spot-Multi", "Cell", "Cell-Multi", "Diode", "Diode-Multi"}


def load(root: Path):
    meta = json.loads(next(root.rglob("module_metadata.json")).read_text())
    X, y, hot = [], [], []
    for k, v in meta.items():
        p = next(root.rglob(v["image_filepath"].split("/")[-1]), None)
        if p is None:
            continue
        g = np.asarray(Image.open(p).convert("L"))
        t = ThermalFrame.from_8bit(g, 20.0, 80.0).celsius     # relative scale: the dataset is not radiometric
        X.append(thermal_features(t))
        y.append(v["anomaly_class"])
        hot.append(v["anomaly_class"] in HOT)
    return np.stack(X), np.array(y), np.array(hot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=resolve_path("dataset/raw/pv_infrared"))
    ap.add_argument("--out", type=Path, default=resolve_path("models/thermal_head.json"))
    args = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, recall_score
    from sklearn.model_selection import train_test_split

    X, y, hot = load(args.root)
    Xtr, Xte, ytr, yte, htr, hte = train_test_split(X, y, hot, test_size=0.2, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    hot_pred = np.isin(pred, list(HOT))
    report = {"hotspot_recall": float(recall_score(hte, hot_pred)), "n_test": int(len(yte)),
              "per_class": classification_report(yte, pred, output_dict=True, zero_division=0)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"classes": clf.classes_.tolist(), "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}))
    out = resolve_path("reports/train/thermal_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"hotspot_recall": report["hotspot_recall"], "n_test": report["n_test"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
