#!/usr/bin/env python3
"""Evaluate condition classification: nearest-neighbour baseline vs trained heads.

Both models are scored on the same split of dataset/processed/manifest.csv:
- baseline: `PrototypeConditionClassifier` (label of the nearest training embedding,
  i.e. the repository's previous approach) built from the clean training split;
- classifier: models/condition_classifier.pt.

Each is evaluated on clean images and on the same images passed once through the
webcam degradation model (ml/augment.py, fixed seed). Outputs in reports/eval/:
summary.md, metrics.json and confusion-matrix PNGs per model / condition / asset type.

Usage: python training/evaluate.py [--config config.yaml] [--split test]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from ml.condition_classifier import (  # noqa: E402
    NORMAL_LABEL, ConditionClassifier, MultiHeadConditionClassifier, PrototypeConditionClassifier,
)
from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.embedding_model import AssetEmbeddingModel  # noqa: E402
from training.common import extract_features, load_manifest, write_json  # noqa: E402

logger = logging.getLogger("evaluate")

DEGRADED_VIEW = 1


def score(clf: ConditionClassifier, feats: np.ndarray, asset_type: str, labels: List[str], y_true: List[str]) -> Dict:
    preds = [p.label for p in clf.predict_embeddings(feats, asset_type)]
    p, r, f, s = precision_recall_fscore_support(y_true, preds, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=labels)
    true_def = np.array([y != NORMAL_LABEL for y in y_true])
    pred_def = np.array([y != NORMAL_LABEL for y in preds])
    return {
        "n": len(y_true),
        "accuracy": float(np.mean(np.array(preds) == np.array(y_true))),
        "macro_f1": float(np.mean(f)),
        "per_class": {
            lbl: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lbl in enumerate(labels)
        },
        "defect_recall": float(pred_def[true_def].mean()) if true_def.any() else None,
        "false_alarm_rate": float(pred_def[~true_def].mean()) if (~true_def).any() else None,
        "confusion": cm.tolist(),
        "labels": labels,
    }


def plot_confusion(m: Dict, title: str, path: Path) -> None:
    cm = np.array(m["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = m["labels"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(labels), 0.8 + 0.7 * len(labels)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    config = load_config(args.config)
    out_dir = resolve_path(cfg(config, "reports", "eval_dir"))
    ckpt = resolve_path(cfg(config, "condition_classifier", "checkpoint"))
    rows = load_manifest(config)

    embedder = AssetEmbeddingModel.from_config(config)
    head_maps = MultiHeadConditionClassifier.load(ckpt).label_maps  # label maps only; backbone applied below
    types = list(head_maps)

    data = {}
    proto_emb, proto_lab = {}, {}
    for t in types:
        tr = [r for r in rows if r.asset_type == t and r.split == "train" and r.condition in head_maps[t]]
        te = [r for r in rows if r.asset_type == t and r.split == args.split and r.condition in head_maps[t]]
        proto_emb[t] = extract_features(embedder, [r.path for r in tr], config, view=0)
        proto_lab[t] = [r.condition for r in tr]
        paths = [r.path for r in te]
        data[t] = {
            "paths": paths,
            "y": [r.condition for r in te],
            "clean": extract_features(embedder, paths, config, view=0),
            "webcam": extract_features(embedder, paths, config, view=DEGRADED_VIEW, tag="eval"),
        }
    baseline = PrototypeConditionClassifier(proto_emb, proto_lab, float(cfg(config, "condition_classifier", "prototype_temperature")))

    classifier = MultiHeadConditionClassifier.load(ckpt, embedder)
    if classifier.meta.get("fine_tune_blocks"):
        tag = f"ft:{classifier.meta['trained_at']}"
        for t in types:
            d = data[t]
            d["clean_ft"] = extract_features(embedder, d["paths"], config, view=0, tag=tag)
            d["webcam_ft"] = extract_features(embedder, d["paths"], config, view=DEGRADED_VIEW, tag=f"eval{tag}")

    results: Dict[str, Dict[str, Dict[str, Dict]]] = {"baseline": {}, "classifier": {}}
    for cond in ("clean", "webcam"):
        results["baseline"][cond], results["classifier"][cond] = {}, {}
        for t in types:
            d = data[t]
            base_labels = [lbl for lbl in head_maps[t] if lbl in baseline.label_maps[t]]
            results["baseline"][cond][t] = score(baseline, d[cond], t, base_labels, d["y"])
            feats = d.get(f"{cond}_ft", d[cond])
            results["classifier"][cond][t] = score(classifier, feats, t, head_maps[t], d["y"])
            for model in ("baseline", "classifier"):
                m = results[model][cond][t]
                plot_confusion(m, f"{model} · {cond} · {t} (acc {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f})",
                               out_dir / f"confusion_{model}_{cond}_{t}.png")

    def overall(model: str, cond: str) -> Dict[str, float]:
        per = results[model][cond]
        n = sum(m["n"] for m in per.values())
        return {
            "accuracy": sum(m["accuracy"] * m["n"] for m in per.values()) / max(n, 1),
            "mean_macro_f1": float(np.mean([m["macro_f1"] for m in per.values()])),
            "n": n,
        }

    summary = {f"{m}_{c}": overall(m, c) for m in results for c in ("clean", "webcam")}
    write_json(out_dir / "metrics.json", {
        "split": args.split, "backbone": embedder.active_model_name, "classifier_meta": classifier.meta,
        "summary": summary, "results": results,
    })

    L = [
        "# Condition classification — evaluation",
        "",
        f"Split: **{args.split}** of `dataset/processed/manifest.csv` · backbone `{embedder.active_model_name}` · "
        f"classifier trained {classifier.meta.get('trained_at')} (manifest {classifier.meta.get('manifest')}).",
        "",
        "Baseline = nearest-neighbour label copy over clean training embeddings (previous approach).",
        "Webcam = each test image degraded once by `ml/augment.py` (blur, JPEG, colour cast, perspective, crop).",
        "",
        "## Overall",
        "",
        "| model | images | accuracy (clean) | mean macro-F1 (clean) | accuracy (webcam) | mean macro-F1 (webcam) |",
        "|---|---|---|---|---|---|",
    ]
    for model in ("baseline", "classifier"):
        c, w = summary[f"{model}_clean"], summary[f"{model}_webcam"]
        L.append(f"| {model} | {c['n']} | {c['accuracy']:.3f} | {c['mean_macro_f1']:.3f} | {w['accuracy']:.3f} | {w['mean_macro_f1']:.3f} |")
    L += ["", "## Per asset type", "",
          "| asset type | model | n | acc clean | macro-F1 clean | acc webcam | macro-F1 webcam | defect recall | false-alarm rate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in types:
        for model in ("baseline", "classifier"):
            c, w = results[model]["clean"][t], results[model]["webcam"][t]
            L.append(f"| {t} | {model} | {c['n']} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} | {w['accuracy']:.3f} | "
                     f"{w['macro_f1']:.3f} | {fmt(c['defect_recall'])} | {fmt(c['false_alarm_rate'])} |")
    L += ["", "## Per class (classifier, clean)", ""]
    for t in types:
        L += [f"### {t}", "", "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
        for lbl, m in results["classifier"]["clean"][t]["per_class"].items():
            L.append(f"| {lbl} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
        L += ["", f"![confusion](confusion_classifier_clean_{t}.png)", ""]
    L += [f"Generated by `training/evaluate.py` in {time.time() - t0:.0f} s.", ""]
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", out_dir / "summary.md")
    for k, v in summary.items():
        logger.info("%s: acc %.3f mean macro-F1 %.3f (n=%d)", k, v["accuracy"], v["mean_macro_f1"], v["n"])


if __name__ == "__main__":
    main()
