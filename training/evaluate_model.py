#!/usr/bin/env python3
"""
Evaluate asset matching, similarity distributions, and change detection.

Change detection supervised metrics require labeled change data; if unavailable,
the script reports that quantitative evaluation cannot be performed.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ml.asset_encoder import AssetEncoder
from ml.change_detector import ChangeDetector
from ml.memory_matcher import AssetMatcher
from ml.similarity import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_manifest(dataset_dir: Path) -> List[Dict]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("manifest.json not found. Run generate_demo_dataset.py first.")
        return []
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f).get("assets", [])


def evaluate_matching(
    encoder: AssetEncoder,
    matcher: AssetMatcher,
    entries: List[Dict],
    threshold: float = 0.75,
) -> Dict:
    """Evaluate top-1/top-5 matching using leave-one-out."""
    by_asset: Dict[str, List] = defaultdict(list)
    for e in entries:
        by_asset[e["asset_id"]].append(e)

    top1_correct = 0
    top5_correct = 0
    total = 0
    same_sims: List[float] = []
    diff_sims: List[float] = []
    false_matches = 0
    new_asset_correct = 0
    new_asset_total = 0

    all_embeddings: Dict[str, np.ndarray] = {}
    for e in entries:
        from PIL import Image

        all_embeddings[e["path"]] = encoder.encode_asset(Image.open(e["path"]))

    paths = list(all_embeddings.keys())
    path_to_asset = {e["path"]: e["asset_id"] for e in entries}

    for query_path, query_emb in all_embeddings.items():
        query_id = path_to_asset[query_path]
        candidates = []
        candidate_ids = []
        for p, emb in all_embeddings.items():
            if p == query_path:
                continue
            candidates.append(emb)
            candidate_ids.append(path_to_asset[p])

        if not candidates:
            continue

        matrix = np.stack(candidates)
        result = matcher.match(query_emb, matrix, candidate_ids, threshold)
        total += 1

        # Top-1: best candidate asset_id
        from ml.similarity import batch_cosine_similarity

        sims = batch_cosine_similarity(query_emb, matrix)
        best_idx = int(np.argmax(sims))
        if candidate_ids[best_idx] == query_id:
            top1_correct += 1

        top5_idx = np.argsort(sims)[::-1][:5]
        if query_id in [candidate_ids[i] for i in top5_idx]:
            top5_correct += 1

        for i, cid in enumerate(candidate_ids):
            sim = float(sims[i])
            if cid == query_id:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

        if result.matched and result.asset_id != query_id:
            false_matches += 1

    # New asset detection: hold out one asset entirely
    asset_ids = list(by_asset.keys())
    if len(asset_ids) >= 2:
        holdout = asset_ids[-1]
        train_entries = [e for e in entries if e["asset_id"] != holdout]
        test_entries = by_asset[holdout]
        train_embs = {e["path"]: all_embeddings[e["path"]] for e in train_entries}
        for te in test_entries:
            qemb = all_embeddings[te["path"]]
            cids = [path_to_asset[p] for p in train_embs]
            cmatrix = np.stack(list(train_embs.values()))
            result = matcher.match(qemb, cmatrix, cids, threshold)
            new_asset_total += 1
            if not result.matched:
                new_asset_correct += 1

    precision = top1_correct / max(total, 1)
    recall = top1_correct / max(total, 1)

    return {
        "top1_accuracy": top1_correct / max(total, 1),
        "top5_accuracy": top5_correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "false_match_rate": false_matches / max(total, 1),
        "new_asset_detection_rate": new_asset_correct / max(new_asset_total, 1),
        "same_asset_similarity_mean": float(np.mean(same_sims)) if same_sims else 0.0,
        "different_asset_similarity_mean": float(np.mean(diff_sims)) if diff_sims else 0.0,
        "same_asset_similarity_std": float(np.std(same_sims)) if same_sims else 0.0,
        "different_asset_similarity_std": float(np.std(diff_sims)) if diff_sims else 0.0,
    }


def evaluate_change_detection(entries: List[Dict], encoder: AssetEncoder) -> Dict:
    """Evaluate change detection if damage labels exist."""
    has_labels = any(e.get("condition_damage", 0) > 0 for e in entries)
    if not has_labels:
        return {
            "note": "No labeled change data available. Quantitative supervised evaluation cannot be performed.",
        }

    detector = ChangeDetector(threshold=0.30)
    tp = fp = tn = fn = 0

    by_asset: Dict[str, List] = defaultdict(list)
    for e in entries:
        by_asset[e["asset_id"]].append(e)

    for asset_id, asset_entries in by_asset.items():
        asset_entries.sort(key=lambda x: x.get("condition_damage", 0))
        from PIL import Image

        baseline = encoder.encode_asset(Image.open(asset_entries[0]["path"]))
        for e in asset_entries[1:]:
            current = encoder.encode_asset(Image.open(e["path"]))
            result = detector.compare(current, [baseline.tolist()])
            actual_changed = e.get("condition_damage", 0) > 0
            if result.changed and actual_changed:
                tp += 1
            elif result.changed and not actual_changed:
                fp += 1
            elif not result.changed and not actual_changed:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "change_detection_accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "note": "Evaluated on DEMO synthetic damage labels — not production metrics.",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AUDI ML components")
    parser.add_argument("--data", type=str, default="dataset")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    entries = load_manifest(Path(args.data))
    if not entries:
        return

    logger.info("Loading embedding model for evaluation")
    encoder = AssetEncoder()
    matcher = AssetMatcher()

    matching_results = evaluate_matching(encoder, matcher, entries, args.threshold)
    change_results = evaluate_change_detection(entries, encoder)

    results = {"matching": matching_results, "change_detection": change_results}
    print(json.dumps(results, indent=2))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
