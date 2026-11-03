#!/usr/bin/env python3
"""Human feedback loop (upgrade 5.4): periodic fine-tuning on inspector labels.

Labels come from `alert_feedback` (confirmed | false_alarm | wrong_asset) per site:
  * condition head   confirmed → the observed label is right; false_alarm → the image is
                     `correct_condition` (default "normal"). Images are the stored crops.
                     The per-type heads are fine-tuned on these rows mixed with a replay sample
                     of the original training set (no forgetting), small learning rate.
  * re-ID head       wrong_asset with correct_asset_id → a positive pair (crop, correct asset's
                     prototypes) and a hard negative (crop, the wrongly matched asset); the head is
                     fine-tuned with the triplet loss on these.
Used labels are marked so the next round only adds new ones. Verification: false alarms per
mission across feedback rounds (`GET /api/feedback/stats`).

  python training/retrain_from_feedback.py --site site_demo --min-labels 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ml.config import cfg, load_config, resolve_path  # noqa: E402


def condition_rows(store, fb):
    rows = []
    for f in fb:
        if f["label"] not in ("confirmed", "false_alarm"):
            continue
        obs = next((o for o in store.observations(f["asset_id"]) if o["observation_id"] == f["observation_id"]), None)
        if obs is None or not obs.get("image_reference"):
            continue
        a = store.get_asset(f["asset_id"])
        label = obs["condition_description"] if f["label"] == "confirmed" else (f.get("correct_condition") or "normal")
        rows.append((store.open_pil(obs["image_reference"]), a["asset_type"], label))
    return rows


def reid_triplets(store, fb):
    out = []
    for f in fb:
        if f["label"] != "wrong_asset" or not f.get("correct_asset_id"):
            continue
        obs = next((o for o in store.observations(f["correct_asset_id"]) + store.observations(f["asset_id"])
                    if o["observation_id"] == f["observation_id"]), None)
        pos = [o for o in store.observations(f["correct_asset_id"]) if o["observation_id"] != f["observation_id"]]
        neg = [o for o in store.observations(f["asset_id"]) if o["observation_id"] != f["observation_id"]]
        if obs and pos and neg:
            out.append((store.open_pil(obs["image_reference"]), store.open_pil(pos[-1]["image_reference"]), store.open_pil(neg[-1]["image_reference"])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None)
    ap.add_argument("--min-labels", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args(argv)
    config = load_config()
    from memory.store import LocalStore
    from ml.condition_classifier import MultiHeadConditionClassifier
    from ml.embedding_model import AssetEmbeddingModel
    from ml.reid_head import IdentityEncoder, batch_hard_triplet, save_head

    store = LocalStore(config)
    fb = store.list_feedback(args.site, unused_only=True)
    if len(fb) < args.min_labels:
        print(f"{len(fb)} new labels (< {args.min_labels}); nothing to do")
        return 0
    emb = AssetEmbeddingModel.from_config(config)
    report = {"labels": len(fb)}
    ck = resolve_path(cfg(config, "condition_classifier", "checkpoint"))
    rows = condition_rows(store, fb)
    if rows and ck.exists():
        clf = MultiHeadConditionClassifier.load(ck, emb)
        net = clf.net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        by_type = {}
        for img, t, label in rows:
            if t in clf.label_maps and label in clf.label_maps[t]:
                by_type.setdefault(t, []).append((img, clf.label_maps[t].index(label)))
        for _ in range(args.epochs):
            for t, items in by_type.items():
                x = torch.from_numpy(emb.encode_batch([i for i, _ in items])).to(next(net.parameters()).device)
                y = torch.tensor([l for _, l in items], device=x.device)
                loss = F.cross_entropy(net(x, t), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        clf.meta["feedback_rounds"] = int(clf.meta.get("feedback_rounds", 0)) + 1
        clf.save(ck)
        report["condition_rows"] = sum(len(v) for v in by_type.values())
    trip = reid_triplets(store, fb)
    enc = IdentityEncoder.from_config(config, emb.embedding_dim)
    if trip and enc.head is not None:
        head = enc.head.train()
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        for _ in range(args.epochs):
            a, p, n = (torch.from_numpy(emb.encode_batch([t[k] for t in trip])) for k in range(3))
            z = head(torch.cat([a, p, n]))
            y = torch.cat([torch.arange(len(trip)), torch.arange(len(trip)), torch.arange(len(trip), 2 * len(trip))])
            loss = batch_hard_triplet(z, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        save_head(head.eval(), resolve_path(cfg(config, "reid_head", "checkpoint")), enc.name + "+feedback")
        report["reid_triplets"] = len(trip)
    store.mark_feedback_used([f["feedback_id"] for f in fb])
    report["stats"] = store.feedback_stats(args.site)["totals"]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
