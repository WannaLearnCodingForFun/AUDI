#!/usr/bin/env python3
"""Copy-paste augmentation for weak classes (upgrade 3.5).

Target: solar `physical_damage` (F1 0.47). Donor defects are cut from existing training
images of the class with the classical crack segmenter (and from the crack datasets, whose
masks are cracks on concrete — recoloured to glass-crack tones), then pasted with random
scale / rotation / position onto clean (`normal`) training panels.

  python training/copy_paste_augment.py            # → data/copy_paste/*.jpg + manifest.json
  python training/train_condition_classifier.py    # picks the rows up (condition_classifier.copy_paste)

Only TRAIN-split images are used as donors and hosts, so validation / test stay untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.damage_synth import copy_paste  # noqa: E402
from ml.segmentation import ClassicalSegmenter  # noqa: E402
from training.common import load_manifest  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)
    config = load_config()
    cp = cfg(config, "condition_classifier")["copy_paste"]
    out_dir = resolve_path(cp["manifest"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(config)
    seg = ClassicalSegmenter(config)
    rng = random.Random(args.seed)
    manifest = []
    for asset_type, target in cp["targets"].items():
        donors_rows = [r for r in rows if r.split == "train" and r.asset_type == asset_type and r.condition == target]
        crack_rows = [r for r in rows if r.split == "train" and r.condition == "crack"]
        hosts = [r for r in rows if r.split == "train" and r.asset_type == asset_type and r.condition == "normal"]
        donors = []
        for r in donors_rows[:400] + crack_rows[:400]:
            with Image.open(ROOT / r.path) as im:
                img = im.convert("RGB")
            m = seg.segment(img, "crack")
            if 0.002 < m.mean() < 0.25:
                donors.append((img, m))
        if not donors or not hosts:
            print(f"{asset_type}/{target}: no donors ({len(donors)}) or hosts ({len(hosts)}); skipped")
            continue
        for i in range(int(cp["per_target"])):
            h = rng.choice(hosts)
            with Image.open(ROOT / h.path) as im:
                host = im.convert("RGB")
            d, m = rng.choice(donors)
            out, mask = copy_paste(host, d, m, rng)
            if mask.sum() < 20:
                continue
            rel = out_dir.relative_to(ROOT) / f"{asset_type}_{target}_{i:05d}.jpg"
            out.save(ROOT / rel, quality=92)
            manifest.append({"path": rel.as_posix(), "asset_type": asset_type, "condition": target, "split": "train",
                             "source": "copy_paste", "host": h.path, "mask_fraction": float(mask.mean())})
        print(f"{asset_type}/{target}: {sum(1 for x in manifest if x['condition'] == target)} synthetic images from {len(donors)} donors")
    resolve_path(cp["manifest"]).write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
