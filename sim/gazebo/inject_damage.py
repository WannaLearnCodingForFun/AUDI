#!/usr/bin/env python3
"""Rewrite the world's asset textures with mission k's damage (cracks, rust decals, soiling).

  python sim/gazebo/inject_damage.py --build sim/gazebo/build --mission 3
Restart the gz world (or PX4 SITL) afterwards so Gazebo reloads the textures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.world import build_world  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path(__file__).parent / "build")
    ap.add_argument("--mission", type=int, required=True)
    args = ap.parse_args(argv)
    m = json.loads((args.build / "manifest.json").read_text())
    world = build_world(seed=m["seed"], missions=max(args.mission + 1, 6))
    truth = {}
    for a in world.assets:
        img, extents, _ = a.appearance(args.mission)
        img.save(args.build / "textures" / f"{a.asset_id}.png")
        truth[a.asset_id] = extents
    (args.build / f"damage_mission_{args.mission}.json").write_text(json.dumps(truth, indent=2))
    print(f"mission {args.mission}: damage written for {sum(1 for v in truth.values() if any(x > 0 for x in v.values()))} assets")


if __name__ == "__main__":
    main()
