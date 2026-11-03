#!/usr/bin/env python3
"""Gazebo (gz-sim) world of the simulated site (upgrade 4.3): solar rows, a pole line and a
building wall at the positions sim.world generates, with per-asset texture images.

  python sim/gazebo/generate_world.py --seed 7 --out sim/gazebo/build
  # → build/audi_site.sdf, build/textures/<asset>.png, build/manifest.json (ground truth)

Between missions run inject_damage.py to rewrite the textures with that mission's damage, then
restart the world. PX4 SITL: see sim/gazebo/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.world import build_world  # noqa: E402

DIMS = {"solar_panel": (1.0, 1.7, 0.04, 0.8), "utility_pole": (0.3, 0.3, 10.0, 5.0), "building": (6.0, 0.4, 4.0, 2.0), "machinery": (1.0, 1.0, 1.2, 0.6)}


def model_sdf(a, tex: str) -> str:
    w, d, h, z = DIMS[a.asset_type]
    tilt = "0.35 0 0" if a.asset_type == "solar_panel" else "0 0 0"
    return f"""
    <model name="{a.asset_id}">
      <static>true</static>
      <pose>{a.east:.3f} {a.north:.3f} {z:.3f} {tilt}</pose>
      <link name="link">
        <visual name="v">
          <geometry><box><size>{w} {d} {h}</size></box></geometry>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <pbr><metal><albedo_map>{tex}</albedo_map></metal></pbr>
          </material>
        </visual>
        <collision name="c"><geometry><box><size>{w} {d} {h}</size></box></geometry></collision>
      </link>
    </model>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--missions", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "build")
    args = ap.parse_args(argv)
    world = build_world(seed=args.seed, missions=args.missions)
    (args.out / "textures").mkdir(parents=True, exist_ok=True)
    models = []
    for a in world.assets:
        tex = args.out / "textures" / f"{a.asset_id}.png"
        a.texture.save(tex)
        models.append(model_sdf(a, f"textures/{a.asset_id}.png"))
    lat, lon = world.origin
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="audi_site">
    <physics name="1ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{lat}</latitude_deg><longitude_deg>{lon}</longitude_deg><elevation>0</elevation>
    </spherical_coordinates>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 50 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.4 0.2 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="l"><visual name="v">
      <geometry><plane><normal>0 0 1</normal><size>400 400</size></plane></geometry>
      <material><diffuse>0.35 0.45 0.25 1</diffuse></material></visual>
      <collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision></link></model>
    {''.join(models)}
  </world>
</sdf>
"""
    (args.out / "audi_site.sdf").write_text(sdf)
    manifest = {"seed": args.seed, "origin": world.origin, "days_between_missions": world.days_between_missions,
                "assets": [{"asset_id": a.asset_id, "asset_type": a.asset_type, "east": a.east, "north": a.north,
                            "latitude": a.latitude, "longitude": a.longitude, "dims": DIMS[a.asset_type],
                            "damage": [{"kind": d.spec.kind, "start_mission": d.start_mission, "law": d.law, "rate": d.rate,
                                        "exponent": d.exponent, "e0": d.e0} for d in a.damages]} for a in world.assets]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'audi_site.sdf'} with {len(world.assets)} assets")


if __name__ == "__main__":
    main()
