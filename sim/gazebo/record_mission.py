#!/usr/bin/env python3
"""ROS 2 recorder: PX4 SITL camera + NavSatFix + pose → the mission-folder format with ground
truth boxes (asset footprints projected into the image with ml.geo_projection.world_to_pixel).

  ros2 run ros_gz_bridge parameter_bridge /camera@sensor_msgs/msg/Image@gz.msgs.Image ...
  python sim/gazebo/record_mission.py --build sim/gazebo/build --mission 3 --out runs/gz/m3
Then: python scripts/run_sim_eval.py --missions-dir runs/gz
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2" / "audi_ros"))


def gt_boxes(manifest, damage, pose, intr, drone_en, frame_size):
    from ml.geo_projection import CameraPose, world_to_pixel

    cp = CameraPose(**{k: pose[k] for k in ("latitude", "longitude", "altitude_agl", "heading_deg", "pitch_deg", "roll_deg")})
    out = []
    W, H = frame_size
    for a in manifest["assets"]:
        w, d, h, z = a["dims"]
        pts = []
        for dx in (-w / 2, w / 2):
            for dy in (-d / 2, d / 2):
                for dz in (z - h / 2, z + h / 2):
                    p = world_to_pixel(a["east"] + dx, a["north"] + dy, dz, cp, intr, *drone_en)
                    if p is not None:
                        pts.append(p)
        if len(pts) < 8:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))]
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            continue
        out.append({"asset_id": a["asset_id"], "asset_type": a["asset_type"], "bbox": [int(v) for v in box],
                    "damage": damage.get(a["asset_id"], {}), "damage_area_px": 0})
    return out


def main(argv=None):
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from PIL import Image
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as RosImage
    from sensor_msgs.msg import NavSatFix

    from audi_ros.core import camera_pose
    from ml.camera_model import CameraIntrinsics

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=Path(__file__).parent / "build")
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hfov", type=float, default=78.0)
    ap.add_argument("--gimbal-pitch", type=float, default=90.0)
    ap.add_argument("--every", type=float, default=1.0, help="seconds between saved frames")
    args = ap.parse_args(argv)
    manifest = json.loads((args.build / "manifest.json").read_text())
    damage = json.loads((args.build / f"damage_mission_{args.mission}.json").read_text())
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("audi_gz_recorder")
    bridge = CvBridge()
    st = {"fix": None, "quat": None, "en": (0.0, 0.0), "alt": None, "last": 0.0, "n": 0}
    out = (args.out / "frames.jsonl").open("w")
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(days=manifest["days_between_missions"] * args.mission)

    def on_pose(m):
        st["quat"] = (m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w)
        st["en"] = (m.pose.position.x, m.pose.position.y)
        st["alt"] = m.pose.position.z

    def on_image(m):
        now = node.get_clock().now().nanoseconds / 1e9
        if st["fix"] is None or st["alt"] is None or now - st["last"] < args.every:
            return
        st["last"] = now
        rgb = bridge.imgmsg_to_cv2(m, desired_encoding="rgb8")
        pose = camera_pose(st["fix"], st["quat"], st["alt"], args.gimbal_pitch)
        intr = CameraIntrinsics.from_hfov(rgb.shape[1], rgb.shape[0], args.hfov)
        gt = gt_boxes(manifest, damage, pose, intr, st["en"], (rgb.shape[1], rgb.shape[0]))
        if not gt:
            return
        rel = f"frames/{st['n']:05d}.jpg"
        Image.fromarray(rgb).save(args.out / rel, quality=92)
        out.write(json.dumps({"frame": rel, "ts": (t0 + timedelta(seconds=now % 86400)).isoformat(), "pose": pose, "gt": gt}) + "\n")
        out.flush()
        st["n"] += 1

    node.create_subscription(NavSatFix, "/mavros/global_position/global", lambda m: st.update(fix={"latitude": m.latitude, "longitude": m.longitude}), qos_profile_sensor_data)
    node.create_subscription(PoseStamped, "/mavros/local_position/pose", on_pose, qos_profile_sensor_data)
    node.create_subscription(RosImage, "/camera/image_raw", on_image, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
