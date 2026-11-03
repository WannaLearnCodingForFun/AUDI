"""perception_node: camera image + MAVROS NavSatFix / pose → AUDI engine → audi_msgs/AssetEvent.

Read-only towards the autopilot: subscribes only, never publishes to /mavros/* setpoints or
calls arming / mode services. Camera topics use sensor-data QoS with a queue depth of 1 (the
newest frame wins; stale frames are dropped, never queued).

Parameters
  image_topic        /camera/image_raw
  fix_topic          /mavros/global_position/global
  rel_alt_topic      /mavros/global_position/rel_alt
  pose_topic         /mavros/local_position/pose
  range_topic        ""  (sensor_msgs/Range from a rangefinder / LiDAR, optional)
  gimbal_pitch_deg   90.0  (down from horizontal; overridden by gimbal_topic)
  gimbal_topic       /mavros/mount_control/orientation
  session_service    audi/start_session (provided by mission_node)
  config             path to config.yaml (default: repository config)
"""

from __future__ import annotations

import json
import threading

import rclpy
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import NavSatFix, Range
from std_msgs.msg import Float64, String

from audi_msgs.msg import AssetEvent
from audi_ros.core import camera_pose, event_fields, navsat_accuracy, quaternion_to_euler_enu


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("audi_perception")
        p = self.declare_parameter
        p("image_topic", "/camera/image_raw")
        p("fix_topic", "/mavros/global_position/global")
        p("rel_alt_topic", "/mavros/global_position/rel_alt")
        p("pose_topic", "/mavros/local_position/pose")
        p("range_topic", "")
        p("gimbal_topic", "/mavros/mount_control/orientation")
        p("gimbal_pitch_deg", 90.0)
        p("config", "")
        g = lambda k: self.get_parameter(k).value

        from audi_ros.engine_host import EngineHost

        self.host = EngineHost.shared(g("config") or None)
        self.fix = None
        self.acc = None
        self.rel_alt = None
        self.quat = None
        self.range = None
        self.gimbal_pitch = float(g("gimbal_pitch_deg"))
        self.busy = threading.Lock()
        sensor = qos_profile_sensor_data
        sensor.depth = 1
        self.create_subscription(RosImage, g("image_topic"), self.on_image, sensor)
        self.create_subscription(NavSatFix, g("fix_topic"), self.on_fix, sensor)
        self.create_subscription(Float64, g("rel_alt_topic"), lambda m: setattr(self, "rel_alt", m.data), sensor)
        self.create_subscription(PoseStamped, g("pose_topic"), self.on_pose, sensor)
        self.create_subscription(QuaternionStamped, g("gimbal_topic"), self.on_gimbal, sensor)
        if g("range_topic"):
            self.create_subscription(Range, g("range_topic"), lambda m: setattr(self, "range", m.range), sensor)
        self.pub = self.create_publisher(AssetEvent, "audi/events", QoSProfile(depth=50))
        self.pub_json = self.create_publisher(String, "audi/events_json", QoSProfile(depth=50))
        from cv_bridge import CvBridge

        self.bridge = CvBridge()
        self.get_logger().info("AUDI perception ready (read-only towards the autopilot)")

    def on_fix(self, m: NavSatFix):
        self.fix = {"latitude": m.latitude, "longitude": m.longitude}
        self.acc = navsat_accuracy(m.position_covariance, m.position_covariance_type)

    def on_pose(self, m: PoseStamped):
        q = m.pose.orientation
        self.quat = (q.x, q.y, q.z, q.w)

    def on_gimbal(self, m: QuaternionStamped):
        q = m.quaternion
        _, pitch, _ = quaternion_to_euler_enu(q.x, q.y, q.z, q.w)
        self.gimbal_pitch = max(0.0, min(90.0, -pitch * 57.29577951308232))

    def on_image(self, msg: RosImage):
        if self.host.session_id is None or not self.busy.acquire(blocking=False):
            return          # no active session, or still processing: drop (queue depth 1)
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pose = None
            if self.fix is not None and self.rel_alt is not None:
                pose = camera_pose(self.fix, self.quat, self.rel_alt, self.gimbal_pitch, gps_accuracy_m=self.acc, range_m=self.range)
            for ev in self.host.process(rgb, pose):
                f = event_fields(ev)
                out = AssetEvent()
                out.header = msg.header
                for k, v in f.items():
                    setattr(out, k, v)
                self.pub.publish(out)
                self.pub_json.publish(String(data=f["json"]))
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the node
            self.get_logger().error(f"frame failed: {exc}")
        finally:
            self.busy.release()


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
