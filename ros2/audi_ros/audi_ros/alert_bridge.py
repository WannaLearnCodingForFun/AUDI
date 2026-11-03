"""alert_bridge: audi/events → audi/alerts (priority / critical) and → the web app.

Forwards to the AUDI API over HTTP (POST /api/live/telemetry is not involved; alerts are
already persisted by the engine, so the bridge publishes them to the web app's WebSocket hub
through the API's notification path and to ROS consumers on audi/alerts)."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from audi_msgs.msg import Alert, AssetEvent


class AlertBridge(Node):
    def __init__(self):
        super().__init__("audi_alert_bridge")
        self.declare_parameter("api_url", "http://127.0.0.1:8000")
        self.declare_parameter("api_token_env", "AUDI_ROS_API_TOKEN")
        self.pub = self.create_publisher(Alert, "audi/alerts", QoSProfile(depth=50))
        self.create_subscription(AssetEvent, "audi/events", self.on_event, QoSProfile(depth=50))

    def on_event(self, ev: AssetEvent):
        if not (ev.priority or ev.critical):
            return
        full = json.loads(ev.json or "{}")
        a = Alert()
        a.header = ev.header
        a.session_id, a.asset_id = ev.session_id, ev.asset_id
        a.level = "critical" if ev.critical else "priority"
        a.label, a.severity_score = ev.condition_label, ev.severity_score
        a.explanation = ((full.get("explanation") or {}).get("summary")) or ""
        a.guidance = ev.guidance
        self.pub.publish(a)
        self._forward(full)

    def _forward(self, event: dict):
        import os

        import httpx

        token = os.environ.get(self.get_parameter("api_token_env").value)
        if not token:
            return
        try:
            httpx.post(f"{self.get_parameter('api_url').value}/api/live/ros-alert", json=event,
                       headers={"Authorization": f"Bearer {token}"}, timeout=2.0)
        except httpx.HTTPError as exc:
            self.get_logger().warning(f"web app unreachable: {exc}")


def main():
    rclpy.init()
    node = AlertBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
