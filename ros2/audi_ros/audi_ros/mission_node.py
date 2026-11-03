"""mission_node: audi/start_session and audi/end_session services (FR-1–4, FR-6/18/20)."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node

from audi_msgs.srv import EndSession, StartSession


class MissionNode(Node):
    def __init__(self):
        super().__init__("audi_mission")
        self.declare_parameter("config", "")
        self.declare_parameter("operator", "ros")
        from audi_ros.engine_host import EngineHost

        self.host = EngineHost.shared(self.get_parameter("config").value or None)
        self.create_service(StartSession, "audi/start_session", self.on_start)
        self.create_service(EndSession, "audi/end_session", self.on_end)

    def on_start(self, req, res):
        try:
            out = self.host.start(req.site_id, self.get_parameter("operator").value, req.simulated)
            res.ok, res.session_id, res.mode = True, out["session"]["session_id"], out["sync"]["mode"]
        except Exception as exc:  # noqa: BLE001
            res.ok, res.message = False, str(exc)
        return res

    def on_end(self, req, res):
        try:
            rep = self.host.end(req.session_id)
            res.ok = True
            res.assets_processed = int(rep["counts"]["assets_processed"])
            res.missing_assets = len((rep.get("coverage") or {}).get("missing", []))
            res.report_json = json.dumps(rep, default=str)
        except Exception as exc:  # noqa: BLE001
            res.ok, res.message = False, str(exc)
        return res


def main():
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
