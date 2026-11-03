"""Launch test: replay a recorded rosbag2 through the nodes and compare the event stream with
scripts/smoke_e2e.py on the same frames (`audi_ros.core.event_signature`).

  ros2 bag record /camera/image_raw /mavros/global_position/global /mavros/global_position/rel_alt /mavros/local_position/pose
  launch_test src/audi_ros/launch/replay_test.launch.py bag:=<bag dir> expected:=reports/smoke/events.jsonl
"""

import json
import os
import unittest

import launch
import launch_testing
import pytest
import rclpy
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from std_msgs.msg import String

from audi_ros.core import event_signature


@pytest.mark.launch_test
def generate_test_description():
    bag = os.environ.get("AUDI_TEST_BAG", "test_bag")
    return launch.LaunchDescription([
        Node(package="audi_ros", executable="mission_node"),
        Node(package="audi_ros", executable="perception_node"),
        ExecuteProcess(cmd=["ros2", "service", "call", "/audi/start_session", "audi_msgs/srv/StartSession", "{site_id: site_demo}"]),
        ExecuteProcess(cmd=["ros2", "bag", "play", bag, "--rate", "0.5"]),
        launch_testing.actions.ReadyToTest(),
    ])


class TestReplay(unittest.TestCase):
    def test_events_match_smoke(self):
        expected = [json.loads(l) for l in open(os.environ.get("AUDI_EXPECTED_EVENTS", "reports/smoke/events.jsonl"))]
        rclpy.init()
        node = rclpy.create_node("replay_checker")
        got = []
        node.create_subscription(String, "audi/events_json", lambda m: got.append(json.loads(m.data)), 50)
        end = node.get_clock().now().nanoseconds + 120e9
        while len(got) < len(expected) and node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(node, timeout_sec=0.5)
        rclpy.shutdown()
        want = [event_signature(e) for e in expected if e.get("type") != "tracked"]
        have = [event_signature(e) for e in got if e.get("type") != "tracked"]
        self.assertEqual(have, want)
