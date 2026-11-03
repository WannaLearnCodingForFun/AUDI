# AUDI ROS 2 packages (upgrade 4.2)

* `audi_msgs` — `AssetEvent`, `Alert` messages; `StartSession`, `EndSession` services.
* `audi_ros` — `perception_node` (image + MAVROS NavSatFix / rel_alt / pose → engine → `audi/events`),
  `mission_node` (`audi/start_session`, `audi/end_session`), `alert_bridge` (`audi/alerts` + web app).

Rules: **read-only towards the autopilot** — the nodes only subscribe to MAVROS; nothing publishes
setpoints or calls arming / mode services. Camera topics use sensor-data QoS with depth 1.

```bash
# in a ROS 2 Humble/Jazzy workspace, with the AUDI repo on PYTHONPATH
ln -s /path/to/AUDI/ros2/audi_msgs /path/to/AUDI/ros2/audi_ros src/
colcon build --packages-select audi_msgs audi_ros && source install/setup.bash
ros2 launch audi_ros audi.launch.py image_topic:=/camera/image_raw
ros2 service call /audi/start_session audi_msgs/srv/StartSession "{site_id: site_demo}"
```

Replay test: record a rosbag2 of the camera + MAVROS topics, run `scripts/smoke_e2e.py` on the
same frames to produce `reports/smoke/events.jsonl`, then
`AUDI_TEST_BAG=<bag> launch_test src/audi_ros/launch/replay_test.launch.py`.
The pure-Python conversions (`audi_ros/core.py`) are unit-tested in the main suite without ROS
(`tests/test_ros_core.py`).
