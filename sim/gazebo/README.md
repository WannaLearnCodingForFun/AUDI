# PX4 SITL + Gazebo testbed (upgrade 4.3)

The pure-Python simulator (`sim/`, `scripts/run_sim_eval.py`) is what CI runs. This folder
puts the **same site and damage schedule** into Gazebo for photo-realistic flights:

```bash
python sim/gazebo/generate_world.py --seed 7 --out sim/gazebo/build      # SDF + textures + manifest
python sim/gazebo/inject_damage.py --build sim/gazebo/build --mission 0  # mission-0 textures

# PX4 v1.15+ with gz-sim (Harmonic): fly a camera drone in the AUDI world
cp sim/gazebo/build/audi_site.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/
cp -r sim/gazebo/build/textures ~/PX4-Autopilot/Tools/simulation/gz/worlds/
cd ~/PX4-Autopilot && PX4_GZ_WORLD=audi_site make px4_sitl gz_x500_mono_cam

# MAVROS + camera bridge, then record the mission with ground truth
ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557
ros2 run ros_gz_bridge parameter_bridge /camera@sensor_msgs/msg/Image@gz.msgs.Image --ros-args -r /camera:=/camera/image_raw
python sim/gazebo/record_mission.py --build sim/gazebo/build --mission 0 --out runs/gz/m0
```

Fly the route from `POST /api/sites/{site}/route-plan?format=plan` (QGroundControl → Plan → Open)
or any survey pattern. Repeat `inject_damage.py --mission k` → restart → record for k = 1, 2, …
and evaluate:

```bash
python scripts/run_sim_eval.py --missions-dir runs/gz
```
