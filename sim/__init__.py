"""Simulation testbed with ground truth (upgrade 4.3).

`sim.world` builds a site (solar farm rows, a line of poles, a building wall, machinery) with
known asset positions and damage that grows between missions; `sim.render` flies missions and
produces frames + drone poses + per-frame ground truth in the mission-folder format that the
Gazebo/PX4 recorder (sim/gazebo/) also writes; `sim.runner` replays missions through the real
SemanticMemoryEngine; `sim.metrics` scores re-ID, change and growth forecasting.
"""
