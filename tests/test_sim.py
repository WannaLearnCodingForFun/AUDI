"""4.3 simulation testbed: ground truth through the real engine, and the mission-folder format."""

import pytest

from sim.metrics import change_metrics, reid_metrics
from sim.render import fly_mission, read_mission, write_mission
from sim.runner import run, sim_config
from sim.world import build_world


def small_world(seed=5, missions=4):
    return build_world(seed=seed, rows=1, panels_per_row=4, poles=2, walls=1, machines=1, damaged_fraction=0.5, missions=missions)


def test_damage_grows_and_repair_resets():
    w = small_world()
    d = next(a for a in w.assets if a.damages).damages[0]
    ext = [d.extent(m) for m in range(6)]
    assert ext[d.start_mission] > 0 and all(b >= a for a, b in zip(ext, ext[1:]))
    d.repaired_at = 5
    assert d.extent(5) == 0.0


def test_mission_folder_roundtrip(tmp_path):
    w = small_world()
    frames = fly_mission(w, 0, seed=1)
    write_mission(frames, tmp_path / "m0")
    back = list(read_mission(tmp_path / "m0"))
    assert len(back) == len(frames) == len(w.assets)
    assert back[0].gt[0]["asset_id"] == frames[0].gt[0]["asset_id"] and back[0].pose == frames[0].pose


@pytest.mark.parametrize("mode", ["appearance", "location_appearance_sift"])
def test_simulated_missions_end_to_end(embedder, tmp_path, mode):
    w = small_world()
    res = run(w, 4, sim_config(tmp=tmp_path, mode=mode), embedder, seed=2)
    m = reid_metrics(res["log"])
    assert m["revisits"] == 3 * len(w.assets)
    if mode == "location_appearance_sift":
        assert m["rank1"] >= 0.8 and m["false_new_rate"] <= 0.15, m
        c = change_metrics(res["log"])
        assert c["pairs"] > 0 and c["mask"]["recall"] is not None
