"""Replay simulated (or Gazebo-recorded) missions through the real engine."""

from __future__ import annotations

import copy
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from memory.store import LocalStore
from ml.condition_classifier import PrototypeConditionClassifier
from ml.config import load_config
from ml.detector import Detection, Detector
from ml.embedding_model import AssetEmbeddingModel
from ml.semantic_memory import SemanticMemoryEngine
from sim.render import SimFrame, fly_mission
from sim.world import KINDS_BY_TYPE, LABEL, SimWorld, build_world

SITE = "site_sim"


class GroundTruthDetector(Detector):
    """Returns the simulator's boxes (slightly jittered): isolates memory from detector errors."""

    def __init__(self, jitter_px: int = 3, seed: int = 0):
        self.current: List[Dict[str, Any]] = []
        self.rng = random.Random(seed)
        self.jitter = jitter_px

    def detect(self, frame):
        out = []
        for g in self.current:
            j = lambda v: v + self.rng.randint(-self.jitter, self.jitter)
            x1, y1, x2, y2 = g["bbox"]
            out.append(Detection((max(0, j(x1)), max(0, j(y1)), min(frame.width, j(x2)), min(frame.height, j(y2))),
                                 g["asset_type"], 0.95, "sim"))
        return out


def sim_classifier(world: SimWorld, embedder: AssetEmbeddingModel, temperature: float = 0.02) -> PrototypeConditionClassifier:
    """Nearest-neighbour condition model fitted on simulator renders (normal vs each damage kind)."""
    from ml.damage_synth import DamageSpec, render_damage

    rng = random.Random(3)
    emb, labels = {}, {}
    for t, kinds in KINDS_BY_TYPE.items():
        examples = [a for a in world.assets if a.asset_type == t][:4] or []
        imgs, lbls = [], []
        for a in examples:
            imgs.append(a.texture)
            lbls.append("normal")
            for k in kinds:
                for ext in (0.35, 0.6):
                    im, _ = render_damage(a.texture, DamageSpec(k, rng.randrange(1 << 30), (0.5, 0.5)), ext)
                    imgs.append(im)
                    lbls.append(LABEL[k])
        if imgs:
            emb[t] = embedder.encode_batch(imgs)
            labels[t] = lbls
    return PrototypeConditionClassifier(emb, labels, temperature)


def sim_config(base: Optional[Dict[str, Any]] = None, tmp: Optional[Path] = None, mode: str = "location_appearance_sift") -> Dict[str, Any]:
    """Engine config for a simulation run. `mode` selects the re-ID ablation arm (7.2)."""
    cfg = copy.deepcopy(base or load_config())
    tmp = Path(tmp or tempfile.mkdtemp(prefix="audi_sim_"))
    cfg["model"]["name"] = cfg["model"].get("sim_backbone", "projection") if base is None else cfg["model"]["name"]
    cfg["storage"]["db_url"] = f"sqlite:///{tmp}/sim.db"
    cfg["storage"]["images_dir"] = str(tmp / "images")
    cfg["change_detection"]["heatmap_dir"] = str(tmp / "heat")
    cfg["detection"]["mode"] = "full_frame"
    cfg["cloud"]["enabled"] = False
    cfg["camera_model"]["calibration_file"] = None
    cfg["camera_model"]["hfov_deg"] = 78.0
    cfg["geo"]["identity_mode"] = "appearance" if mode == "appearance" else "location_appearance"
    cfg["matching"]["geometric_verification"] = mode != "location_appearance"
    cfg["quality_gate"]["enabled"] = True
    return cfg


def run(world: SimWorld, missions: int, config: Dict[str, Any], embedder=None, seed: int = 0, coverage: float = 1.0,
        frames_by_mission: Optional[List[List[SimFrame]]] = None) -> Dict[str, Any]:
    embedder = embedder or AssetEmbeddingModel.from_config(config)
    clf = sim_classifier(world, embedder)
    store = LocalStore(config)
    store.ensure_site(SITE, "Simulated site")
    det = GroundTruthDetector(seed=seed)
    engine = SemanticMemoryEngine(config, store, embedder, clf, det)
    log: List[Dict[str, Any]] = []
    for m in range(missions):
        sid = store.create_session(SITE, "sim", "offline", {}, simulated=True)["session_id"]
        frames = frames_by_mission[m] if frames_by_mission else fly_mission(world, m, seed, coverage if m else 1.0)
        for fr in frames:
            det.current = fr.gt
            for ev, g in zip(engine.process_frame(fr.image, sid, pose=fr.pose, timestamp=fr.ts), fr.gt):
                log.append({"mission": m, "session_id": sid, "gt": g, "event": ev.to_dict()})
        store.recompute_revisit(SITE)
        store.update_session(sid, status="ended", ended_ts=frames[-1].ts if frames else None)
    return {"log": log, "store": store, "engine": engine, "world": world}
