"""Change detector: condition delta, patch-token visual delta, homography warp."""

import numpy as np

from ml.asset_identity import GeometricVerifier
from ml.change_detector import ConditionVisualChangeDetector, ReferenceCondition
from ml.condition_classifier import prediction_from_scores
from tests.conftest import add_crack, make_panel

LABELS = ["normal", "crack"]
NORMAL_REF = ReferenceCondition("normal", {"normal": 1.0, "crack": 0.0}, False, None)


def _pred(p_normal: float):
    return prediction_from_scores("solar_panel", LABELS, np.array([p_normal, 1 - p_normal]))


def test_identical_view_has_zero_change(config, embedder):
    det = ConditionVisualChangeDetector(config, embedder)
    img = make_panel(1)
    res = det.assess(_pred(1.0), NORMAL_REF, img, img, None)
    assert res.visual_delta < 1e-5
    assert res.magnitude < 1e-5
    assert not res.significant


def test_synthetic_crack_gives_positive_visual_delta(config, embedder):
    """Regression: the patch term was clipped to 0 by an absolute floor above the signal."""
    det = ConditionVisualChangeDetector(config, embedder)
    ref = make_panel(1)
    res = det.assess(_pred(1.0), NORMAL_REF, add_crack(ref), ref, None)
    assert res.visual_delta is not None and res.visual_delta > 0.0
    assert res.heatmap is not None and res.heatmap.max() > 0.0


def test_crack_visual_delta_survives_homography_warp(config, embedder):
    det = ConditionVisualChangeDetector(config, embedder)
    verifier = GeometricVerifier(config["matching"])
    ref = make_panel(3)
    moved = add_crack(ref.rotate(4, expand=False))
    inliers, ratio, H = verifier.verify(verifier.extract(moved), verifier.extract(ref))
    assert verifier.accepts(inliers, ratio)
    res = det.assess(_pred(1.0), NORMAL_REF, moved, ref, H)
    assert res.warped
    assert res.visual_delta > 0.0


def test_defect_flip_makes_change_significant(config, embedder):
    det = ConditionVisualChangeDetector(config, embedder)
    img = make_panel(2)
    res = det.assess(_pred(0.0), NORMAL_REF, img, None, None)
    assert res.condition_delta == 1.0
    assert res.magnitude == 1.0
    assert res.significant


def test_magnitude_bounded(config, embedder):
    det = ConditionVisualChangeDetector(config, embedder)
    for p in np.linspace(0, 1, 6):
        ref = make_panel(5)
        res = det.assess(_pred(float(p)), NORMAL_REF, add_crack(ref, 12), ref, None)
        assert 0.0 <= res.magnitude <= 1.0
