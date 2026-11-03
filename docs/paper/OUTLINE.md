# Persistent Semantic Memory for Longitudinal UAV Infrastructure Inspection

Working outline (upgrade 7.3). Target: an IEEE robotics or computer-vision conference / workshop
(e.g. IROS / ICRA workshop on field robotics, or WACV / CVPR EarthVision-style workshops).

## Contributions
1. **Identity separated from condition, anchored in space.** Ground-projected detections gate the
   visual shortlist (1.1), a site orthomosaic gives footprints (1.2), a metric-learning identity head
   is trained to ignore damage (1.3); merge / split repairs keep history consistent (1.4).
2. **Defect-level change tracking and growth forecasting.** Aligned pixel change masks with metric
   area (2.1), per-defect instances matched across visits (2.2), deterministic per-family growth laws
   that predict the date a defect becomes critical and drive revisit priority (2.3).
3. **A longitudinal dataset** of the same assets over 3–5 visits with identities, defects and
   changes (7.1), plus a simulator with ground truth (4.3) for controlled ablations.

## Sections
1. Introduction — inspection is longitudinal; classifiers forget; what goes wrong without identity (8/12 returning assets registered as new in the original benchmark).
2. Related work — UAV inspection (PV, power lines, concrete), re-ID / place recognition, bitemporal change detection (BIT, ChangeFormer), deterioration models (Paris law), semantic maps.
3. System — the engine pipeline; data model (assets, observations, defect instances, measurements, relations); edge / cloud sync; ROS 2 integration.
4. Spatial identity — projection, gating radius from uncertainty, footprint registration; identity head training (synthetic changes that must not change identity).
5. Change and growth — mask computation (robust photometric fit, shift-tolerant difference), metric scale, instance matching, growth models, forecast-driven revisit priority.
6. Experiments
   * Simulated missions (`reports/sim/ablation.md`, 3 seeds × 6 missions)
   * Real multi-visit dataset (`scripts/run_sim_eval.py --missions-dir release/v1`)
   * Tables: re-ID arms; change detectors; forecast error in days; latency on Jetson (4.1)
7. Discussion / limitations — flat-ground projection; periodic textures; weak classifier classes; staged vs natural damage.
8. Conclusion.

## Current simulated results (weight-free projection embedder; lower bound for DINOv2)

| Re-ID arm | Rank-1 | False-new | False-match |
|---|---|---|---|
| appearance only | 0.750 | 0.070 | 0.180 |
| location + appearance | 0.920 | 0.017 | 0.063 |
| location + appearance + SIFT | 0.910 | 0.033 | 0.057 |

| Change detector | Precision | Recall | F1 | False / 100 unchanged |
|---|---|---|---|---|
| label difference | 0.938 | 0.185 | 0.412 | 0.4 |
| patch difference | 0.579 | 0.826 | 0.671 | 23.4 |
| pixel mask | 0.797 | 0.739 | 0.753 | 6.8 |

| Forecast | MAE to critical (days) |
|---|---|
| severity-slope extrapolation | 124.1 |
| per-defect growth model | 38.3 |

Regenerate with `make sim-eval`; numbers on real data and with DINOv2 replace these before submission.

## Open items before submission
- Collect and label the real dataset (7.1); run every table on it.
- Train the identity head (1.3) and Siamese change net (2.1) on real data; add them as arms.
- Jetson Orin Nano numbers (4.1): fps, p95, watts, accuracy loss FP16 / INT8.
- Inspector study for explainable alerts (5.3), feedback rounds (5.4), expert ranking for the health index (5.5).
