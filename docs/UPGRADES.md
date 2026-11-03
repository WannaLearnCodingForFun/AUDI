# Upgrade implementation status

Maps every item of the *AUDI — Upgrade Implementation Plan* (Nov 3, 2026) to code, tests and
verification. **Status** says exactly what was checked:

* **verified** — the plan's verification ran here and passed (test or script output below);
* **verified (sim)** — verified on the simulator / synthetic data; the real-data run is still open;
* **built** — implemented and unit-tested, but the plan's verification needs hardware, trained
  weights, a dataset or people that this environment does not have.

Suite: `pytest -q` → **169 passed, 1 skipped** (the skip needs YOLO-World weights). Lint: `ruff check .` clean.
Simulator numbers: `reports/sim/ablation.md` (3 seeds × 6 missions, weight-free projection embedder).

## Phase 1 — Trustworthy memory

| # | Built | Verification | Status |
|---|---|---|---|
| 1.1 Spatial identity | `ml/geo_projection.py` (ray–ground intersection from GPS, height, heading, pitch, roll, intrinsics; uncertainty), gate before the shortlist in `ml/asset_identity.py` (radius = `geo.match_radius_m` widened to the fix uncertainty), appearance fallback without GPS | `tests/test_geo_identity.py` (projection maths, identical panels 15 m apart separated by the gate, merged by appearance only); ablation in `reports/sim/ablation.md` | **verified (sim)**: Rank-1 0.75 → 0.92, false-new 0.07 → 0.017 |
| 1.2 Site map | `ml/site_map.py` (GeoTIFF geotransform, SIFT + RANSAC frame → map registration with GPS window, footprint lookup, detector on tiles), `site_maps` table, `assets.footprint`, `scripts/build_site_map.py` (OpenDroneMap, register, detect, verify), API `/api/sites/{id}/map`, `PUT /api/assets/{id}/footprint` | `tests/test_geo_identity.py::test_frame_registers_to_map_and_hits_footprint`, `::test_engine_uses_site_map_footprint`; `build_site_map.py verify` computes "detections inside the correct polygon" | **built**: ≥95 % target needs a real orthomosaic + flight |
| 1.3 Re-ID head | `ml/reid_head.py` (256-d head, ArcFace / batch-hard triplet, Rank-k / mAP), `training/train_reid_head.py` (same asset across visits + synthetic lighting / viewpoint / cracks / rust / dust / snow), engine uses it when `reid_head.checkpoint` exists | `tests/test_sensing.py::test_reid_head_training_beats_or_matches_baseline_on_synthetic` | **built**: Rank-1 / mAP vs baseline needs DINOv2 + real re-ID set (`make reid-capture`, `make train-reid`) |
| 1.4 Merge / split | `memory/store_ext.py` (observations, prototypes, defects folded by type, instances, work orders, feedback, relations), `audit_log`, replayable `asset_op` records (edge + cloud), API + admin UI | `tests/test_merge_split.py` (nothing lost or duplicated, split ↔ merge round trip, replay on a second device), `tests/test_platform.py::test_merge_split_and_audit_over_api` | **verified** |

## Phase 2 — Core differentiators

| # | Built | Verification | Status |
|---|---|---|---|
| 2.1 Change detection | `ml/pixel_change.py` (homography warp, robust photometric fit, shift-tolerant Lab + gradient difference, change type, cm² via metric scale and homography area scale; feeds M), `ml/siamese_change.py` BIT-style network, `training/train_change_detector.py` (synthetic pairs, LEVIR-CD option) | `tests/test_defect_growth.py` (crack recall / precision, lighting is not change, rust / soiling types), `tests/test_sensing.py::test_siamese_change_net_learns_synthetic_changes`; sim: mask F1 0.75, AUC 0.93, 6.8 false changes / 100 unchanged | **verified (sim)** |
| 2.2 Defect tracking | `defect_instances` + `defect_measurements`, Hungarian matching on position (`ml/defect_tracking.py`), area / length history, miss counting → resolved | `tests/test_defect_growth.py::test_defect_matching_accuracy_and_size_error_on_growth_sequences` (≥ 95 % matching, ≤ 10 % length error), `::test_engine_tracks_one_growing_crack_and_forecasts` | **verified (sim)** |
| 2.3 Forecast | `ml/forecast.py` (crack power law, corrosion linear, soiling linear with resets; least squares, deterministic), predicted critical date drives T in revisit priority | `tests/test_defect_growth.py` (parameter recovery, critical date, resets, determinism, held-out error); sim: MAE to critical 38 days vs 124 for the severity slope | **verified (sim)** |
| 2.4 Guidance | `ml/guidance.py` (required px/cm per type, move-closer distance, tilt / turn), on events, overlay bottom band, live page | `tests/test_platform.py::test_guidance_message`, `::test_telemetry_pose_reaches_live_pipeline_and_guidance` | **built**: "share of revisits reaching the resolution" needs flights with and without guidance |
| 2.5 Coverage | `ml/coverage.py`, `/api/sessions/{id}/coverage`, `/api/sites/{id}/unseen`, report `coverage`, live "check before landing" | `tests/test_platform.py::test_coverage_twin_route_and_health` | **built**: "share seen per mission rises" needs missions over time |

## Phase 3 — Sensing and perception

| # | Built | Verification | Status |
|---|---|---|---|
| 3.1 Thermal + RGB | `ml/thermal.py` (ECC registration on edges, hotspots by ΔT classes, fused features), engine `thermal=` input, `training/train_thermal.py` (`marcosgabriel/infrared-solar-modules`) | `tests/test_sensing.py::test_thermal_hotspot_and_registration`, `::test_engine_reports_thermal_hotspot` | **built**: hotspot recall needs the IR dataset download + a thermal camera |
| 3.2 Fine-tuned detector | `training/train_detector.py` (YOLO11 / v8, mAP@0.5 per class, latency), `FineTunedDetector`, `detection.mode: hybrid` (YOLO-World only when the fine-tuned model finds nothing) | build_detector paths covered by the existing detection tests | **built**: needs `dataset/processed/detection/` + GPU |
| 3.3 Segmentation | `ml/segmentation.py` (classical crack / rust / soiling masks with hysteresis and structural-line removal; YOLO-seg and SAM2 backends), `training/train_defect_seg.py` (mask IoU, length error mm) | `tests/test_defect_growth.py::test_classical_crack_segmentation_length` | **verified (sim)** for the classical backend; learned backends **built** |
| 3.4 Quality gate | `ml/quality_gate.py` (blur, exposure, clipping, distance, incidence angle, size) on every frame and detection; counters in metrics | `tests/test_sensing.py::test_quality_gate_rejects_blur_and_exposure` | **built**: "fewer false changes" needs real footage |
| 3.5 Weak classes | focal loss (`condition_classifier.focal_gamma`), `training/copy_paste_augment.py` (solar physical_damage) | `tests/test_sensing.py::test_focal_loss_downweights_easy_examples`, `tests/test_defect_growth.py::test_copy_paste_augmentation_keeps_mask_consistent` | **built**: F1 0.47 → ≥ 0.80 needs a retrain on the datasets (`make copy-paste && make train && make eval`) |
| 3.6 Calibrated scale | `ml/calibration.py`, `scripts/calibrate_camera.py` (chessboard, tape-measure check), `ml/camera_model.py` (range / altitude / size scale) | `tests/test_sensing.py::test_calibration_recovers_focal_length_from_synthetic_views` (fx within 3 %), `tests/test_geo_identity.py::test_metric_scale_sources` | **built**: tape-measure comparison needs the real camera |

## Phase 4 — Edge and robotics

| # | Built | Verification | Status |
|---|---|---|---|
| 4.1 Edge device | `scripts/export_edge.py` (ONNX for backbone, heads, re-ID, change net, detector; TensorRT FP16 / INT8 via trtexec; onnxruntime fps / p95 / cosine vs PyTorch) | `tests/test_sensing.py::test_edge_export_onnx` (ONNX matches PyTorch to 1e-5) | **built**: Jetson fps / watts / INT8 loss need the device |
| 4.2 ROS 2 | `ros2/audi_msgs` (AssetEvent, Alert, StartSession, EndSession), `ros2/audi_ros` (perception_node, mission_node, alert_bridge; read-only to the autopilot, sensor QoS depth 1), rosbag2 replay launch test vs `smoke_e2e.py` events | `tests/test_ros_core.py` (MAVROS → pose, message fields) | **built**: launch test needs a ROS 2 workspace + recorded bag |
| 4.3 Simulation | `sim/` (site, growing damage, camera geometry, mission-folder format, GT detector, metrics), `scripts/run_sim_eval.py` (+ `--quick` CI gates), `sim/gazebo/` (SDF world, damage injection, PX4 SITL recorder) | `tests/test_sim.py`; CI runs `run_sim_eval.py --quick` | **verified** (Python simulator in CI); Gazebo path **built** |
| 4.4 3D twin | `ml/twin.py` (ENU scene, defects on asset faces, mesh reference), `/api/sites/{id}/twin`, three.js page; click a defect → its history | `tests/test_platform.py::test_coverage_twin_route_and_health`; browser check (page renders, no console errors) | **verified** with the box model; photogrammetry mesh loading **built** |
| 4.5 Revisit plan | `ml/route_plan.py` (nearest neighbour + 2-opt, standoff per type, ROI + waypoint items, QGC `.plan`), API + UI download | `tests/test_platform.py::test_coverage_twin_route_and_health`; `compare_with_manual()` | **built**: comparison with a manual plan needs a real mission |

## Phase 5 — Semantic layer

| # | Built | Verification | Status |
|---|---|---|---|
| 5.1 Knowledge graph | `asset_relations`, recursive-CTE ancestors / descendants, priority propagation into R | `tests/test_platform.py::test_graph_propagation_changes_ranking` | **verified** |
| 5.2 NL queries | `ml/nl_query.py` (rules translator; optional Claude Haiku 5.5 translator with schema-constrained output; deterministic executor), `/api/query`, UI | `tests/test_nl_query.py`: 30-question benchmark, exact match **30/30** (rules) | **verified** |
| 5.3 Explainable alerts | `ml/explain.py` on every event (before / after / mask, M / C / Cf terms, reasons), live alert cards | engine tests assert `explanation.terms.S == severity` | **built**: inspector survey pending |
| 5.4 Feedback loop | `alert_feedback`, API + UI buttons, `/api/feedback/stats` (false alarms per mission), `training/retrain_from_feedback.py` | `tests/test_platform.py::test_feedback_and_stats` | **built**: drop across rounds needs inspectors |
| 5.5 Health index | `ml/health_index.py` (0–100, site roll-up), on assets, map, owner report | `tests/test_defect_growth.py`, `tests/test_platform.py` | **built**: Spearman vs expert ranking pending |

## Phase 6 — Platform and operations

| # | Built | Verification | Status |
|---|---|---|---|
| 6.1 Cloud | images through the outbox to an S3-compatible / local object store, Postgres URL, `docker-compose.yml` (Postgres, MinIO, Caddy HTTPS, Prometheus, Grafana), `docs/DEPLOY.md` | `tests/test_platform.py::test_cloud_receives_images_and_merge` (offline → online syncs records **and** images) | **verified** in-process; compose stack validated by CI (`docker compose config`) |
| 6.2 Work orders | `work_orders` (Open → Assigned → Fixed → Verified, history), auto-created for critical defects, auto-Verified by the next normal observation | `tests/test_platform.py::test_work_order_full_cycle` | **verified** |
| 6.3 Multi-site, multi-drone | site onboarding (`POST /api/admin/sites`), drone registry with per-drone sites, `sessions.drone_id`; after every pull, assets two drones registered separately (same type, ≤ 3 m, prototype cosine ≥ 0.95) are merged through the audited merge path | `tests/test_platform.py::test_two_drones_produce_one_consistent_memory`, `::test_drone_keys_scopes_and_signed_frames`, `tests/test_merge_split.py::test_asset_ops_replay_on_second_device` | **verified** (two simulated drones → one memory) |
| 6.4 Owner reports | `ml/owner_report.py`, `scripts/owner_report.py` (one command), `/api/sites/{id}/owner-report?format=pdf` | `tests/test_platform.py::test_owner_report_from_real_data` | **verified** |
| 6.5 Security | HMAC-signed uploads, Fernet image encryption at rest, audit log, scoped revocable drone keys, rate limiting, OWASP API Top 10 review (`docs/DEPLOY.md`) | `tests/test_platform.py::test_drone_keys_scopes_and_signed_frames`, `::test_rate_limit_and_metrics`, `::test_images_encrypted_at_rest` | **verified** |
| 6.6 Observability | `/metrics` (Prometheus text format: HTTP, live fps / latency, outbox depth, quality gate), JSON logs, Grafana dashboard `deploy/grafana/dashboards/audi_live.json` | `tests/test_platform.py::test_rate_limit_and_metrics` | **built**: dashboard on a live mission needs the deployment |
| 6.7 CI/CD | `.github/workflows/ci.yml` (ruff, pytest, simulated-mission gates, frontend build, compose validation, cloud image build), badge in the README | runs on push | **built**: first green run happens on GitHub |

## Phase 7 — Evidence and publication

| # | Built | Status |
|---|---|---|
| 7.1 Real dataset | `docs/DATASET_PROTOCOL.md`, `docs/DATASHEET_TEMPLATE.md`, `scripts/multivisit_dataset.py` (package / prelabel / import), evaluator runs on recorded missions | **open**: the visits have to be flown |
| 7.2 Ablations | `scripts/run_sim_eval.py`: re-ID (3 arms), change (label / patch / mask / engine), forecast (slope / growth model) | **verified (sim)**: `reports/sim/ablation.md`; real-data tables after 7.1 |
| 7.3 Paper | `docs/paper/OUTLINE.md` (contributions, sections, current tables, open items) | **open** |

## Checklist (ticked = verified as above; "sim" = simulator / synthetic only)

**Top five**
- [x] 1.1 Spatial (geo-anchored) identity — sim
- [x] 2.1 Change detection, with 2.2 defect-level tracking — sim
- [x] 2.3 Deterioration forecast — sim
- [x] 4.3 Simulation testbed with ground truth
- [ ] 7.1 Real multi-visit dataset — tooling and protocol ready, collection open

**Next**
- [ ] 1.2 Site map — built; real orthomosaic run open
- [ ] 1.3 Re-ID model — built; training on DINOv2 + real set open
- [ ] 3.2 Fine-tuned detector — built; training open
- [x] 3.3 Defect segmentation — classical backend, sim
- [ ] 3.4 Image quality gate — built; real-footage effect open
- [ ] 3.5 Weak classes — built; retrain open
- [ ] 3.6 Calibrated scale — built; tape-measure check open
- [ ] 4.2 ROS 2 — built; rosbag launch test open
- [ ] 5.3 Explainable alerts — built; inspector survey open
- [x] 6.1 Cloud deployment — in-process sync incl. images; compose validated

**Then**
- [x] 1.4 Merge and split tools
- [ ] 2.4 In-flight guidance — built; flight comparison open
- [ ] 2.5 Coverage map — built; mission trend open
- [ ] 3.1 Thermal + RGB — built; IR data open
- [ ] 4.1 Edge device — built; Jetson numbers open
- [x] 4.4 3D digital twin — box model; mesh open
- [ ] 4.5 Revisit route plan — built; manual comparison open
- [x] 5.1 Knowledge graph
- [x] 5.2 Natural-language queries — 30/30 benchmark
- [ ] 5.4 Feedback loop — built; feedback rounds open
- [ ] 5.5 Health index — built; expert ranking open
- [x] 6.2 Work orders
- [x] 6.3 Multi-site, multi-drone
- [x] 6.4 Owner reports
- [x] 6.5 Security hardening
- [ ] 6.6 Observability — built; live dashboard open
- [ ] 6.7 CI/CD — built; first GitHub run open
- [x] 7.2 Ablation studies — sim
- [ ] 7.3 Paper — outline
