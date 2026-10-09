# SRS traceability

Generated from `tests/test_srs_traceability.py` (which fails if a requirement loses its test or this file stops citing it).
Status "verified" = the listed tests pass in `pytest -q`; numeric NFR results are in `reports/bench.md`.

## Functional and non-functional requirements

| ID | Requirement | Module / file | Tests | Status |
|---|---|---|---|---|
| FR-1 | Check Cloud DB connectivity at mission start | memory/sync.py `SyncManager.check_connectivity`, ml/mission.py | `tests/test_sync_mission.py::test_offline_queue_then_flush_exactly_once`<br>`tests/test_sync_mission.py::test_online_end_sync_and_pull_on_second_device` | verified |
| FR-2 | If online, pull the site's latest memory | memory/sync.py `start_sync`, memory/cloud_store_interface.py `pull_site`, cloud/main.py | `tests/test_sync_mission.py::test_online_end_sync_and_pull_on_second_device` | verified |
| FR-3 | If offline, run from last-synced Local DB | memory/sync.py `_bounded`, ml/mission.py `start` | `tests/test_sync_mission.py::test_offline_queue_then_flush_exactly_once`<br>`tests/test_sync_mission.py::test_nfr3_slow_cloud_times_out_to_offline` | verified |
| FR-4 | Pre-flight briefing ranked by R + unresolved defects | ml/mission.py `briefing`, api/routers/sessions.py, frontend Briefing page | `tests/test_sync_mission.py::test_fr4_briefing_and_fr18_report`<br>`tests/test_api.py::test_session_flow_d6_d16_report` | verified |
| FR-5 | Update Local DB after every processed asset | memory/store.py `commit_observation` (one transaction per asset) | `tests/test_engine.py::test_restart_recovers_state_and_index`<br>`tests/test_engine.py::test_crash_mid_commit_rolls_back` | verified |
| FR-6 | Sync Local→Cloud at end of flight | memory/sync.py `end_sync`, outbox in memory/store.py | `tests/test_sync_mission.py::test_online_end_sync_and_pull_on_second_device` | verified |
| FR-7 | Queue offline updates, sync at next connection | memory/sync.py background flush, `sync_outbox` table | `tests/test_sync_mission.py::test_offline_queue_then_flush_exactly_once` | verified |
| FR-8 | Continuously capture live camera frames | ml/live/camera.py `CameraReader`/`PushFrameSource`, ml/live/devices.py | `tests/test_live_camera.py::test_camera_reconnects_after_unplug`<br>`tests/test_live_camera.py::test_camera_keeps_only_latest_frame`<br>`tests/test_live_camera.py::test_video_file_source_finishes`<br>`tests/test_api.py::test_live_browser_mode_over_websocket` | verified |
| FR-9 | Detect candidate assets in each frame | ml/detector.py `YoloWorldDetector`/`FullFrameDetector` | `tests/test_detection.py::test_yolo_world_postprocessing_maps_prompts_to_asset_types`<br>`tests/test_detection.py::test_every_detection_in_a_frame_becomes_an_asset` | verified |
| FR-10 | Match each detection against memory | ml/asset_identity.py cascade, memory/vector_memory.py | `tests/test_engine.py::test_new_then_matched_across_sessions`<br>`tests/test_engine.py::test_d2_damaged_asset_still_matches`<br>`tests/test_engine.py::test_aruco_marker_identity`<br>`tests/test_engine.py::test_d3_location_gating_separates_identical_panels` | verified |
| FR-11 | No match → register new asset | ml/semantic_memory.py `_process`/`_persist` | `tests/test_engine.py::test_new_then_matched_across_sessions`<br>`tests/test_engine.py::test_distinct_assets_not_merged` | verified |
| FR-12 | Match → retrieve history, prior images, defect notes | ml/semantic_memory.py `_history_summary`, `get_asset_history`, GET /api/assets/{id}/history | `tests/test_engine.py::test_d2_damaged_asset_still_matches`<br>`tests/test_api.py::test_session_flow_d6_d16_report` | verified |
| FR-13 | Decide significant change vs history | ml/change_detector.py | `tests/test_change_detection.py::test_identical_view_has_zero_change`<br>`tests/test_change_detection.py::test_synthetic_crack_gives_positive_visual_delta`<br>`tests/test_change_detection.py::test_crack_visual_delta_survives_homography_warp`<br>`tests/test_change_detection.py::test_defect_flip_makes_change_significant` | verified |
| FR-14 | Severity S per §8.1 | ml/severity.py | `tests/test_severity.py::test_formula_matches_srs`<br>`tests/test_severity.py::test_d1_severity_monotonic_in_change`<br>`tests/test_severity.py::test_levels` | verified |
| FR-15 | Priority flag + visual/audio alert without obstructing view | ml/live/pipeline.py `_alert_for`, ml/live/overlay.py, frontend LiveFlight page | `tests/test_live_camera.py::test_overlay_is_non_obstructive`<br>`tests/test_live_camera.py::test_nfr1_alert_latency_measured_from_capture` | verified |
| FR-16 | No change → confirmed, refresh last_checked_ts | ml/severity.py `gated_severity`, ml/semantic_memory.py | `tests/test_engine.py::test_d4_fr16_unchanged_asset_confirmed_with_zero_severity` | verified |
| FR-17 | Log defects with type, severity, trend | memory/store.py `_apply_defects`, `defects`/`defect_trend` tables, api/routers/defects.py | `tests/test_engine.py::test_fr17_defect_lifecycle_and_resolution_suggestion`<br>`tests/test_api.py::test_fr19_critical_notifies_only_subscribed_maintenance_over_ws` | verified |
| FR-18 | Post-flight report | ml/mission.py `report`, ml/reporting.py (JSON/HTML/PDF) | `tests/test_sync_mission.py::test_fr4_briefing_and_fr18_report`<br>`tests/test_api.py::test_session_flow_d6_d16_report` | verified |
| FR-19 | Critical → escalate to authorised maintenance | ml/escalation.py, api/ws.py, api/routers/notifications.py | `tests/test_sync_mission.py::test_fr19_escalation_only_to_subscribed_maintenance`<br>`tests/test_api.py::test_fr19_critical_notifies_only_subscribed_maintenance_over_ws` | verified |
| FR-20 | Recompute R for every site asset after each mission | memory/store.py `recompute_revisit`, ml/revisit_priority.py | `tests/test_engine.py::test_fr20_recompute_includes_unseen_assets` | verified |
| NFR-1 | Alert latency ≤ 500 ms p95 | ml/live/pipeline.py `LatencyStats`; measured by scripts/bench.py → reports/bench.md | `tests/test_live_camera.py::test_nfr1_alert_latency_measured_from_capture`<br>`tests/test_live_camera.py::test_pipeline_skips_duplicate_frames_and_frame_skip` | verified |
| NFR-2 | Matching ≤ 50 ms p95 @ 5,000 assets | memory/vector_memory.py, ml/asset_identity.py; measured by scripts/bench.py | `tests/test_vector_memory.py::test_nfr2_index_search_with_5000_assets` | verified |
| NFR-3 | Sync ≤ 10 s then proceed offline | memory/sync.py `_bounded` (cloud.sync_timeout_s); measured by scripts/bench.py | `tests/test_sync_mission.py::test_nfr3_slow_cloud_times_out_to_offline` | verified |

## Defects found in the original implementation

| ID | Defect | Fix | Tests | Status |
|---|---|---|---|---|
| D1 | Severity inverted, could not reach 0.5 | Cf = observation confidence (min of detector and classifier) | `tests/test_severity.py::test_d1_severity_monotonic_in_change`<br>`tests/test_severity.py::test_d1_severity_can_exceed_priority_threshold`<br>`tests/test_severity.py::test_d1_cf_is_min_of_detector_and_classifier` | fixed |
| D2 | Damaged asset re-registered as new | identity cascade with geometric verification, separate from condition | `tests/test_engine.py::test_d2_damaged_asset_still_matches` | fixed |
| D3 | Matching ignored location | ArUco + location gate before visual matching; location history averaged | `tests/test_engine.py::test_d3_location_gating_separates_identical_panels`<br>`tests/test_vector_memory.py::test_location_allowed_subset` | fixed |
| D4 | Non-zero severity floor for unchanged assets | M-gate: S = 0 when M < change_detection.threshold | `tests/test_severity.py::test_d4_gate_zeroes_insignificant_change`<br>`tests/test_engine.py::test_d4_fr16_unchanged_asset_confirmed_with_zero_severity` | fixed |
| D5 | Simulated history bypassed matching | build_semantic_memory.py runs the real pipeline; real re-ID set + evaluate_reid.py | `tests/test_reid_eval.py::test_reid_protocol_cascade_vs_global` | fixed |
| D6 | /api/analyze ignored add_to_memory | persist flag honoured end to end | `tests/test_engine.py::test_d6_no_persist_writes_nothing`<br>`tests/test_api.py::test_session_flow_d6_d16_report` | fixed |
| D7 | Observations had no session | every observation requires an active session | `tests/test_engine.py::test_d7_observation_requires_session` | fixed |
| D8 | Duplicate-frame check only logged | MAD frame skip, tracker (IoU + embedding + thumbnail NCC; a track whose region is taken over ends), one observation per asset per session | `tests/test_engine.py::test_d8_one_observation_per_asset_per_session`<br>`tests/test_engine.py::test_d8_tracker_processes_track_once`<br>`tests/test_engine.py::test_d8_tracker_does_not_merge_different_assets_in_same_box`<br>`tests/test_live_camera.py::test_pipeline_skips_duplicate_frames_and_frame_skip`<br>`tests/test_live_camera.py::test_tracked_asset_processed_once_per_track` | fixed |
| D9 | R not recomputed / D never grew | FR-20 recompute for all assets; D computed from now at briefing | `tests/test_revisit_priority.py::test_d9_time_decay_grows_with_now`<br>`tests/test_engine.py::test_fr20_recompute_includes_unseen_assets` | fixed |
| D10 | Gallery rebuilt per frame | bounded per-asset prototypes in a partitioned vector index | `tests/test_vector_memory.py::test_prototypes_bounded_and_novel`<br>`tests/test_vector_memory.py::test_partition_filters_by_site_and_type` | fixed |
| D11 | Non-atomic JSON store | SQLite WAL + transactions; migrate_json_store.py | `tests/test_engine.py::test_crash_mid_commit_rolls_back`<br>`tests/test_engine.py::test_restart_recovers_state_and_index` | fixed |
| D12 | Hard-coded embedding_dim | dimension derived from the loaded backbone | `tests/test_embedding.py::test_d12_dim_derived_from_model` | fixed |
| D13 | Defects were strings | Defect records with Open → In Progress → Resolved | `tests/test_engine.py::test_fr17_defect_lifecycle_and_resolution_suggestion` | fixed |
| D14 | No auth, CORS * | JWT + roles on every route; CORS from api.cors_origins | `tests/test_api.py::test_every_route_requires_auth`<br>`tests/test_api.py::test_role_guards`<br>`tests/test_api.py::test_cors_restricted_to_config_origins` | fixed |
| D15 | Trend used all history | slope over last revisit_priority.trend_window missions, clipped | `tests/test_revisit_priority.py::test_d15_trend_uses_only_last_window`<br>`tests/test_revisit_priority.py::test_d15_trend_clipped` | fixed |
| D16 | Wrong missions count | /api/summary counts sessions | `tests/test_api.py::test_session_flow_d6_d16_report` | fixed |

Hardware-dependent evidence (live Philips webcam, real re-ID set) is listed under "Run on your machine" in PROGRESS.md.
