"""SRS traceability: every FR / NFR (and every defect D1–D16) maps to tests that exist.

Fails if a requirement has no test, if a listed test does not exist, or if
docs/SRS_TRACEABILITY.md does not mention the requirement.
"""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

TRACE = {
    "FR-1": ["test_sync_mission::test_offline_queue_then_flush_exactly_once", "test_sync_mission::test_online_end_sync_and_pull_on_second_device"],
    "FR-2": ["test_sync_mission::test_online_end_sync_and_pull_on_second_device"],
    "FR-3": ["test_sync_mission::test_offline_queue_then_flush_exactly_once", "test_sync_mission::test_nfr3_slow_cloud_times_out_to_offline"],
    "FR-4": ["test_sync_mission::test_fr4_briefing_and_fr18_report", "test_api::test_session_flow_d6_d16_report"],
    "FR-5": ["test_engine::test_restart_recovers_state_and_index", "test_engine::test_crash_mid_commit_rolls_back"],
    "FR-6": ["test_sync_mission::test_online_end_sync_and_pull_on_second_device"],
    "FR-7": ["test_sync_mission::test_offline_queue_then_flush_exactly_once"],
    "FR-8": ["test_live_camera::test_camera_reconnects_after_unplug", "test_live_camera::test_camera_keeps_only_latest_frame",
             "test_live_camera::test_video_file_source_finishes", "test_api::test_live_browser_mode_over_websocket"],
    "FR-9": ["test_detection::test_yolo_world_postprocessing_maps_prompts_to_asset_types", "test_detection::test_every_detection_in_a_frame_becomes_an_asset"],
    "FR-10": ["test_engine::test_new_then_matched_across_sessions", "test_engine::test_d2_damaged_asset_still_matches",
              "test_engine::test_aruco_marker_identity", "test_engine::test_d3_location_gating_separates_identical_panels"],
    "FR-11": ["test_engine::test_new_then_matched_across_sessions", "test_engine::test_distinct_assets_not_merged"],
    "FR-12": ["test_engine::test_d2_damaged_asset_still_matches", "test_api::test_session_flow_d6_d16_report"],
    "FR-13": ["test_change_detection::test_identical_view_has_zero_change", "test_change_detection::test_synthetic_crack_gives_positive_visual_delta",
              "test_change_detection::test_crack_visual_delta_survives_homography_warp", "test_change_detection::test_defect_flip_makes_change_significant"],
    "FR-14": ["test_severity::test_formula_matches_srs", "test_severity::test_d1_severity_monotonic_in_change", "test_severity::test_levels"],
    "FR-15": ["test_live_camera::test_overlay_is_non_obstructive", "test_live_camera::test_nfr1_alert_latency_measured_from_capture"],
    "FR-16": ["test_engine::test_d4_fr16_unchanged_asset_confirmed_with_zero_severity"],
    "FR-17": ["test_engine::test_fr17_defect_lifecycle_and_resolution_suggestion", "test_api::test_fr19_critical_notifies_only_subscribed_maintenance_over_ws"],
    "FR-18": ["test_sync_mission::test_fr4_briefing_and_fr18_report", "test_api::test_session_flow_d6_d16_report"],
    "FR-19": ["test_sync_mission::test_fr19_escalation_only_to_subscribed_maintenance", "test_api::test_fr19_critical_notifies_only_subscribed_maintenance_over_ws"],
    "FR-20": ["test_engine::test_fr20_recompute_includes_unseen_assets"],
    "NFR-1": ["test_live_camera::test_nfr1_alert_latency_measured_from_capture", "test_live_camera::test_pipeline_skips_duplicate_frames_and_frame_skip"],
    "NFR-2": ["test_vector_memory::test_nfr2_index_search_with_5000_assets"],
    "NFR-3": ["test_sync_mission::test_nfr3_slow_cloud_times_out_to_offline"],
}

DEFECTS = {
    "D1": ["test_severity::test_d1_severity_monotonic_in_change", "test_severity::test_d1_severity_can_exceed_priority_threshold",
           "test_severity::test_d1_cf_is_min_of_detector_and_classifier"],
    "D2": ["test_engine::test_d2_damaged_asset_still_matches"],
    "D3": ["test_engine::test_d3_location_gating_separates_identical_panels", "test_vector_memory::test_location_allowed_subset"],
    "D4": ["test_severity::test_d4_gate_zeroes_insignificant_change", "test_engine::test_d4_fr16_unchanged_asset_confirmed_with_zero_severity"],
    "D5": ["test_reid_eval::test_reid_protocol_cascade_vs_global"],
    "D6": ["test_engine::test_d6_no_persist_writes_nothing", "test_api::test_session_flow_d6_d16_report"],
    "D7": ["test_engine::test_d7_observation_requires_session"],
    "D8": ["test_engine::test_d8_one_observation_per_asset_per_session", "test_engine::test_d8_tracker_processes_track_once",
           "test_engine::test_d8_tracker_does_not_merge_different_assets_in_same_box",
           "test_live_camera::test_pipeline_skips_duplicate_frames_and_frame_skip", "test_live_camera::test_tracked_asset_processed_once_per_track"],
    "D9": ["test_revisit_priority::test_d9_time_decay_grows_with_now", "test_engine::test_fr20_recompute_includes_unseen_assets"],
    "D10": ["test_vector_memory::test_prototypes_bounded_and_novel", "test_vector_memory::test_partition_filters_by_site_and_type"],
    "D11": ["test_engine::test_crash_mid_commit_rolls_back", "test_engine::test_restart_recovers_state_and_index"],
    "D12": ["test_embedding::test_d12_dim_derived_from_model"],
    "D13": ["test_engine::test_fr17_defect_lifecycle_and_resolution_suggestion"],
    "D14": ["test_api::test_every_route_requires_auth", "test_api::test_role_guards", "test_api::test_cors_restricted_to_config_origins"],
    "D15": ["test_revisit_priority::test_d15_trend_uses_only_last_window", "test_revisit_priority::test_d15_trend_clipped"],
    "D16": ["test_api::test_session_flow_d6_d16_report"],
}

REQUIRED = [f"FR-{i}" for i in range(1, 21)] + [f"NFR-{i}" for i in range(1, 4)]


def _exists(ref: str) -> bool:
    module, func = ref.split("::")
    return callable(getattr(importlib.import_module(f"tests.{module}"), func, None))


def test_every_requirement_has_a_test():
    missing = [r for r in REQUIRED if not TRACE.get(r)]
    assert not missing, f"requirements without tests: {missing}"
    assert sorted(DEFECTS, key=lambda d: int(d[1:])) == [f"D{i}" for i in range(1, 17)]


@pytest.mark.parametrize("req", REQUIRED + list(DEFECTS))
def test_listed_tests_exist(req):
    refs = TRACE.get(req) or DEFECTS[req]
    absent = [r for r in refs if not _exists(r)]
    assert not absent, f"{req}: tests not found {absent}"


def test_traceability_doc_covers_every_requirement():
    doc = (ROOT / "docs" / "SRS_TRACEABILITY.md").read_text(encoding="utf-8")
    absent = [r for r in REQUIRED + list(DEFECTS) if f"| {r} |" not in doc]
    assert not absent, f"docs/SRS_TRACEABILITY.md misses {absent}"
    for refs in list(TRACE.values()) + list(DEFECTS.values()):
        for ref in refs:
            assert ref.split("::")[1] in doc, f"docs/SRS_TRACEABILITY.md does not cite {ref}"
