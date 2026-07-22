#!/usr/bin/env python3
"""Golden tests for the n3mapping v2 evaluation artifact tools."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
FIXTURE_CONTRACT = REPO_ROOT / "test" / "fixtures" / "eval_v2" / "frozen_contract.yaml"
SYNTHETIC_FIXTURE_CONTRACT = (
    REPO_ROOT / "test" / "fixtures" / "eval_v2" / "synthetic_frozen_contract.yaml"
)
sys.path.insert(0, str(TOOLS_DIR))

from n3mapping_dataset_readiness import build_report, spatial_coverage  # noqa: E402
from n3mapping_episode_benchmark import (  # noqa: E402
    _finalize_benchmark_output,
    _verify_hashed_output,
    _verify_benchmark_output,
    _verify_manifest,
    classify_contract,
    classify_episode,
)
from n3mapping_episode_diagnose import diagnose_benchmark  # noqa: E402
from n3mapping_episode_freeze import freeze_episodes  # noqa: E402
from n3mapping_episode_review_prepare import (  # noqa: E402
    _matrix_from_xyzw,
    _xyzw_from_rotation,
)
from n3mapping_surface_overlap_audit import classify_surface_overlap  # noqa: E402
from n3mapping_runtime_signal_audit import summarize_ranges  # noqa: E402
from n3mapping_multiview_free_space_audit import (  # noqa: E402
    classify_map_ray_observations,
)
from n3mapping_eval_compare import compare_runs  # noqa: E402
from n3mapping_eval_validate import load_contract, sha256_file, validate_run  # noqa: E402
from n3mapping_synthetic_eval_gate import SyntheticGateError, run_synthetic_gate  # noqa: E402


def _write_json(path: Path, value: Any, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(run_dir: Path, required: list[str]) -> None:
    lines = []
    for relative in sorted(set(required) - {"checksums.sha256", "COMPLETE"}):
        lines.append(f"{sha256_file(run_dir / relative)}  {relative}")
    (run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run(
    parent: Path,
    name: str,
    *,
    manifest_overrides: dict[str, Any] | None = None,
    metrics_overrides: dict[str, Any] | None = None,
    query_rows: list[dict[str, Any]] | None = None,
    attempt_rows: list[dict[str, Any]] | None = None,
    runtime_event: dict[str, Any] | None = None,
    write_complete: bool = True,
    allow_nan_metrics: bool = False,
) -> Path:
    run_dir = parent / name
    run_dir.mkdir(parents=True)
    contract = load_contract(FIXTURE_CONTRACT)
    required = list(contract["required_artifacts"])
    effective_odom_source = (metrics_overrides or {}).get("odom_source", "oracle_gt_odom")
    effective_query_source = (metrics_overrides or {}).get(
        "query_source", "global_map_render"
    )
    effective_evidence_class = (metrics_overrides or {}).get(
        "evidence_class", "synthetic_map_render"
    )
    effective_gt_runtime_access = (manifest_overrides or {}).get("gt_runtime_access", True)
    shutil.copyfile(FIXTURE_CONTRACT, run_dir / "frozen_contract.yaml")
    _write_json(
        run_dir / "resolved_config.yaml",
        {
            "schema_version": 2,
            "mode": "relocalization",
            "evidence_class": effective_evidence_class,
            "odom_source": effective_odom_source,
            "query_source": effective_query_source,
            "gt_runtime_access": effective_gt_runtime_access,
            "verdict": "SHADOW_ONLY",
            "eval_profile": "product_default",
            "synthetic_frame_period_s": 1.0,
            "reloc_temporal_window_size": 5,
            "required_temporal_window_size": 5,
            "minimum_query_points": 100,
            "minimum_occupied_ray_bins": 100,
            "pose_translation_threshold_m": 1.0,
            "pose_yaw_threshold_deg": 10.0,
            "pose_roll_pitch_threshold_deg": 5.0,
        },
    )
    _write_json(
        run_dir / "dataset_manifest.json",
        {
            "schema_version": 2,
            "dataset_id": "fixture_route",
            "session": "fixture_session",
            "odom_source": effective_odom_source,
            "query_source": effective_query_source,
            "evidence_class": effective_evidence_class,
            "gt_runtime_access": effective_gt_runtime_access,
        },
    )
    command = "n3mapping_fixture_eval --mode relocalization"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(run_dir / "environment.json", {"cpu_model": "fixture-cpu", "thread_count": 1})
    (run_dir / "stdout.log").write_text("fixture completed\n", encoding="utf-8")

    if runtime_event is None:
        runtime_event = {
            "query_id": "q0",
            "runtime_stage": "SEARCHING",
            "runtime_reason": "candidate_search",
        }
    runtime_path = run_dir / "raw" / "runtime_events.jsonl"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime_event, sort_keys=True) + "\n", encoding="utf-8")

    if query_rows is None:
        query_rows = [
            {"query_id": "q0", "attempt_id": "a0", "query_outcome": "CORRECT_FULL_LOCK"},
            {"query_id": "q1", "attempt_id": "a1", "query_outcome": "TEMPORAL_DEFER"},
        ]
    query_rows = [dict(row) for row in query_rows]
    for row in query_rows:
        row.setdefault(
            "correct_lock_within_10s",
            row.get("query_outcome") == "CORRECT_FULL_LOCK",
        )
    _write_csv(
        run_dir / "raw" / "relocalization_queries.csv",
        ["query_id", "attempt_id", "query_outcome", "correct_lock_within_10s"],
        query_rows,
    )
    if attempt_rows is None:
        attempt_rows = [
            {"attempt_id": "a0", "attempt_outcome": "correct", "ever_false_lock": "false"},
            {"attempt_id": "a1", "attempt_outcome": "timeout", "ever_false_lock": "false"},
        ]
    _write_csv(
        run_dir / "raw" / "relocalization_attempts.csv",
        ["attempt_id", "attempt_outcome", "ever_false_lock"],
        attempt_rows,
    )
    _write_csv(
        run_dir / "raw" / "stage_timing.csv",
        ["query_id", "total_ms", "start_ns", "end_ns"],
        [
            {"query_id": "q0", "total_ms": 9.0, "start_ns": 100, "end_ns": 109},
            {"query_id": "q1", "total_ms": 10.0, "start_ns": 200, "end_ns": 210},
        ],
    )

    query_counts = {"CORRECT_FULL_LOCK": 1, "TEMPORAL_DEFER": 1}
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "mode": "relocalization",
        "odom_source": effective_odom_source,
        "query_source": effective_query_source,
        "evidence_class": effective_evidence_class,
        "gt_runtime_access": effective_gt_runtime_access,
        "attempt_count": 2,
        "correct_attempt_count": 1,
        "false_attempt_count": 0,
        "timeout_attempt_count": 1,
        "eligible_query_count": 2,
        "query_outcome_counts": query_counts,
        "correct_lock_event_count": 1,
        "false_lock_event_count": 0,
        "correct_lock_by_10s_count": 1,
        "correct_lock_by_10s": 0.5,
        "false_lock_attempt_rate": 0.0,
        "lock_precision": 1.0,
        "false_lock_given_lock": 0.0,
        "p95_total_ms": 10.0,
    }
    if metrics_overrides:
        metrics.update(metrics_overrides)
    _write_json(run_dir / "summary" / "metrics.json", metrics, allow_nan=allow_nan_metrics)
    _write_json(
        run_dir / "summary" / "failure_counts.json",
        {"schema_version": 2, "query_outcome_counts": metrics["query_outcome_counts"]},
    )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "run_id": name,
        "mode": "relocalization",
        "repo_sha": "0123456789abcdef0123456789abcdef01234567",
        "branch": "research/relocalization-evidence-v2",
        "dirty": False,
        "start_time_asia_shanghai": "2026-07-21T12:00:00+08:00",
        "end_time_asia_shanghai": "2026-07-21T12:00:01+08:00",
        "command": command,
        "exact_argv": ["n3mapping_fixture_eval", "--mode", "relocalization"],
        "exit_code": 0,
        "dataset_id": "fixture_route",
        "odom_id": "fixture_oracle",
        "gt_id": "fixture_gt",
        "odom_source": metrics["odom_source"],
        "query_source": metrics["query_source"],
        "gt_runtime_access": effective_gt_runtime_access,
        "evidence_class": metrics["evidence_class"],
        "candidate_budget": {
            "top_k": 5,
            "yaw_hypotheses": 4,
            "reloc_temporal_window_size": 5,
        },
        "icp_budget": {"max_iterations": 20},
        "cpu_model": "fixture-cpu",
        "thread_count": 1,
        "memory_max_bytes": 1073741824,
        "memory_swap_max_bytes": 0,
        "evaluator_version": "fixture-v2",
        "analyzer_version": None,
        "label_thresholds": {
            "pose_translation_threshold_m": 1.0,
            "pose_yaw_threshold_deg": 10.0,
            "pose_roll_pitch_threshold_deg": 5.0,
        },
        "feature_flags": (
            {
                "measurement_only": True,
                "odom_derived_from_gt": False,
                "synthetic_from_target_map": False,
                "gt_passed_to_localizer": False,
            }
            if effective_evidence_class == "recorded_cross_session"
            else {
                "measurement_only": True,
                "odom_derived_from_gt": True,
                "synthetic_from_target_map": True,
                "gt_passed_to_localizer": False,
            }
        ),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "dataset_manifest_sha256": sha256_file(run_dir / "dataset_manifest.json"),
        "frozen_contract_sha256": sha256_file(run_dir / "frozen_contract.yaml"),
        "required_artifacts": required,
        "complete": True,
        "verdict_ceiling": "SHADOW_ONLY",
        "synthetic_frame_period_s": 1.0,
        "required_temporal_window_size": 5,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    _write_json(run_dir / "manifest.json", manifest)
    _write_checksums(run_dir, required)
    if write_complete:
        (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return run_dir


def _error_codes(run_dir: Path) -> set[str]:
    return {item["code"] for item in validate_run(run_dir, strict=True).errors}


def _refresh_integrity(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_config_sha256"] = sha256_file(run_dir / "resolved_config.yaml")
    manifest["dataset_manifest_sha256"] = sha256_file(run_dir / "dataset_manifest.json")
    manifest["frozen_contract_sha256"] = sha256_file(run_dir / "frozen_contract.yaml")
    _write_json(manifest_path, manifest)
    _write_checksums(run_dir, list(manifest["required_artifacts"]))


def _write_episode_pose_manifest(path: Path, second_episode_frames: int = 5) -> None:
    rows: list[dict[str, Any]] = []
    query_id = 0
    for episode_id, frame_count in ((10, 5), (20, second_episode_frames)):
        for frame_index in range(frame_count):
            rows.append(
                {
                    "query_id": query_id,
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "x_m": query_id * 0.1,
                    "y_m": episode_id * 0.01,
                    "z_m": 0.5,
                    "roll_deg": 0.0,
                    "pitch_deg": 0.0,
                    "yaw_deg": frame_index * 2.0,
                }
            )
            query_id += 1
    _write_csv(
        path,
        [
            "query_id",
            "episode_id",
            "frame_index",
            "x_m",
            "y_m",
            "z_m",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ],
        rows,
    )


def _write_fake_synthetic_producer(path: Path) -> None:
    source = r'''#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

FINGERPRINT_ALGORITHM = "fnv1a64_xyz_intensity_float32_le_v1"

parser = argparse.ArgumentParser()
parser.add_argument("--map", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--query_pose_manifest", required=True)
parser.add_argument("--eval_profile", default="product_default")
parser.add_argument("--fingerprint-salt", type=int, default=0)
parser.add_argument("--fake-temporal-window", type=int, default=5)
parser.add_argument("--omit-fingerprint", action="store_true")
parser.add_argument("--mismatch-resolved-fingerprint", action="store_true")
parser.add_argument("--raycast-off", action="store_true")
parser.add_argument("--break-state-chain", action="store_true")
parser.add_argument("--false-reentry", action="store_true")
parser.add_argument("--delayed-lock", action="store_true")
parser.add_argument("--invalid-hide-lock", action="store_true")
parser.add_argument("--valid-counts", type=int, default=100)
args = parser.parse_args()

output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
with Path(args.query_pose_manifest).open("r", encoding="utf-8", newline="") as stream:
    pose_rows = list(csv.DictReader(stream))

per_query_fields = [
    "query_id", "episode_id", "frame_index", "reference_keyframe_id", "renderer_seed",
    "generation_valid", "generation_reason", "localization_attempted", "tracking_processed",
    "locked_before_frame", "locked_after_frame", "ever_locked_before_frame",
    "lock_event_type", "tracking_loss_event", "runtime_stage", "success",
    "relocalization_locked", "matched_keyframe_id", "pose_accurate",
    "translation_error_m", "yaw_error_deg", "roll_pitch_error_deg",
    "renderer_elapsed_ms", "localizer_elapsed_ms", "scorer_elapsed_ms",
    "query_x_m", "query_y_m", "query_z_m", "query_roll_deg", "query_pitch_deg",
    "query_yaw_deg", "input_map_points", "finite_points", "range_eligible_points",
    "fov_eligible_points", "azimuth_bins", "vertical_bins", "occupied_ray_bins",
    "same_ray_occluded_points", "dropout_suppressed_points", "occlusion_suppressed_bins",
    "query_points_before_voxel", "num_query_points", "raycast_enabled", "envelope_valid",
    "envelope_azimuth_full", "envelope_azimuth_start_deg", "envelope_azimuth_span_deg",
    "envelope_vertical_min_deg", "envelope_vertical_max_deg",
    "query_cloud_fingerprint_available", "query_cloud_fingerprint_algorithm",
    "query_cloud_fingerprint_fnv1a64",
]
if args.omit_fingerprint:
    per_query_fields.remove("query_cloud_fingerprint_fnv1a64")

first_episode = pose_rows[0]["episode_id"]
per_query_rows = []
resolved_rows = []
for pose in pose_rows:
    query_id = int(pose["query_id"])
    episode_id = pose["episode_id"]
    frame_index = int(pose["frame_index"])
    is_first_episode = episode_id == first_episode
    state = "initial_localization"
    attempted = True
    tracking = False
    locked_before = False
    locked_after = False
    ever_before = False
    lock_type = "none"
    loss = False
    success = False
    if is_first_episode and frame_index == 0:
        state = "initial_lock"
        locked_after = True
        lock_type = "initial_lock"
        success = True
    elif is_first_episode and frame_index == 1:
        state = "tracking_loss"
        attempted = False
        tracking = True
        locked_before = True
        ever_before = True
        loss = True
    elif is_first_episode and frame_index == 2:
        state = "reentry_lock"
        ever_before = True
        locked_after = True
        lock_type = "reentry_lock"
        success = True
        if args.break_state_chain:
            locked_before = True
            tracking = True
    elif is_first_episode and frame_index == 3:
        state = "tracking"
        attempted = False
        tracking = True
        locked_before = True
        locked_after = True
        ever_before = True
        success = True
    elif is_first_episode and frame_index == 4:
        state = "reentry_lock"
        attempted = True
        tracking = True
        locked_before = True
        locked_after = True
        ever_before = True
        lock_type = "reentry_lock"
        success = True
    elif not is_first_episode and args.delayed_lock and frame_index == 10:
        state = "initial_lock"
        locked_after = True
        lock_type = "initial_lock"
        success = True

    lock_event = lock_type != "none"
    false_lock = bool(args.false_reentry and is_first_episode and frame_index == 2)
    pose_accurate = lock_event and not false_lock
    translation_error = 2.0 if false_lock else (0.2 if lock_event else "nan")
    yaw_error = 1.0 if lock_event else "nan"
    roll_pitch_error = 0.5 if lock_event else "nan"
    fingerprint = f"{((query_id + args.fingerprint_salt + 1) * 1099511628211) & ((1 << 64) - 1):016x}"
    raycast = not args.raycast_off
    if raycast:
        visible_count = args.valid_counts
        counts = {
            "input_map_points": visible_count * 3,
            "finite_points": visible_count * 3,
            "range_eligible_points": visible_count * 2 + 50,
            "fov_eligible_points": visible_count * 2,
            "azimuth_bins": 20,
            "vertical_bins": 20,
            "occupied_ray_bins": visible_count,
            "same_ray_occluded_points": visible_count,
            "dropout_suppressed_points": 0,
            "occlusion_suppressed_bins": 0,
            "query_points_before_voxel": visible_count,
            "num_query_points": visible_count,
        }
    else:
        counts = {
            "input_map_points": 100,
            "finite_points": 100,
            "range_eligible_points": 90,
            "fov_eligible_points": 80,
            "azimuth_bins": 0,
            "vertical_bins": 0,
            "occupied_ray_bins": 0,
            "same_ray_occluded_points": 0,
            "dropout_suppressed_points": 2,
            "occlusion_suppressed_bins": 0,
            "query_points_before_voxel": 78,
            "num_query_points": 70,
        }
    generation_valid = not (args.invalid_hide_lock and is_first_episode and frame_index == 2)
    row = {
        "query_id": query_id,
        "episode_id": episode_id,
        "frame_index": frame_index,
        "reference_keyframe_id": 100 + query_id,
        "renderer_seed": 1000 + query_id,
        "generation_valid": generation_valid,
        "generation_reason": "ok" if generation_valid else "hidden_event",
        "localization_attempted": attempted,
        "tracking_processed": tracking,
        "locked_before_frame": locked_before,
        "locked_after_frame": locked_after,
        "ever_locked_before_frame": ever_before,
        "lock_event_type": lock_type,
        "tracking_loss_event": loss,
        "runtime_stage": state,
        "success": success,
        "relocalization_locked": lock_event,
        "matched_keyframe_id": (100 + query_id) if lock_event else -1,
        "pose_accurate": pose_accurate,
        "translation_error_m": translation_error,
        "yaw_error_deg": yaw_error,
        "roll_pitch_error_deg": roll_pitch_error,
        "renderer_elapsed_ms": 1.0,
        "localizer_elapsed_ms": 2.0,
        "scorer_elapsed_ms": 0.5 if lock_event else 0.0,
        "query_x_m": pose["x_m"],
        "query_y_m": pose["y_m"],
        "query_z_m": pose["z_m"],
        "query_roll_deg": pose["roll_deg"],
        "query_pitch_deg": pose["pitch_deg"],
        "query_yaw_deg": pose["yaw_deg"],
        **counts,
        "raycast_enabled": raycast,
        "envelope_valid": True,
        "envelope_azimuth_full": True,
        "envelope_azimuth_start_deg": 0.0,
        "envelope_azimuth_span_deg": 360.0,
        "envelope_vertical_min_deg": -30.0,
        "envelope_vertical_max_deg": 30.0,
        "query_cloud_fingerprint_available": True,
        "query_cloud_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "query_cloud_fingerprint_fnv1a64": fingerprint,
    }
    per_query_rows.append(row)
    resolved_fingerprint = fingerprint
    if args.mismatch_resolved_fingerprint and query_id == 0:
        resolved_fingerprint = "ffffffffffffffff"
    resolved_rows.append({
        "query_id": query_id,
        "episode_id": episode_id,
        "frame_index": frame_index,
        "reference_keyframe_id": 100 + query_id,
        "renderer_seed": 1000 + query_id,
        "x_m": pose["x_m"],
        "y_m": pose["y_m"],
        "z_m": pose["z_m"],
        "roll_deg": pose["roll_deg"],
        "pitch_deg": pose["pitch_deg"],
        "yaw_deg": pose["yaw_deg"],
        "generation_valid": generation_valid,
        "generation_reason": "ok" if generation_valid else "hidden_event",
        "query_cloud_fingerprint_available": True,
        "query_cloud_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "query_cloud_fingerprint_fnv1a64": resolved_fingerprint,
    })

with (output / "per_query.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=per_query_fields)
    writer.writeheader()
    writer.writerows([{field: row[field] for field in per_query_fields} for row in per_query_rows])
resolved_fields = list(resolved_rows[0])
if args.omit_fingerprint:
    resolved_fields.remove("query_cloud_fingerprint_fnv1a64")
with (output / "resolved_queries.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=resolved_fields)
    writer.writeheader()
    writer.writerows([{field: row[field] for field in resolved_fields} for row in resolved_rows])

config = {
    "schema_version": 2,
    "evidence_class": "synthetic_map_render",
    "odom_source": "oracle_gt_odom",
    "gt_runtime_access": True,
    "verdict": "SHADOW_ONLY",
    "eval_profile": args.eval_profile,
    "query_source": "global_map_render",
    "visibility_model": "map_conditioned_point_zbuffer_v1",
    "normal_visibility": "not_modeled",
    "sensor_profile": "reference_envelope_only",
    "scan_pattern_model": "not_modeled",
    "motion_distortion_model": "not_modeled",
    "query_cloud_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
    "raycast_azimuth_resolution_deg": 0.0 if args.raycast_off else 1.0,
    "raycast_vertical_resolution_deg": 0.0 if args.raycast_off else 1.0,
    "rhpd_num_candidates": 5,
    "rhpd_preselect_candidates": 20,
    "reloc_num_candidates": 5,
    "reloc_temporal_window_size": args.fake_temporal_window,
    "minimum_query_points": 100,
    "minimum_occupied_ray_bins": 100,
    "gicp_max_iterations": 20,
    "gicp_max_correspondence_distance": 1.0,
    "gicp_fitness_threshold": 0.5,
    "gicp_submap_size": 3,
    "pose_translation_threshold_m": 1.0,
    "pose_yaw_threshold_deg": 10.0,
    "pose_roll_pitch_threshold_deg": 5.0,
}
for filename, value in (
    ("renderer_config.json", config),
    ("config_used.json", config),
    ("metrics.json", {"schema_version": 2, "tested": len(per_query_rows)}),
    ("summary.json", {"schema_version": 2, "tested": len(per_query_rows)}),
):
    (output / filename).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
(output / "failed_queries.txt").write_text("", encoding="utf-8")
(output / "EVAL_COMPLETE").write_text("status=complete\n", encoding="utf-8")
print("fake synthetic producer complete")
'''
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    path.chmod(0o755)


def _make_synthetic_gate_inputs(
    root: Path,
    second_episode_frames: int = 5,
) -> tuple[Path, Path, Path]:
    producer = root / "fake_synthetic_producer.py"
    map_path = root / "target.n3map"
    pose_manifest = root / "query_pose_manifest.csv"
    _write_fake_synthetic_producer(producer)
    map_path.write_bytes(b"synthetic-map-fixture\n")
    _write_episode_pose_manifest(pose_manifest, second_episode_frames=second_episode_frames)
    return producer, map_path, pose_manifest


def _run_fake_gate(
    root: Path,
    name: str,
    *,
    producer_args: list[str] | None = None,
    frame_period_s: float = 1.0,
    second_episode_frames: int = 5,
) -> Path:
    producer, map_path, pose_manifest = _make_synthetic_gate_inputs(
        root,
        second_episode_frames=second_episode_frames,
    )
    output = root / name
    run_synthetic_gate(
        producer=producer,
        output=output,
        map_path=map_path,
        query_pose_manifest=pose_manifest,
        frozen_contract=SYNTHETIC_FIXTURE_CONTRACT,
        dataset_id="synthetic_fixture_route",
        map_provenance="unit_test_fixture",
        repo=REPO_ROOT,
        producer_args=list(producer_args or []),
        split="p0s_shadow",
        frame_period_s=frame_period_s,
        build_type="test",
    )
    return output


class DatasetReadinessTest(unittest.TestCase):
    def test_multiview_free_space_distinguishes_support_free_and_unknown(self) -> None:
        map_points = np.array(
            [
                [10.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ]
        )
        map_ranges = np.linalg.norm(map_points, axis=1)
        map_directions = map_points / map_ranges[:, None]
        query_world = np.array(
            [
                [10.0, 0.0, 0.0],  # same endpoint: supported
                [5.0, 0.0, 0.0],   # map ray passed through: explicit free conflict
                [15.0, 0.0, 0.0],  # behind map return: occluded/unknown
                [5.0, 2.0, 0.0],   # no matching observed ray: unknown
            ]
        )

        support, free_conflict = classify_map_ray_observations(
            query_world,
            np.eye(4),
            cKDTree(map_directions),
            map_ranges,
            0.5,
        )

        np.testing.assert_array_equal(support, [True, False, False, False])
        np.testing.assert_array_equal(free_conflict, [False, True, False, False])

    def test_runtime_signal_range_summary_does_not_invent_threshold(self) -> None:
        rows = [
            {"analysis_group": "positive_correct_lock", "signal": 0.2},
            {"analysis_group": "positive_correct_lock", "signal": 0.6},
            {"analysis_group": "hard_negative_false_lock", "signal": 0.5},
            {"analysis_group": "hard_negative_false_lock", "signal": 0.9},
        ]
        overlap = summarize_ranges(rows, "signal")
        self.assertFalse(overlap["range_disjoint"])
        rows[2]["signal"] = 0.7
        disjoint = summarize_ranges(rows, "signal")
        self.assertTrue(disjoint["range_disjoint"])
        self.assertNotIn("threshold", disjoint)

    def test_review_pose_quaternion_round_trip(self) -> None:
        quaternion = (0.2, -0.3, 0.4, 0.8)
        pose = _matrix_from_xyzw((1.0, 2.0, 3.0), quaternion)
        recovered = _xyzw_from_rotation(pose[:3, :3])
        expected_norm = sum(value * value for value in quaternion) ** 0.5
        expected = tuple(value / expected_norm for value in quaternion)
        self.assertAlmostEqual(abs(sum(a * b for a, b in zip(recovered, expected))), 1.0)
        self.assertEqual(tuple(pose[:3, 3]), (1.0, 2.0, 3.0))

    def test_episode_outcome_is_attempt_level_and_false_lock_dominates(self) -> None:
        self.assertEqual(classify_episode({"correct_lock_count": 1}), "correct_lock")
        self.assertEqual(classify_episode({"correct_lock_count": 0}), "no_lock")
        self.assertEqual(
            classify_episode({"correct_lock_count": 1, "false_lock_count": 1}),
            "false_lock",
        )
        self.assertEqual(classify_contract("correct_lock", "lock"), "pass")
        self.assertEqual(classify_contract("no_lock", "lock"), "fail")
        self.assertEqual(classify_contract("no_lock", "abstain"), "pass")
        self.assertEqual(classify_contract("correct_lock", "abstain"), "fail")

    def test_oracle_candidate_diagnosis_separates_main_and_motion_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "manifest"
            benchmark_dir = root / "benchmark"
            manifest_dir.mkdir()
            (benchmark_dir / "map").mkdir(parents=True)
            episode_dir = benchmark_dir / "episodes" / "query_000"
            episode_dir.mkdir(parents=True)
            _write_json(
                manifest_dir / "dataset_manifest.json",
                {
                    "dataset": "kitti360",
                    "evidence_class": "oracle_gt_same_session",
                    "overlap_radius_m": 1.0,
                },
            )
            _write_csv(
                manifest_dir / "episode_frames.csv",
                ["episode_id", "role", "frame_token", "x", "y", "z"],
                [
                    {"episode_id": "map", "role": "map", "frame_token": "10", "x": 0, "y": 0, "z": 0},
                    {"episode_id": "map", "role": "map", "frame_token": "20", "x": 10, "y": 0, "z": 0},
                    {"episode_id": "query_000", "role": "query", "frame_token": "30", "x": 0.5, "y": 0, "z": 0},
                    {"episode_id": "query_000", "role": "query", "frame_token": "31", "x": 9.5, "y": 0, "z": 0},
                ],
            )
            _write_csv(
                benchmark_dir / "map" / "keyframes_gt.csv",
                ["keyframe_id", "frame_id", "x", "y", "z"],
                [
                    {"keyframe_id": 0, "frame_id": 10, "x": 0, "y": 0, "z": 0},
                    {"keyframe_id": 1, "frame_id": 20, "x": 10, "y": 0, "z": 0},
                ],
            )
            events = [
                {
                    "record_type": "relocalize",
                    "top_candidates": [{"match_id": 0}],
                    "motion_query_top_candidates": [{"match_id": 1}],
                    "reject_reason": "temporal_window_pending",
                },
                {
                    "record_type": "relocalize",
                    "top_candidates": [{"match_id": 0}],
                    "motion_query_top_candidates": [{"match_id": 1}],
                    "reject_reason": "no_valid_icp_hypothesis",
                },
            ]
            (episode_dir / "relocalization_debug.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            _write_json(
                benchmark_dir / "summary.json",
                {"schema_version": 1, "attempt_count": 1},
            )
            _write_csv(
                benchmark_dir / "episodes.csv",
                ["episode_id", "outcome"],
                [{"episode_id": "query_000", "outcome": "no_lock"}],
            )
            _finalize_benchmark_output(benchmark_dir)

            diagnostic_dir = root / "diagnostics"
            report = diagnose_benchmark(
                manifest_dir, benchmark_dir, diagnostic_dir
            )
            self.assertEqual(report["oracle_overlap_frame_count"], 2)
            self.assertEqual(report["main_top1_recall_rate"], 0.5)
            self.assertEqual(report["main_topk_recall_rate"], 0.5)
            self.assertEqual(report["motion_topk_recall_rate"], 0.5)
            self.assertEqual(
                report["decision_counts"],
                {"no_valid_icp_hypothesis": 1, "temporal_window_pending": 1},
            )
            self.assertTrue((diagnostic_dir / "oracle_candidate_frames.csv").is_file())
            _verify_hashed_output(
                diagnostic_dir,
                {
                    "oracle_candidate_diagnostics.json",
                    "oracle_candidate_frames.csv",
                },
            )
            _verify_benchmark_output(benchmark_dir)

    def test_benchmark_finalization_detects_incomplete_and_mutated_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _write_json(output / "summary.json", {"attempt_count": 1})
            _write_csv(
                output / "episodes.csv",
                ["episode_id", "outcome"],
                [{"episode_id": "query_000", "outcome": "correct_lock"}],
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _verify_benchmark_output(output)
            _finalize_benchmark_output(output)
            _verify_benchmark_output(output)
            self.assertTrue((output / "COMPLETE").is_file())
            (output / "summary.json").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                _verify_benchmark_output(output)

    def test_spatial_coverage_counts_only_queries_inside_radius(self) -> None:
        report = spatial_coverage(
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
            [(0.5, 0.0, 0.0), (20.0, 0.0, 0.0)],
            1.0,
        )
        self.assertEqual(report["covered_query_count"], 1)
        self.assertEqual(report["covered_query_rate"], 0.5)

    def test_build_report_discovers_kitti_and_m2dgr_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kitti = root / "KITTI360"
            for sequence, offset in (("drive_0000_sync", 0.0), ("drive_0001_sync", 0.2)):
                lidar_dir = kitti / "data_3d_raw" / sequence / "velodyne_points" / "data"
                lidar_dir.mkdir(parents=True)
                pose_dir = kitti / "data_poses" / sequence
                pose_dir.mkdir(parents=True)
                pose_lines = []
                for frame_id, x in enumerate((0.0, 10.0, 0.5, 50.0)):
                    (lidar_dir / f"{frame_id:010d}.bin").write_bytes(b"")
                    pose_lines.append(
                        f"{frame_id} 1 0 0 {x + offset} 0 1 0 0 0 0 1 0"
                    )
                (pose_dir / "poses.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
            calibration = kitti / "calibration"
            calibration.mkdir()
            (calibration / "calib_cam_to_velo.txt").write_text("1 0 0 0\n", encoding="utf-8")

            m2dgr = root / "M2DGR"
            lidar_dir = m2dgr / "gate_02" / "velodyne_points"
            lidar_dir.mkdir(parents=True)
            gt_lines = []
            for timestamp, x in ((1.0, 0.0), (2.0, 10.0), (3.0, 0.5), (4.0, 50.0)):
                (lidar_dir / f"{timestamp:.3f}.bin").write_bytes(b"")
                gt_lines.append(f"{timestamp:.3f} {x} 0 0 0 0 0 1")
            (m2dgr / "gate_02" / "gate_02.txt").write_text(
                "\n".join(gt_lines) + "\n", encoding="utf-8"
            )

            report = build_report(kitti, m2dgr, 1.0, 0.05, 1, 0.2)
            self.assertFalse(report["formal_gate_ready"])
            self.assertEqual(len(report["kitti360"]["sequences"]), 2)
            self.assertEqual(
                report["kitti360"]["sequences"][0]["same_session_half_split"][
                    "covered_query_count"
                ],
                1,
            )
            self.assertTrue(
                report["kitti360"]["cross_sequence_overlap"][0][
                    "positive_pair_candidate"
                ]
            )
            gate = report["m2dgr"]["sequences"][0]
            self.assertEqual(gate["aligned_frame_count"], 4)
            self.assertEqual(gate["orientation_quality"], "measured_quaternion")

    def test_freeze_episodes_writes_disjoint_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kitti = root / "KITTI360"
            sequence = "drive_0009_sync"
            lidar_dir = kitti / "data_3d_raw" / sequence / "velodyne_points" / "data"
            pose_dir = kitti / "data_poses" / sequence
            lidar_dir.mkdir(parents=True)
            pose_dir.mkdir(parents=True)
            pose_lines = []
            for frame_id in range(20):
                x = float(frame_id) if frame_id < 10 else float(frame_id - 10) + 0.1
                (lidar_dir / f"{frame_id:010d}.bin").write_bytes(
                    frame_id.to_bytes(4, byteorder="little")
                )
                pose_lines.append(f"{frame_id} 1 0 0 {x} 0 1 0 0 0 0 1 0")
            (pose_dir / "poses.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
            calibration_dir = kitti / "calibration"
            calibration_dir.mkdir()
            for calibration_name in ("calib_cam_to_pose.txt", "calib_cam_to_velo.txt"):
                (calibration_dir / calibration_name).write_text(
                    "1 0 0 0\n0 1 0 0\n0 0 1 0\n", encoding="utf-8"
                )
            output = root / "episodes"
            manifest = freeze_episodes(
                dataset="kitti360",
                root=kitti,
                map_sequence=sequence,
                query_sequence=sequence,
                output=output,
                overlap_radius_m=1.0,
                map_context_radius_m=5.0,
                query_episode_frames=5,
                max_episodes=2,
                max_map_frames=10,
                map_stride=1,
                m2dgr_max_time_diff_s=0.05,
            )
            self.assertTrue(manifest["map_query_disjoint"])
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(manifest["expected_behavior"], "lock")
            self.assertEqual(manifest["source_covered_query_frame_count"], 10)
            self.assertEqual(manifest["frozen_covered_query_frame_count"], 10)
            self.assertEqual(manifest["query_episode_count"], 2)
            self.assertEqual(manifest["query_frame_count"], 10)
            self.assertEqual(manifest["kitti360_calibration"]["mode"], "official")
            self.assertEqual(len(manifest["kitti360_calibration"]["files"]), 2)
            with (output / "episode_frames.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(row["role"] == "query" for row in rows), 10)
            self.assertTrue(all(len(row["cloud_sha256"]) == 64 for row in rows))
            self.assertTrue((output / "checksums.sha256").is_file())

            _verify_manifest(output)
            original_gt = (pose_dir / "poses.txt").read_text(encoding="utf-8")
            (pose_dir / "poses.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ground-truth hash mismatch"):
                _verify_manifest(output)
            (pose_dir / "poses.txt").write_text(original_gt, encoding="utf-8")
            (calibration_dir / "calib_cam_to_velo.txt").write_text(
                "changed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "calibration hash mismatch"):
                _verify_manifest(output)

    def test_freeze_uncovered_cross_drive_abstain_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "KITTI360"
            for sequence, offset in (("drive_map_sync", 0.0), ("drive_query_sync", 100.0)):
                lidar_dir = root / "data_3d_raw" / sequence / "velodyne_points" / "data"
                pose_dir = root / "data_poses" / sequence
                lidar_dir.mkdir(parents=True)
                pose_dir.mkdir(parents=True)
                pose_lines = []
                for frame_id in range(12):
                    (lidar_dir / f"{frame_id:010d}.bin").write_bytes(
                        frame_id.to_bytes(4, byteorder="little")
                    )
                    pose_lines.append(
                        f"{frame_id} 1 0 0 {offset + frame_id} 0 1 0 0 0 0 1 0"
                    )
                (pose_dir / "poses.txt").write_text(
                    "\n".join(pose_lines) + "\n", encoding="utf-8"
                )
            calibration_dir = root / "calibration"
            calibration_dir.mkdir()
            for name in ("calib_cam_to_pose.txt", "calib_cam_to_velo.txt"):
                (calibration_dir / name).write_text(
                    "1 0 0 0\n0 1 0 0\n0 0 1 0\n", encoding="utf-8"
                )

            manifest = freeze_episodes(
                dataset="kitti360",
                root=root,
                map_sequence="drive_map_sync",
                query_sequence="drive_query_sync",
                output=Path(temporary) / "episodes",
                overlap_radius_m=5.0,
                map_context_radius_m=30.0,
                query_episode_frames=4,
                max_episodes=2,
                max_map_frames=6,
                map_stride=1,
                m2dgr_max_time_diff_s=0.05,
                expected_behavior="abstain",
            )
            self.assertEqual(manifest["expected_behavior"], "abstain")
            self.assertEqual(manifest["spatial_relationship"], "uncovered_within_radius")
            self.assertEqual(manifest["source_covered_query_frame_count"], 0)
            self.assertEqual(manifest["frozen_covered_query_frame_count"], 0)
            self.assertEqual(manifest["query_episode_count"], 2)
            self.assertEqual(manifest["map_frame_count"], 6)
            _verify_manifest(Path(temporary) / "episodes")

    def test_verify_kitti_manifest_requires_frozen_calibration_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_dir = root / "manifest"
            manifest_dir.mkdir()
            cloud = root / "cloud.bin"
            cloud.write_bytes(b"cloud")
            gt = root / "poses.txt"
            gt.write_text("0 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            _write_csv(
                manifest_dir / "episode_frames.csv",
                ["episode_id", "role", "relative_cloud_path", "cloud_sha256"],
                [
                    {
                        "episode_id": "map",
                        "role": "map",
                        "relative_cloud_path": "cloud.bin",
                        "cloud_sha256": sha256_file(cloud),
                    }
                ],
            )
            _write_json(
                manifest_dir / "dataset_manifest.json",
                {
                    "schema_version": 1,
                    "dataset": "kitti360",
                    "root": str(root),
                    "map_gt_path": str(gt),
                    "query_gt_path": str(gt),
                    "map_gt_sha256": sha256_file(gt),
                    "query_gt_sha256": sha256_file(gt),
                    "episode_frames_sha256": sha256_file(
                        manifest_dir / "episode_frames.csv"
                    ),
                    "map_query_disjoint": True,
                },
            )

            with self.assertRaisesRegex(ValueError, "no frozen official calibration"):
                _verify_manifest(manifest_dir)
            _verify_manifest(
                manifest_dir, allow_legacy_unhashed_calibration=True
            )

    def test_freeze_m2dgr_records_alignment_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "M2DGR"
            sequence = "gate_02"
            lidar_dir = root / sequence / "velodyne_points"
            lidar_dir.mkdir(parents=True)
            gt_lines = []
            for frame_id in range(20):
                stamp = float(frame_id)
                x = float(frame_id) if frame_id < 10 else float(frame_id - 10) + 0.1
                (lidar_dir / f"{stamp:.3f}.bin").write_bytes(
                    frame_id.to_bytes(4, byteorder="little")
                )
                gt_lines.append(f"{stamp:.3f} {x} 0 0 0 0 0 1")
            (root / sequence / f"{sequence}.txt").write_text(
                "\n".join(gt_lines) + "\n", encoding="utf-8"
            )

            manifest = freeze_episodes(
                dataset="m2dgr",
                root=root,
                map_sequence=sequence,
                query_sequence=sequence,
                output=Path(temporary) / "episodes",
                overlap_radius_m=1.0,
                map_context_radius_m=5.0,
                query_episode_frames=5,
                max_episodes=1,
                max_map_frames=10,
                map_stride=1,
                m2dgr_max_time_diff_s=0.005,
            )
            self.assertEqual(manifest["m2dgr_max_time_diff_s"], 0.005)

    def test_surface_overlap_labels_trajectory_near_positive_and_hard_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "KITTI360"
            map_sequence = "drive_map_sync"
            query_sequence = "drive_query_sync"
            map_lidar = root / "data_3d_raw" / map_sequence / "velodyne_points" / "data"
            query_lidar = root / "data_3d_raw" / query_sequence / "velodyne_points" / "data"
            map_lidar.mkdir(parents=True)
            query_lidar.mkdir(parents=True)
            grid = np.asarray(
                [[x, y, 0.0, 1.0] for x in np.arange(0.0, 4.0, 0.5) for y in np.arange(0.0, 4.0, 0.5)],
                dtype="<f4",
            )
            map_cloud = map_lidar / "0000000000.bin"
            positive_cloud = query_lidar / "0000000000.bin"
            negative_cloud = query_lidar / "0000000001.bin"
            ambiguous_cloud = query_lidar / "0000000002.bin"
            grid.tofile(map_cloud)
            grid.tofile(positive_cloud)
            displaced = grid.copy()
            displaced[:, 2] += 10.0
            displaced.tofile(negative_cloud)
            ambiguous = grid.copy()
            ambiguous[::2, 2] += 10.0
            ambiguous.tofile(ambiguous_cloud)
            pose_line = "1 0 0 0 0 1 0 0 0 0 1 0"
            for sequence, lines in (
                (map_sequence, [f"0 {pose_line}"]),
                (query_sequence, [f"0 {pose_line}", f"1 {pose_line}", f"2 {pose_line}"]),
            ):
                pose_dir = root / "data_poses" / sequence
                pose_dir.mkdir(parents=True)
                (pose_dir / "poses.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            calibration = root / "calibration"
            calibration.mkdir()
            (calibration / "calib_cam_to_velo.txt").write_text(pose_line + "\n", encoding="utf-8")
            (calibration / "calib_cam_to_pose.txt").write_text(
                "image_00: " + pose_line + "\n", encoding="utf-8"
            )

            candidate = Path(temporary) / "candidate"
            candidate.mkdir()
            rows = [
                {
                    "episode_id": "map",
                    "role": "map",
                    "frame_token": "0",
                    "relative_cloud_path": map_cloud.relative_to(root).as_posix(),
                    "cloud_sha256": sha256_file(map_cloud),
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                }
            ]
            rows.extend(
                {
                    "episode_id": episode_id,
                    "role": "query",
                    "frame_token": str(frame_token),
                    "relative_cloud_path": cloud.relative_to(root).as_posix(),
                    "cloud_sha256": sha256_file(cloud),
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                }
                for episode_id, frame_token, cloud in (
                    ("query_000", 0, positive_cloud),
                    ("query_001", 1, negative_cloud),
                    ("query_002", 2, ambiguous_cloud),
                )
            )
            _write_csv(candidate / "episode_frames.csv", list(rows[0]), rows)
            calibration_files = []
            for relative in (
                "calibration/calib_cam_to_pose.txt",
                "calibration/calib_cam_to_velo.txt",
            ):
                calibration_files.append(
                    {"relative_path": relative, "sha256": sha256_file(root / relative)}
                )
            _write_json(
                candidate / "dataset_manifest.json",
                {
                    "schema_version": 3,
                    "dataset": "kitti360",
                    "root": str(root),
                    "map_sequence": map_sequence,
                    "query_sequence": query_sequence,
                    "evidence_class": "oracle_gt_cross_drive",
                    "map_query_disjoint": True,
                    "map_frame_count": 1,
                    "query_episode_count": 3,
                    "query_frame_count": 3,
                    "expected_behavior": "lock",
                    "spatial_relationship": "covered_within_radius",
                    "frozen_covered_query_frame_count": 3,
                    "overlap_radius_m": 5.0,
                    "map_gt_path": str(root / "data_poses" / map_sequence / "poses.txt"),
                    "query_gt_path": str(root / "data_poses" / query_sequence / "poses.txt"),
                    "map_gt_sha256": sha256_file(root / "data_poses" / map_sequence / "poses.txt"),
                    "query_gt_sha256": sha256_file(root / "data_poses" / query_sequence / "poses.txt"),
                    "episode_frames_sha256": sha256_file(candidate / "episode_frames.csv"),
                    "kitti360_calibration": {"mode": "official", "files": calibration_files},
                },
            )

            labeled = Path(temporary) / "labeled"
            manifest = classify_surface_overlap(manifest_dir=candidate, output=labeled)
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["expected_behavior"], "mixed")
            self.assertEqual(
                manifest["expected_behavior_by_episode"],
                {"query_000": "lock", "query_001": "abstain"},
            )
            self.assertEqual(manifest["excluded_ambiguous_episode_ids"], ["query_002"])
            with (labeled / "episode_frames.csv").open(encoding="utf-8", newline="") as stream:
                labeled_rows = list(csv.DictReader(stream))
            self.assertNotIn("query_002", {row["episode_id"] for row in labeled_rows})
            _verify_manifest(labeled)
            report_path = labeled / "surface_overlap.json"
            report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence hash mismatch"):
                _verify_manifest(labeled)


class EvalValidateTest(unittest.TestCase):
    def test_valid_synthetic_run_passes_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary), "valid")
            report = validate_run(run, strict=True)
            self.assertTrue(report.valid, report.to_dict())

    def test_missing_complete_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary), "missing_complete", write_complete=False)
            self.assertIn("missing_artifact", _error_codes(run))

    def test_precomplete_accepts_payload_only_and_rejects_existing_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_run(root, "payload", write_complete=False)
            preflight = validate_run(payload, strict=True, pre_complete=True)
            self.assertTrue(preflight.valid, preflight.to_dict())
            self.assertFalse(validate_run(payload, strict=True).valid)

            complete = _make_run(root, "already_complete")
            report = validate_run(complete, strict=True, pre_complete=True)
            self.assertFalse(report.valid)
            self.assertIn("precomplete_has_complete", {item["code"] for item in report.errors})

    def test_synthetic_honesty_flags_are_independently_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary), "dishonest_flags")
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            manifest["feature_flags"]["gt_passed_to_localizer"] = True
            _write_json(run / "manifest.json", manifest)
            _refresh_integrity(run)
            self.assertIn("synthetic_honesty", _error_codes(run))

    def test_synthetic_product_profile_is_independently_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary), "relaxed_profile")
            config = json.loads((run / "resolved_config.yaml").read_text(encoding="utf-8"))
            config["eval_profile"] = "relaxed_smoke"
            _write_json(run / "resolved_config.yaml", config)
            _refresh_integrity(run)
            self.assertIn("synthetic_eval_profile", _error_codes(run))

    def test_query_evidence_drives_sticky_attempt_and_lock_event_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            query_rows = [
                {
                    "query_id": "q0",
                    "attempt_id": "a0",
                    "query_outcome": "FALSE_FULL_LOCK",
                    "correct_lock_within_10s": False,
                },
                {
                    "query_id": "q1",
                    "attempt_id": "a1",
                    "query_outcome": "TEMPORAL_DEFER",
                    "correct_lock_within_10s": False,
                },
            ]
            run = _make_run(
                Path(temporary),
                "forged_attempt",
                query_rows=query_rows,
                metrics_overrides={
                    "query_outcome_counts": {"FALSE_FULL_LOCK": 1, "TEMPORAL_DEFER": 1},
                },
            )
            codes = _error_codes(run)
            self.assertIn("attempt_evidence", codes)
            self.assertIn("lock_event_count", codes)
            self.assertIn("correct_lock_by_10s", codes)

    def test_formal_cross_artifact_semantics_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_overrides = {
                "odom_source": "recorded_lio_odom",
                "query_source": "independent_real_scan",
                "gt_runtime_access": False,
                "evidence_class": "recorded_cross_session",
            }
            metrics_overrides = {
                "odom_source": "recorded_lio_odom",
                "query_source": "independent_real_scan",
                "evidence_class": "recorded_cross_session",
                "gt_runtime_access": False,
            }
            run = _make_run(
                root,
                "formal_mismatch",
                manifest_overrides=manifest_overrides,
                metrics_overrides=metrics_overrides,
            )
            config = json.loads((run / "resolved_config.yaml").read_text(encoding="utf-8"))
            config["query_source"] = "global_map_render"
            _write_json(run / "resolved_config.yaml", config)
            _refresh_integrity(run)
            self.assertIn("cross_artifact_semantics", _error_codes(run))

    def test_nonfinite_json_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(
                Path(temporary),
                "nonfinite",
                metrics_overrides={"p95_total_ms": float("nan")},
                allow_nan_metrics=True,
            )
            self.assertIn("json_parse", _error_codes(run))

    def test_duplicate_query_and_wrong_count_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = [
                {"query_id": "q0", "attempt_id": "a0", "query_outcome": "CORRECT_FULL_LOCK"},
                {"query_id": "q0", "attempt_id": "a1", "query_outcome": "TEMPORAL_DEFER"},
            ]
            run = _make_run(Path(temporary), "duplicate_query", query_rows=rows)
            self.assertIn("duplicate_query_id", _error_codes(run))

    def test_duplicate_attempt_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempts = [
                {"attempt_id": "a0", "attempt_outcome": "correct", "ever_false_lock": "false"},
                {"attempt_id": "a0", "attempt_outcome": "timeout", "ever_false_lock": "false"},
            ]
            run = _make_run(Path(temporary), "duplicate_attempt", attempt_rows=attempts)
            self.assertIn("duplicate_attempt_id", _error_codes(run))

    def test_false_lock_stays_false_after_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempts = [
                {"attempt_id": "a0", "attempt_outcome": "correct", "ever_false_lock": "true"},
                {"attempt_id": "a1", "attempt_outcome": "timeout", "ever_false_lock": "false"},
            ]
            run = _make_run(Path(temporary), "false_then_recover", attempt_rows=attempts)
            self.assertIn("false_lock_sticky", _error_codes(run))

    def test_runtime_gt_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(
                Path(temporary),
                "gt_leak",
                runtime_event={
                    "query_id": "q0",
                    "runtime_stage": "SEARCHING",
                    "query_outcome": "CORRECT_FULL_LOCK",
                },
            )
            self.assertIn("runtime_gt_leak", _error_codes(run))

    def test_recorded_lio_requires_gt_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(
                Path(temporary),
                "recorded_gt_leak",
                manifest_overrides={"odom_source": "recorded_lio_odom", "gt_runtime_access": True},
                metrics_overrides={"odom_source": "recorded_lio_odom"},
            )
            self.assertIn("gt_isolation", _error_codes(run))

    def test_noncanonical_source_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(
                Path(temporary),
                "legacy_names",
                manifest_overrides={"odom_source": "oracle_gt", "query_source": "global_map"},
                metrics_overrides={"odom_source": "oracle_gt", "query_source": "global_map"},
            )
            codes = _error_codes(run)
            self.assertIn("odom_source", codes)
            self.assertIn("query_source", codes)

    def test_checksum_tamper_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(Path(temporary), "checksum_tamper")
            (run / "stdout.log").write_text("modified after COMPLETE\n", encoding="utf-8")
            self.assertIn("checksum_mismatch", _error_codes(run))

    def test_zero_lock_precision_must_be_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _make_run(
                Path(temporary),
                "zero_lock",
                metrics_overrides={
                    "correct_lock_event_count": 0,
                    "false_lock_event_count": 0,
                    "correct_lock_by_10s_count": 0,
                    "correct_lock_by_10s": 0.0,
                    "lock_precision": 0.0,
                    "false_lock_given_lock": 0.0,
                },
            )
            self.assertIn("zero_denominator", _error_codes(run))


class SyntheticEvalGateTest(unittest.TestCase):
    def test_wrapper_publishes_a_strict_valid_shadow_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run_fake_gate(Path(temporary), "synthetic_run")
            validation = validate_run(run, strict=True)
            self.assertTrue(validation.valid, validation.to_dict())
            self.assertTrue((run / "COMPLETE").is_file())
            self.assertFalse((run / "summary" / "gate_report.json").exists())
            self.assertFalse((run / "raw" / "producer_stage").exists())

            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["verdict_ceiling"], "SHADOW_ONLY")
            self.assertEqual(manifest["split"], "p0s_shadow")
            self.assertEqual(manifest["query_identity_capability"], "per_query_fnv1a64")
            self.assertEqual(manifest["required_temporal_window_size"], 5)
            self.assertTrue(manifest["feature_flags"]["odom_derived_from_gt"])
            self.assertTrue(manifest["feature_flags"]["synthetic_from_target_map"])
            self.assertFalse(manifest["feature_flags"]["gt_passed_to_localizer"])
            self.assertEqual(manifest["render_map_sha256"], manifest["target_map_sha256"])

            metrics = json.loads((run / "summary" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["planned_render_query_count"], 10)
            self.assertEqual(metrics["valid_render_query_count"], 10)
            self.assertEqual(metrics["invalid_render_query_count"], 0)
            self.assertEqual(metrics["eligible_query_count"], 8)
            self.assertEqual(metrics["tracking_processed_query_count"], 3)
            self.assertEqual(metrics["attempt_count"], 2)
            self.assertEqual(metrics["correct_attempt_count"], 1)
            self.assertEqual(metrics["timeout_attempt_count"], 1)
            self.assertEqual(metrics["correct_lock_event_count"], 3)

            with (run / "raw" / "relocalization_queries.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                eligible_ids = {row["query_id"] for row in csv.DictReader(stream)}
            self.assertEqual(eligible_ids, {"0", "2", "4", "5", "6", "7", "8", "9"})
            with (run / "raw" / "relocalization_attempts.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                attempts = {row["attempt_id"]: row for row in csv.DictReader(stream)}
            self.assertEqual(attempts["10"]["attempt_outcome"], "correct")
            self.assertEqual(attempts["20"]["attempt_outcome"], "timeout")
            with (run / "raw" / "renderer_visibility.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                first_visibility = next(csv.DictReader(stream))
            self.assertEqual(int(first_visibility["num_query_points"]), 100)
            self.assertEqual(int(first_visibility["occupied_ray_bins"]), 100)

    def test_false_lock_is_sticky_across_a_later_correct_reentry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run_fake_gate(
                Path(temporary),
                "sticky_false",
                producer_args=["--false-reentry"],
            )
            metrics = json.loads((run / "summary" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["false_attempt_count"], 1)
            self.assertEqual(metrics["correct_attempt_count"], 0)
            self.assertEqual(metrics["false_lock_event_count"], 1)
            self.assertEqual(metrics["correct_lock_event_count"], 2)
            with (run / "raw" / "relocalization_attempts.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                attempts = {row["attempt_id"]: row for row in csv.DictReader(stream)}
            self.assertEqual(attempts["10"]["attempt_outcome"], "false")
            self.assertEqual(attempts["10"]["ever_false_lock"].lower(), "true")

    def test_wrapper_rejects_incomplete_producer_schema_and_preserves_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(SyntheticGateError) as caught:
                _run_fake_gate(root, "missing_fingerprint", producer_args=["--omit-fingerprint"])
            self.assertFalse((root / "missing_fingerprint").exists())
            self.assertIsNotNone(caught.exception.staging_dir)
            staging = caught.exception.staging_dir
            assert staging is not None
            self.assertTrue((staging / "WRAPPER_FAILURE.json").is_file())

    def test_wrapper_rejects_non_gate_profile_window_raycast_and_state_chain(self) -> None:
        cases = (
            ("relaxed", ["--eval_profile", "relaxed_smoke"], "eval_profile"),
            ("window", ["--fake-temporal-window", "1"], "reloc_temporal_window_size"),
            ("raycast", ["--raycast-off"], "raycast_azimuth_resolution_deg"),
            ("state_chain", ["--break-state-chain"], "locked_before_frame"),
            ("hidden_event", ["--invalid-hide-lock"], "generation-invalid"),
            ("visibility_99", ["--valid-counts", "99"], "below frozen minimum"),
        )
        for name, producer_args, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(SyntheticGateError, expected):
                    _run_fake_gate(Path(temporary), name, producer_args=producer_args)

    def test_correct_lock_after_deadline_keeps_event_but_attempt_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run_fake_gate(
                Path(temporary),
                "delayed_lock",
                producer_args=["--delayed-lock"],
                second_episode_frames=11,
            )
            metrics = json.loads((run / "summary" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["correct_lock_event_count"], 4)
            self.assertEqual(metrics["correct_attempt_count"], 1)
            self.assertEqual(metrics["timeout_attempt_count"], 1)
            self.assertEqual(metrics["correct_lock_by_10s_count"], 1)
            with (run / "raw" / "relocalization_attempts.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                attempts = {row["attempt_id"]: row for row in csv.DictReader(stream)}
            self.assertEqual(attempts["20"]["attempt_outcome"], "timeout")

    def test_wrapper_rejects_non_one_second_deadline_basis_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(SyntheticGateError, "must equal 1.0"):
                _run_fake_gate(root, "wrong_period", frame_period_s=0.5)

            producer, map_path, pose_manifest = _make_synthetic_gate_inputs(root)
            output = root / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(SyntheticGateError, "refusing to overwrite"):
                run_synthetic_gate(
                    producer=producer,
                    output=output,
                    map_path=map_path,
                    query_pose_manifest=pose_manifest,
                    frozen_contract=SYNTHETIC_FIXTURE_CONTRACT,
                    dataset_id="synthetic_fixture_route",
                    map_provenance="unit_test_fixture",
                    repo=REPO_ROOT,
                    producer_args=[],
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_strict_validator_rejects_tampered_synthetic_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run_fake_gate(Path(temporary), "tampered_fingerprint")
            resolved_path = run / "raw" / "resolved_queries.csv"
            with resolved_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["query_cloud_fingerprint_fnv1a64"] = "ffffffffffffffff"
            _write_csv(resolved_path, headers, rows)
            _refresh_integrity(run)
            self.assertIn("synthetic_fingerprint", _error_codes(run))

    def test_wrapper_cli_requires_explicit_repo(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOLS_DIR / "n3mapping_synthetic_eval_gate.py"),
                "--producer",
                "producer",
                "--output",
                "output",
                "--map",
                "map",
                "--query-pose-manifest",
                "poses",
                "--frozen-contract",
                "contract",
                "--dataset-id",
                "dataset",
                "--map-provenance",
                "fixture",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("--repo", process.stderr)


class EvalCompareTest(unittest.TestCase):
    def test_synthetic_pair_is_capped_at_shadow_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_run(root, "baseline")
            candidate = _make_run(root, "candidate")
            output = root / "comparison"
            report = compare_runs(baseline, candidate, FIXTURE_CONTRACT, output)
            self.assertEqual(report["verdict"], "SHADOW_ONLY", report)
            self.assertTrue((output / "paired_diff.csv").is_file())
            self.assertTrue((output / "gate_report.json").is_file())
            self.assertEqual(report["gates"]["evidence"]["status"], "SHADOW_ONLY")

    def test_utility_regression_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_run(root, "baseline")
            candidate = _make_run(
                root,
                "candidate",
                query_rows=[
                    {
                        "query_id": "q0",
                        "attempt_id": "a0",
                        "query_outcome": "CORRECT_FULL_LOCK",
                        "correct_lock_within_10s": False,
                    },
                    {
                        "query_id": "q1",
                        "attempt_id": "a1",
                        "query_outcome": "TEMPORAL_DEFER",
                        "correct_lock_within_10s": False,
                    },
                ],
                attempt_rows=[
                    {"attempt_id": "a0", "attempt_outcome": "timeout", "ever_false_lock": "false"},
                    {"attempt_id": "a1", "attempt_outcome": "timeout", "ever_false_lock": "false"},
                ],
                metrics_overrides={
                    "correct_attempt_count": 0,
                    "timeout_attempt_count": 2,
                    "correct_lock_by_10s_count": 0,
                    "correct_lock_by_10s": 0.0,
                },
            )
            report = compare_runs(baseline, candidate, FIXTURE_CONTRACT, root / "comparison")
            self.assertEqual(report["verdict"], "FAIL", report)
            self.assertEqual(report["gates"]["utility"]["status"], "FAIL")

    def test_formal_recorded_cross_session_pair_can_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_overrides = {
                "odom_source": "recorded_lio_odom",
                "query_source": "independent_real_scan",
                "gt_runtime_access": False,
                "evidence_class": "recorded_cross_session",
            }
            metrics_overrides = {
                "odom_source": "recorded_lio_odom",
                "query_source": "independent_real_scan",
                "evidence_class": "recorded_cross_session",
            }
            baseline = _make_run(
                root,
                "baseline",
                manifest_overrides=manifest_overrides,
                metrics_overrides=metrics_overrides,
            )
            candidate = _make_run(
                root,
                "candidate",
                manifest_overrides=manifest_overrides,
                metrics_overrides=metrics_overrides,
            )
            report = compare_runs(baseline, candidate, FIXTURE_CONTRACT, root / "comparison")
            self.assertEqual(report["verdict"], "PASS", report)

    def test_budget_mismatch_is_invalid_not_a_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_run(root, "baseline")
            candidate = _make_run(
                root,
                "candidate",
                manifest_overrides={"candidate_budget": {"top_k": 10, "yaw_hypotheses": 4}},
            )
            report = compare_runs(baseline, candidate, FIXTURE_CONTRACT, root / "comparison")
            self.assertEqual(report["verdict"], "INVALID", report)

    def test_existing_nonempty_compare_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_run(root, "baseline")
            candidate = _make_run(root, "candidate")
            output = root / "comparison"
            output.mkdir()
            sentinel = output / "gate_report.json"
            sentinel.write_text("old evidence\n", encoding="utf-8")
            report = compare_runs(baseline, candidate, FIXTURE_CONTRACT, output)
            self.assertEqual(report["verdict"], "INVALID", report)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "old evidence\n")
            self.assertFalse((output / "paired_diff.csv").exists())

    def test_synthetic_comparator_requires_exact_per_query_fnv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _run_fake_gate(root, "baseline")
            candidate = _run_fake_gate(
                root,
                "candidate",
                producer_args=["--fingerprint-salt", "100"],
            )
            report = compare_runs(
                baseline,
                candidate,
                SYNTHETIC_FIXTURE_CONTRACT,
                root / "comparison",
            )
            self.assertEqual(report["verdict"], "INVALID", report)
            fnv_check = next(
                item for item in report["compatibility"] if item["field"] == "per_query_fnv1a64"
            )
            self.assertFalse(fnv_check["equal"])

    def test_comparator_rejects_self_compare_and_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_run(root, "baseline")
            self_report = compare_runs(
                baseline,
                baseline,
                FIXTURE_CONTRACT,
                root / "self_comparison",
            )
            self.assertEqual(self_report["verdict"], "INVALID", self_report)

            candidate = _make_run(root, "candidate")
            config = json.loads((candidate / "resolved_config.yaml").read_text(encoding="utf-8"))
            config["candidate_only_tag"] = True
            _write_json(candidate / "resolved_config.yaml", config)
            _refresh_integrity(candidate)
            mismatch = compare_runs(
                baseline,
                candidate,
                FIXTURE_CONTRACT,
                root / "config_comparison",
            )
            self.assertEqual(mismatch["verdict"], "INVALID", mismatch)

    def test_formal_gate_rejects_non_independent_query_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_overrides = {
                "odom_source": "recorded_lio_odom",
                "query_source": "global_map_render",
                "gt_runtime_access": False,
                "evidence_class": "recorded_cross_session",
            }
            metrics_overrides = {
                "odom_source": "recorded_lio_odom",
                "query_source": "global_map_render",
                "evidence_class": "recorded_cross_session",
                "gt_runtime_access": False,
            }
            baseline = _make_run(
                root,
                "baseline",
                manifest_overrides=manifest_overrides,
                metrics_overrides=metrics_overrides,
            )
            candidate = _make_run(
                root,
                "candidate",
                manifest_overrides=manifest_overrides,
                metrics_overrides=metrics_overrides,
            )
            report = compare_runs(baseline, candidate, FIXTURE_CONTRACT, root / "comparison")
            self.assertEqual(report["verdict"], "INVALID", report)

    def test_cli_entrypoints_accept_a_valid_fixture_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _make_run(root, "baseline")
            candidate = _make_run(root, "candidate")
            validate_process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(TOOLS_DIR / "n3mapping_eval_validate.py"),
                    "--run",
                    str(baseline),
                    "--strict",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validate_process.returncode, 0, validate_process.stderr + validate_process.stdout)
            self.assertTrue(json.loads(validate_process.stdout)["valid"])
            compare_process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(TOOLS_DIR / "n3mapping_eval_compare.py"),
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--contract",
                    str(FIXTURE_CONTRACT),
                    "--output",
                    str(root / "cli_comparison"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(compare_process.returncode, 0, compare_process.stderr + compare_process.stdout)
            self.assertEqual(json.loads(compare_process.stdout)["verdict"], "SHADOW_ONLY")


if __name__ == "__main__":
    unittest.main()
