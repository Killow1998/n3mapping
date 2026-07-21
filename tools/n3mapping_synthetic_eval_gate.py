#!/usr/bin/env python3
"""Wrap the synthetic C++ producer in the strict n3mapping v2 run contract.

The wrapper never overwrites a run. It executes the producer in a unique
staging directory, maps its flat outputs into strict raw/summary artifacts,
performs payload validation without ``COMPLETE``, creates ``COMPLETE``
atomically, validates again, and only then publishes the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from n3mapping_eval_validate import (
    QUERY_OUTCOMES,
    load_contract,
    load_json_strict,
    sha256_file,
    validate_run,
)


WRAPPER_VERSION = "n3mapping_synthetic_gate_v1"
ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")
CANONICAL_QUERY_SOURCES = {"global_map_render", "local_submap_render"}
FINGERPRINT_ALGORITHM = "fnv1a64_xyz_intensity_float32_le_v1"
PRODUCT_EVAL_PROFILE = "product_default"
FIXED_FRAME_PERIOD_S = 1.0
FRAME_PERIOD_TOLERANCE = 1e-12
REQUIRED_TEMPORAL_WINDOW_SIZE = 5
REQUIRED_MINIMUM_QUERY_POINTS = 100
REQUIRED_MINIMUM_OCCUPIED_RAY_BINS = 100
PRODUCER_REQUIRED_FILES = {
    "per_query.csv": "raw/producer_per_query.source",
    "resolved_queries.csv": "raw/producer_resolved_queries.source",
    "metrics.json": "raw/producer_metrics.json",
    "summary.json": "raw/producer_summary.json",
    "config_used.json": "raw/producer_config_used.json",
    "renderer_config.json": "raw/producer_renderer_config.json",
    "failed_queries.txt": "raw/producer_failed_queries.txt",
    "EVAL_COMPLETE": "raw/producer_eval_complete.txt",
}
SYNTHETIC_REQUIRED_ARTIFACTS = {
    "manifest.json",
    "resolved_config.yaml",
    "dataset_manifest.json",
    "frozen_contract.yaml",
    "command.txt",
    "environment.json",
    "stdout.log",
    "raw/runtime_events.jsonl",
    "raw/relocalization_queries.csv",
    "raw/relocalization_attempts.csv",
    "raw/stage_timing.csv",
    "raw/renderer_visibility.csv",
    "raw/resolved_queries.csv",
    *PRODUCER_REQUIRED_FILES.values(),
    "raw/producer_stderr.log",
    "labels/query_pose_manifest.csv",
    "summary/metrics.json",
    "summary/failure_counts.json",
    "summary/renderer_metrics.json",
    "checksums.sha256",
    "COMPLETE",
}
PRODUCER_REQUIRED_COLUMNS = {
    "query_id",
    "episode_id",
    "frame_index",
    "reference_keyframe_id",
    "renderer_seed",
    "generation_valid",
    "generation_reason",
    "localization_attempted",
    "tracking_processed",
    "locked_before_frame",
    "locked_after_frame",
    "ever_locked_before_frame",
    "lock_event_type",
    "tracking_loss_event",
    "runtime_stage",
    "success",
    "relocalization_locked",
    "matched_keyframe_id",
    "pose_accurate",
    "translation_error_m",
    "yaw_error_deg",
    "roll_pitch_error_deg",
    "renderer_elapsed_ms",
    "localizer_elapsed_ms",
    "scorer_elapsed_ms",
    "query_x_m",
    "query_y_m",
    "query_z_m",
    "query_roll_deg",
    "query_pitch_deg",
    "query_yaw_deg",
    "input_map_points",
    "range_eligible_points",
    "fov_eligible_points",
    "occupied_ray_bins",
    "same_ray_occluded_points",
    "dropout_suppressed_points",
    "occlusion_suppressed_bins",
    "query_points_before_voxel",
    "num_query_points",
    "raycast_enabled",
    "envelope_valid",
    "envelope_azimuth_full",
    "envelope_azimuth_start_deg",
    "envelope_azimuth_span_deg",
    "envelope_vertical_min_deg",
    "envelope_vertical_max_deg",
    "query_cloud_fingerprint_fnv1a64",
    "query_cloud_fingerprint_available",
    "query_cloud_fingerprint_algorithm",
    "finite_points",
    "azimuth_bins",
    "vertical_bins",
}


class SyntheticGateError(RuntimeError):
    def __init__(self, message: str, staging_dir: Path | None = None) -> None:
        super().__init__(message)
        self.staging_dir = staging_dir


def _timestamp() -> str:
    return datetime.now(ASIA_SHANGHAI).isoformat(timespec="microseconds")


def _ensure_new_file(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise SyntheticGateError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_new_file(path)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise SyntheticGateError(f"input must be a regular non-symlink file: {source}")
    _ensure_new_file(target)
    shutil.copyfile(source, target)


def _move_producer_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise SyntheticGateError(f"producer is missing required artifact: {source.name}")
    _ensure_new_file(target)
    os.replace(source, target)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SyntheticGateError(f"producer CSV has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise SyntheticGateError(f"producer CSV has duplicate columns: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise SyntheticGateError(f"producer CSV has malformed rows: {path}")
    return list(reader.fieldnames), rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    _ensure_new_file(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(raw: Any, owner: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise SyntheticGateError(f"{owner} must be boolean, got {raw!r}")


def _parse_int(raw: Any, owner: str, nonnegative: bool = False) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise SyntheticGateError(f"{owner} must be an integer, got {raw!r}") from exc
    if nonnegative and value < 0:
        raise SyntheticGateError(f"{owner} must be non-negative, got {value}")
    return value


def _parse_float(raw: Any, owner: str, nullable: bool = False) -> float | None:
    text = str(raw).strip().lower()
    if nullable and text in {"", "nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise SyntheticGateError(f"{owner} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        if nullable:
            return None
        raise SyntheticGateError(f"{owner} must be finite, got {raw!r}")
    return value


def _percentile(values: list[float], q: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    index = max(0.0, min(1.0, q)) * (len(finite) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return finite[low]
    alpha = index - low
    return finite[low] * (1.0 - alpha) + finite[high] * alpha


def _git_output(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise SyntheticGateError(
            f"git {' '.join(args)} failed: {process.stderr.strip() or process.stdout.strip()}"
        )
    return process.stdout.strip()


def _git_metadata(repo: Path) -> dict[str, Any]:
    return {
        "repo_sha": _git_output(repo, "rev-parse", "HEAD"),
        "branch": _git_output(repo, "branch", "--show-current") or "DETACHED",
        "dirty": bool(_git_output(repo, "status", "--porcelain")),
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read_cgroup_value(relative: str) -> str | None:
    cgroup = Path("/proc/self/cgroup")
    if not cgroup.is_file():
        return None
    for line in cgroup.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            path = Path("/sys/fs/cgroup") / parts[2].lstrip("/") / relative
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


def _limit_as_int(raw: str | None) -> int:
    if raw is None or raw == "max":
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


def _environment() -> dict[str, Any]:
    memory_max = _read_cgroup_value("memory.max")
    memory_swap_max = _read_cgroup_value("memory.swap.max")
    cpu_max = _read_cgroup_value("cpu.max")
    try:
        thread_count = len(os.sched_getaffinity(0))
    except AttributeError:
        thread_count = os.cpu_count() or 1
    return {
        "cpu_model": _cpu_model(),
        "thread_count": thread_count,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cgroup_v2": {
            "memory.max": memory_max,
            "memory.swap.max": memory_swap_max,
            "cpu.max": cpu_max,
        },
        "memory_max_bytes": _limit_as_int(memory_max),
        "memory_swap_max_bytes": _limit_as_int(memory_swap_max),
    }


def _validate_contract(
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    list[str],
    bool,
    str,
    dict[str, float],
    bool,
    int,
    dict[str, int],
]:
    contract = load_contract(contract_path)
    if contract.get("schema_version") != 2:
        raise SyntheticGateError("synthetic frozen contract must have schema_version=2")
    required = contract.get("required_artifacts")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise SyntheticGateError("synthetic frozen contract required_artifacts must be a string list")
    missing = sorted(SYNTHETIC_REQUIRED_ARTIFACTS - set(required))
    if missing:
        raise SyntheticGateError(
            f"frozen contract predates the synthetic wrapper; missing required artifacts: {missing}"
        )
    synthetic_requirements = contract.get("synthetic_requirements", {})
    if not isinstance(synthetic_requirements, dict):
        raise SyntheticGateError("synthetic_requirements must be an object when present")
    require_query_fingerprint = synthetic_requirements.get(
        "require_query_cloud_fingerprint_fnv1a64",
        False,
    )
    if not isinstance(require_query_fingerprint, bool):
        raise SyntheticGateError(
            "synthetic_requirements.require_query_cloud_fingerprint_fnv1a64 must be boolean"
        )
    required_eval_profile = contract.get("required_eval_profile")
    if required_eval_profile != PRODUCT_EVAL_PROFILE:
        raise SyntheticGateError(
            "synthetic frozen contract required_eval_profile must be 'product_default'"
        )
    required_frame_period = contract.get("required_frame_period_s")
    if (
        isinstance(required_frame_period, bool)
        or not isinstance(required_frame_period, (int, float))
        or not math.isfinite(float(required_frame_period))
        or not math.isclose(
            float(required_frame_period),
            FIXED_FRAME_PERIOD_S,
            rel_tol=0.0,
            abs_tol=FRAME_PERIOD_TOLERANCE,
        )
    ):
        raise SyntheticGateError("synthetic frozen contract required_frame_period_s must be 1.0")
    required_temporal_window = contract.get("required_temporal_window_size")
    if (
        isinstance(required_temporal_window, bool)
        or not isinstance(required_temporal_window, int)
        or required_temporal_window != REQUIRED_TEMPORAL_WINDOW_SIZE
    ):
        raise SyntheticGateError(
            "synthetic frozen contract required_temporal_window_size must be integer 5"
        )
    visibility_thresholds = {
        "minimum_query_points": contract.get("minimum_query_points"),
        "minimum_occupied_ray_bins": contract.get("minimum_occupied_ray_bins"),
    }
    expected_visibility_thresholds = {
        "minimum_query_points": REQUIRED_MINIMUM_QUERY_POINTS,
        "minimum_occupied_ray_bins": REQUIRED_MINIMUM_OCCUPIED_RAY_BINS,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value != expected_visibility_thresholds[field]
        for field, value in visibility_thresholds.items()
    ):
        raise SyntheticGateError(
            "synthetic frozen contract must freeze minimum_query_points=100 and "
            "minimum_occupied_ray_bins=100"
        )
    required_thresholds_raw = contract.get("required_label_thresholds")
    threshold_fields = (
        "pose_translation_threshold_m",
        "pose_yaw_threshold_deg",
        "pose_roll_pitch_threshold_deg",
    )
    if not isinstance(required_thresholds_raw, dict) or set(required_thresholds_raw) != set(
        threshold_fields
    ):
        raise SyntheticGateError(
            "synthetic frozen contract required_label_thresholds must freeze exactly "
            f"{list(threshold_fields)}"
        )
    required_thresholds: dict[str, float] = {}
    for field in threshold_fields:
        value = required_thresholds_raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SyntheticGateError(f"required_label_thresholds.{field} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise SyntheticGateError(
                f"required_label_thresholds.{field} must be finite and non-negative"
            )
        required_thresholds[field] = number
    require_raycast = synthetic_requirements.get("require_raycast_visibility", False)
    if not isinstance(require_raycast, bool):
        raise SyntheticGateError(
            "synthetic_requirements.require_raycast_visibility must be boolean"
        )
    return (
        contract,
        required,
        require_query_fingerprint,
        required_eval_profile,
        required_thresholds,
        require_raycast,
        required_temporal_window,
        visibility_thresholds,
    )


def _resolved_query_records(path: Path, required: bool) -> dict[str, dict[str, Any]]:
    headers, rows = _read_csv(path)
    identity_fields = {
        "query_id",
        "episode_id",
        "frame_index",
        "reference_keyframe_id",
        "renderer_seed",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "generation_valid",
        "generation_reason",
    }
    missing_identity = identity_fields - set(headers)
    if missing_identity:
        raise SyntheticGateError(
            "producer resolved_queries.csv is missing identity fields: "
            f"{sorted(missing_identity)}"
        )
    fingerprint_fields = {
        "query_cloud_fingerprint_fnv1a64",
        "query_cloud_fingerprint_available",
        "query_cloud_fingerprint_algorithm",
    }
    if not fingerprint_fields.issubset(headers):
        if required:
            raise SyntheticGateError(
                "producer resolved_queries.csv is missing contract-required fields: "
                f"{sorted(fingerprint_fields - set(headers))}"
            )
    records: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=2):
        owner = f"producer resolved_queries.csv:{line_number}"
        query_id_int = _parse_int(row.get("query_id", ""), f"{owner}.query_id", nonnegative=True)
        query_id = str(query_id_int)
        if query_id in records:
            raise SyntheticGateError(
                f"{owner} has duplicate query_id {query_id}"
            )
        record: dict[str, Any] = {
            "query_id": query_id,
            "episode_id": str(
                _parse_int(row["episode_id"], f"{owner}.episode_id", nonnegative=True)
            ),
            "frame_index": _parse_int(
                row["frame_index"], f"{owner}.frame_index", nonnegative=True
            ),
            "reference_keyframe_id": _parse_int(
                row["reference_keyframe_id"], f"{owner}.reference_keyframe_id"
            ),
            "renderer_seed": _parse_int(
                row["renderer_seed"], f"{owner}.renderer_seed", nonnegative=True
            ),
            "generation_valid": _parse_bool(
                row["generation_valid"], f"{owner}.generation_valid"
            ),
            "generation_reason": row["generation_reason"].strip(),
        }
        for field in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"):
            value = _parse_float(row[field], f"{owner}.{field}")
            assert value is not None
            record[field] = value
        if fingerprint_fields.issubset(headers):
            fingerprint = row.get("query_cloud_fingerprint_fnv1a64", "").strip()
            if re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None:
                raise SyntheticGateError(
                    f"{owner}.query_cloud_fingerprint_fnv1a64 must be 16 lowercase hex"
                )
            available = _parse_bool(
                row.get("query_cloud_fingerprint_available", ""),
                f"{owner}.query_cloud_fingerprint_available",
            )
            algorithm = row.get("query_cloud_fingerprint_algorithm", "").strip()
            if algorithm != FINGERPRINT_ALGORITHM:
                raise SyntheticGateError(f"{owner} has the wrong fingerprint algorithm")
            record.update(
                {
                    "query_cloud_fingerprint_available": available,
                    "query_cloud_fingerprint_algorithm": algorithm,
                    "query_cloud_fingerprint_fnv1a64": fingerprint,
                }
            )
        records[query_id] = record
    return records


def _query_pose_manifest_records(path: Path) -> dict[str, dict[str, Any]]:
    headers, rows = _read_csv(path)
    expected = [
        "query_id",
        "episode_id",
        "frame_index",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]
    if headers != expected:
        raise SyntheticGateError(
            "product_default query pose manifest must use the exact episode header: "
            + ",".join(expected)
        )
    records: dict[str, dict[str, Any]] = {}
    active_episode: str | None = None
    closed_episodes: set[str] = set()
    expected_frame_index = 0
    for line_number, row in enumerate(rows, start=2):
        owner = f"query pose manifest:{line_number}"
        query_id = str(_parse_int(row["query_id"], f"{owner}.query_id", nonnegative=True))
        episode_id = str(
            _parse_int(row["episode_id"], f"{owner}.episode_id", nonnegative=True)
        )
        frame_index = _parse_int(
            row["frame_index"], f"{owner}.frame_index", nonnegative=True
        )
        if query_id in records:
            raise SyntheticGateError(f"{owner} has duplicate query_id {query_id}")
        if episode_id != active_episode:
            if episode_id in closed_episodes:
                raise SyntheticGateError(f"{owner} episode rows are not contiguous")
            if active_episode is not None:
                closed_episodes.add(active_episode)
            active_episode = episode_id
            expected_frame_index = 0
        if frame_index != expected_frame_index:
            raise SyntheticGateError(
                f"{owner}.frame_index expected {expected_frame_index}, got {frame_index}"
            )
        expected_frame_index += 1
        record: dict[str, Any] = {
            "query_id": query_id,
            "episode_id": episode_id,
            "frame_index": frame_index,
        }
        for field in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"):
            value = _parse_float(row[field], f"{owner}.{field}")
            assert value is not None
            record[field] = value
        records[query_id] = record
    if not records:
        raise SyntheticGateError("query pose manifest contains no query rows")
    return records


def _verify_query_identity(
    pose_records: dict[str, dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
    resolved_records: dict[str, dict[str, Any]],
    require_query_fingerprint: bool,
) -> None:
    per_query = {row["query_id"]: row for row in normalized_rows}
    expected_ids = set(pose_records)
    if set(per_query) != expected_ids or set(resolved_records) != expected_ids:
        raise SyntheticGateError(
            "query_id sets differ across pose manifest, per_query.csv, and resolved_queries.csv"
        )
    pose_field_pairs = (
        ("query_x_m", "x_m"),
        ("query_y_m", "y_m"),
        ("query_z_m", "z_m"),
        ("query_roll_deg", "roll_deg"),
        ("query_pitch_deg", "pitch_deg"),
        ("query_yaw_deg", "yaw_deg"),
    )
    for query_id in sorted(expected_ids, key=int):
        pose = pose_records[query_id]
        produced = per_query[query_id]
        resolved = resolved_records[query_id]
        for field in ("episode_id", "frame_index"):
            if produced[field] != pose[field] or resolved[field] != pose[field]:
                raise SyntheticGateError(
                    f"query {query_id} {field} differs across pose manifest and producer outputs"
                )
        for produced_field, pose_field in pose_field_pairs:
            if not math.isclose(
                float(produced[produced_field]),
                float(pose[pose_field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                float(resolved[pose_field]),
                float(pose[pose_field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SyntheticGateError(
                    f"query {query_id} pose field {pose_field} differs from input pose manifest"
                )
        for field in (
            "reference_keyframe_id",
            "renderer_seed",
            "generation_valid",
            "generation_reason",
        ):
            if produced[field] != resolved[field]:
                raise SyntheticGateError(
                    f"query {query_id} {field} differs between per_query and resolved_queries"
                )
        if require_query_fingerprint:
            for field in (
                "query_cloud_fingerprint_available",
                "query_cloud_fingerprint_algorithm",
                "query_cloud_fingerprint_fnv1a64",
            ):
                if produced[field] != resolved.get(field):
                    raise SyntheticGateError(
                        f"query {query_id} {field} differs between per_query and resolved_queries"
                    )


def _producer_command(
    producer: Path,
    map_path: Path,
    pose_manifest: Path,
    producer_output: Path,
    producer_args: list[str],
) -> list[str]:
    protected = {"--map", "--output", "--query_pose_manifest"}
    conflicts = sorted(protected & set(producer_args))
    if conflicts:
        raise SyntheticGateError(f"wrapper-owned producer arguments may not be overridden: {conflicts}")
    return [
        str(producer),
        "--map",
        str(map_path),
        "--output",
        str(producer_output),
        "--query_pose_manifest",
        str(pose_manifest),
        *producer_args,
    ]


def _require_producer_config(
    config: dict[str, Any],
    required_eval_profile: str,
    required_thresholds: dict[str, float],
    require_raycast: bool,
    required_temporal_window: int,
    visibility_thresholds: dict[str, int],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    expected = {
        "schema_version": 2,
        "evidence_class": "synthetic_map_render",
        "odom_source": "oracle_gt_odom",
        "gt_runtime_access": True,
        "verdict": "SHADOW_ONLY",
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise SyntheticGateError(
                f"producer renderer_config {field} must be {value!r}, got {config.get(field)!r}"
            )
    if config.get("eval_profile") != required_eval_profile:
        raise SyntheticGateError(
            "producer renderer_config eval_profile must match frozen contract "
            f"{required_eval_profile!r}, got {config.get('eval_profile')!r}"
        )
    query_source = config.get("query_source")
    if query_source not in CANONICAL_QUERY_SOURCES:
        raise SyntheticGateError(
            "synthetic Gate requires producer query_source global_map_render or local_submap_render"
        )
    candidate_fields = (
        "rhpd_num_candidates",
        "rhpd_preselect_candidates",
        "reloc_num_candidates",
        "reloc_temporal_window_size",
    )
    missing_candidate = [field for field in candidate_fields if field not in config]
    if missing_candidate:
        raise SyntheticGateError(
            f"producer renderer_config omits candidate budget fields: {missing_candidate}"
        )
    candidate_budget = {field: config[field] for field in candidate_fields}
    temporal_window = config.get("reloc_temporal_window_size")
    if (
        isinstance(temporal_window, bool)
        or not isinstance(temporal_window, int)
        or temporal_window != required_temporal_window
    ):
        raise SyntheticGateError(
            "producer renderer_config reloc_temporal_window_size must match frozen "
            f"required_temporal_window_size={required_temporal_window}"
        )
    icp_fields = (
        "gicp_max_iterations",
        "gicp_max_correspondence_distance",
        "gicp_fitness_threshold",
        "gicp_submap_size",
    )
    reported_icp = {field: config[field] for field in icp_fields if field in config}
    icp_budget = {
        "detail_complete": len(reported_icp) == len(icp_fields),
        "reported": reported_icp,
    }
    threshold_fields = (
        "pose_translation_threshold_m",
        "pose_yaw_threshold_deg",
        "pose_roll_pitch_threshold_deg",
    )
    missing_thresholds = [field for field in threshold_fields if field not in config]
    if missing_thresholds:
        raise SyntheticGateError(
            f"producer renderer_config omits label thresholds: {missing_thresholds}"
        )
    thresholds: dict[str, float] = {}
    for field in threshold_fields:
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SyntheticGateError(f"producer renderer_config {field} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise SyntheticGateError(
                f"producer renderer_config {field} must be finite and non-negative"
            )
        if not math.isclose(
            number,
            required_thresholds[field],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SyntheticGateError(
                f"producer renderer_config {field}={number} differs from frozen "
                f"required_label_thresholds value {required_thresholds[field]}"
            )
        thresholds[field] = number
    renderer_contract = {
        "visibility_model": "map_conditioned_point_zbuffer_v1",
        "normal_visibility": "not_modeled",
        "sensor_profile": "reference_envelope_only",
        "scan_pattern_model": "not_modeled",
        "motion_distortion_model": "not_modeled",
        "query_cloud_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
    }
    for field, expected_value in renderer_contract.items():
        if config.get(field) != expected_value:
            raise SyntheticGateError(
                f"producer renderer_config {field} must be {expected_value!r}, got {config.get(field)!r}"
            )
    if require_raycast:
        for field in (
            "raycast_azimuth_resolution_deg",
            "raycast_vertical_resolution_deg",
        ):
            value = config.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SyntheticGateError(f"producer renderer_config {field} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise SyntheticGateError(
                    f"producer renderer_config {field} must be finite and positive for the Gate"
                )
    for field, expected_value in visibility_thresholds.items():
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected_value:
            raise SyntheticGateError(
                f"producer renderer_config {field} must match frozen value {expected_value}"
            )
    return query_source, candidate_budget, icp_budget, thresholds, renderer_contract


def _normalize_producer_rows(
    headers: list[str],
    raw_rows: list[dict[str, str]],
    frame_period_s: float,
    thresholds: dict[str, float],
    require_raycast: bool,
    visibility_thresholds: dict[str, int],
) -> dict[str, Any]:
    missing = sorted(PRODUCER_REQUIRED_COLUMNS - set(headers))
    if missing:
        raise SyntheticGateError(f"producer per_query.csv is missing required columns: {missing}")
    rows: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_times: list[float] = []
    for line_number, raw in enumerate(raw_rows, start=2):
        owner = f"producer per_query.csv:{line_number}"
        query_id = str(_parse_int(raw["query_id"], f"{owner}.query_id", nonnegative=True))
        episode_id = str(
            _parse_int(raw["episode_id"], f"{owner}.episode_id", nonnegative=True)
        )
        if query_id in query_ids:
            raise SyntheticGateError(f"{owner} has duplicate query_id {query_id!r}")
        query_ids.add(query_id)
        generation_valid = _parse_bool(raw["generation_valid"], f"{owner}.generation_valid")
        attempted = _parse_bool(raw["localization_attempted"], f"{owner}.localization_attempted")
        tracking = _parse_bool(raw["tracking_processed"], f"{owner}.tracking_processed")
        success = _parse_bool(raw["success"], f"{owner}.success")
        locked = _parse_bool(raw["relocalization_locked"], f"{owner}.relocalization_locked")
        locked_before = _parse_bool(raw["locked_before_frame"], f"{owner}.locked_before_frame")
        locked_after = _parse_bool(raw["locked_after_frame"], f"{owner}.locked_after_frame")
        ever_locked_before = _parse_bool(
            raw["ever_locked_before_frame"], f"{owner}.ever_locked_before_frame"
        )
        tracking_loss_event = _parse_bool(
            raw["tracking_loss_event"], f"{owner}.tracking_loss_event"
        )
        lock_event_type = raw["lock_event_type"].strip()
        if lock_event_type not in {"none", "initial_lock", "reentry_lock"}:
            raise SyntheticGateError(f"{owner}.lock_event_type is unsupported")
        pose_accurate = _parse_bool(raw["pose_accurate"], f"{owner}.pose_accurate")
        if generation_valid and not (attempted or tracking):
            raise SyntheticGateError(
                f"{owner} valid row must have localization_attempted or tracking_processed"
            )
        if attempted and tracking and not (
            lock_event_type == "reentry_lock" and locked_before and locked_after
        ):
            raise SyntheticGateError(
                f"{owner} both processing flags are only valid for an in-frame reentry_lock"
            )
        lock_event = lock_event_type != "none"
        if lock_event_type == "initial_lock" and not (
            attempted
            and not tracking
            and not locked_before
            and locked_after
            and not ever_locked_before
        ):
            raise SyntheticGateError(f"{owner} initial_lock transition flags are inconsistent")
        if lock_event_type == "reentry_lock" and not (
            attempted
            and locked_after
            and ever_locked_before
            and tracking == locked_before
        ):
            raise SyntheticGateError(f"{owner} reentry_lock transition flags are inconsistent")
        if tracking_loss_event and not (tracking and locked_before and not locked_after):
            raise SyntheticGateError(f"{owner} tracking loss transition flags are inconsistent")
        if lock_event and tracking_loss_event:
            raise SyntheticGateError(f"{owner} cannot be both a lock and tracking-loss event")
        if locked != lock_event:
            raise SyntheticGateError(
                f"{owner}.relocalization_locked must identify the explicit lock event"
            )
        if not generation_valid and any(
            (attempted, tracking, lock_event, tracking_loss_event, success)
        ):
            raise SyntheticGateError(
                f"{owner} generation-invalid row must not contain runtime processing or events"
            )
        if not generation_valid and locked_before != locked_after:
            raise SyntheticGateError(
                f"{owner} generation-invalid row must preserve the episode lock state"
            )
        localizer_ms = _parse_float(raw["localizer_elapsed_ms"], f"{owner}.localizer_elapsed_ms")
        assert localizer_ms is not None
        renderer_ms = _parse_float(raw["renderer_elapsed_ms"], f"{owner}.renderer_elapsed_ms")
        scorer_ms = _parse_float(raw["scorer_elapsed_ms"], f"{owner}.scorer_elapsed_ms")
        assert renderer_ms is not None and scorer_ms is not None
        if min(localizer_ms, renderer_ms, scorer_ms) < 0.0:
            raise SyntheticGateError(f"{owner} renderer/localizer/scorer timing must be non-negative")
        total_ms = renderer_ms + localizer_ms + scorer_ms
        if generation_valid:
            total_times.append(total_ms)
        fingerprint = raw["query_cloud_fingerprint_fnv1a64"].strip()
        if re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None:
            raise SyntheticGateError(
                f"{owner}.query_cloud_fingerprint_fnv1a64 must be 16 lowercase hex"
            )
        fingerprint_available = _parse_bool(
            raw["query_cloud_fingerprint_available"],
            f"{owner}.query_cloud_fingerprint_available",
        )
        if generation_valid and not fingerprint_available:
            raise SyntheticGateError(f"{owner} valid query must have an available cloud fingerprint")
        if raw["query_cloud_fingerprint_algorithm"].strip() != FINGERPRINT_ALGORITHM:
            raise SyntheticGateError(f"{owner} has the wrong query cloud fingerprint algorithm")
        point_fields = (
            "input_map_points",
            "finite_points",
            "range_eligible_points",
            "fov_eligible_points",
            "azimuth_bins",
            "vertical_bins",
            "occupied_ray_bins",
            "same_ray_occluded_points",
            "dropout_suppressed_points",
            "occlusion_suppressed_bins",
            "query_points_before_voxel",
            "num_query_points",
        )
        counts = {
            field: _parse_int(raw[field], f"{owner}.{field}", nonnegative=True)
            for field in point_fields
        }
        if not (
            counts["input_map_points"]
            >= counts["finite_points"]
            >= counts["range_eligible_points"]
            >= counts["fov_eligible_points"]
        ):
            raise SyntheticGateError(
                f"{owner} violates input >= finite >= range >= FOV visibility counts"
            )
        if counts["query_points_before_voxel"] < counts["num_query_points"]:
            raise SyntheticGateError(f"{owner} voxel output exceeds pre-voxel query points")
        raycast_enabled = _parse_bool(raw["raycast_enabled"], f"{owner}.raycast_enabled")
        if require_raycast and generation_valid and not raycast_enabled:
            raise SyntheticGateError(f"{owner} valid Gate query must have raycast_enabled=true")
        if generation_valid:
            for field, threshold in (
                ("num_query_points", visibility_thresholds["minimum_query_points"]),
                (
                    "occupied_ray_bins",
                    visibility_thresholds["minimum_occupied_ray_bins"],
                ),
            ):
                if counts[field] < threshold:
                    raise SyntheticGateError(
                        f"{owner} generation-valid {field}={counts[field]} is below frozen "
                        f"minimum {threshold}"
                    )
        if raycast_enabled:
            if counts["azimuth_bins"] <= 0 or counts["vertical_bins"] <= 0:
                raise SyntheticGateError(f"{owner} raycast bin dimensions must be positive")
            if counts["occupied_ray_bins"] > counts["azimuth_bins"] * counts["vertical_bins"]:
                raise SyntheticGateError(f"{owner} occupied rays exceed the ray grid")
            if counts["occupied_ray_bins"] != (
                counts["fov_eligible_points"] - counts["same_ray_occluded_points"]
            ):
                raise SyntheticGateError(f"{owner} ray occupancy count is not conserved")
            expected_prevoxel = (
                counts["occupied_ray_bins"]
                - counts["dropout_suppressed_points"]
                - counts["occlusion_suppressed_bins"]
            )
        else:
            if any(
                counts[field] != 0
                for field in (
                    "azimuth_bins",
                    "vertical_bins",
                    "occupied_ray_bins",
                    "same_ray_occluded_points",
                    "occlusion_suppressed_bins",
                )
            ):
                raise SyntheticGateError(f"{owner} no-raycast counters are inconsistent")
            expected_prevoxel = (
                counts["fov_eligible_points"] - counts["dropout_suppressed_points"]
            )
        if expected_prevoxel < 0 or counts["query_points_before_voxel"] != expected_prevoxel:
            raise SyntheticGateError(f"{owner} pre-voxel visibility count is not conserved")
        row: dict[str, Any] = {
            "query_id": query_id,
            "episode_id": episode_id,
            "frame_index": _parse_int(raw["frame_index"], f"{owner}.frame_index", nonnegative=True),
            "reference_keyframe_id": _parse_int(
                raw["reference_keyframe_id"], f"{owner}.reference_keyframe_id"
            ),
            "renderer_seed": _parse_int(raw["renderer_seed"], f"{owner}.renderer_seed", nonnegative=True),
            "generation_valid": generation_valid,
            "generation_reason": raw["generation_reason"].strip(),
            "localization_attempted": attempted,
            "tracking_processed": tracking,
            "locked_before_frame": locked_before,
            "locked_after_frame": locked_after,
            "ever_locked_before_frame": ever_locked_before,
            "lock_event_type": lock_event_type,
            "tracking_loss_event": tracking_loss_event,
            "runtime_stage": raw["runtime_stage"].strip(),
            "success": success,
            "lock_event": lock_event,
            "producer_relocalization_locked": locked,
            "matched_keyframe_id": _parse_int(
                raw["matched_keyframe_id"], f"{owner}.matched_keyframe_id"
            ),
            "pose_accurate": pose_accurate,
            "translation_error_m": _parse_float(
                raw["translation_error_m"], f"{owner}.translation_error_m", nullable=True
            ),
            "yaw_error_deg": _parse_float(
                raw["yaw_error_deg"], f"{owner}.yaw_error_deg", nullable=True
            ),
            "roll_pitch_error_deg": _parse_float(
                raw["roll_pitch_error_deg"], f"{owner}.roll_pitch_error_deg", nullable=True
            ),
            "localizer_ms": localizer_ms,
            "renderer_ms": renderer_ms,
            "scorer_ms": scorer_ms,
            "total_ms": total_ms,
            "query_cloud_fingerprint_fnv1a64": fingerprint,
            "query_cloud_fingerprint_available": fingerprint_available,
            "query_cloud_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "query_x_m": _parse_float(raw["query_x_m"], f"{owner}.query_x_m"),
            "query_y_m": _parse_float(raw["query_y_m"], f"{owner}.query_y_m"),
            "query_z_m": _parse_float(raw["query_z_m"], f"{owner}.query_z_m"),
            "query_roll_deg": _parse_float(raw["query_roll_deg"], f"{owner}.query_roll_deg"),
            "query_pitch_deg": _parse_float(raw["query_pitch_deg"], f"{owner}.query_pitch_deg"),
            "query_yaw_deg": _parse_float(raw["query_yaw_deg"], f"{owner}.query_yaw_deg"),
            "raycast_enabled": raycast_enabled,
            "envelope_valid": _parse_bool(raw["envelope_valid"], f"{owner}.envelope_valid"),
            "envelope_azimuth_full": _parse_bool(
                raw["envelope_azimuth_full"], f"{owner}.envelope_azimuth_full"
            ),
            "envelope_azimuth_start_deg": _parse_float(
                raw["envelope_azimuth_start_deg"], f"{owner}.envelope_azimuth_start_deg"
            ),
            "envelope_azimuth_span_deg": _parse_float(
                raw["envelope_azimuth_span_deg"], f"{owner}.envelope_azimuth_span_deg"
            ),
            "envelope_vertical_min_deg": _parse_float(
                raw["envelope_vertical_min_deg"], f"{owner}.envelope_vertical_min_deg"
            ),
            "envelope_vertical_max_deg": _parse_float(
                raw["envelope_vertical_max_deg"], f"{owner}.envelope_vertical_max_deg"
            ),
            **counts,
        }
        if lock_event and any(
            row[field] is None
            for field in ("translation_error_m", "yaw_error_deg", "roll_pitch_error_deg")
        ):
            raise SyntheticGateError(f"{owner} lock event is missing offline pose errors")
        expected_pose_accurate = bool(
            lock_event
            and row["translation_error_m"] <= thresholds["pose_translation_threshold_m"]
            and row["yaw_error_deg"] <= thresholds["pose_yaw_threshold_deg"]
            and row["roll_pitch_error_deg"] <= thresholds["pose_roll_pitch_threshold_deg"]
        )
        if pose_accurate != expected_pose_accurate:
            raise SyntheticGateError(
                f"{owner}.pose_accurate disagrees with frozen label thresholds"
            )
        rows.append(row)
        episodes[episode_id].append(row)

    for episode_id, episode_rows in episodes.items():
        invoked_rows = sorted(
            (row for row in episode_rows if row["generation_valid"]),
            key=lambda row: row["frame_index"],
        )
        previous_locked_after = False
        ever_locked = False
        for index, row in enumerate(invoked_rows):
            if row["locked_before_frame"] != previous_locked_after:
                raise SyntheticGateError(
                    f"episode {episode_id} frame {row['frame_index']} locked_before_frame "
                    "does not continue the previous invoked frame"
                )
            if row["ever_locked_before_frame"] != ever_locked:
                raise SyntheticGateError(
                    f"episode {episode_id} frame {row['frame_index']} "
                    "ever_locked_before_frame disagrees with prior lock history"
                )
            if index == 0 and (row["locked_before_frame"] or row["ever_locked_before_frame"]):
                raise SyntheticGateError(
                    f"episode {episode_id} first invoked frame must start unlocked"
                )
            if (
                row["tracking_processed"]
                and row["locked_before_frame"]
                and not row["locked_after_frame"]
                and not row["tracking_loss_event"]
            ):
                raise SyntheticGateError(
                    f"episode {episode_id} frame {row['frame_index']} lost tracking "
                    "without tracking_loss_event=true"
                )
            ordinary_tracking = row["tracking_processed"] and not row["localization_attempted"]
            if ordinary_tracking and not row["tracking_loss_event"] and not (
                row["locked_before_frame"]
                and row["locked_after_frame"]
                and row["lock_event_type"] == "none"
            ):
                raise SyntheticGateError(
                    f"episode {episode_id} frame {row['frame_index']} ordinary tracking "
                    "transition flags are inconsistent"
                )
            if row["lock_event"]:
                ever_locked = True
            previous_locked_after = row["locked_after_frame"]

    attempt_rows: list[dict[str, Any]] = []
    correct_attempts = 0
    false_attempts = 0
    timeout_attempts = 0
    correct_events = 0
    false_events = 0
    correct_by_deadline = 0
    for episode_id, episode_rows in episodes.items():
        valid_rows = [row for row in episode_rows if row["generation_valid"]]
        if not valid_rows:
            continue
        attempted_rows = [row for row in valid_rows if row["localization_attempted"]]
        if not attempted_rows:
            continue
        locks = [row for row in attempted_rows if row["lock_event"]]
        false_locks = [row for row in locks if not row["pose_accurate"]]
        correct_locks = [row for row in locks if row["pose_accurate"]]
        timely_correct_locks = [
            row
            for row in correct_locks
            if (row["frame_index"] + 1) * frame_period_s <= 10.0
        ]
        false_events += len(false_locks)
        correct_events += len(correct_locks)
        if false_locks:
            attempt_outcome = "false"
            false_attempts += 1
        elif timely_correct_locks:
            attempt_outcome = "correct"
            correct_attempts += 1
            correct_by_deadline += 1
        else:
            attempt_outcome = "timeout"
            timeout_attempts += 1
        attempt_rows.append(
            {
                "attempt_id": episode_id,
                "attempt_outcome": attempt_outcome,
                "ever_false_lock": bool(false_locks),
            }
        )

    query_rows: list[dict[str, Any]] = []
    runtime_events: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for episode_id, episode_rows in episodes.items():
        valid_rows = [row for row in episode_rows if row["generation_valid"]]
        for row in valid_rows:
            if row["lock_event"]:
                outcome = "CORRECT_FULL_LOCK" if row["pose_accurate"] else "FALSE_FULL_LOCK"
            else:
                outcome = "TEMPORAL_DEFER"
            if row["localization_attempted"]:
                if outcome not in QUERY_OUTCOMES:
                    raise SyntheticGateError(f"wrapper produced unsupported query outcome {outcome}")
                query_rows.append(
                    {
                        "query_id": row["query_id"],
                        "attempt_id": episode_id,
                        "query_outcome": outcome,
                        "correct_lock_within_10s": bool(
                            row["lock_event"]
                            and row["pose_accurate"]
                            and (row["frame_index"] + 1) * frame_period_s <= 10.0
                        ),
                        "translation_error_m": row["translation_error_m"],
                        "yaw_error_deg": row["yaw_error_deg"],
                        "roll_pitch_error_deg": row["roll_pitch_error_deg"],
                    }
                )
            runtime_events.append(
                {
                    "query_id": row["query_id"],
                    "attempt_id": episode_id,
                    "runtime_stage": row["runtime_stage"],
                    "runtime_reason": (
                        row["lock_event_type"]
                        if row["lock_event"]
                        else (
                            "tracking_loss"
                            if row["tracking_loss_event"]
                            else ("tracking" if row["tracking_processed"] else "no_lock")
                        )
                    ),
                    "success": row["success"],
                    "lock_event": row["lock_event"],
                    "lock_event_type": row["lock_event_type"],
                    "tracking_loss_event": row["tracking_loss_event"],
                    "locked_before_frame": row["locked_before_frame"],
                    "locked_after_frame": row["locked_after_frame"],
                    "ever_locked_before_frame": row["ever_locked_before_frame"],
                    "matched_keyframe_id": row["matched_keyframe_id"],
                }
            )
            timing_rows.append(
                {
                    "query_id": row["query_id"],
                    "renderer_ms": row["renderer_ms"],
                    "localizer_ms": row["localizer_ms"],
                    "scorer_ms": row["scorer_ms"],
                    "total_ms": row["total_ms"],
                }
            )

    query_counts = Counter(row["query_outcome"] for row in query_rows)
    attempt_count = len(attempt_rows)
    lock_event_count = correct_events + false_events
    metrics = {
        "attempt_count": attempt_count,
        "correct_attempt_count": correct_attempts,
        "false_attempt_count": false_attempts,
        "timeout_attempt_count": timeout_attempts,
        "eligible_query_count": len(query_rows),
        "query_outcome_counts": dict(sorted(query_counts.items())),
        "correct_lock_event_count": correct_events,
        "false_lock_event_count": false_events,
        "correct_lock_by_10s_count": correct_by_deadline,
        "correct_lock_by_10s": correct_by_deadline / attempt_count if attempt_count else None,
        "false_lock_attempt_rate": false_attempts / attempt_count if attempt_count else None,
        "lock_precision": correct_events / lock_event_count if lock_event_count else None,
        "false_lock_given_lock": false_events / lock_event_count if lock_event_count else None,
        "p95_total_ms": _percentile(total_times, 0.95),
        "planned_render_query_count": len(rows),
        "valid_render_query_count": sum(1 for row in rows if row["generation_valid"]),
        "invalid_render_query_count": sum(1 for row in rows if not row["generation_valid"]),
        "tracking_processed_query_count": sum(1 for row in rows if row["tracking_processed"]),
    }
    return {
        "rows": rows,
        "query_rows": query_rows,
        "attempt_rows": attempt_rows,
        "runtime_events": runtime_events,
        "timing_rows": timing_rows,
        "metrics": metrics,
        "query_identity_capability": "per_query_fnv1a64",
    }


def _write_mapped_artifacts(
    staging: Path,
    normalized: dict[str, Any],
    query_source: str,
    frame_period_s: float,
    renderer_contract: dict[str, str],
) -> dict[str, Any]:
    rows = normalized["rows"]
    visibility_fields = [
        "query_id",
        "episode_id",
        "frame_index",
        "reference_keyframe_id",
        "renderer_seed",
        "generation_valid",
        "generation_reason",
        "query_cloud_fingerprint_available",
        "query_cloud_fingerprint_algorithm",
        "query_cloud_fingerprint_fnv1a64",
        "input_map_points",
        "finite_points",
        "range_eligible_points",
        "fov_eligible_points",
        "azimuth_bins",
        "vertical_bins",
        "occupied_ray_bins",
        "same_ray_occluded_points",
        "dropout_suppressed_points",
        "occlusion_suppressed_bins",
        "query_points_before_voxel",
        "num_query_points",
        "raycast_enabled",
        "envelope_valid",
        "envelope_azimuth_full",
        "envelope_azimuth_start_deg",
        "envelope_azimuth_span_deg",
        "envelope_vertical_min_deg",
        "envelope_vertical_max_deg",
    ]
    _write_csv(
        staging / "raw" / "renderer_visibility.csv",
        visibility_fields,
        [{field: row.get(field) for field in visibility_fields} for row in rows],
    )
    query_fields = [
        "query_id",
        "attempt_id",
        "query_outcome",
        "correct_lock_within_10s",
        "translation_error_m",
        "yaw_error_deg",
        "roll_pitch_error_deg",
    ]
    _write_csv(staging / "raw" / "relocalization_queries.csv", query_fields, normalized["query_rows"])
    _write_csv(
        staging / "raw" / "relocalization_attempts.csv",
        ["attempt_id", "attempt_outcome", "ever_false_lock"],
        normalized["attempt_rows"],
    )
    _write_csv(
        staging / "raw" / "stage_timing.csv",
        ["query_id", "renderer_ms", "localizer_ms", "scorer_ms", "total_ms"],
        normalized["timing_rows"],
    )
    runtime_path = staging / "raw" / "runtime_events.jsonl"
    _ensure_new_file(runtime_path)
    with runtime_path.open("w", encoding="utf-8") as stream:
        for event in normalized["runtime_events"]:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    metrics = {
        "schema_version": 2,
        "mode": "relocalization",
        "odom_source": "oracle_gt_odom",
        "query_source": query_source,
        "evidence_class": "synthetic_map_render",
        "gt_runtime_access": True,
        "verdict_ceiling": "SHADOW_ONLY",
        "synthetic_from_target_map": True,
        "synthetic_frame_period_s": frame_period_s,
        **normalized["metrics"],
    }
    _write_json(staging / "summary" / "metrics.json", metrics)
    _write_json(
        staging / "summary" / "failure_counts.json",
        {
            "schema_version": 2,
            "query_outcome_counts": metrics["query_outcome_counts"],
        },
    )
    sums: dict[str, int] = {}
    for field in (
        "input_map_points",
        "finite_points",
        "range_eligible_points",
        "fov_eligible_points",
        "azimuth_bins",
        "vertical_bins",
        "occupied_ray_bins",
        "same_ray_occluded_points",
        "dropout_suppressed_points",
        "occlusion_suppressed_bins",
        "query_points_before_voxel",
        "num_query_points",
    ):
        sums[field] = sum(int(row[field]) for row in rows)
    renderer_metrics = {
        "schema_version": 2,
        "planned_render_query_count": len(rows),
        "valid_render_query_count": sum(1 for row in rows if row["generation_valid"]),
        "invalid_render_query_count": sum(1 for row in rows if not row["generation_valid"]),
        "eligible_query_count": len(normalized["query_rows"]),
        "tracking_processed_query_count": sum(
            1 for row in rows if row["tracking_processed"]
        ),
        "stage_count_sums": sums,
        "query_identity_capability": normalized["query_identity_capability"],
        "determinism_claim": "per_query_fnv1a64_available",
        "renderer_timing_capability": "reported",
        "localizer_timing_capability": "reported",
        "scorer_timing_capability": "reported",
        "algorithm_observation": (
            "FALSIFIER_FALSE_LOCK"
            if metrics["false_attempt_count"] > 0
            else (
                "FALSIFIER_NO_CORRECT_LOCK"
                if metrics["correct_attempt_count"] == 0
                else "CORRECT_LOCK_OBSERVED"
            )
        ),
        **renderer_contract,
    }
    _write_json(staging / "summary" / "renderer_metrics.json", renderer_metrics)
    return {"metrics": metrics, "renderer_metrics": renderer_metrics}


def _write_checksums(staging: Path, required: list[str]) -> None:
    lines: list[str] = []
    for relative in sorted(set(required) - {"checksums.sha256", "COMPLETE"}):
        path = staging / relative
        if not path.is_file() or path.is_symlink():
            raise SyntheticGateError(f"required artifact is missing before checksums: {relative}")
        lines.append(f"{sha256_file(path)}  {relative}")
    _write_text(staging / "checksums.sha256", "\n".join(lines) + "\n")


def _create_complete_atomically(staging: Path) -> None:
    complete = staging / "COMPLETE"
    if complete.exists():
        raise SyntheticGateError("COMPLETE already exists before payload preflight")
    temporary = staging / f".COMPLETE.tmp-{uuid.uuid4().hex}"
    payload = json.dumps(
        {
            "schema_version": 2,
            "status": "complete",
            "verdict_ceiling": "SHADOW_ONLY",
        },
        sort_keys=True,
    ) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, complete)


def _failure_report(staging: Path, message: str, start_time: str) -> None:
    path = staging / "WRAPPER_FAILURE.json"
    if path.exists():
        return
    _write_json(
        path,
        {
            "schema_version": 2,
            "wrapper_version": WRAPPER_VERSION,
            "status": "INVALID",
            "error": message,
            "start_time_asia_shanghai": start_time,
            "failure_time_asia_shanghai": _timestamp(),
            "staging_dir": str(staging),
        },
    )


def run_synthetic_gate(
    *,
    producer: Path | str,
    output: Path | str,
    map_path: Path | str,
    query_pose_manifest: Path | str,
    frozen_contract: Path | str,
    dataset_id: str,
    map_provenance: str,
    repo: Path | str,
    producer_args: list[str],
    split: str = "p0s_shadow",
    frame_period_s: float = 1.0,
    build_type: str = "unknown",
) -> dict[str, Any]:
    producer_path = Path(producer).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    map_file = Path(map_path).expanduser().resolve()
    pose_file = Path(query_pose_manifest).expanduser().resolve()
    contract_file = Path(frozen_contract).expanduser().resolve()
    repo_path = Path(repo).expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise SyntheticGateError(f"output already exists; refusing to overwrite: {output_path}")
    if not dataset_id.strip() or not map_provenance.strip() or not split.strip():
        raise SyntheticGateError("dataset_id, map_provenance, and split must be non-empty")
    if (
        not math.isfinite(frame_period_s)
        or not math.isclose(
            frame_period_s,
            FIXED_FRAME_PERIOD_S,
            rel_tol=0.0,
            abs_tol=FRAME_PERIOD_TOLERANCE,
        )
    ):
        raise SyntheticGateError(
            "frame_period_s must equal 1.0 because the producer timestamps frames at fixed 1s steps"
        )
    for name, path in (
        ("producer", producer_path),
        ("map", map_file),
        ("query pose manifest", pose_file),
        ("frozen contract", contract_file),
    ):
        if not path.is_file() or path.is_symlink():
            raise SyntheticGateError(f"{name} must be a regular non-symlink file: {path}")
    if not os.access(producer_path, os.X_OK):
        raise SyntheticGateError(f"producer is not executable: {producer_path}")
    if not repo_path.is_dir():
        raise SyntheticGateError(f"repo directory does not exist: {repo_path}")
    (
        _,
        required,
        require_query_fingerprint,
        required_eval_profile,
        required_thresholds,
        require_raycast,
        required_temporal_window,
        visibility_thresholds,
    ) = _validate_contract(contract_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    start_time = _timestamp()
    try:
        producer_output = staging / "raw"
        command = _producer_command(
            producer_path,
            map_file,
            pose_file,
            producer_output,
            list(producer_args),
        )
        command_text = shlex.join(command)
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        _write_text(staging / "stdout.log", process.stdout)
        _write_text(staging / "raw" / "producer_stderr.log", process.stderr)
        _write_text(staging / "command.txt", command_text + "\n")
        if process.returncode != 0:
            raise SyntheticGateError(
                f"producer returned {process.returncode}; see stdout.log and raw/producer_stderr.log",
                staging,
            )
        for source_name, target_relative in PRODUCER_REQUIRED_FILES.items():
            _move_producer_file(producer_output / source_name, staging / target_relative)

        config_path = staging / "raw" / "producer_renderer_config.json"
        config = load_json_strict(config_path)
        if not isinstance(config, dict):
            raise SyntheticGateError("producer renderer_config.json must contain an object")
        (
            query_source,
            candidate_budget,
            icp_budget,
            thresholds,
            renderer_contract,
        ) = _require_producer_config(
            config,
            required_eval_profile,
            required_thresholds,
            require_raycast,
            required_temporal_window,
            visibility_thresholds,
        )
        headers, raw_rows = _read_csv(staging / "raw" / "producer_per_query.source")
        if not raw_rows:
            raise SyntheticGateError("producer per_query.csv contains no query rows")
        normalized = _normalize_producer_rows(
            headers,
            raw_rows,
            frame_period_s,
            thresholds,
            require_raycast,
            visibility_thresholds,
        )
        resolved_records = _resolved_query_records(
            staging / "raw" / "producer_resolved_queries.source",
            require_query_fingerprint,
        )
        pose_records = _query_pose_manifest_records(pose_file)
        _verify_query_identity(
            pose_records,
            normalized["rows"],
            resolved_records,
            require_query_fingerprint,
        )

        _copy_file(pose_file, staging / "labels" / "query_pose_manifest.csv")
        _copy_file(contract_file, staging / "frozen_contract.yaml")
        _write_json(
            staging / "resolved_config.yaml",
            {
                **config,
                "synthetic_frame_period_s": frame_period_s,
                "required_temporal_window_size": required_temporal_window,
                **visibility_thresholds,
                "wrapper_version": WRAPPER_VERSION,
            },
        )
        _copy_file(
            staging / "raw" / "producer_resolved_queries.source",
            staging / "raw" / "resolved_queries.csv",
        )
        mapped = _write_mapped_artifacts(
            staging,
            normalized,
            query_source,
            frame_period_s,
            renderer_contract,
        )

        environment = _environment()
        _write_json(staging / "environment.json", environment)
        map_hash = sha256_file(map_file)
        pose_hash = sha256_file(pose_file)
        config_hash = sha256_file(staging / "resolved_config.yaml")
        contract_hash = sha256_file(staging / "frozen_contract.yaml")
        dataset_manifest = {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "split": split,
            "evidence_class": "synthetic_map_render",
            "odom_source": "oracle_gt_odom",
            "gt_runtime_access": True,
            "map_path": str(map_file),
            "map_sha256": map_hash,
            "render_map_sha256": map_hash,
            "target_map_sha256": map_hash,
            "map_provenance": map_provenance,
            "query_pose_manifest_path": str(pose_file),
            "query_pose_manifest_sha256": pose_hash,
            "query_source": query_source,
            "synthetic_from_target_map": True,
            "required_eval_profile": required_eval_profile,
            "required_temporal_window_size": required_temporal_window,
            "visibility_thresholds": visibility_thresholds,
            "synthetic_frame_period_s": frame_period_s,
        }
        _write_json(staging / "dataset_manifest.json", dataset_manifest)

        git = _git_metadata(repo_path)
        end_time = _timestamp()
        manifest = {
            "schema_version": 2,
            "run_id": output_path.name,
            "mode": "relocalization",
            **git,
            "start_time_asia_shanghai": start_time,
            "end_time_asia_shanghai": end_time,
            "command": command_text,
            "exact_argv": command,
            "exit_code": process.returncode,
            "dataset_id": dataset_id,
            "split": split,
            "odom_id": "synthetic_oracle_from_pose_manifest",
            "gt_id": f"sha256:{pose_hash}",
            "odom_source": "oracle_gt_odom",
            "query_source": query_source,
            "gt_runtime_access": True,
            "evidence_class": "synthetic_map_render",
            "synthetic_from_target_map": True,
            "verdict_ceiling": "SHADOW_ONLY",
            "candidate_budget": candidate_budget,
            "icp_budget": icp_budget,
            "cpu_model": environment["cpu_model"],
            "thread_count": environment["thread_count"],
            "memory_max_bytes": environment["memory_max_bytes"],
            "memory_swap_max_bytes": environment["memory_swap_max_bytes"],
            "evaluator_version": str(config.get("schema_version")),
            "analyzer_version": WRAPPER_VERSION,
            "label_thresholds": thresholds,
            "feature_flags": {
                "synthetic_gate_wrapper": True,
                "odom_derived_from_gt": True,
                "synthetic_from_target_map": True,
                "gt_passed_to_localizer": False,
                "query_identity_capability": normalized["query_identity_capability"],
                "contract_requires_query_fingerprint": require_query_fingerprint,
                "renderer_timing_reported": True,
                "localizer_timing_reported": True,
                "scorer_timing_reported": True,
                "raycast_visibility_required": require_raycast,
                "eval_profile": required_eval_profile,
            },
            "resolved_config_sha256": config_hash,
            "dataset_manifest_sha256": sha256_file(staging / "dataset_manifest.json"),
            "frozen_contract_sha256": contract_hash,
            "required_artifacts": required,
            "complete": True,
            "producer_path": str(producer_path),
            "producer_sha256": sha256_file(producer_path),
            "map_path": str(map_file),
            "map_sha256": map_hash,
            "render_map_sha256": map_hash,
            "target_map_sha256": map_hash,
            "map_provenance": map_provenance,
            "query_pose_manifest_path": str(pose_file),
            "query_pose_manifest_sha256": pose_hash,
            "renderer_config_sha256": config_hash,
            "producer_renderer_config_sha256": sha256_file(config_path),
            "build_type": build_type,
            "visibility_model": mapped["renderer_metrics"]["visibility_model"],
            "sensor_profile": "reference_envelope_only",
            "scan_pattern_model": "not_modeled",
            "motion_distortion_model": "not_modeled",
            "query_identity_capability": normalized["query_identity_capability"],
            "determinism_claim": mapped["renderer_metrics"]["determinism_claim"],
            "synthetic_frame_period_s": frame_period_s,
            "required_eval_profile": required_eval_profile,
            "required_temporal_window_size": required_temporal_window,
            "visibility_thresholds": visibility_thresholds,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging, required)

        payload_report = validate_run(staging, strict=True, pre_complete=True).to_dict()
        if not payload_report["valid"]:
            raise SyntheticGateError(
                f"strict payload preflight failed: {payload_report['errors']}",
                staging,
            )
        _create_complete_atomically(staging)
        complete_report = validate_run(staging, strict=True).to_dict()
        if not complete_report["valid"]:
            raise SyntheticGateError(
                f"normal strict validation failed after COMPLETE: {complete_report['errors']}",
                staging,
            )
        if output_path.exists() or output_path.is_symlink():
            raise SyntheticGateError(
                f"output appeared during execution; refusing to overwrite: {output_path}",
                staging,
            )
        os.rename(staging, output_path)
        final_report = validate_run(output_path, strict=True).to_dict()
        if not final_report["valid"]:
            raise SyntheticGateError(
                f"published run failed final path validation: {final_report['errors']}",
                output_path,
            )
        return {
            "schema_version": 2,
            "valid": True,
            "verdict": "SHADOW_ONLY",
            "run_dir": str(output_path),
            "payload_preflight": payload_report,
            "strict_validation": final_report,
            "query_identity_capability": normalized["query_identity_capability"],
            "determinism_claim": mapped["renderer_metrics"]["determinism_claim"],
        }
    except Exception as exc:
        if staging.exists():
            _failure_report(staging, str(exc), start_time)
        if isinstance(exc, SyntheticGateError):
            if exc.staging_dir is None:
                exc.staging_dir = staging
            raise
        raise SyntheticGateError(str(exc), staging) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--query-pose-manifest", required=True, type=Path)
    parser.add_argument("--frozen-contract", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--map-provenance", required=True)
    parser.add_argument("--split", default="p0s_shadow")
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Explicit Git repository recorded in manifest",
    )
    parser.add_argument("--frame-period-s", type=float, default=1.0)
    parser.add_argument("--build-type", default="unknown")
    parser.add_argument(
        "producer_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded verbatim to the C++ producer",
    )
    args = parser.parse_args(argv)
    producer_args = list(args.producer_args)
    if producer_args and producer_args[0] == "--":
        producer_args = producer_args[1:]
    try:
        report = run_synthetic_gate(
            producer=args.producer,
            output=args.output,
            map_path=args.map,
            query_pose_manifest=args.query_pose_manifest,
            frozen_contract=args.frozen_contract,
            dataset_id=args.dataset_id,
            map_provenance=args.map_provenance,
            repo=args.repo,
            producer_args=producer_args,
            split=args.split,
            frame_period_s=args.frame_period_s,
            build_type=args.build_type,
        )
    except SyntheticGateError as exc:
        report = {
            "schema_version": 2,
            "valid": False,
            "verdict": "INVALID",
            "error": str(exc),
            "staging_dir": str(exc.staging_dir) if exc.staging_dir else None,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
