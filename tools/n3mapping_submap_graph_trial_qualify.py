#!/usr/bin/env python3
"""Fail-closed qualification for shadow submap-graph trial JSONL.

``QUALIFIED_FOR_REVIEW`` means that the latest record is a complete final
multi-submap checkpoint from the expected verified build.  It does not mean
that the shadow graph is physically correct or authorized for writeback.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


INPUT_SCHEMA = "n3mapping_submap_graph_trial_v1"
REPORT_SCHEMA = "n3mapping_submap_graph_trial_qualification_v1"
QUALIFIED = "QUALIFIED_FOR_REVIEW"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {QUALIFIED: 0, INSUFFICIENT: 2, INVALID: 3}
FINAL_CONTEXTS = {"save_map", "save_extended_map"}
RUNTIME_CONTEXTS = {
    "mapping": {
        ("core", "loop_commit"),
        ("core", "save_map"),
    },
    "map_extension": {
        ("mapping_resuming", "cross_session_loop"),
        ("mapping_resuming", "save_extended_map"),
        # The ROS wrappers save the shared resumed session through N3MappingCore.
        ("core", "save_map"),
    },
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ABS_TOL = 1e-9
REL_TOL = 1e-8
MAX_ERRORS = 100

COUNTS = (
    "snapshot_node_count", "snapshot_owned_keyframe_count",
    "snapshot_source_edge_count", "snapshot_intra_edge_count",
    "snapshot_cross_edge_count", "snapshot_unassigned_keyframe_count",
    "snapshot_unassigned_edge_count", "snapshot_floor_count",
    "snapshot_assigned_floor_count", "snapshot_unassigned_floor_count",
    "node_count", "gauge_anchor_count", "active_edge_factor_count",
    "intra_submap_constant_edge_count", "full_6d_factor_count",
    "xy_yaw_lifted_factor_count", "robust_factor_count",
    "session_odometry_factor_count", "explicit_information_factor_count",
    "fallback_noise_factor_count", "floor_factor_count",
    "initial_keyframe_reference_count", "optimized_keyframe_reference_count",
)
NONNEGATIVE = (
    "snapshot_mean_cross_translation_residual_m",
    "snapshot_max_cross_translation_residual_m",
    "snapshot_mean_cross_rotation_residual_rad",
    "snapshot_max_cross_rotation_residual_rad",
    "max_translation_delta_m", "max_rotation_delta_rad",
    "initial_keyframe_reference_mean_translation_error_m",
    "initial_keyframe_reference_p95_translation_error_m",
    "initial_keyframe_reference_max_translation_error_m",
    "initial_keyframe_reference_mean_rotation_error_rad",
    "initial_keyframe_reference_p95_rotation_error_rad",
    "initial_keyframe_reference_max_rotation_error_rad",
    "optimized_keyframe_reference_mean_translation_error_m",
    "optimized_keyframe_reference_p95_translation_error_m",
    "optimized_keyframe_reference_max_translation_error_m",
    "optimized_keyframe_reference_mean_rotation_error_rad",
    "optimized_keyframe_reference_p95_rotation_error_rad",
    "optimized_keyframe_reference_max_rotation_error_rad",
)
STRINGS = (
    "schema", "record_type", "runtime_source", "context", "mode",
    "product_commit", "product_profile_sha256", "product_build_type",
    "product_research_tools", "snapshot_failure_reason", "failure_reason",
)
BOOLEANS = (
    "product_verified", "no_writeback", "snapshot_valid",
    "valid", "attempted", "solved",
)
OBJECTIVES = (
    "initial_nonlinear_error", "final_nonlinear_error",
    "nonlinear_error_reduction",
)
REQUIRED = {
    *STRINGS,
    *BOOLEANS,
    *COUNTS,
    *NONNEGATIVE,
    *OBJECTIVES,
    "nodes",
    "keyframes",
}


class DuplicateKeyError(ValueError):
    pass


class Issues:
    def __init__(self) -> None:
        self.count = 0
        self.items: list[dict[str, Any]] = []

    def add(self, code: str, message: str, line: int | None = None) -> None:
        self.count += 1
        if len(self.items) >= MAX_ERRORS:
            return
        issue: dict[str, Any] = {"code": code, "message": message}
        if line is not None:
            issue["line"] = line
        self.items.append(issue)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def pose(
    value: Any, owner: str, issues: Issues, line: int
) -> tuple[list[float], list[float]] | None:
    if not isinstance(value, dict):
        issues.add("invalid_pose", f"{owner} must be an object", line)
        return None
    translation = value.get("translation")
    quaternion = value.get("quaternion_xyzw")
    if not isinstance(translation, list) or len(translation) != 3:
        issues.add("invalid_pose", f"{owner}.translation must have length 3", line)
        return None
    if not isinstance(quaternion, list) or len(quaternion) != 4:
        issues.add("invalid_pose", f"{owner}.quaternion_xyzw must have length 4", line)
        return None
    if not all(finite(item) for item in translation + quaternion):
        issues.add("invalid_pose", f"{owner} contains non-finite values", line)
        return None
    translation = [float(item) for item in translation]
    quaternion = [float(item) for item in quaternion]
    norm = math.sqrt(sum(item * item for item in quaternion))
    if abs(norm - 1.0) > 1e-6:
        issues.add("invalid_pose", f"{owner} quaternion is not normalized", line)
        return None
    return translation, [item / norm for item in quaternion]


def pose_error(
    reference: tuple[list[float], list[float]],
    candidate: tuple[list[float], list[float]],
) -> tuple[float, float]:
    translation = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(reference[0], candidate[0]))
    )
    rx, ry, rz, rw = reference[1]
    cx, cy, cz, cw = candidate[1]
    relative_xyz = (
        rw * cx - rx * cw - ry * cz + rz * cy,
        rw * cy + rx * cz - ry * cw - rz * cx,
        rw * cz - rx * cy + ry * cx - rz * cw,
    )
    relative_w = rw * cw + rx * cx + ry * cy + rz * cz
    rotation = 2.0 * math.atan2(
        math.sqrt(sum(value * value for value in relative_xyz)),
        abs(relative_w),
    )
    return translation, rotation


def check_metric(
    observed: Any,
    expected: float,
    name: str,
    issues: Issues,
    line: int,
) -> None:
    if not finite(observed) or not close(float(observed), expected):
        issues.add(
            "inconsistent_metric",
            f"{name}={observed!r}, recomputed={expected:.17g}",
            line,
        )


def check_stats(
    record: dict[str, Any],
    prefix: str,
    translations: list[float],
    rotations: list[float],
    issues: Issues,
    line: int,
) -> None:
    if record[f"{prefix}_count"] != len(translations):
        issues.add("inconsistent_count", f"{prefix}_count is inconsistent", line)
    for dimension, values in (
        ("translation_error_m", translations),
        ("rotation_error_rad", rotations),
    ):
        expected = (
            sum(values) / len(values) if values else 0.0,
            percentile95(values),
            max(values, default=0.0),
        )
        for statistic, result in zip(("mean", "p95", "max"), expected):
            field = f"{prefix}_{statistic}_{dimension}"
            check_metric(record[field], result, field, issues, line)


def record_summary(record: Any, line: int) -> dict[str, Any]:
    get = record.get if isinstance(record, dict) else lambda unused: None
    return {
        "line": line,
        "context": get("context"),
        "mode": get("mode"),
        "snapshot_valid": get("snapshot_valid"),
        "trial_valid": get("valid"),
        "attempted": get("attempted"),
        "solved": get("solved"),
        "failure_reason": get("failure_reason"),
        "node_count": get("node_count"),
        "owned_keyframe_count": get("snapshot_owned_keyframe_count"),
        "cross_edge_count": get("snapshot_cross_edge_count"),
        "initial_nonlinear_error": get("initial_nonlinear_error"),
        "final_nonlinear_error": get("final_nonlinear_error"),
        "initial_keyframe_reference_p95_translation_error_m": get(
            "initial_keyframe_reference_p95_translation_error_m"
        ),
        "optimized_keyframe_reference_p95_translation_error_m": get(
            "optimized_keyframe_reference_p95_translation_error_m"
        ),
        "initial_keyframe_reference_p95_rotation_error_rad": get(
            "initial_keyframe_reference_p95_rotation_error_rad"
        ),
        "optimized_keyframe_reference_p95_rotation_error_rad": get(
            "optimized_keyframe_reference_p95_rotation_error_rad"
        ),
        "structurally_valid": False,
        "complete_final_multi_submap": False,
    }


def validate_base(record: dict[str, Any], line: int, issues: Issues) -> bool:
    before = issues.count
    for field in sorted(REQUIRED - record.keys()):
        issues.add("missing_field", f"missing required field {field!r}", line)
    if issues.count != before:
        return False
    for field in STRINGS:
        if not isinstance(record[field], str):
            issues.add("invalid_type", f"{field} must be a string", line)
    for field in BOOLEANS:
        if not isinstance(record[field], bool):
            issues.add("invalid_type", f"{field} must be a boolean", line)
    for field in COUNTS:
        if not is_int(record[field]) or record[field] < 0:
            issues.add("invalid_count", f"{field} must be a non-negative integer", line)
    for field in NONNEGATIVE:
        if not finite(record[field]) or float(record[field]) < 0.0:
            issues.add("invalid_number", f"{field} must be finite and non-negative", line)
    for field in OBJECTIVES:
        if record[field] is not None and not finite(record[field]):
            issues.add("invalid_number", f"{field} must be finite or null", line)
    for field in ("nodes", "keyframes"):
        if not isinstance(record[field], list):
            issues.add("invalid_type", f"{field} must be an array", line)
    return issues.count == before


def validate_solved(record: dict[str, Any], line: int, issues: Issues) -> None:
    expected_counts = (
        ("node_count", record["snapshot_node_count"]),
        ("node_count", len(record["nodes"])),
        ("snapshot_owned_keyframe_count", len(record["keyframes"])),
        ("gauge_anchor_count", 1),
        ("active_edge_factor_count", record["snapshot_cross_edge_count"]),
        ("intra_submap_constant_edge_count", record["snapshot_intra_edge_count"]),
        ("floor_factor_count", record["snapshot_assigned_floor_count"]),
    )
    for field, expected in expected_counts:
        if record[field] != expected:
            issues.add("inconsistent_count", f"{field} != {expected}", line)
    if any(
        record[field]
        for field in (
            "snapshot_unassigned_keyframe_count",
            "snapshot_unassigned_edge_count",
            "snapshot_unassigned_floor_count",
        )
    ):
        issues.add("incomplete_snapshot", "solved snapshot has unassigned inputs", line)
    if record["node_count"] < 1 or not record["keyframes"]:
        issues.add("empty_solution", "solved trial has no nodes or keyframes", line)
    if (
        record["full_6d_factor_count"]
        + record["xy_yaw_lifted_factor_count"]
        != record["active_edge_factor_count"]
    ):
        issues.add("inconsistent_count", "factor-mode partition is incomplete", line)
    if (
        record["explicit_information_factor_count"]
        + record["fallback_noise_factor_count"]
        != record["active_edge_factor_count"]
    ):
        issues.add("inconsistent_count", "noise-provenance partition is incomplete", line)
    for field in ("robust_factor_count", "session_odometry_factor_count"):
        if record[field] > record["active_edge_factor_count"]:
            issues.add("inconsistent_count", f"{field} exceeds active factors", line)

    if not all(finite(record[field]) for field in OBJECTIVES):
        issues.add("invalid_objective", "solved objective fields must be finite", line)
    else:
        initial = float(record["initial_nonlinear_error"])
        final = float(record["final_nonlinear_error"])
        if initial < 0.0 or final < 0.0:
            issues.add("invalid_objective", "nonlinear errors must be non-negative", line)
        if final > initial + 1e-9 * max(1.0, initial):
            issues.add("objective_increased", "final objective exceeds initial", line)
        check_metric(
            record["nonlinear_error_reduction"],
            initial - final,
            "nonlinear_error_reduction",
            issues,
            line,
        )

    node_ids: list[int] = []
    gauge_ids: list[int] = []
    node_translation: list[float] = []
    node_rotation: list[float] = []
    for index, node in enumerate(record["nodes"]):
        owner = f"nodes[{index}]"
        required = {
            "submap_id",
            "gauge_anchor",
            "initial_pose",
            "optimized_pose",
            "translation_delta_m",
            "rotation_delta_rad",
        }
        if not isinstance(node, dict) or not required.issubset(node):
            issues.add("invalid_node", f"{owner} is incomplete", line)
            continue
        submap_id = node["submap_id"]
        if (
            not is_int(submap_id)
            or submap_id < 0
            or not isinstance(node["gauge_anchor"], bool)
        ):
            issues.add("invalid_node", f"{owner} has invalid identity fields", line)
            continue
        initial = pose(node["initial_pose"], f"{owner}.initial_pose", issues, line)
        optimized = pose(node["optimized_pose"], f"{owner}.optimized_pose", issues, line)
        if initial is None or optimized is None:
            continue
        translation, rotation = pose_error(initial, optimized)
        check_metric(
            node["translation_delta_m"],
            translation,
            f"{owner}.translation_delta_m",
            issues,
            line,
        )
        check_metric(
            node["rotation_delta_rad"],
            rotation,
            f"{owner}.rotation_delta_rad",
            issues,
            line,
        )
        node_ids.append(submap_id)
        node_translation.append(translation)
        node_rotation.append(rotation)
        if node["gauge_anchor"]:
            gauge_ids.append(submap_id)
            if translation > ABS_TOL or rotation > ABS_TOL:
                issues.add("gauge_moved", f"{owner} gauge anchor moved", line)
    if node_ids != sorted(set(node_ids)):
        issues.add("invalid_node_order", "submap IDs must be unique and sorted", line)
    if len(gauge_ids) != 1 or (node_ids and gauge_ids[0] != min(node_ids)):
        issues.add("invalid_gauge", "gauge must be the minimum submap ID", line)
    check_metric(
        record["max_translation_delta_m"],
        max(node_translation, default=0.0),
        "max_translation_delta_m",
        issues,
        line,
    )
    check_metric(
        record["max_rotation_delta_rad"],
        max(node_rotation, default=0.0),
        "max_rotation_delta_rad",
        issues,
        line,
    )

    keyframe_ids: list[int] = []
    error_columns: list[list[float]] = [[], [], [], []]
    for index, keyframe in enumerate(record["keyframes"]):
        owner = f"keyframes[{index}]"
        pose_fields = (
            "reference_pose",
            "initial_shadow_pose",
            "optimized_shadow_pose",
        )
        error_fields = (
            "initial_translation_error_m",
            "initial_rotation_error_rad",
            "optimized_translation_error_m",
            "optimized_rotation_error_rad",
        )
        required = {"keyframe_id", "submap_id", *pose_fields, *error_fields}
        if not isinstance(keyframe, dict) or not required.issubset(keyframe):
            issues.add("invalid_keyframe", f"{owner} is incomplete", line)
            continue
        if (
            not is_int(keyframe["keyframe_id"])
            or keyframe["keyframe_id"] < 0
            or not is_int(keyframe["submap_id"])
            or keyframe["submap_id"] not in set(node_ids)
        ):
            issues.add("invalid_keyframe", f"{owner} has invalid identity fields", line)
            continue
        poses = [
            pose(keyframe[field], f"{owner}.{field}", issues, line)
            for field in pose_fields
        ]
        if any(item is None for item in poses):
            continue
        reference, initial_shadow, optimized_shadow = poses
        assert reference is not None
        assert initial_shadow is not None
        assert optimized_shadow is not None
        initial_error = pose_error(reference, initial_shadow)
        optimized_error = pose_error(reference, optimized_shadow)
        for column, field, expected in zip(
            error_columns,
            error_fields,
            (*initial_error, *optimized_error),
        ):
            check_metric(keyframe[field], expected, f"{owner}.{field}", issues, line)
            column.append(expected)
        keyframe_ids.append(keyframe["keyframe_id"])
    if keyframe_ids != sorted(set(keyframe_ids)):
        issues.add("invalid_keyframe_order", "keyframe IDs must be unique and sorted", line)
    check_stats(
        record,
        "initial_keyframe_reference",
        error_columns[0],
        error_columns[1],
        issues,
        line,
    )
    check_stats(
        record,
        "optimized_keyframe_reference",
        error_columns[2],
        error_columns[3],
        issues,
        line,
    )


def validate_record(
    record: Any,
    line: int,
    expected_commit: str,
    expected_profile: str | None,
    issues: Issues,
) -> dict[str, Any]:
    before = issues.count
    summary = record_summary(record, line)
    if not isinstance(record, dict):
        issues.add("invalid_record", "JSONL record must be an object", line)
        return summary
    if not validate_base(record, line, issues):
        return summary

    if record["schema"] != INPUT_SCHEMA or record["record_type"] != "checkpoint":
        issues.add("schema_mismatch", "unexpected schema or record_type", line)
    mode = record["mode"]
    if mode not in RUNTIME_CONTEXTS:
        issues.add("invalid_mode", f"unsupported mode {mode!r}", line)
    elif (record["runtime_source"], record["context"]) not in RUNTIME_CONTEXTS[mode]:
        issues.add("runtime_context_mismatch", "mode/source/context do not match", line)
    commit = record["product_commit"]
    if not COMMIT_RE.fullmatch(commit) or commit != expected_commit:
        issues.add(
            "commit_mismatch",
            f"product_commit {commit!r} != expected {expected_commit}",
            line,
        )
    profile = record["product_profile_sha256"]
    if not SHA256_RE.fullmatch(profile) or (
        expected_profile is not None and profile != expected_profile
    ):
        issues.add("profile_mismatch", "product profile lineage does not match", line)
    if (
        record["product_build_type"] != "Release"
        or record["product_research_tools"] != "OFF"
    ):
        issues.add(
            "build_profile_mismatch",
            "build must be Release with research tools OFF",
            line,
        )
    if not record["product_verified"] or not record["no_writeback"]:
        issues.add("authority_mismatch", "verified no-writeback evidence is required", line)
    if record["snapshot_valid"] == bool(record["snapshot_failure_reason"]):
        issues.add("snapshot_state_mismatch", "snapshot state/reason mismatch", line)
    if record["valid"] != record["solved"] or (
        record["solved"] and not record["attempted"]
    ):
        issues.add("trial_state_mismatch", "trial valid/attempted/solved mismatch", line)
    if record["valid"] == bool(record["failure_reason"]) or (
        record["solved"] and not record["snapshot_valid"]
    ):
        issues.add("trial_state_mismatch", "trial validity/reason/snapshot mismatch", line)
    if record["snapshot_source_edge_count"] != (
        record["snapshot_intra_edge_count"]
        + record["snapshot_cross_edge_count"]
        + record["snapshot_unassigned_edge_count"]
    ):
        issues.add("inconsistent_count", "snapshot edge partition is incomplete", line)
    if record["snapshot_floor_count"] != (
        record["snapshot_assigned_floor_count"]
        + record["snapshot_unassigned_floor_count"]
    ):
        issues.add("inconsistent_count", "snapshot floor partition is incomplete", line)

    if record["solved"]:
        validate_solved(record, line, issues)
    else:
        trial_counts = COUNTS[10:]
        trial_metrics = NONNEGATIVE[4:]
        partial = (
            any(record[field] for field in trial_counts)
            or any(record[field] for field in trial_metrics)
            or bool(record["nodes"])
            or bool(record["keyframes"])
            or any(record[field] is not None for field in OBJECTIVES)
        )
        if partial:
            issues.add("partial_failed_trial", "unsolved trial exposes partial results", line)

    summary["structurally_valid"] = issues.count == before
    summary["complete_final_multi_submap"] = bool(
        summary["structurally_valid"]
        and record["context"] in FINAL_CONTEXTS
        and record["solved"]
        and record["node_count"] >= 2
        and record["snapshot_cross_edge_count"] >= 1
        and record["snapshot_owned_keyframe_count"] >= 2
    )
    return summary


def qualify(
    input_path: Path,
    expected_commit: str,
    expected_profile_sha256: str | None = None,
) -> dict[str, Any]:
    issues = Issues()
    records: list[Any] = []
    payload: bytes | None = None
    if not COMMIT_RE.fullmatch(expected_commit):
        issues.add("invalid_expected_commit", "expected commit must be 40 lowercase hex")
    if expected_profile_sha256 is not None and not SHA256_RE.fullmatch(
        expected_profile_sha256
    ):
        issues.add(
            "invalid_expected_profile",
            "expected profile must be 64 lowercase hex",
        )
    try:
        if input_path.is_symlink() or not input_path.is_file():
            raise ValueError("input must be a regular, non-symlink file")
        payload = input_path.read_bytes()
        if not payload:
            raise ValueError("input JSONL is empty")
        if not payload.endswith(b"\n"):
            raise ValueError("input JSONL must end with a newline")
        for line, source in enumerate(payload.decode("utf-8").splitlines(), start=1):
            if not source.strip():
                issues.add("blank_line", "JSONL contains a blank line", line)
                records.append(None)
                continue
            try:
                records.append(
                    json.loads(
                        source,
                        object_pairs_hook=strict_object,
                        parse_constant=reject_constant,
                    )
                )
            except (DuplicateKeyError, ValueError, json.JSONDecodeError) as error:
                issues.add("json_parse_error", str(error), line)
                records.append(None)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        issues.add("input_error", str(error))

    summaries = [
        validate_record(
            record,
            line,
            expected_commit,
            expected_profile_sha256,
            issues,
        )
        for line, record in enumerate(records, start=1)
    ]
    mappings = [record for record in records if isinstance(record, dict)]
    profiles = sorted(
        {
            record.get("product_profile_sha256")
            for record in mappings
            if isinstance(record.get("product_profile_sha256"), str)
        }
    )
    modes = sorted(
        {
            record.get("mode")
            for record in mappings
            if isinstance(record.get("mode"), str)
        }
    )
    if len(profiles) > 1:
        issues.add("mixed_profile_lineage", "JSONL contains multiple product profiles")
    if len(modes) > 1:
        issues.add("mixed_runtime_lineage", "JSONL contains multiple runtime modes")

    latest = summaries[-1] if summaries else None
    if issues.count:
        classification = INVALID
        reason = "schema_lineage_or_consistency_validation_failed"
        selected = None
    elif latest and latest["complete_final_multi_submap"]:
        classification = QUALIFIED
        reason = "latest_checkpoint_is_complete_final_multi_submap_trial"
        selected = latest
    else:
        classification = INSUFFICIENT
        reason = "latest_checkpoint_is_not_complete_final_multi_submap_trial"
        selected = None
    contexts = Counter(
        summary["context"]
        for summary in summaries
        if isinstance(summary["context"], str)
    )
    return {
        "schema": REPORT_SCHEMA,
        "classification": classification,
        "decision_reason": reason,
        "input": {
            "path": str(input_path.absolute()),
            "bytes": len(payload) if payload is not None else None,
            "sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
            "newline_terminated": bool(payload and payload.endswith(b"\n")),
        },
        "lineage": {
            "expected_commit": expected_commit,
            "expected_profile_sha256": expected_profile_sha256,
            "observed_profile_sha256": profiles[0] if len(profiles) == 1 else None,
            "verified_release_research_off_required": True,
        },
        "record_count": len(summaries),
        "solved_record_count": sum(
            item["solved"] is True for item in summaries
        ),
        "failed_trial_record_count": sum(
            item["trial_valid"] is False for item in summaries
        ),
        "complete_final_multi_submap_record_count": sum(
            item["complete_final_multi_submap"] is True
            for item in summaries
        ),
        "contexts": dict(sorted(contexts.items())),
        "records": summaries,
        "selected_record": selected,
        "error_count": issues.count,
        "errors_truncated": issues.count > len(issues.items),
        "errors": issues.items,
        "limitations": [
            "reference_is_current_keyframe_graph_not_external_ground_truth",
            "qualification_is_artifact_readiness_not_algorithm_acceptance",
            "no_numeric_quality_thresholds_were_applied",
            "jsonl_v1_has_no_run_identifier_or_operator_truth_label",
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-profile-sha256")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("--input and --output must be different files")
    report = qualify(
        arguments.input,
        arguments.expected_commit,
        arguments.expected_profile_sha256,
    )
    try:
        write_report(arguments.output, report)
    except OSError as error:
        print(f"cannot write qualification report: {error}", file=sys.stderr)
        return EXIT_CODES[INVALID]
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return EXIT_CODES[report["classification"]]


if __name__ == "__main__":
    raise SystemExit(main())
