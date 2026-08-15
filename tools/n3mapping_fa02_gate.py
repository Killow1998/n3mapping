#!/usr/bin/env python3
"""Fail-closed FA-02 ground-truth loop-closure acceptance gate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


CONTRACT_SCHEMA = "n3mapping_final_acceptance_contract_v1"
INPUT_SCHEMA = "n3mapping_fa02_input_manifest_v1"
RUN_SCHEMA = "n3mapping_fa02_episode_run_v1"
REPORT_SCHEMA = "n3mapping_fa02_acceptance_v1"
PASS = "PASS"
FAIL = "FAIL_FA02_QUALITY"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {PASS: 0, FAIL: 1, INVALID: 3}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DuplicateKeyError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []
        self.counts: Counter[str] = Counter()

    def add(self, category: str, code: str, message: str) -> None:
        self.counts[category] += 1
        if len(self.items) < 200:
            self.items.append({"category": category, "code": code, "message": message})


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def mat_mul(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        sum(lhs[row * 3 + k] * rhs[k * 3 + column] for k in range(3))
        for row in range(3)
        for column in range(3)
    )


def mat_transpose(matrix: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(matrix[column * 3 + row] for row in range(3) for column in range(3))


def mat_vec(matrix: tuple[float, ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row * 3 + column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def quat_matrix(qx: float, qy: float, qz: float, qw: float) -> tuple[float, ...]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid zero/non-finite quaternion")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )


def rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    )


def pose_inverse(pose: dict[str, Any]) -> dict[str, Any]:
    rotation = mat_transpose(pose["rotation"])
    translated = mat_vec(rotation, tuple(-value for value in pose["translation"]))
    return {"translation": translated, "rotation": rotation}


def pose_compose(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    rotated = mat_vec(lhs["rotation"], rhs["translation"])
    return {
        "translation": tuple(a + b for a, b in zip(lhs["translation"], rotated)),
        "rotation": mat_mul(lhs["rotation"], rhs["rotation"]),
    }


def relative_pose(match: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    return pose_compose(pose_inverse(match), query)


def translation_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def rotation_angle_deg(rotation: tuple[float, ...]) -> float:
    cosine = max(-1.0, min(1.0, (rotation[0] + rotation[4] + rotation[8] - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def yaw_difference_deg(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    lhs_yaw = math.atan2(lhs["rotation"][3], lhs["rotation"][0])
    rhs_yaw = math.atan2(rhs["rotation"][3], rhs["rotation"][0])
    return abs(math.degrees(math.atan2(
        math.sin(lhs_yaw - rhs_yaw), math.cos(lhs_yaw - rhs_yaw)
    )))


def pose_error(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float]:
    error = pose_compose(pose_inverse(reference), candidate)
    return translation_norm(error["translation"]), rotation_angle_deg(error["rotation"])


def load_trajectory(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 trajectory fields")
        key = fields[0]
        values = [float(value) for value in fields[1:]]
        if key in seen or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: duplicate key or non-finite pose")
        seen.add(key)
        x, y, z, qx, qy, qz, qw = values
        rows.append(
            (key, {"translation": (x, y, z), "rotation": quat_matrix(qx, qy, qz, qw)})
        )
    if not rows:
        raise ValueError(f"{path}: trajectory is empty")
    return rows


def load_keyframes(path: Path) -> dict[int, dict[str, Any]]:
    poses: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"keyframe_id", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing keyframe GT columns")
        for row in reader:
            keyframe_id = int(row["keyframe_id"])
            values = [float(row[name]) for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            if keyframe_id in poses or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: duplicate keyframe or non-finite pose")
            x, y, z, qx, qy, qz, qw = values
            poses[keyframe_id] = {
                "translation": (x, y, z),
                "rotation": quat_matrix(qx, qy, qz, qw),
            }
    if not poses:
        raise ValueError(f"{path}: no keyframes")
    return poses


def load_accepted_pairs(path: Path) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"query_id", "match_id"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing accepted-loop columns")
        for row in reader:
            pair = (int(row["query_id"]), int(row["match_id"]))
            if pair in pairs:
                raise ValueError(f"{path}: duplicate accepted loop {pair}")
            pairs.add(pair)
    return pairs


def load_candidates(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(
            line,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
        if not isinstance(event, dict):
            raise ValueError(f"{path}:{line_number}: event is not an object")
        if event.get("record_type") != "candidate":
            continue
        pair = (int(event["query_id"]), int(event["match_id"]))
        if pair in candidates and event.get("gate_result") == "accepted":
            raise ValueError(f"{path}:{line_number}: duplicate accepted candidate {pair}")
        if pair not in candidates or event.get("gate_result") == "accepted":
            candidates[pair] = event
    return candidates


def measurement_pose(event: dict[str, Any]) -> dict[str, Any]:
    values = [float(event[f"measurement_{name}"]) for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("accepted loop measurement is non-finite")
    x, y, z, roll, pitch, yaw = values
    return {"translation": (x, y, z), "rotation": rpy_matrix(roll, pitch, yaw)}


def opportunity_segments(
    poses: dict[int, dict[str, Any]], minimum_gap: int, radius: float,
    cluster_gap: int, maximum_yaw_difference_deg: float | None = None,
) -> list[list[int]]:
    ids = sorted(poses)
    queries: list[int] = []
    for query_index, query_id in enumerate(ids):
        for match_id in ids[:query_index]:
            if query_id - match_id < minimum_gap:
                continue
            distance = translation_norm(
                tuple(a - b for a, b in zip(poses[query_id]["translation"], poses[match_id]["translation"]))
            )
            heading_eligible = (
                maximum_yaw_difference_deg is None
                or yaw_difference_deg(poses[query_id], poses[match_id])
                <= maximum_yaw_difference_deg
            )
            if distance <= radius and heading_eligible:
                queries.append(query_id)
                break
    segments: list[list[int]] = []
    for query_id in queries:
        if not segments or query_id - segments[-1][-1] > cluster_gap:
            segments.append([query_id])
        else:
            segments[-1].append(query_id)
    return segments


def trajectory_metrics(gt_path: Path, estimate_path: Path) -> dict[str, Any]:
    gt = load_trajectory(gt_path)
    estimate = load_trajectory(estimate_path)
    if [key for key, _ in gt] != [key for key, _ in estimate]:
        raise ValueError("optimized and GT trajectory keys/order differ")
    absolute_translation: list[float] = []
    absolute_rotation: list[float] = []
    relative_translation: list[float] = []
    relative_rotation: list[float] = []
    for (_, gt_pose), (_, estimate_pose) in zip(gt, estimate):
        translation, rotation = pose_error(gt_pose, estimate_pose)
        absolute_translation.append(translation)
        absolute_rotation.append(rotation)
    for (_, gt_a), (_, gt_b), (_, est_a), (_, est_b) in zip(gt, gt[1:], estimate, estimate[1:]):
        translation, rotation = pose_error(
            relative_pose(gt_a, gt_b), relative_pose(est_a, est_b)
        )
        relative_translation.append(translation)
        relative_rotation.append(rotation)
    return {
        "pose_count": len(gt),
        "ate_translation_rmse_m": rmse(absolute_translation),
        "ate_translation_p95_m": percentile(absolute_translation, 0.95),
        "ate_rotation_rmse_deg": rmse(absolute_rotation),
        "rpe_translation_rmse_m": rmse(relative_translation),
        "rpe_rotation_rmse_deg": rmse(relative_rotation),
    }


def load_map_helpers() -> Any:
    path = Path(__file__).with_name("n3mapping_runtime_performance_gate.py")
    spec = importlib.util.spec_from_file_location("n3mapping_fa02_map_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import map helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_run_provenance(
    label: str,
    manifest: dict[str, Any],
    episode: dict[str, Any],
    expected: dict[str, str],
    issues: Issues,
) -> None:
    required = {
        "episode_id": label,
        "dataset": str(episode["dataset"]),
        "sequence": str(episode["sequence"]),
        "source_commit": expected["source_commit"],
        "input_manifest_sha256": expected["input_manifest_sha256"],
        "episode_frames_sha256": expected["episode_frames_sha256"],
    }
    for field, value in required.items():
        if manifest.get(field) != value:
            issues.add(
                "evidence", "run_provenance",
                f"{label}: run manifest {field}={manifest.get(field)!r}, expected {value!r}",
            )
    if manifest.get("missing_artifacts") != []:
        issues.add("evidence", "missing_artifacts", f"{label}: run reports missing artifacts")

    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, dict):
        issues.add("evidence", "evaluator_fingerprint", f"{label}: evaluator fingerprint missing")
        return
    path_value = evaluator.get("path")
    expected_size = evaluator.get("size_bytes")
    expected_hash = evaluator.get("sha256")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_size, int)
        or not isinstance(expected_hash, str)
        or not SHA256_RE.fullmatch(expected_hash)
    ):
        issues.add("evidence", "evaluator_fingerprint", f"{label}: malformed evaluator fingerprint")
        return
    path = Path(path_value)
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_hash
    ):
        issues.add("evidence", "evaluator_fingerprint", f"{label}: evaluator binary changed or missing")
    command = manifest.get("command")
    if not isinstance(command, list) or not command or command[0] != path_value:
        issues.add("evidence", "evaluator_command", f"{label}: command does not use fingerprinted evaluator")


def evaluate_episode(
    episode: dict[str, Any], directory: Path, oracle: dict[str, Any],
    thresholds: dict[str, Any], map_helpers: Any, proto_module: Any,
    issues: Issues, provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    label = str(episode["id"])
    manifest = load_json(directory / "run_manifest.json")
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "COMPLETE":
        issues.add("evidence", "run_manifest", f"{label}: incomplete run manifest")
    if manifest.get("process_return_code") != 0:
        issues.add("evidence", "process_return_code", f"{label}: evaluator return code is not zero")
    if provenance is not None:
        verify_run_provenance(label, manifest, episode, provenance, issues)
    metrics = load_json(directory / "metrics.json")
    expected_frames = int(episode["max_frames"])
    if metrics.get("frames_processed") != expected_frames:
        issues.add("evidence", "frame_count", f"{label}: evaluator frame count mismatch")
    if metrics.get("backend_input_contract") != "gt_pose_plus_lidar":
        issues.add("evidence", "backend_contract", f"{label}: backend input contract mismatch")
    if metrics.get("trajectory_optimized_semantics") != "final_dense_after_all_loop_updates":
        issues.add("evidence", "trajectory_semantics", f"{label}: final trajectory semantics missing")
    if episode["dataset"] == "m2dgr":
        if metrics.get("gt_sensor_frame") != episode["gt_sensor_frame"]:
            issues.add("evidence", "gt_sensor_frame", f"{label}: M2DGR GT sensor frame mismatch")
        if metrics.get("gt_to_lidar_calibration_applied") is not True:
            issues.add("evidence", "m2dgr_calibration", f"{label}: M2DGR calibration not applied")

    keyframes = load_keyframes(directory / "keyframes_gt.csv")
    accepted = load_accepted_pairs(directory / "accepted_loops.csv")
    candidates = load_candidates(directory / "loop_debug.jsonl")
    attitude_authoritative = bool(episode["attitude_authoritative"])
    correct_pairs: set[tuple[int, int]] = set()
    accepted_details: list[dict[str, Any]] = []
    catastrophic_count = 0
    for pair in sorted(accepted):
        query_id, match_id = pair
        if query_id not in keyframes or match_id not in keyframes:
            issues.add("evidence", "accepted_pair_missing_gt", f"{label}: accepted pair {pair} lacks GT")
            continue
        event = candidates.get(pair)
        if event is None or event.get("gate_result") != "accepted":
            issues.add("evidence", "accepted_pair_missing_debug", f"{label}: accepted pair {pair} lacks accepted debug event")
            continue
        try:
            measurement = measurement_pose(event)
        except (KeyError, TypeError, ValueError) as exc:
            issues.add("evidence", "measurement", f"{label}: accepted pair {pair}: {exc}")
            continue
        gt_relative = relative_pose(keyframes[match_id], keyframes[query_id])
        measurement_translation_error, measurement_rotation_error = pose_error(
            gt_relative, measurement
        )
        place_distance = translation_norm(
            tuple(
                a - b
                for a, b in zip(
                    keyframes[query_id]["translation"], keyframes[match_id]["translation"]
                )
            )
        )
        measurement_correct = (
            measurement_translation_error
            <= float(oracle["correct_measurement_translation_error_m_max"])
            and measurement_rotation_error
            <= float(oracle["correct_measurement_rotation_error_deg_max"])
        )
        position_consistent = (
            place_distance <= float(oracle["place_translation_threshold_m"])
        )
        correct = measurement_correct if attitude_authoritative else position_consistent
        catastrophic = (
            (
                attitude_authoritative
                and (
                    measurement_translation_error
                    > float(oracle["catastrophic_measurement_translation_error_m_min"])
                    or measurement_rotation_error
                    > float(oracle["catastrophic_measurement_rotation_error_deg_min"])
                )
            )
            or (
                not attitude_authoritative
                and place_distance
                > float(oracle["catastrophic_place_distance_m_min"])
            )
        )
        if correct:
            correct_pairs.add(pair)
        if catastrophic:
            catastrophic_count += 1
        accepted_details.append(
            {
                "query_id": query_id,
                "match_id": match_id,
                "place_distance_m": place_distance,
                "measurement_translation_error_m": measurement_translation_error,
                "measurement_rotation_error_deg": measurement_rotation_error,
                "measurement_correct": measurement_correct if attitude_authoritative else None,
                "position_consistent": position_consistent,
                "correct_basis": "se3_measurement" if attitude_authoritative else "position_only",
                "correct": correct,
                "catastrophic": catastrophic,
            }
        )

    segments = opportunity_segments(
        keyframes,
        int(oracle["minimum_keyframe_id_gap"]),
        float(oracle["place_translation_threshold_m"]),
        int(oracle["revisit_segment_max_query_id_gap"]),
        float(oracle["same_heading_yaw_threshold_deg"])
        if attitude_authoritative else None,
    )
    correct_queries = {query_id for query_id, _ in correct_pairs}
    hit_segments = sum(any(query_id in correct_queries for query_id in segment) for segment in segments)
    expected_role = str(episode["expected_role"])
    if expected_role.startswith("positive_revisit"):
        if not segments:
            issues.add("evidence", "no_gt_revisit", f"{label}: positive episode has no GT revisit segment")
        if len(correct_pairs) < int(thresholds["minimum_correct_accepted_loops_per_positive_episode"]):
            issues.add("quality", "positive_loop_miss", f"{label}: no correct accepted loop")

    trajectory = trajectory_metrics(
        directory / "trajectory_gt.txt", directory / "trajectory_optimized.txt"
    )
    for metric, threshold_name in (
        ("ate_translation_rmse_m", "ate_translation_rmse_m_max"),
        ("ate_translation_p95_m", "ate_translation_p95_m_max"),
        ("rpe_translation_rmse_m", "rpe_translation_rmse_m_max"),
    ):
        if trajectory[metric] is None or trajectory[metric] > float(thresholds[threshold_name]):
            issues.add("quality", metric, f"{label}: {metric}={trajectory[metric]}")
    if attitude_authoritative:
        for metric, threshold_name in (
            ("ate_rotation_rmse_deg", "ate_rotation_rmse_deg_max"),
            ("rpe_rotation_rmse_deg", "rpe_rotation_rmse_deg_max"),
        ):
            if trajectory[metric] is None or trajectory[metric] > float(thresholds[threshold_name]):
                issues.add("quality", metric, f"{label}: {metric}={trajectory[metric]}")

    map_proto = map_helpers.parse_map(directory / "n3map.pbstream", proto_module)
    structure = map_helpers.map_structure(map_proto, proto_module)
    expected_keyframes = int(metrics["accepted_keyframes"])
    invariants = {
        "keyframes": expected_keyframes,
        "odometry": max(0, expected_keyframes - 1),
        "loop": len(accepted),
        "session_anchor": 0,
        "unknown_edges": 0,
        "dense_trajectory": expected_frames,
        "metadata_match": True,
        "duplicate_keyframes": 0,
        "dangling_edges": 0,
    }
    for field, expected in invariants.items():
        if structure.get(field) != expected:
            issues.add(
                "quality", "graph_structure",
                f"{label}: map {field}={structure.get(field)!r}, expected {expected!r}",
            )
    stderr = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
    optimizer_error_count = sum(
        bool(re.search(r"(?:optimizer|optimization).*(?:exception|failed|fatal)", line, re.I))
        for line in stderr.splitlines()
    )
    if optimizer_error_count > int(thresholds["optimizer_error_count_max"]):
        issues.add("quality", "optimizer_error", f"{label}: optimizer errors={optimizer_error_count}")
    nonfinite_pose_count = 0
    if nonfinite_pose_count > int(thresholds["nonfinite_pose_count_max"]):
        issues.add("quality", "nonfinite_pose", f"{label}: non-finite poses={nonfinite_pose_count}")
    return {
        "id": label,
        "dataset": episode["dataset"],
        "expected_role": expected_role,
        "attitude_authoritative": attitude_authoritative,
        "keyframe_count": len(keyframes),
        "accepted_loop_count": len(accepted),
        "correct_accepted_loop_count": len(correct_pairs),
        "measurement_authoritative_accepted_loop_count": (
            len(accepted_details) if attitude_authoritative else 0
        ),
        "measurement_correct_accepted_loop_count": (
            sum(bool(detail["measurement_correct"]) for detail in accepted_details)
            if attitude_authoritative else 0
        ),
        "position_only_accepted_loop_count": (
            len(accepted_details) if not attitude_authoritative else 0
        ),
        "position_consistent_accepted_loop_count": (
            sum(bool(detail["position_consistent"]) for detail in accepted_details)
            if not attitude_authoritative else 0
        ),
        "catastrophic_false_loop_count": catastrophic_count,
        "revisit_segment_count": len(segments),
        "revisit_segment_hit_count": hit_segments,
        "accepted_loops": accepted_details,
        "trajectory": trajectory,
        "map_structure": structure,
        "optimizer_error_count": optimizer_error_count,
        "nonfinite_pose_count": nonfinite_pose_count,
    }


def verify_inputs(
    contract_path: Path,
    contract: dict[str, Any],
    root: Path,
    issues: Issues,
    *,
    section: str = "fa02",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_json(root / "input_manifest.json")
    if manifest.get("schema") != INPUT_SCHEMA:
        issues.add("evidence", "input_schema", "FA-02 input manifest schema mismatch")
    if manifest.get("contract", {}).get("sha256") != sha256_file(contract_path):
        issues.add("evidence", "contract_hash", "frozen contract SHA-256 mismatch")
    frames_path = root / "episode_frames.csv"
    if manifest.get("episode_frames", {}).get("sha256") != sha256_file(frames_path):
        issues.add("evidence", "frame_manifest_hash", "episode frame manifest SHA-256 mismatch")
    by_id = {str(episode["id"]): episode for episode in manifest.get("episodes", [])}
    for episode in contract[section]["episodes"]:
        episode_id = str(episode["id"])
        frozen = by_id.get(episode_id)
        if frozen is None:
            issues.add("evidence", "missing_frozen_episode", f"{episode_id}: input manifest missing")
            continue
        clouds = frozen.get("selected_clouds", [])
        if len(clouds) != int(episode["max_frames"]):
            issues.add("evidence", "frozen_frame_count", f"{episode_id}: frozen frame count mismatch")
        for cloud in clouds:
            path = Path(cloud["path"])
            if not path.is_file() or path.stat().st_size != cloud.get("size_bytes"):
                issues.add("evidence", "cloud_missing", f"{episode_id}: cloud missing/size mismatch: {path}")
                continue
            if sha256_file(path) != cloud.get("sha256"):
                issues.add("evidence", "cloud_hash", f"{episode_id}: cloud SHA-256 mismatch: {path}")
        for category in ("gt",):
            evidence = frozen.get("inputs", {}).get(category, {})
            path = Path(evidence.get("path", ""))
            if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
                issues.add("evidence", f"{category}_hash", f"{episode_id}: {category} SHA-256 mismatch")
        for evidence in frozen.get("inputs", {}).get("calibration", []):
            path = Path(evidence.get("path", ""))
            if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
                issues.add("evidence", "calibration_hash", f"{episode_id}: calibration SHA-256 mismatch")
    return manifest, by_id


def evaluate(contract_path: Path, root: Path, proto_path: Path) -> dict[str, Any]:
    issues = Issues()
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        issues.add("evidence", "contract_schema", "acceptance contract schema mismatch")
    fa02 = contract.get("fa02")
    if not isinstance(fa02, dict):
        raise ValueError("contract has no FA-02 section")
    input_manifest, _ = verify_inputs(contract_path, contract, root, issues)
    provenance = {
        "source_commit": str(input_manifest.get("source_commit", "")),
        "input_manifest_sha256": sha256_file(root / "input_manifest.json"),
        "episode_frames_sha256": sha256_file(root / "episode_frames.csv"),
    }
    map_helpers = load_map_helpers()
    proto_module = map_helpers.load_proto_module(proto_path)
    episode_reports: list[dict[str, Any]] = []
    for episode in fa02["episodes"]:
        try:
            episode_reports.append(
                evaluate_episode(
                    episode, root / "runs" / str(episode["id"]), fa02["oracle"],
                    fa02["thresholds"], map_helpers, proto_module, issues, provenance,
                )
            )
        except Exception as exc:
            issues.add("evidence", "episode_exception", f"{episode['id']}: {exc}")

    total_accepted = sum(report["accepted_loop_count"] for report in episode_reports)
    total_correct = sum(report["correct_accepted_loop_count"] for report in episode_reports)
    measurement_accepted = sum(
        report["measurement_authoritative_accepted_loop_count"]
        for report in episode_reports
    )
    measurement_correct = sum(
        report["measurement_correct_accepted_loop_count"]
        for report in episode_reports
    )
    position_only_accepted = sum(
        report["position_only_accepted_loop_count"] for report in episode_reports
    )
    position_consistent = sum(
        report["position_consistent_accepted_loop_count"] for report in episode_reports
    )
    total_catastrophic = sum(report["catastrophic_false_loop_count"] for report in episode_reports)
    positive_reports = [
        report for report in episode_reports
        if str(report["expected_role"]).startswith("positive_revisit")
    ]
    total_segments = sum(report["revisit_segment_count"] for report in positive_reports)
    hit_segments = sum(report["revisit_segment_hit_count"] for report in positive_reports)
    precision = measurement_correct / measurement_accepted if measurement_accepted else None
    position_precision = (
        position_consistent / position_only_accepted if position_only_accepted else None
    )
    segment_recall = hit_segments / total_segments if total_segments else None
    thresholds = fa02["thresholds"]
    if total_catastrophic > int(thresholds["catastrophic_false_loop_count_max"]):
        issues.add("quality", "catastrophic_false_loop", f"catastrophic false loops={total_catastrophic}")
    if precision is None or precision < float(thresholds["accepted_measurement_precision_min"]):
        issues.add("quality", "accepted_precision", f"accepted measurement precision={precision}")
    if segment_recall is None or segment_recall < float(thresholds["revisit_segment_recall_min"]):
        issues.add("quality", "segment_recall", f"revisit segment recall={segment_recall}")

    status = INVALID if issues.counts["evidence"] else FAIL if issues.counts["quality"] else PASS
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "source_commit": input_manifest.get("source_commit"),
        "aggregate": {
            "accepted_loop_count": total_accepted,
            "correct_accepted_loop_count": total_correct,
            "measurement_authoritative_accepted_loop_count": measurement_accepted,
            "measurement_correct_accepted_loop_count": measurement_correct,
            "catastrophic_false_loop_count": total_catastrophic,
            "accepted_measurement_precision": precision,
            "position_only_accepted_loop_count": position_only_accepted,
            "position_consistent_accepted_loop_count": position_consistent,
            "accepted_position_precision": position_precision,
            "positive_revisit_segment_count": total_segments,
            "positive_revisit_segment_hit_count": hit_segments,
            "revisit_segment_recall": segment_recall,
        },
        "episodes": episode_reports,
        "issue_counts": dict(issues.counts),
        "issues": issues.items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        report = evaluate(
            args.contract.resolve(strict=True),
            args.evidence_root.resolve(strict=True),
            args.proto.resolve(strict=True),
        )
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}, sort_keys=True))
        return EXIT_CODES[report["status"]]
    except Exception as exc:
        print(f"n3mapping_fa02_gate: {exc}", file=sys.stderr)
        return EXIT_CODES[INVALID]


if __name__ == "__main__":
    raise SystemExit(main())
