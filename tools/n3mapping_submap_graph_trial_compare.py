#!/usr/bin/env python3
"""Build a review artifact for a qualified shadow submap-graph trial.

The report compares the shadow lift-back against two persisted surfaces:

* the map's native dense trajectory at each keyframe timestamp; and
* the same local cloud samples transformed by reference and shadow poses.

Neither surface is external ground truth.  ``REVIEW_READY`` therefore means
that the comparison is complete and auditable, not that shadow writeback is
safe or beneficial.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable


REPORT_SCHEMA = "n3mapping_submap_graph_trial_comparison_v1"
QUALIFICATION_SCHEMA = "n3mapping_submap_graph_trial_qualification_v1"
TRIAL_SCHEMA = "n3mapping_submap_graph_trial_v1"
QUALIFIED = "QUALIFIED_FOR_REVIEW"
READY = "REVIEW_READY"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {READY: 0, INSUFFICIENT: 2, INVALID: 3}
PAIRING_TRANSLATION_TOLERANCE_M = 1e-6
PAIRING_ROTATION_TOLERANCE_RAD = 1e-6

Pose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
Point = tuple[float, float, float, float]


class ComparisonError(RuntimeError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def load_json_bytes(payload: bytes, owner: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ComparisonError(f"{owner} is not strict JSON: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def normalized_quaternion(values: Iterable[Any], owner: str) -> tuple[float, float, float, float]:
    raw = tuple(values)
    if len(raw) != 4 or not all(finite(value) for value in raw):
        raise ComparisonError(f"{owner} quaternion must contain four finite numbers")
    quaternion = tuple(float(value) for value in raw)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12 or abs(norm - 1.0) > 1e-6:
        raise ComparisonError(f"{owner} quaternion is not normalized")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def json_pose(value: Any, owner: str) -> Pose:
    if not isinstance(value, dict):
        raise ComparisonError(f"{owner} must be an object")
    translation = value.get("translation")
    quaternion = value.get("quaternion_xyzw")
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or not all(finite(item) for item in translation)
    ):
        raise ComparisonError(f"{owner} translation must contain three finite numbers")
    return (
        tuple(float(item) for item in translation),  # type: ignore[arg-type]
        normalized_quaternion(quaternion or (), owner),
    )


def proto_pose(value: Any, owner: str) -> Pose:
    translation = (value.tx, value.ty, value.tz)
    if not all(finite(item) for item in translation):
        raise ComparisonError(f"{owner} translation is not finite")
    return (
        tuple(float(item) for item in translation),
        normalized_quaternion((value.qx, value.qy, value.qz, value.qw), owner),
    )


def pose_error(reference: Pose, candidate: Pose) -> tuple[float, float]:
    translation = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(reference[0], candidate[0]))
    )
    dot = abs(sum(left * right for left, right in zip(reference[1], candidate[1])))
    rotation = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    return translation, rotation


def rotation_matrix(pose: Pose) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = pose[1]
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def transform_point(pose: Pose, rotation: tuple[tuple[float, float, float], ...], point: Point) -> Point:
    x, y, z, intensity = point
    return (
        pose[0][0] + rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z,
        pose[0][1] + rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z,
        pose[0][2] + rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z,
        intensity,
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def paired_statistics(translations: list[float], rotations: list[float]) -> dict[str, Any]:
    return {
        "translation_error_m": statistics(translations),
        "rotation_error_rad": statistics(rotations),
        "rotation_error_deg": statistics([math.degrees(value) for value in rotations]),
    }


def load_qualification(path: Path, trial_payload: bytes) -> dict[str, Any]:
    payload = path.read_bytes()
    report = load_json_bytes(payload, "qualification")
    if not isinstance(report, dict) or report.get("schema") != QUALIFICATION_SCHEMA:
        raise ComparisonError("qualification schema is invalid")
    if report.get("classification") != QUALIFIED:
        raise ComparisonError("trial is not QUALIFIED_FOR_REVIEW")
    if report.get("error_count") != 0:
        raise ComparisonError("qualification contains validation errors")
    source = report.get("input")
    if not isinstance(source, dict) or source.get("sha256") != sha256_bytes(trial_payload):
        raise ComparisonError("qualification does not bind the supplied trial JSONL")
    selected = report.get("selected_record")
    if not isinstance(selected, dict) or not isinstance(selected.get("line"), int):
        raise ComparisonError("qualification has no selected record line")
    line_count = len(trial_payload.splitlines())
    if report.get("record_count") != line_count or selected["line"] != line_count:
        raise ComparisonError("qualification does not select the latest trial record")
    if selected.get("complete_final_multi_submap") is not True:
        raise ComparisonError("qualification selected record is not complete")
    return report


def selected_trial_record(payload: bytes, qualification: dict[str, Any]) -> dict[str, Any]:
    lines = payload.splitlines()
    selected = qualification["selected_record"]
    line_number = selected["line"]
    if line_number < 1 or line_number > len(lines):
        raise ComparisonError("qualification selected line is outside the trial JSONL")
    record = load_json_bytes(lines[line_number - 1], "selected trial record")
    if not isinstance(record, dict) or record.get("schema") != TRIAL_SCHEMA:
        raise ComparisonError("selected trial record schema is invalid")
    if (
        record.get("no_writeback") is not True
        or record.get("solved") is not True
        or record.get("product_verified") is not True
        or record.get("product_build_type") != "Release"
        or record.get("product_research_tools") != "OFF"
    ):
        raise ComparisonError("selected trial record is not a solved no-writeback trial")
    lineage = qualification.get("lineage")
    if not isinstance(lineage, dict):
        raise ComparisonError("qualification lineage is missing")
    if record.get("product_commit") != lineage.get("expected_commit"):
        raise ComparisonError("selected trial commit does not match qualification")
    expected_profile = lineage.get("expected_profile_sha256")
    if expected_profile is not None and record.get("product_profile_sha256") != expected_profile:
        raise ComparisonError("selected trial profile does not match qualification")
    for record_field, selected_field in (
        ("node_count", "node_count"),
        ("snapshot_owned_keyframe_count", "owned_keyframe_count"),
        ("snapshot_cross_edge_count", "cross_edge_count"),
    ):
        if record.get(record_field) != selected.get(selected_field):
            raise ComparisonError(f"selected trial summary mismatch for {record_field}")
    if not isinstance(record.get("nodes"), list) or len(record["nodes"]) != record["node_count"]:
        raise ComparisonError("selected trial node list is inconsistent")
    if (
        not isinstance(record.get("keyframes"), list)
        or len(record["keyframes"]) != record["snapshot_owned_keyframe_count"]
    ):
        raise ComparisonError("selected trial keyframe list is inconsistent")
    return record


def load_proto_map(map_path: Path, proto_module_dir: Path) -> Any:
    module_path = proto_module_dir / "n3map_pb2.py"
    if not module_path.is_file():
        raise ComparisonError(f"generated protobuf module is missing: {module_path}")
    spec = importlib.util.spec_from_file_location("n3mapping_sg09_n3map_pb2", module_path)
    if spec is None or spec.loader is None:
        raise ComparisonError(f"cannot load generated protobuf module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        result = module.N3Map()
        required_fields = {"keyframes", "dense_optimized_trajectory", "submaps"}
        if not required_fields.issubset(result.DESCRIPTOR.fields_by_name):
            missing = sorted(required_fields - set(result.DESCRIPTOR.fields_by_name))
            raise ComparisonError(
                "generated protobuf module predates required map fields: "
                + ",".join(missing)
            )
        result.ParseFromString(map_path.read_bytes())
    except ComparisonError:
        raise
    except Exception as error:  # protobuf raises implementation-specific errors
        raise ComparisonError(f"cannot parse map protobuf: {error}") from error
    return result


def sample_indices(point_count: int, limit: int) -> list[int]:
    count = min(point_count, limit)
    if count <= 0:
        return []
    return [(index * point_count) // count for index in range(count)]


def nearest_dense_pose(dense: list[tuple[float, Pose]], timestamps: list[float], stamp: float) -> tuple[float, Pose]:
    position = bisect.bisect_left(timestamps, stamp)
    candidates = []
    if position < len(dense):
        candidates.append(dense[position])
    if position > 0:
        candidates.append(dense[position - 1])
    if not candidates:
        raise ComparisonError("map has no native dense trajectory")
    return min(candidates, key=lambda item: abs(item[0] - stamp))


def write_pcd(path: Path, points: list[Point]) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for point in points:
            stream.write(struct.pack("<ffff", *point))


def thin_points(points: list[Point], limit: int) -> list[Point]:
    if len(points) <= limit:
        return points
    return [points[index] for index in sample_indices(len(points), limit)]


def write_svg(
    path: Path,
    reference_points: list[Point],
    shadow_points: list[Point],
    trajectory_rows: list[dict[str, Any]],
    preview_limit: int,
    vector_magnification: float,
) -> None:
    reference_preview = thin_points(reference_points, preview_limit)
    shadow_preview = thin_points(shadow_points, preview_limit)
    xy = [(point[0], point[1]) for point in reference_preview + shadow_preview]
    if not xy:
        raise ComparisonError("cannot render an empty geometry preview")
    min_x = min(point[0] for point in xy)
    max_x = max(point[0] for point in xy)
    min_y = min(point[1] for point in xy)
    max_y = max(point[1] for point in xy)
    width, height, margin = 1200.0, 900.0, 55.0
    scale = min(
        (width - 2.0 * margin) / max(max_x - min_x, 1e-6),
        (height - 2.0 * margin) / max(max_y - min_y, 1e-6),
    )

    def project(x: float, y: float) -> tuple[float, float]:
        return margin + (x - min_x) * scale, height - margin - (y - min_y) * scale

    def point_path(points: list[Point]) -> str:
        return " ".join(
            f"M{project(point[0], point[1])[0]:.2f},{project(point[0], point[1])[1]:.2f}h0.4"
            for point in points
        )

    def polyline(name: str) -> str:
        values = []
        for row in trajectory_rows:
            pose = row[name]
            x, y = project(pose[0][0], pose[0][1])
            values.append(f"{x:.2f},{y:.2f}")
        return " ".join(values)

    vectors = []
    for row in trajectory_rows:
        reference = row["reference_pose"]
        shadow = row["optimized_shadow_pose"]
        endpoint = (
            reference[0][0] + vector_magnification * (shadow[0][0] - reference[0][0]),
            reference[0][1] + vector_magnification * (shadow[0][1] - reference[0][1]),
        )
        x1, y1 = project(reference[0][0], reference[0][1])
        x2, y2 = project(*endpoint)
        vectors.append(f"M{x1:.2f},{y1:.2f}L{x2:.2f},{y2:.2f}")

    document = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900">
<rect width="1200" height="900" fill="#11151a"/>
<path d="{point_path(reference_preview)}" stroke="#d0d5db" stroke-width="1.0" opacity="0.42"/>
<path d="{point_path(shadow_preview)}" stroke="#ff6b55" stroke-width="1.0" opacity="0.30"/>
<polyline points="{polyline('reference_pose')}" fill="none" stroke="#55b7ff" stroke-width="2.0"/>
<polyline points="{polyline('optimized_shadow_pose')}" fill="none" stroke="#ffb347" stroke-width="2.0"/>
<path d="{' '.join(vectors)}" fill="none" stroke="#ff3d8d" stroke-width="1.1" opacity="0.9"/>
<rect x="20" y="18" width="535" height="105" rx="6" fill="#11151a" stroke="#8b949e" opacity="0.94"/>
<text x="35" y="43" fill="#f0f3f6" font-family="monospace" font-size="16">SG-09 shadow review (top view)</text>
<text x="35" y="68" fill="#55b7ff" font-family="monospace" font-size="14">blue: reference keyframe trajectory</text>
<text x="35" y="89" fill="#ffb347" font-family="monospace" font-size="14">orange: optimized shadow trajectory</text>
<text x="35" y="110" fill="#ff3d8d" font-family="monospace" font-size="14">pink vectors: reference to shadow, x{html.escape(f'{vector_magnification:g}')}</text>
</svg>\n"""
    path.write_text(document, encoding="utf-8")


def write_trajectory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "keyframe_id", "submap_id", "timestamp", "dense_timestamp_error_ms",
        "reference_dense_translation_m", "reference_dense_rotation_deg",
        "initial_dense_translation_m", "initial_dense_rotation_deg",
        "optimized_dense_translation_m", "optimized_dense_rotation_deg",
        "reference_shadow_translation_m", "reference_shadow_rotation_deg",
        "reference_tx", "reference_ty", "reference_tz",
        "shadow_tx", "shadow_ty", "shadow_tz",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_geometry_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "keyframe_id", "submap_id", "cloud_point_count", "sample_count",
        "same_point_displacement_mean_m", "same_point_displacement_p50_m",
        "same_point_displacement_p95_m", "same_point_displacement_max_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(
    qualification_path: Path,
    trial_path: Path,
    map_path: Path,
    proto_module_dir: Path,
    output_dir: Path,
    max_dense_time_error_ms: float,
    max_points_per_keyframe: int = 512,
    preview_points: int = 20000,
    vector_magnification: float = 20.0,
) -> dict[str, Any]:
    if max_dense_time_error_ms <= 0.0 or not math.isfinite(max_dense_time_error_ms):
        raise ComparisonError("max dense time error must be finite and positive")
    if max_points_per_keyframe <= 0 or preview_points <= 0:
        raise ComparisonError("point sampling limits must be positive")
    if vector_magnification <= 0.0 or not math.isfinite(vector_magnification):
        raise ComparisonError("vector magnification must be finite and positive")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ComparisonError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_payload = trial_path.read_bytes()
    qualification = load_qualification(qualification_path, trial_payload)
    trial = selected_trial_record(trial_payload, qualification)
    map_payload = map_path.read_bytes()
    n3map = load_proto_map(map_path, proto_module_dir)

    trial_keyframes = trial.get("keyframes")
    if not isinstance(trial_keyframes, list) or not trial_keyframes:
        raise ComparisonError("selected trial has no keyframe comparisons")
    trial_by_id: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(trial_keyframes):
        if not isinstance(record, dict) or not isinstance(record.get("keyframe_id"), int):
            raise ComparisonError(f"trial keyframes[{index}] is invalid")
        keyframe_id = record["keyframe_id"]
        if keyframe_id in trial_by_id:
            raise ComparisonError(f"duplicate trial keyframe id {keyframe_id}")
        trial_by_id[keyframe_id] = record

    map_by_id: dict[int, Any] = {}
    for keyframe in n3map.keyframes:
        if keyframe.id in map_by_id:
            raise ComparisonError(f"duplicate map keyframe id {keyframe.id}")
        map_by_id[keyframe.id] = keyframe
    if set(map_by_id) != set(trial_by_id):
        raise ComparisonError("map and selected trial keyframe ids do not match exactly")
    if n3map.metadata.num_keyframes != len(map_by_id):
        raise ComparisonError("map metadata keyframe count is inconsistent")

    trial_nodes = trial.get("nodes")
    trial_submap_ids: set[int] = set()
    for index, node in enumerate(trial_nodes):
        if not isinstance(node, dict) or not isinstance(node.get("submap_id"), int):
            raise ComparisonError(f"trial nodes[{index}] is invalid")
        if node["submap_id"] in trial_submap_ids:
            raise ComparisonError(f"duplicate trial submap id {node['submap_id']}")
        trial_submap_ids.add(node["submap_id"])
    map_submap_owners: dict[int, int] = {}
    map_submap_ids: set[int] = set()
    for submap in n3map.submaps:
        submap_id = int(submap.id)
        if submap_id in map_submap_ids:
            raise ComparisonError(f"duplicate map submap id {submap_id}")
        map_submap_ids.add(submap_id)
        for keyframe_id in submap.keyframe_ids:
            if keyframe_id in map_submap_owners:
                raise ComparisonError(f"map keyframe {keyframe_id} has duplicate submap ownership")
            map_submap_owners[int(keyframe_id)] = submap_id
    if map_submap_ids != trial_submap_ids:
        raise ComparisonError("map and selected trial submap ids do not match exactly")
    if set(map_submap_owners) != set(map_by_id):
        raise ComparisonError("map submaps do not own every keyframe exactly once")
    for keyframe_id, record in trial_by_id.items():
        if map_submap_owners[keyframe_id] != record.get("submap_id"):
            raise ComparisonError(f"map submap ownership mismatch for keyframe {keyframe_id}")

    dense: list[tuple[float, Pose]] = []
    for expected_seq, item in enumerate(n3map.dense_optimized_trajectory):
        timestamp = float(item.timestamp)
        if item.seq != expected_seq or not math.isfinite(timestamp):
            raise ComparisonError("map dense trajectory sequence or timestamp is invalid")
        dense.append(
            (timestamp, proto_pose(item.pose_world_lidar, f"dense pose {item.seq}"))
        )
    dense_timestamps = [item[0] for item in dense]
    if any(right < left for left, right in zip(dense_timestamps, dense_timestamps[1:])):
        raise ComparisonError("map dense trajectory timestamps are not ordered")

    trajectory_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    reference_points: list[Point] = []
    shadow_points: list[Point] = []
    all_displacements: list[float] = []
    displacements_by_submap: dict[int, list[float]] = {}
    map_pair_translation: list[float] = []
    map_pair_rotation: list[float] = []
    reference_dense_translation: list[float] = []
    reference_dense_rotation: list[float] = []
    initial_dense_translation: list[float] = []
    initial_dense_rotation: list[float] = []
    optimized_dense_translation: list[float] = []
    optimized_dense_rotation: list[float] = []
    dense_time_errors_ms: list[float] = []
    optimized_proxy_better = 0
    optimized_proxy_worse = 0
    optimized_proxy_tied = 0
    empty_cloud_keyframe_count = 0

    for keyframe_id in sorted(map_by_id):
        map_keyframe = map_by_id[keyframe_id]
        trial_keyframe = trial_by_id[keyframe_id]
        submap_id = trial_keyframe.get("submap_id")
        if not isinstance(submap_id, int):
            raise ComparisonError(f"keyframe {keyframe_id} has invalid submap id")
        reference = json_pose(trial_keyframe.get("reference_pose"), f"keyframe {keyframe_id} reference")
        initial = json_pose(trial_keyframe.get("initial_shadow_pose"), f"keyframe {keyframe_id} initial shadow")
        optimized = json_pose(trial_keyframe.get("optimized_shadow_pose"), f"keyframe {keyframe_id} optimized shadow")
        map_reference = proto_pose(map_keyframe.pose_optimized, f"map keyframe {keyframe_id}")
        pair_translation, pair_rotation = pose_error(reference, map_reference)
        map_pair_translation.append(pair_translation)
        map_pair_rotation.append(pair_rotation)
        if (
            pair_translation > PAIRING_TRANSLATION_TOLERANCE_M
            or pair_rotation > PAIRING_ROTATION_TOLERANCE_RAD
        ):
            raise ComparisonError(f"map pose does not match trial reference for keyframe {keyframe_id}")

        dense_stamp, dense_pose = nearest_dense_pose(dense, dense_timestamps, float(map_keyframe.timestamp))
        dense_error_ms = abs(dense_stamp - float(map_keyframe.timestamp)) * 1000.0
        dense_time_errors_ms.append(dense_error_ms)
        reference_error = pose_error(dense_pose, reference)
        initial_error = pose_error(dense_pose, initial)
        optimized_error = pose_error(dense_pose, optimized)
        shadow_delta = pose_error(reference, optimized)
        reference_dense_translation.append(reference_error[0])
        reference_dense_rotation.append(reference_error[1])
        initial_dense_translation.append(initial_error[0])
        initial_dense_rotation.append(initial_error[1])
        optimized_dense_translation.append(optimized_error[0])
        optimized_dense_rotation.append(optimized_error[1])
        proxy_delta = optimized_error[0] - reference_error[0]
        if proxy_delta < -1e-12:
            optimized_proxy_better += 1
        elif proxy_delta > 1e-12:
            optimized_proxy_worse += 1
        else:
            optimized_proxy_tied += 1

        cloud = map_keyframe.cloud
        point_count = int(cloud.num_points)
        if point_count < 0 or len(cloud.points) != point_count * 4:
            raise ComparisonError(f"keyframe {keyframe_id} cloud shape is inconsistent")
        reference_rotation = rotation_matrix(reference)
        optimized_rotation = rotation_matrix(optimized)
        keyframe_displacements: list[float] = []
        if point_count == 0:
            empty_cloud_keyframe_count += 1
        for point_index in sample_indices(point_count, max_points_per_keyframe):
            offset = point_index * 4
            local = tuple(float(cloud.points[offset + axis]) for axis in range(4))
            if not all(math.isfinite(value) for value in local):
                raise ComparisonError(f"keyframe {keyframe_id} cloud contains non-finite points")
            reference_point = transform_point(reference, reference_rotation, local)  # type: ignore[arg-type]
            shadow_point = transform_point(optimized, optimized_rotation, local)  # type: ignore[arg-type]
            displacement = math.sqrt(
                sum((reference_point[axis] - shadow_point[axis]) ** 2 for axis in range(3))
            )
            reference_points.append(reference_point)
            shadow_points.append(shadow_point)
            keyframe_displacements.append(displacement)
            all_displacements.append(displacement)
            displacements_by_submap.setdefault(submap_id, []).append(displacement)
        displacement_stats = statistics(keyframe_displacements)
        geometry_rows.append(
            {
                "keyframe_id": keyframe_id,
                "submap_id": submap_id,
                "cloud_point_count": point_count,
                "sample_count": len(keyframe_displacements),
                "same_point_displacement_mean_m": displacement_stats["mean"],
                "same_point_displacement_p50_m": displacement_stats["p50"],
                "same_point_displacement_p95_m": displacement_stats["p95"],
                "same_point_displacement_max_m": displacement_stats["max"],
            }
        )
        trajectory_rows.append(
            {
                "keyframe_id": keyframe_id,
                "submap_id": submap_id,
                "timestamp": float(map_keyframe.timestamp),
                "dense_timestamp_error_ms": dense_error_ms,
                "reference_dense_translation_m": reference_error[0],
                "reference_dense_rotation_deg": math.degrees(reference_error[1]),
                "initial_dense_translation_m": initial_error[0],
                "initial_dense_rotation_deg": math.degrees(initial_error[1]),
                "optimized_dense_translation_m": optimized_error[0],
                "optimized_dense_rotation_deg": math.degrees(optimized_error[1]),
                "reference_shadow_translation_m": shadow_delta[0],
                "reference_shadow_rotation_deg": math.degrees(shadow_delta[1]),
                "reference_tx": reference[0][0],
                "reference_ty": reference[0][1],
                "reference_tz": reference[0][2],
                "shadow_tx": optimized[0][0],
                "shadow_ty": optimized[0][1],
                "shadow_tz": optimized[0][2],
                "reference_pose": reference,
                "optimized_shadow_pose": optimized,
            }
        )

    dense_complete = (
        bool(dense_time_errors_ms)
        and max(dense_time_errors_ms) <= max_dense_time_error_ms
    )
    geometry_complete = bool(all_displacements) and empty_cloud_keyframe_count == 0
    comparison_complete = dense_complete and geometry_complete
    classification = READY if comparison_complete else INSUFFICIENT
    if comparison_complete:
        decision_reason = "qualified_trial_map_geometry_and_native_dense_proxy_are_complete"
    elif not dense_complete:
        decision_reason = "native_dense_proxy_does_not_cover_every_keyframe_within_declared_time_bound"
    else:
        decision_reason = "map_geometry_samples_do_not_cover_every_keyframe"

    trajectory_csv = output_dir / "trajectory_comparison.csv"
    geometry_csv = output_dir / "map_geometry_displacement.csv"
    reference_pcd = output_dir / "reference_map_samples.pcd"
    shadow_pcd = output_dir / "optimized_shadow_map_samples.pcd"
    overview_svg = output_dir / "top_view_overlay.svg"
    write_trajectory_csv(trajectory_csv, trajectory_rows)
    write_geometry_csv(geometry_csv, geometry_rows)
    write_pcd(reference_pcd, reference_points)
    write_pcd(shadow_pcd, shadow_points)
    write_svg(
        overview_svg,
        reference_points,
        shadow_points,
        trajectory_rows,
        preview_points,
        vector_magnification,
    )

    report = {
        "schema": REPORT_SCHEMA,
        "classification": classification,
        "decision_reason": decision_reason,
        "writeback_authorized": False,
        "inputs": {
            "qualification": {
                "path": str(qualification_path.resolve()),
                "sha256": sha256_bytes(qualification_path.read_bytes()),
                "selected_line": qualification["selected_record"]["line"],
            },
            "trial_jsonl": {
                "path": str(trial_path.resolve()),
                "sha256": sha256_bytes(trial_payload),
                "product_commit": trial["product_commit"],
                "product_profile_sha256": trial["product_profile_sha256"],
                "mode": trial.get("mode"),
                "context": trial.get("context"),
            },
            "map": {
                "path": str(map_path.resolve()),
                "sha256": sha256_bytes(map_payload),
                "version": n3map.metadata.version,
                "keyframe_count": len(map_by_id),
                "dense_trajectory_count": len(dense),
            },
        },
        "pairing": {
            "keyframe_id_sets_match_exactly": True,
            "submap_id_sets_and_keyframe_ownership_match_exactly": True,
            "map_reference_pose_translation_tolerance_m": PAIRING_TRANSLATION_TOLERANCE_M,
            "map_reference_pose_rotation_tolerance_rad": PAIRING_ROTATION_TOLERANCE_RAD,
            "map_reference_pose_translation_m": statistics(map_pair_translation),
            "map_reference_pose_rotation_rad": statistics(map_pair_rotation),
        },
        "trajectory_proxy": {
            "proxy": "map_native_dense_optimized_trajectory",
            "declared_max_timestamp_error_ms": max_dense_time_error_ms,
            "complete": dense_complete,
            "timestamp_error_ms": statistics(dense_time_errors_ms),
            "reference_vs_dense": paired_statistics(reference_dense_translation, reference_dense_rotation),
            "initial_shadow_vs_dense": paired_statistics(initial_dense_translation, initial_dense_rotation),
            "optimized_shadow_vs_dense": paired_statistics(optimized_dense_translation, optimized_dense_rotation),
            "optimized_translation_proxy_change_counts": {
                "lower_than_reference": optimized_proxy_better,
                "higher_than_reference": optimized_proxy_worse,
                "equal_to_reference": optimized_proxy_tied,
            },
        },
        "map_geometry_deformation": {
            "interpretation": "same_local_points_transformed_by_reference_and_optimized_shadow_poses",
            "complete": geometry_complete,
            "empty_cloud_keyframe_count": empty_cloud_keyframe_count,
            "max_points_per_keyframe": max_points_per_keyframe,
            "source_cloud_point_count": sum(row["cloud_point_count"] for row in geometry_rows),
            "sample_count": len(all_displacements),
            "same_point_displacement_m": statistics(all_displacements),
            "by_submap": [
                {
                    "submap_id": submap_id,
                    "same_point_displacement_m": statistics(values),
                }
                for submap_id, values in sorted(displacements_by_submap.items())
            ],
        },
        "artifacts": {
            "trajectory_csv": str(trajectory_csv.resolve()),
            "map_geometry_csv": str(geometry_csv.resolve()),
            "reference_map_samples_pcd": str(reference_pcd.resolve()),
            "optimized_shadow_map_samples_pcd": str(shadow_pcd.resolve()),
            "top_view_overlay_svg": str(overview_svg.resolve()),
            "preview_point_limit_per_surface": preview_points,
            "displacement_vector_magnification": vector_magnification,
        },
        "limitations": [
            "native_dense_trajectory_is_same_map_internal_consistency_not_independent_ground_truth",
            "same_point_displacement_measures_deformation_not_quality",
            "lower_nonlinear_objective_does_not_prove_better_physical_geometry",
            "trial_jsonl_v1_does_not_embed_the_reviewed_map_hash",
            "review_ready_does_not_authorize_shadow_writeback_or_default_changes",
        ],
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--trial-jsonl", type=Path, required=True)
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--proto-module-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-dense-time-error-ms", type=float, required=True)
    parser.add_argument("--max-points-per-keyframe", type=int, default=512)
    parser.add_argument("--preview-points", type=int, default=20000)
    parser.add_argument("--vector-magnification", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = compare(
            args.qualification,
            args.trial_jsonl,
            args.map_path,
            args.proto_module_dir,
            args.output_dir,
            args.max_dense_time_error_ms,
            args.max_points_per_keyframe,
            args.preview_points,
            args.vector_magnification,
        )
    except (ComparisonError, OSError) as error:
        failure = {
            "schema": REPORT_SCHEMA,
            "classification": INVALID,
            "decision_reason": str(error),
            "writeback_authorized": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return EXIT_CODES[INVALID]
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_CODES[report["classification"]]


if __name__ == "__main__":
    raise SystemExit(main())
