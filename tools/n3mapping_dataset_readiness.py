#!/usr/bin/env python3
"""Inventory relocalization datasets and prove map/query spatial overlap.

The report is deliberately metadata-only: point-cloud payloads are not read.  A
LiDAR inventory hash covers sorted relative paths and file sizes; small GT and
calibration files receive full content hashes.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{path.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def bounds(points: list[tuple[float, float, float]]) -> dict[str, list[float]] | None:
    if not points:
        return None
    return {
        "min_xyz": [min(point[i] for point in points) for i in range(3)],
        "max_xyz": [max(point[i] for point in points) for i in range(3)],
    }


def spatial_coverage(
    map_points: list[tuple[float, float, float]],
    query_points: list[tuple[float, float, float]],
    radius_m: float,
) -> dict[str, Any]:
    if not map_points or not query_points:
        return {
            "map_count": len(map_points),
            "query_count": len(query_points),
            "covered_query_count": 0,
            "covered_query_rate": 0.0,
            "radius_m": radius_m,
        }
    cell_size = radius_m
    grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for point in map_points:
        cell = tuple(math.floor(value / cell_size) for value in point)
        grid.setdefault(cell, []).append(point)
    radius_sq = radius_m * radius_m
    covered = 0
    for query in query_points:
        center = tuple(math.floor(value / cell_size) for value in query)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for point in grid.get(
                        (center[0] + dx, center[1] + dy, center[2] + dz), []
                    ):
                        distance_sq = sum((query[i] - point[i]) ** 2 for i in range(3))
                        if distance_sq <= radius_sq:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        covered += int(found)
    return {
        "map_count": len(map_points),
        "query_count": len(query_points),
        "covered_query_count": covered,
        "covered_query_rate": covered / len(query_points),
        "radius_m": radius_m,
    }


def coverage_is_sufficient(
    coverage: dict[str, Any], min_overlap_count: int, min_overlap_rate: float
) -> bool:
    return (
        coverage["covered_query_count"] >= min_overlap_count
        and coverage["covered_query_rate"] >= min_overlap_rate
    )


def read_kitti_poses(path: Path) -> dict[int, tuple[float, float, float]]:
    poses: dict[int, tuple[float, float, float]] = {}
    if not path.is_file():
        return poses
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 13:
            continue
        try:
            frame_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        poses[frame_id] = (values[3], values[7], values[11])
    return poses


def list_kitti_lidar(sequence_dir: Path) -> dict[int, Path]:
    data_dir = sequence_dir / "velodyne_points" / "data"
    result: dict[int, Path] = {}
    if not data_dir.is_dir():
        return result
    for path in data_dir.glob("*.bin"):
        try:
            result[int(path.stem)] = path
        except ValueError:
            continue
    return result


def inspect_kitti(
    root: Path, radius_m: float, min_overlap_count: int, min_overlap_rate: float
) -> dict[str, Any]:
    raw_root = root / "data_3d_raw"
    pose_root = root / "data_poses"
    sequences: list[dict[str, Any]] = []
    positions_by_sequence: dict[str, list[tuple[float, float, float]]] = {}
    for sequence_dir in sorted(raw_root.glob("*_sync")) if raw_root.is_dir() else []:
        sequence = sequence_dir.name
        lidar = list_kitti_lidar(sequence_dir)
        pose_path = pose_root / sequence / "poses.txt"
        poses = read_kitti_poses(pose_path)
        common_ids = sorted(set(lidar) & set(poses))
        positions = [poses[frame_id] for frame_id in common_ids]
        positions_by_sequence[sequence] = positions
        split = max(1, len(positions) // 2) if positions else 0
        coverage = spatial_coverage(positions[:split], positions[split:], radius_m)
        coverage["benchmark_candidate"] = coverage_is_sufficient(
            coverage, min_overlap_count, min_overlap_rate
        )
        lidar_paths = list(lidar.values())
        sequences.append(
            {
                "sequence": sequence,
                "lidar_dir": str(sequence_dir / "velodyne_points" / "data"),
                "lidar_frame_count": len(lidar),
                "gt_path": str(pose_path),
                "gt_pose_count": len(poses),
                "common_frame_count": len(common_ids),
                "dropped_lidar_count": len(set(lidar) - set(poses)),
                "lidar_inventory_sha256": inventory_sha256(lidar_paths, root),
                "lidar_hash_scope": "relative_path_and_size",
                "gt_sha256": sha256_file(pose_path) if pose_path.is_file() else None,
                "position_bounds": bounds(positions),
                "same_session_half_split": coverage,
                "recorded_lio": False,
                "real_timestamps": False,
                "formal_ready": False,
                "evidence_role": "oracle_gt_same_session",
            }
        )

    overlaps: list[dict[str, Any]] = []
    names = sorted(positions_by_sequence)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            first_to_second = spatial_coverage(
                positions_by_sequence[second], positions_by_sequence[first], radius_m
            )
            second_to_first = spatial_coverage(
                positions_by_sequence[first], positions_by_sequence[second], radius_m
            )
            a_as_query = coverage_is_sufficient(
                first_to_second, min_overlap_count, min_overlap_rate
            )
            b_as_query = coverage_is_sufficient(
                second_to_first, min_overlap_count, min_overlap_rate
            )
            overlaps.append(
                {
                    "sequence_a": first,
                    "sequence_b": second,
                    "a_covered_by_b_count": first_to_second["covered_query_count"],
                    "a_covered_by_b_rate": first_to_second["covered_query_rate"],
                    "b_covered_by_a_count": second_to_first["covered_query_count"],
                    "b_covered_by_a_rate": second_to_first["covered_query_rate"],
                    "radius_m": radius_m,
                    "a_as_query_candidate": a_as_query,
                    "b_as_query_candidate": b_as_query,
                    "positive_pair_candidate": a_as_query or b_as_query,
                }
            )
    calibration_files = sorted((root / "calibration").glob("*"))
    calibration_files = [path for path in calibration_files if path.is_file()]
    return {
        "root": str(root),
        "available": raw_root.is_dir() and pose_root.is_dir(),
        "sequences": sequences,
        "cross_sequence_overlap": overlaps,
        "overlap_acceptance": {
            "min_covered_query_count": min_overlap_count,
            "min_covered_query_rate": min_overlap_rate,
            "radius_m": radius_m,
        },
        "calibration": [
            {"path": str(path), "sha256": sha256_file(path)} for path in calibration_files
        ],
        "limitations": [
            "local copy has no recorded LIO trajectory",
            "local copy has no original IMU/OXTS stream or real LiDAR timestamps",
            "same-session splits are diagnostic, not cross-session product evidence",
        ],
    }


def read_tum(path: Path) -> list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]]:
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 8:
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        records.append((values[0], tuple(values[1:4]), tuple(values[4:8])))
    return sorted(records, key=lambda item: item[0])


def align_m2dgr(
    lidar: list[tuple[float, Path]],
    gt: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]],
    max_time_diff_s: float,
) -> list[tuple[float, float, tuple[float, float, float]]]:
    stamps = [record[0] for record in gt]
    aligned = []
    for lidar_stamp, _ in lidar:
        position = bisect.bisect_left(stamps, lidar_stamp)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(gt)]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(stamps[index] - lidar_stamp))
        difference = abs(stamps[best] - lidar_stamp)
        if difference <= max_time_diff_s:
            aligned.append((lidar_stamp, difference, gt[best][1]))
    return aligned


def inspect_m2dgr(
    root: Path,
    radius_m: float,
    max_time_diff_s: float,
    min_overlap_count: int,
    min_overlap_rate: float,
) -> dict[str, Any]:
    sequences = []
    for sequence_dir in sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []:
        lidar_dir = sequence_dir / "velodyne_points"
        lidar_paths = sorted(list(lidar_dir.glob("*.bin")) + list(lidar_dir.glob("*.pcd")))
        lidar = []
        for path in lidar_paths:
            try:
                lidar.append((float(path.stem), path))
            except ValueError:
                continue
        raw_gt_path = sequence_dir / f"{sequence_dir.name}.txt"
        fallback_path = sequence_dir / "groundtruth_yaw_fallback.txt"
        eval_gt_path = fallback_path if fallback_path.is_file() else raw_gt_path
        raw_gt = read_tum(raw_gt_path)
        eval_gt = read_tum(eval_gt_path)
        valid_raw_quaternions = sum(
            0.5 <= math.sqrt(sum(value * value for value in record[2])) <= 1.5
            for record in raw_gt
        )
        aligned = align_m2dgr(lidar, eval_gt, max_time_diff_s)
        positions = [record[2] for record in aligned]
        split = max(1, len(positions) // 2) if positions else 0
        coverage = spatial_coverage(positions[:split], positions[split:], radius_m)
        coverage["benchmark_candidate"] = coverage_is_sufficient(
            coverage, min_overlap_count, min_overlap_rate
        )
        derived_yaw = fallback_path.is_file()
        sequences.append(
            {
                "sequence": sequence_dir.name,
                "lidar_dir": str(lidar_dir),
                "lidar_frame_count": len(lidar),
                "raw_gt_path": str(raw_gt_path),
                "eval_gt_path": str(eval_gt_path),
                "raw_gt_pose_count": len(raw_gt),
                "raw_gt_valid_quaternion_count": valid_raw_quaternions,
                "eval_gt_pose_count": len(eval_gt),
                "aligned_frame_count": len(aligned),
                "alignment_rate": len(aligned) / len(lidar) if lidar else 0.0,
                "max_time_diff_s": max_time_diff_s,
                "lidar_inventory_sha256": inventory_sha256(lidar_paths, root),
                "lidar_hash_scope": "relative_path_and_size",
                "raw_gt_sha256": sha256_file(raw_gt_path) if raw_gt_path.is_file() else None,
                "eval_gt_sha256": sha256_file(eval_gt_path) if eval_gt_path.is_file() else None,
                "orientation_quality": (
                    "missing"
                    if not raw_gt
                    else "trajectory_derived_yaw"
                    if derived_yaw
                    else "measured_quaternion"
                ),
                "position_bounds": bounds(positions),
                "same_session_half_split": coverage,
                "recorded_lio": False,
                "extrinsic_frozen": False,
                "formal_ready": False,
                "evidence_role": (
                    "position_first_oracle_same_session"
                    if derived_yaw
                    else "provisional_oracle_same_session"
                ),
            }
        )
    return {
        "root": str(root),
        "available": root.is_dir(),
        "sequences": sequences,
        "limitations": [
            "sessions do not contain a recorded LIO trajectory",
            "LiDAR-IMU extrinsic and frontend configuration are not frozen",
            "hall_05 orientation is trajectory-derived and cannot grade full attitude",
            "hall_05 and gate_02 are different scenes and cannot form a positive map/query pair",
        ],
    }


def build_report(
    kitti_root: Path | None,
    m2dgr_root: Path | None,
    overlap_radius_m: float,
    m2dgr_max_time_diff_s: float,
    min_overlap_count: int = 100,
    min_overlap_rate: float = 0.02,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "relocalization_dataset_readiness",
        "hash_contract": {
            "small_metadata": "full_file_sha256",
            "lidar": "sha256_of_sorted_relative_path_and_size_inventory",
            "note": "selected frozen episodes must add full payload hashes before formal use",
        },
        "kitti360": (
            inspect_kitti(kitti_root, overlap_radius_m, min_overlap_count, min_overlap_rate)
            if kitti_root
            else None
        ),
        "m2dgr": (
            inspect_m2dgr(
                m2dgr_root,
                overlap_radius_m,
                m2dgr_max_time_diff_s,
                min_overlap_count,
                min_overlap_rate,
            )
            if m2dgr_root
            else None
        ),
        "formal_gate_ready": False,
        "formal_gate_blockers": [
            "no independent cross-session map/query pair with frozen alignment",
            "no recorded LIO odometry artifact",
            "selected episodes do not yet have full point-cloud payload hashes",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", type=Path)
    parser.add_argument("--m2dgr-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-radius-m", type=float, default=5.0)
    parser.add_argument("--m2dgr-max-time-diff-s", type=float, default=0.05)
    parser.add_argument("--min-overlap-count", type=int, default=100)
    parser.add_argument("--min-overlap-rate", type=float, default=0.02)
    args = parser.parse_args()
    if not args.kitti_root and not args.m2dgr_root:
        parser.error("at least one of --kitti-root or --m2dgr-root is required")
    if args.overlap_radius_m <= 0.0:
        parser.error("--overlap-radius-m must be positive")
    if args.m2dgr_max_time_diff_s < 0.0:
        parser.error("--m2dgr-max-time-diff-s must be non-negative")
    if args.min_overlap_count <= 0:
        parser.error("--min-overlap-count must be positive")
    if not 0.0 < args.min_overlap_rate <= 1.0:
        parser.error("--min-overlap-rate must be in (0, 1]")
    return args


def main() -> int:
    args = parse_args()
    report = build_report(
        args.kitti_root,
        args.m2dgr_root,
        args.overlap_radius_m,
        args.m2dgr_max_time_diff_s,
        args.min_overlap_count,
        args.min_overlap_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "formal_gate_ready": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
