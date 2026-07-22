#!/usr/bin/env python3
"""Prepare one frozen episode for the multicolor real-scan review evaluator."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import (
    _expected_behaviors,
    _verify_benchmark_output,
    _verify_manifest,
)


def _ensure_fresh(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"output must be a fresh directory: {path}")
    else:
        path.mkdir(parents=True)


def _matrix_from_twelve(values: list[float]) -> np.ndarray:
    if len(values) != 12 or not all(math.isfinite(value) for value in values):
        raise ValueError("pose matrix must contain 12 finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return result


def _matrix_from_xyzw(
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid pose quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = translation
    return result


def _xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _read_kitti_world_lidar(root: Path, sequence: str) -> dict[str, np.ndarray]:
    cam_to_velo_values = [
        float(value)
        for value in (root / "calibration/calib_cam_to_velo.txt").read_text(
            encoding="utf-8"
        ).split()[:12]
    ]
    T_cam_velo = _matrix_from_twelve(cam_to_velo_values)
    T_pose_cam = None
    for line in (root / "calibration/calib_cam_to_pose.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("image_00:"):
            T_pose_cam = _matrix_from_twelve(
                [float(value) for value in line.split(":", 1)[1].split()[:12]]
            )
            break
    if T_pose_cam is None:
        raise ValueError("KITTI-360 official calibration has no image_00 pose")
    T_velo_cam = np.linalg.inv(T_cam_velo)
    poses = {}
    pose_path = root / "data_poses" / sequence / "poses.txt"
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 13:
            continue
        poses[str(int(fields[0]))] = (
            _matrix_from_twelve([float(value) for value in fields[1:]])
            @ T_pose_cam
            @ T_velo_cam
        )
    return poses


def _ecef_to_enu_rotation(origin: np.ndarray) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    x, y, z = (float(value) for value in origin)
    longitude = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    latitude = math.atan2(z, horizontal * (1.0 - e2))
    for _ in range(8):
        sin_latitude = math.sin(latitude)
        prime_vertical = a / math.sqrt(1.0 - e2 * sin_latitude * sin_latitude)
        height = horizontal / math.cos(latitude) - prime_vertical
        latitude = math.atan2(
            z, horizontal * (1.0 - e2 * prime_vertical / (prime_vertical + height))
        )
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    return np.asarray(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def _read_m2dgr_world_lidar(gt_path: Path) -> tuple[list[float], list[np.ndarray]]:
    stamps = []
    poses = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 8:
            continue
        values = [float(value) for value in fields]
        stamps.append(values[0])
        poses.append(_matrix_from_xyzw(tuple(values[1:4]), tuple(values[4:8])))
    if not poses:
        raise ValueError(f"M2DGR ground truth is empty: {gt_path}")
    origin = poses[0][:3, 3].copy()
    if float(np.linalg.norm(origin)) > 1.0e6:
        R_enu_ecef = _ecef_to_enu_rotation(origin)
        for pose in poses:
            pose[:3, 3] = R_enu_ecef @ (pose[:3, 3] - origin)
    T_local_world = np.linalg.inv(poses[0])
    return stamps, [T_local_world @ pose for pose in poses]


def _write_binary_pcd(path: Path, cloud: np.ndarray) -> None:
    cloud = np.asarray(cloud, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {cloud.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {cloud.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(cloud.tobytes(order="C"))


def _convert_cloud(source: Path, destination: Path) -> None:
    if source.suffix == ".pcd":
        shutil.copyfile(source, destination)
        return
    cloud = np.fromfile(source, dtype="<f4")
    if cloud.size % 4 != 0:
        raise ValueError(f"LiDAR bin does not contain XYZI float32 records: {source}")
    _write_binary_pcd(destination, cloud.reshape(-1, 4))


def prepare_review(
    manifest_dir: Path, benchmark_dir: Path, episode_id: str, output: Path
) -> dict[str, Any]:
    _ensure_fresh(output)
    manifest, rows = _verify_manifest(manifest_dir)
    _verify_benchmark_output(benchmark_dir)
    query_rows = [
        row
        for row in rows
        if row.get("role") == "query" and row.get("episode_id") == episode_id
    ]
    if not query_rows:
        raise ValueError(f"episode is absent from frozen manifest: {episode_id}")
    with (benchmark_dir / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        episode_results = {
            row["episode_id"]: row for row in csv.DictReader(stream)
        }
    if episode_id not in episode_results:
        raise ValueError(f"episode is absent from baseline output: {episode_id}")
    summary = json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))
    fake = summary["fake_map_to_odom"]
    yaw = math.radians(float(fake["yaw_deg"]))
    T_map_odom = np.eye(4, dtype=np.float64)
    T_map_odom[:3, :3] = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    T_map_odom[:3, 3] = [float(fake["x_m"]), float(fake["y_m"]), 0.0]

    root = Path(manifest["root"])
    if manifest["dataset"] == "kitti360":
        world_poses = _read_kitti_world_lidar(root, manifest["query_sequence"])
        pose_for = lambda token: world_poses[str(int(token))]
        stamp_for = lambda token, index: int(token) * 100_000_000
    elif manifest["dataset"] == "m2dgr":
        stamps, world_pose_list = _read_m2dgr_world_lidar(Path(manifest["query_gt_path"]))

        def pose_for(token: str) -> np.ndarray:
            stamp = float(token)
            insertion = bisect.bisect_left(stamps, stamp)
            candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(stamps)]
            best = min(candidates, key=lambda i: abs(stamps[i] - stamp))
            if abs(stamps[best] - stamp) > float(manifest["m2dgr_max_time_diff_s"]):
                raise ValueError(f"M2DGR review frame exceeds frozen alignment tolerance: {token}")
            return world_pose_list[best]

        stamp_for = lambda token, index: int(round(float(token) * 1_000_000_000))
    else:
        raise ValueError(f"unsupported dataset: {manifest['dataset']}")

    review_rows = []
    for index, row in enumerate(query_rows):
        source = root / row["relative_cloud_path"]
        pcd_name = f"frame_{index:03d}.pcd"
        destination = output / pcd_name
        _convert_cloud(source, destination)
        T_odom_lidar = np.linalg.inv(T_map_odom) @ pose_for(row["frame_token"])
        qx, qy, qz, qw = _xyzw_from_rotation(T_odom_lidar[:3, :3])
        review_rows.append(
            {
                "episode_id": episode_id,
                "frame_index": index,
                "stamp_ns": stamp_for(row["frame_token"], index),
                "pcd_path": pcd_name,
                "tx": T_odom_lidar[0, 3],
                "ty": T_odom_lidar[1, 3],
                "tz": T_odom_lidar[2, 3],
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
            }
        )
    frames_path = output / "frames.csv"
    with frames_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    metadata = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "episode_id": episode_id,
        "baseline_outcome": episode_results[episode_id]["outcome"],
        "expected_behavior": _expected_behaviors(manifest, rows)[episode_id],
        "frame_count": len(review_rows),
        "source_manifest_sha256": sha256_file(manifest_dir / "dataset_manifest.json"),
        "source_baseline_summary_sha256": sha256_file(benchmark_dir / "summary.json"),
        "frames_sha256": sha256_file(frames_path),
        "pose_contract": "evaluator_identical_gt_to_fake_odom_transform",
    }
    (output / "prepare.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_review(
                args.manifest_dir, args.benchmark_dir, args.episode_id, args.output
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
