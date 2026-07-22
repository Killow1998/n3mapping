#!/usr/bin/env python3
"""Freeze spatially valid relocalization map/query episodes with payload hashes."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from n3mapping_dataset_readiness import read_kitti_poses, read_tum, sha256_file


KITTI360_OFFICIAL_CALIBRATION_FILES = (
    "calibration/calib_cam_to_pose.txt",
    "calibration/calib_cam_to_velo.txt",
)


@dataclass(frozen=True)
class Frame:
    token: str
    cloud: Path
    position: tuple[float, float, float]


def _kitti_frames(root: Path, sequence: str) -> tuple[list[Frame], Path]:
    lidar_dir = root / "data_3d_raw" / sequence / "velodyne_points" / "data"
    gt_path = root / "data_poses" / sequence / "poses.txt"
    poses = read_kitti_poses(gt_path)
    frames = []
    for cloud in sorted(lidar_dir.glob("*.bin")):
        try:
            frame_id = int(cloud.stem)
        except ValueError:
            continue
        if frame_id in poses:
            frames.append(Frame(str(frame_id), cloud, poses[frame_id]))
    return frames, gt_path


def _m2dgr_frames(
    root: Path, sequence: str, max_time_diff_s: float
) -> tuple[list[Frame], Path]:
    sequence_dir = root / sequence
    lidar_dir = sequence_dir / "velodyne_points"
    fallback = sequence_dir / "groundtruth_yaw_fallback.txt"
    gt_path = fallback if fallback.is_file() else sequence_dir / f"{sequence}.txt"
    gt = read_tum(gt_path)
    gt_stamps = [record[0] for record in gt]
    frames = []
    for cloud in sorted(list(lidar_dir.glob("*.bin")) + list(lidar_dir.glob("*.pcd"))):
        try:
            stamp = float(cloud.stem)
        except ValueError:
            continue
        insertion = bisect.bisect_left(gt_stamps, stamp)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(gt)]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(gt_stamps[index] - stamp))
        if abs(gt_stamps[best] - stamp) <= max_time_diff_s:
            frames.append(Frame(cloud.stem, cloud, gt[best][1]))
    return frames, gt_path


def _near_any(
    point: tuple[float, float, float],
    references: Iterable[tuple[float, float, float]],
    radius_m: float,
) -> bool:
    radius_sq = radius_m * radius_m
    return any(sum((point[i] - other[i]) ** 2 for i in range(3)) <= radius_sq for other in references)


def _covered_mask(map_frames: list[Frame], query_frames: list[Frame], radius_m: float) -> list[bool]:
    if not map_frames:
        return [False] * len(query_frames)
    cell_size = radius_m
    grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for frame in map_frames:
        cell = tuple(math.floor(value / cell_size) for value in frame.position)
        grid.setdefault(cell, []).append(frame.position)
    radius_sq = radius_m * radius_m
    result = []
    for frame in query_frames:
        center = tuple(math.floor(value / cell_size) for value in frame.position)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for point in grid.get(
                        (center[0] + dx, center[1] + dy, center[2] + dz), []
                    ):
                        if sum((frame.position[i] - point[i]) ** 2 for i in range(3)) <= radius_sq:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        result.append(found)
    return result


def _consecutive_windows(mask: list[bool], size: int, max_episodes: int) -> list[tuple[int, int]]:
    windows = []
    run_start = None
    for index in range(len(mask) + 1):
        covered = index < len(mask) and mask[index]
        if covered and run_start is None:
            run_start = index
        if not covered and run_start is not None:
            cursor = run_start
            while cursor + size <= index and len(windows) < max_episodes:
                windows.append((cursor, cursor + size))
                cursor += size
            run_start = None
        if len(windows) >= max_episodes:
            break
    return windows


def _uniform_cap(frames: list[Frame], limit: int) -> list[Frame]:
    if len(frames) <= limit:
        return frames
    if limit == 1:
        return [frames[0]]
    indices = sorted({round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)})
    return [frames[index] for index in indices]


def freeze_episodes(
    *,
    dataset: str,
    root: Path,
    map_sequence: str,
    query_sequence: str,
    output: Path,
    overlap_radius_m: float,
    map_context_radius_m: float,
    query_episode_frames: int,
    max_episodes: int,
    max_map_frames: int,
    map_stride: int,
    m2dgr_max_time_diff_s: float,
    expected_behavior: str = "lock",
) -> dict:
    if expected_behavior not in {"lock", "abstain"}:
        raise ValueError(f"unsupported expected behavior: {expected_behavior}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output}")
    if dataset == "kitti360":
        calibration_files = [
            root / relative for relative in KITTI360_OFFICIAL_CALIBRATION_FILES
        ]
        missing_calibration = [path for path in calibration_files if not path.is_file()]
        if missing_calibration:
            raise ValueError(
                "KITTI-360 official calibration is incomplete: "
                + ", ".join(str(path) for path in missing_calibration)
            )
        map_all, map_gt = _kitti_frames(root, map_sequence)
        query_all, query_gt = _kitti_frames(root, query_sequence)
    elif dataset == "m2dgr":
        if expected_behavior == "abstain":
            raise ValueError(
                "M2DGR cross-scene coordinates are not frozen; abstain episodes require KITTI-360"
            )
        if map_sequence != query_sequence:
            raise ValueError("M2DGR cross-sequence coordinates are not frozen; use one sequence")
        map_all, map_gt = _m2dgr_frames(root, map_sequence, m2dgr_max_time_diff_s)
        query_all, query_gt = map_all, map_gt
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not map_all or not query_all:
        raise ValueError("map/query sequence has no aligned frames")
    if expected_behavior == "abstain" and map_sequence == query_sequence:
        raise ValueError("abstain episodes require distinct map/query sequences")

    if map_sequence == query_sequence:
        split = len(map_all) // 2
        map_pool = map_all[:split]
        query_pool = query_all[split:]
    else:
        split = None
        map_pool = map_all
        query_pool = query_all

    covered = _covered_mask(map_pool, query_pool, overlap_radius_m)
    selection_mask = (
        covered if expected_behavior == "lock" else [not value for value in covered]
    )
    windows = _consecutive_windows(selection_mask, query_episode_frames, max_episodes)
    if not windows:
        relationship = "covered" if expected_behavior == "lock" else "uncovered"
        raise ValueError(f"no consecutive spatially {relationship} query episode found")
    episodes = [query_pool[start:end] for start, end in windows]
    query_positions = [frame.position for episode in episodes for frame in episode]
    if expected_behavior == "lock":
        contextual_map = [
            frame
            for frame in map_pool
            if _near_any(frame.position, query_positions, map_context_radius_m)
        ]
    else:
        contextual_map = map_pool
    contextual_map = contextual_map[::map_stride]
    contextual_map = _uniform_cap(contextual_map, max_map_frames)
    if len(contextual_map) < 5:
        raise ValueError("fewer than five map frames remain after spatial selection")
    frozen_query_frames = [frame for episode in episodes for frame in episode]
    frozen_covered = _covered_mask(
        contextual_map, frozen_query_frames, overlap_radius_m
    )
    if expected_behavior == "lock" and not all(frozen_covered):
        raise ValueError(
            "frozen map selection does not cover every query frame; "
            "increase map budget or reduce map stride"
        )
    if expected_behavior == "abstain" and any(frozen_covered):
        raise ValueError("frozen abstain episode unexpectedly overlaps the map")

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    payload_hashes: dict[str, str] = {}
    for frame in contextual_map:
        relative = frame.cloud.relative_to(root).as_posix()
        payload_hashes[relative] = sha256_file(frame.cloud)
        rows.append(
            {
                "episode_id": "map",
                "role": "map",
                "frame_token": frame.token,
                "relative_cloud_path": relative,
                "cloud_sha256": payload_hashes[relative],
                "x": frame.position[0],
                "y": frame.position[1],
                "z": frame.position[2],
            }
        )
    for episode_index, episode in enumerate(episodes):
        episode_id = f"query_{episode_index:03d}"
        for frame in episode:
            relative = frame.cloud.relative_to(root).as_posix()
            payload_hashes.setdefault(relative, sha256_file(frame.cloud))
            rows.append(
                {
                    "episode_id": episode_id,
                    "role": "query",
                    "frame_token": frame.token,
                    "relative_cloud_path": relative,
                    "cloud_sha256": payload_hashes[relative],
                    "x": frame.position[0],
                    "y": frame.position[1],
                    "z": frame.position[2],
                }
            )

    manifest_csv = output / "episode_frames.csv"
    fieldnames = [
        "episode_id",
        "role",
        "frame_token",
        "relative_cloud_path",
        "cloud_sha256",
        "x",
        "y",
        "z",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    map_tokens = {frame.token for frame in contextual_map}
    query_tokens = {frame.token for episode in episodes for frame in episode}
    manifest = {
        "schema_version": 3,
        "dataset": dataset,
        "root": str(root),
        "map_sequence": map_sequence,
        "query_sequence": query_sequence,
        "evidence_class": (
            "oracle_gt_same_session"
            if map_sequence == query_sequence
            else "oracle_gt_cross_drive"
        ),
        "map_query_disjoint": map_tokens.isdisjoint(query_tokens)
        if map_sequence == query_sequence
        else True,
        "same_sequence_split_index": split,
        "map_frame_count": len(contextual_map),
        "query_episode_count": len(episodes),
        "query_frames_per_episode": query_episode_frames,
        "query_frame_count": sum(len(episode) for episode in episodes),
        "expected_behavior": expected_behavior,
        "spatial_relationship": (
            "covered_within_radius" if expected_behavior == "lock" else "uncovered_within_radius"
        ),
        "source_covered_query_frame_count": sum(
            covered[start:end].count(True) for start, end in windows
        ),
        "frozen_covered_query_frame_count": frozen_covered.count(True),
        "overlap_radius_m": overlap_radius_m,
        "map_context_radius_m": map_context_radius_m,
        "map_stride": map_stride,
        "map_gt_path": str(map_gt),
        "query_gt_path": str(query_gt),
        "map_gt_sha256": sha256_file(map_gt),
        "query_gt_sha256": sha256_file(query_gt),
        "episode_frames_sha256": sha256_file(manifest_csv),
        "recorded_lio": False,
        "gt_runtime_access": True,
        "formal_gate_ready": False,
    }
    if dataset == "kitti360":
        manifest["kitti360_calibration"] = {
            "mode": "official",
            "files": [
                {
                    "relative_path": relative,
                    "sha256": sha256_file(root / relative),
                }
                for relative in KITTI360_OFFICIAL_CALIBRATION_FILES
            ],
        }
    if dataset == "m2dgr":
        manifest["m2dgr_max_time_diff_s"] = m2dgr_max_time_diff_s
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_lines = [
        f"{sha256_file(manifest_path)}  dataset_manifest.json",
        f"{sha256_file(manifest_csv)}  episode_frames.csv",
    ]
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kitti360", "m2dgr"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--map-sequence", required=True)
    parser.add_argument("--query-sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-radius-m", type=float, default=5.0)
    parser.add_argument("--map-context-radius-m", type=float, default=30.0)
    parser.add_argument("--query-episode-frames", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--max-map-frames", type=int, default=500)
    parser.add_argument("--map-stride", type=int, default=5)
    parser.add_argument("--m2dgr-max-time-diff-s", type=float, default=0.05)
    parser.add_argument(
        "--expected-behavior",
        choices=("lock", "abstain"),
        default="lock",
        help="Freeze spatially covered lock episodes or uncovered abstain episodes.",
    )
    args = parser.parse_args()
    positive = {
        "overlap_radius_m": args.overlap_radius_m,
        "map_context_radius_m": args.map_context_radius_m,
        "query_episode_frames": args.query_episode_frames,
        "max_episodes": args.max_episodes,
        "max_map_frames": args.max_map_frames,
        "map_stride": args.map_stride,
        "m2dgr_max_time_diff_s": args.m2dgr_max_time_diff_s,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    manifest = freeze_episodes(
        dataset=args.dataset,
        root=args.root,
        map_sequence=args.map_sequence,
        query_sequence=args.query_sequence,
        output=args.output,
        overlap_radius_m=args.overlap_radius_m,
        map_context_radius_m=args.map_context_radius_m,
        query_episode_frames=args.query_episode_frames,
        max_episodes=args.max_episodes,
        max_map_frames=args.max_map_frames,
        map_stride=args.map_stride,
        m2dgr_max_time_diff_s=args.m2dgr_max_time_diff_s,
        expected_behavior=args.expected_behavior,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
