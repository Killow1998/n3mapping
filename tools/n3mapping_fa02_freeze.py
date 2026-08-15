#!/usr/bin/env python3
"""Freeze FA-02 dataset episodes and full selected-cloud content hashes."""

from __future__ import annotations

import argparse
import bisect
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo


CONTRACT_SCHEMA = "n3mapping_final_acceptance_contract_v1"
MANIFEST_SCHEMA = "n3mapping_fa02_input_manifest_v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"input is not a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def select_kitti360(root: Path, episode: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    sequence = str(episode["sequence"])
    lidar_dir = root / "data_3d_raw" / sequence / "velodyne_points" / "data"
    gt_path = root / "data_poses" / sequence / "poses.txt"
    lidar_by_id: dict[int, Path] = {}
    for path in sorted(lidar_dir.glob("*.bin")):
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        if frame_id in lidar_by_id:
            raise ValueError(f"duplicate KITTI-360 frame id {frame_id}")
        lidar_by_id[frame_id] = path
    gt_ids: set[int] = set()
    for line_number, line in enumerate(gt_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            frame_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"{gt_path}:{line_number}: invalid frame id") from exc
        if frame_id in gt_ids:
            raise ValueError(f"duplicate KITTI-360 GT frame id {frame_id}")
        gt_ids.add(frame_id)
    common = sorted(set(lidar_by_id) & gt_ids)
    stride = int(episode["stride"])
    start = int(episode["start_index"])
    maximum = int(episode["max_frames"])
    selected_ids = common[::stride][start : start + maximum]
    if len(selected_ids) != maximum:
        raise ValueError(
            f"{episode['id']}: selected {len(selected_ids)} KITTI-360 frames, expected {maximum}"
        )
    calibration_dir = root / "calibration"
    calibration_files = [
        file_evidence(path)
        for path in sorted(calibration_dir.glob("*.txt"))
    ]
    required = {"calib_cam_to_velo.txt", "calib_cam_to_pose.txt"}
    present = {Path(item["path"]).name for item in calibration_files}
    if not required.issubset(present):
        raise ValueError(f"KITTI-360 official calibration is incomplete: {required - present}")
    return [lidar_by_id[frame_id] for frame_id in selected_ids], {
        "gt": file_evidence(gt_path),
        "calibration": calibration_files,
        "alignment_input_lidar_count": len(lidar_by_id),
        "alignment_input_gt_count": len(gt_ids),
        "alignment_common_count": len(common),
    }


def read_tum_stamps(path: Path) -> list[float]:
    stamps: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 TUM fields")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite TUM value")
        stamps.append(values[0])
    if not stamps or stamps != sorted(stamps) or len(stamps) != len(set(stamps)):
        raise ValueError(f"{path}: GT stamps must be non-empty, unique, and ordered")
    return stamps


def select_m2dgr(
    root: Path,
    source_repo: Path,
    episode: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    sequence = str(episode["sequence"])
    sequence_dir = root / sequence
    lidar_dir = sequence_dir / "velodyne_points"
    lidar: list[tuple[float, Path]] = []
    for path in sorted(list(lidar_dir.glob("*.bin")) + list(lidar_dir.glob("*.pcd"))):
        try:
            stamp = float(path.stem)
        except ValueError:
            continue
        if not math.isfinite(stamp):
            raise ValueError(f"non-finite M2DGR cloud timestamp: {path}")
        lidar.append((stamp, path))
    lidar.sort(key=lambda item: item[0])
    if not lidar or len({stamp for stamp, _ in lidar}) != len(lidar):
        raise ValueError(f"{sequence}: LiDAR timestamps must be non-empty and unique")
    gt_path = sequence_dir / str(episode["gt_file"])
    gt_stamps = read_tum_stamps(gt_path)
    max_time_diff = float(episode["max_time_diff_s"])
    aligned: list[Path] = []
    pose_index = 0
    for stamp, path in lidar:
        while (
            pose_index + 1 < len(gt_stamps)
            and abs(gt_stamps[pose_index + 1] - stamp)
            <= abs(gt_stamps[pose_index] - stamp)
        ):
            pose_index += 1
        if abs(gt_stamps[pose_index] - stamp) <= max_time_diff:
            aligned.append(path)
    stride = int(episode["stride"])
    start = int(episode["start_index"])
    maximum = int(episode["max_frames"])
    selected = aligned[::stride][start : start + maximum]
    if len(selected) != maximum:
        raise ValueError(
            f"{episode['id']}: selected {len(selected)} M2DGR frames, expected {maximum}"
        )
    calibration_path = source_repo / str(episode["calibration_file"])
    return selected, {
        "gt": file_evidence(gt_path),
        "calibration": [file_evidence(calibration_path)],
        "gt_sensor_frame": str(episode["gt_sensor_frame"]),
        "alignment_input_lidar_count": len(lidar),
        "alignment_input_gt_count": len(gt_stamps),
        "alignment_common_count": len(aligned),
        "max_time_diff_s": max_time_diff,
    }


def aggregate_cloud_digest(clouds: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for cloud in clouds:
        digest.update(
            f"{cloud['frame_token']}\0{cloud['size_bytes']}\0{cloud['sha256']}\n".encode()
        )
    return digest.hexdigest()


def write_frame_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["episode_id", "role", "frame_token"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def freeze(contract_path: Path, output: Path, source_repo: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    source_repo = source_repo.resolve(strict=True)
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("acceptance contract schema mismatch")
    fa02 = contract.get("fa02")
    if not isinstance(fa02, dict):
        raise ValueError("acceptance contract has no FA-02 section")
    status = git_output(source_repo, "status", "--porcelain")
    if status:
        raise ValueError("source repository must be clean before freezing FA-02 inputs")
    source_commit = git_output(source_repo, "rev-parse", "HEAD")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    roots = {
        name: Path(value).resolve(strict=True)
        for name, value in fa02["dataset_roots"].items()
    }
    episode_rows: list[dict[str, str]] = []
    episode_manifests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for episode in fa02["episodes"]:
        episode_id = str(episode["id"])
        if episode_id in seen_ids:
            raise ValueError(f"duplicate FA-02 episode id: {episode_id}")
        seen_ids.add(episode_id)
        dataset = str(episode["dataset"])
        if dataset == "kitti360":
            paths, inputs = select_kitti360(roots[dataset], episode)
        elif dataset == "m2dgr":
            paths, inputs = select_m2dgr(roots[dataset], source_repo, episode)
        else:
            raise ValueError(f"unsupported FA-02 dataset: {dataset}")
        clouds: list[dict[str, Any]] = []
        for ordinal, path in enumerate(paths):
            evidence = file_evidence(path)
            evidence.update(
                {
                    "ordinal": ordinal,
                    "frame_token": path.stem,
                }
            )
            clouds.append(evidence)
            episode_rows.append(
                {"episode_id": episode_id, "role": "map", "frame_token": path.stem}
            )
        episode_manifests.append(
            {
                "id": episode_id,
                "dataset": dataset,
                "sequence": str(episode["sequence"]),
                "expected_role": str(episode["expected_role"]),
                "attitude_authoritative": bool(episode["attitude_authoritative"]),
                "selection": {
                    "start_index": int(episode["start_index"]),
                    "stride": int(episode["stride"]),
                    "max_frames": int(episode["max_frames"]),
                    "input_voxel_size_m": float(episode["input_voxel_size_m"]),
                },
                "inputs": inputs,
                "selected_cloud_count": len(clouds),
                "selected_clouds_sha256": aggregate_cloud_digest(clouds),
                "selected_clouds": clouds,
            }
        )

    frames_path = output / "episode_frames.csv"
    write_frame_manifest(frames_path, episode_rows)
    timezone = ZoneInfo(str(contract.get("recorded_timezone", "America/Los_Angeles")))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone).isoformat(),
        "contract": file_evidence(contract_path),
        "source_repo": str(source_repo),
        "source_commit": source_commit,
        "source_clean": True,
        "episode_frames": file_evidence(frames_path),
        "episodes": episode_manifests,
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        manifest = freeze(args.contract, args.output, args.source_repo)
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "source_commit": manifest["source_commit"],
                    "episodes": {
                        episode["id"]: episode["selected_cloud_count"]
                        for episode in manifest["episodes"]
                    },
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"n3mapping_fa02_freeze: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
