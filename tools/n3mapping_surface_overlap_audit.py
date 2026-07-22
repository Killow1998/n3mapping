#!/usr/bin/env python3
"""Classify frozen relocalization episodes from GT-aligned visible surfaces."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import _verify_manifest
from n3mapping_episode_review_prepare import (
    _read_kitti_world_lidar,
    _read_m2dgr_world_lidar,
)


def _ensure_fresh(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"output must be a fresh directory: {path}")
    else:
        path.mkdir(parents=True)


def _load_pcd_xyz(path: Path) -> np.ndarray:
    metadata: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA declaration: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            metadata[key.upper()] = values
            if key.upper() == "DATA":
                break
        fields = metadata.get("FIELDS", [])
        if not {"x", "y", "z"}.issubset(fields):
            raise ValueError(f"PCD must contain x/y/z fields: {path}")
        data_kind = metadata["DATA"][0].lower()
        if data_kind == "ascii":
            values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
            return values[:, [fields.index(axis) for axis in ("x", "y", "z")]]
        if data_kind != "binary":
            raise ValueError(f"unsupported PCD DATA mode {data_kind!r}: {path}")
        sizes = [int(value) for value in metadata.get("SIZE", [])]
        types = metadata.get("TYPE", [])
        counts = [int(value) for value in metadata.get("COUNT", ["1"] * len(fields))]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"malformed PCD field metadata: {path}")
        dtype_fields = []
        for name, size, scalar_type, count in zip(fields, sizes, types, counts):
            type_code = {"F": "f", "I": "i", "U": "u"}.get(scalar_type.upper())
            if type_code is None or size not in (1, 2, 4, 8) or count <= 0:
                raise ValueError(f"unsupported PCD field {name!r}: {path}")
            scalar = np.dtype(f"<{type_code}{size}")
            dtype_fields.append((name, scalar, (count,)) if count > 1 else (name, scalar))
        points = int(metadata.get("POINTS", metadata.get("WIDTH", ["0"]))[0])
        records = np.frombuffer(stream.read(), dtype=np.dtype(dtype_fields), count=points)
        return np.column_stack([records[axis] for axis in ("x", "y", "z")]).astype(
            np.float64, copy=False
        )


def _load_xyz(path: Path) -> np.ndarray:
    if path.suffix == ".pcd":
        points = _load_pcd_xyz(path)
    elif path.suffix == ".bin":
        raw = np.fromfile(path, dtype="<f4")
        if raw.size % 4 != 0:
            raise ValueError(f"LiDAR bin does not contain XYZI float32 records: {path}")
        points = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    else:
        raise ValueError(f"unsupported point-cloud payload: {path}")
    points = points[np.all(np.isfinite(points), axis=1)]
    if not points.size:
        raise ValueError(f"point cloud has no finite XYZ samples: {path}")
    return points


def _voxel_centers(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    occupied = np.unique(keys, axis=0)
    return (occupied.astype(np.float64) + 0.5) * voxel_size_m


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _pose_resolver(manifest: dict[str, Any]) -> Callable[[str, str], np.ndarray]:
    if manifest["dataset"] == "kitti360":
        by_sequence = {
            sequence: _read_kitti_world_lidar(Path(manifest["root"]), sequence)
            for sequence in {manifest["map_sequence"], manifest["query_sequence"]}
        }

        def resolve(sequence: str, token: str) -> np.ndarray:
            try:
                return by_sequence[sequence][str(int(token))]
            except KeyError as error:
                raise ValueError(f"KITTI-360 pose is absent for {sequence}/{token}") from error

        return resolve
    if manifest["dataset"] == "m2dgr":
        stamps, poses = _read_m2dgr_world_lidar(Path(manifest["query_gt_path"]))
        tolerance = float(manifest["m2dgr_max_time_diff_s"])

        def resolve(sequence: str, token: str) -> np.ndarray:
            del sequence
            stamp = float(token)
            insertion = bisect.bisect_left(stamps, stamp)
            candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(stamps)]
            if not candidates:
                raise ValueError(f"M2DGR pose is absent for {token}")
            best = min(candidates, key=lambda index: abs(stamps[index] - stamp))
            if abs(stamps[best] - stamp) > tolerance:
                raise ValueError(f"M2DGR pose exceeds frozen alignment tolerance: {token}")
            return poses[best]

        return resolve
    raise ValueError(f"unsupported dataset: {manifest['dataset']}")


def _radius_field(radius_m: float) -> str:
    return f"coverage_within_{radius_m:g}m".replace(".", "p")


def classify_surface_overlap(
    *,
    manifest_dir: Path,
    output: Path,
    voxel_size_m: float = 0.5,
    neighbor_radii_m: tuple[float, ...] = (0.3, 0.5, 1.0),
    decision_radius_m: float = 0.5,
    positive_min_frame_coverage: float = 0.8,
    negative_max_frame_coverage: float = 0.4,
) -> dict[str, Any]:
    _ensure_fresh(output)
    manifest, rows = _verify_manifest(manifest_dir)
    if not 0.0 <= negative_max_frame_coverage < positive_min_frame_coverage <= 1.0:
        raise ValueError("coverage labels require 0 <= negative < positive <= 1")
    radii = tuple(sorted(set(float(radius) for radius in neighbor_radii_m)))
    if (
        voxel_size_m <= 0.0
        or not radii
        or any(not math.isfinite(radius) or radius <= 0.0 for radius in radii)
        or decision_radius_m not in radii
    ):
        raise ValueError("voxel size and neighbor radii must be positive and include decision radius")

    root = Path(manifest["root"])
    pose_for = _pose_resolver(manifest)
    map_surfaces = []
    for row in rows:
        if row.get("role") != "map":
            continue
        points = _load_xyz(root / row["relative_cloud_path"])
        pose = pose_for(manifest["map_sequence"], row["frame_token"])
        map_surfaces.append(_transform(points, pose))
    if not map_surfaces:
        raise ValueError("manifest has no map frames")
    map_surface = _voxel_centers(np.concatenate(map_surfaces), voxel_size_m)
    tree = cKDTree(map_surface)

    frame_rows: list[dict[str, Any]] = []
    query_rows = [row for row in rows if row.get("role") == "query"]
    for row in query_rows:
        points = _load_xyz(root / row["relative_cloud_path"])
        pose = pose_for(manifest["query_sequence"], row["frame_token"])
        query_surface = _voxel_centers(_transform(points, pose), voxel_size_m)
        distances = tree.query(query_surface, k=1, workers=1)[0]
        result: dict[str, Any] = {
            "episode_id": row["episode_id"],
            "frame_token": row["frame_token"],
            "query_voxel_count": int(query_surface.shape[0]),
        }
        for radius in radii:
            result[_radius_field(radius)] = float(np.mean(distances <= radius))
        frame_rows.append(result)

    episode_ids = sorted({row["episode_id"] for row in frame_rows})
    decision_field = _radius_field(decision_radius_m)
    episode_rows: list[dict[str, Any]] = []
    expected_by_episode: dict[str, str] = {}
    for episode_id in episode_ids:
        coverages = [
            float(row[decision_field])
            for row in frame_rows
            if row["episode_id"] == episode_id
        ]
        minimum = min(coverages)
        maximum = max(coverages)
        mean = sum(coverages) / len(coverages)
        if minimum >= positive_min_frame_coverage:
            expected = "lock"
            label = "surface_repeat_positive"
        elif maximum <= negative_max_frame_coverage:
            expected = "abstain"
            label = "surface_nonoverlap_hard_negative"
        else:
            expected = "exclude"
            label = "ambiguous_surface_overlap"
        episode_rows.append(
            {
                "episode_id": episode_id,
                "label": label,
                "expected_behavior": expected,
                "frame_coverage_min": minimum,
                "frame_coverage_mean": mean,
                "frame_coverage_max": maximum,
            }
        )
        if expected != "exclude":
            expected_by_episode[episode_id] = expected
    ambiguous = [row["episode_id"] for row in episode_rows if row["expected_behavior"] == "exclude"]
    if not expected_by_episode:
        raise ValueError("every candidate episode has ambiguous surface overlap")

    labeled_rows = [
        row
        for row in rows
        if row.get("role") == "map" or row.get("episode_id") in expected_by_episode
    ]
    frames_output = output / "episode_frames.csv"
    with frames_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(labeled_rows)
    shutil.copyfile(
        manifest_dir / "dataset_manifest.json", output / "source_candidate_manifest.json"
    )
    shutil.copyfile(
        manifest_dir / "episode_frames.csv", output / "source_candidate_episode_frames.csv"
    )
    frames_path = output / "surface_overlap_frames.csv"
    with frames_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    episodes_path = output / "surface_overlap_episodes.csv"
    with episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    report = {
        "schema_version": 1,
        "evidence_class": "oracle_gt_visible_surface_overlap",
        "source_candidate_manifest_sha256": sha256_file(
            output / "source_candidate_manifest.json"
        ),
        "source_candidate_episode_frames_sha256": sha256_file(
            output / "source_candidate_episode_frames.csv"
        ),
        "episode_frames_sha256": sha256_file(frames_output),
        "map_voxel_count": int(map_surface.shape[0]),
        "source_query_frame_count": len(frame_rows),
        "labeled_query_frame_count": sum(
            row["episode_id"] in expected_by_episode for row in frame_rows
        ),
        "excluded_ambiguous_episode_ids": ambiguous,
        "classification_contract": {
            "voxel_size_m": voxel_size_m,
            "neighbor_radii_m": list(radii),
            "decision_radius_m": decision_radius_m,
            "positive_rule": "minimum frame coverage >= positive_min_frame_coverage",
            "positive_min_frame_coverage": positive_min_frame_coverage,
            "negative_rule": "maximum frame coverage <= negative_max_frame_coverage",
            "negative_max_frame_coverage": negative_max_frame_coverage,
            "ambiguous_policy": "exclude_from_derived_manifest",
        },
        "expected_behavior_by_episode": expected_by_episode,
        "surface_overlap_frames_sha256": sha256_file(frames_path),
        "surface_overlap_episodes_sha256": sha256_file(episodes_path),
    }
    report_path = output / "surface_overlap.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    behaviors = set(expected_by_episode.values())
    labeled_manifest = dict(manifest)
    map_positions = [
        tuple(float(row[axis]) for axis in ("x", "y", "z"))
        for row in labeled_rows
        if row.get("role") == "map"
    ]
    radius_sq = float(manifest["overlap_radius_m"]) ** 2
    labeled_query_rows = [row for row in labeled_rows if row.get("role") == "query"]
    frozen_covered = sum(
        any(
            sum((float(row[axis]) - mapped[index]) ** 2 for index, axis in enumerate(("x", "y", "z")))
            <= radius_sq
            for mapped in map_positions
        )
        for row in labeled_query_rows
    )
    labeled_manifest.pop("source_covered_query_frame_count", None)
    labeled_manifest.update(
        {
            "schema_version": 4,
            "expected_behavior": next(iter(behaviors)) if len(behaviors) == 1 else "mixed",
            "expected_behavior_by_episode": expected_by_episode,
            "spatial_relationship": "gt_visible_surface_classified",
            "query_episode_count": len(expected_by_episode),
            "query_frame_count": len(labeled_query_rows),
            "frozen_covered_query_frame_count": frozen_covered,
            "episode_frames_sha256": sha256_file(frames_output),
            "source_candidate_manifest_sha256": report["source_candidate_manifest_sha256"],
            "excluded_ambiguous_episode_ids": ambiguous,
            "surface_overlap_evidence": {
                "report": "surface_overlap.json",
                "report_sha256": sha256_file(report_path),
                "frames": "surface_overlap_frames.csv",
                "frames_sha256": sha256_file(frames_path),
                "episodes": "surface_overlap_episodes.csv",
                "episodes_sha256": sha256_file(episodes_path),
            },
        }
    )
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(labeled_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payloads = (
        "dataset_manifest.json",
        "episode_frames.csv",
        "source_candidate_manifest.json",
        "source_candidate_episode_frames.csv",
        "surface_overlap.json",
        "surface_overlap_episodes.csv",
        "surface_overlap_frames.csv",
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256_file(output / relative)}  {relative}\n" for relative in payloads),
        encoding="utf-8",
    )
    return labeled_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.5)
    parser.add_argument("--neighbor-radii-m", type=float, nargs="+", default=(0.3, 0.5, 1.0))
    parser.add_argument("--decision-radius-m", type=float, default=0.5)
    parser.add_argument("--positive-min-frame-coverage", type=float, default=0.8)
    parser.add_argument("--negative-max-frame-coverage", type=float, default=0.4)
    args = parser.parse_args()
    result = classify_surface_overlap(
        manifest_dir=args.manifest_dir,
        output=args.output,
        voxel_size_m=args.voxel_size_m,
        neighbor_radii_m=tuple(args.neighbor_radii_m),
        decision_radius_m=args.decision_radius_m,
        positive_min_frame_coverage=args.positive_min_frame_coverage,
        negative_max_frame_coverage=args.negative_max_frame_coverage,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
