#!/usr/bin/env python3
"""Audit lock poses against explicit free space from frozen map scan origins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import (
    _expected_behaviors,
    _finalize_hashed_output,
    _verify_benchmark_output,
    _verify_hashed_output,
    _verify_manifest,
)
from n3mapping_keyframe_surface_support_audit import (
    _lock_event,
    _pose_from_hypothesis,
    _winner,
)
from n3mapping_runtime_signal_audit import summarize_ranges
from n3mapping_surface_overlap_audit import _load_xyz, _pose_resolver, _transform


SIGNALS = (
    "observed_support_fraction",
    "observed_free_conflict_fraction",
    "query_any_support_fraction",
    "query_any_free_conflict_fraction",
    "query_free_without_support_fraction",
    "nearest_view_observed_support_fraction",
    "nearest_view_observed_free_conflict_fraction",
    "nearest_view_query_support_fraction",
    "nearest_view_query_free_conflict_fraction",
    "nearest_view_origin_distance_m",
)


def _voxel_representatives(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def classify_map_ray_observations(
    query_world: np.ndarray,
    T_world_map_sensor: np.ndarray,
    map_direction_tree: cKDTree,
    map_ranges: np.ndarray,
    geometry_scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    T_sensor_world = np.linalg.inv(T_world_map_sensor)
    query_sensor = _transform(query_world, T_sensor_world)
    query_ranges = np.linalg.norm(query_sensor, axis=1)
    valid = query_ranges > 1.0e-6
    directions = np.zeros_like(query_sensor)
    directions[valid] = query_sensor[valid] / query_ranges[valid, None]
    chord, nearest = map_direction_tree.query(directions, k=1, workers=1)
    laterally_observed = valid & (chord * query_ranges <= geometry_scale_m)
    range_delta = map_ranges[nearest] - query_ranges
    support = laterally_observed & (np.abs(range_delta) <= geometry_scale_m)
    free_conflict = laterally_observed & (range_delta > geometry_scale_m)
    return support, free_conflict


def audit_multiview_free_space(
    labeled_manifest_dirs: list[Path],
    benchmark_dirs: list[Path],
    output: Path,
    *,
    geometry_scale_m: float = 0.5,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite free-space audit: {output}")
    if not labeled_manifest_dirs or len(labeled_manifest_dirs) != len(benchmark_dirs):
        raise ValueError("provide the same non-zero number of manifests and benchmarks")
    if geometry_scale_m <= 0.0:
        raise ValueError("geometry scale must be positive")

    feature_rows = []
    label_hashes = []
    benchmark_hashes = []
    for manifest_dir, benchmark_dir in zip(labeled_manifest_dirs, benchmark_dirs):
        manifest, rows = _verify_manifest(manifest_dir)
        _verify_benchmark_output(benchmark_dir)
        benchmark_summary = json.loads(
            (benchmark_dir / "summary.json").read_text(encoding="utf-8")
        )
        label_hash = sha256_file(manifest_dir / "dataset_manifest.json")
        if benchmark_summary.get("manifest_sha256") not in {
            label_hash,
            manifest.get("source_candidate_manifest_sha256"),
        }:
            raise ValueError("benchmark is unrelated to labeled manifest")
        expected = _expected_behaviors(manifest, rows)
        with (benchmark_dir / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            outcomes = {row["episode_id"]: row["outcome"] for row in csv.DictReader(stream)}
        root = Path(manifest["root"])
        pose_for = _pose_resolver(manifest)
        map_views = []
        for row in rows:
            if row.get("role") != "map":
                continue
            points = _load_xyz(root / row["relative_cloud_path"])
            ranges = np.linalg.norm(points, axis=1)
            valid = ranges > 1.0e-6
            directions = points[valid] / ranges[valid, None]
            map_views.append(
                (
                    pose_for(manifest["map_sequence"], row["frame_token"]),
                    cKDTree(directions),
                    ranges[valid],
                )
            )
        if not map_views:
            raise ValueError("manifest contains no map views")
        query_by_episode: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            if row.get("role") == "query":
                query_by_episode.setdefault(row["episode_id"], []).append(row)

        for episode_id in sorted(expected):
            debug = benchmark_dir / "episodes" / episode_id / "relocalization_debug.jsonl"
            events = [json.loads(line) for line in debug.read_text().splitlines() if line.strip()]
            accepted = _lock_event(events)
            outcome = outcomes[episode_id]
            if expected[episode_id] == "lock" and outcome == "correct_lock":
                group = "positive_correct_lock"
            elif expected[episode_id] == "abstain" and outcome == "false_lock":
                group = "hard_negative_false_lock"
            else:
                group = "not_locked_or_not_comparable"
            result: dict[str, Any] = {
                "case_id": manifest_dir.name,
                "episode_id": episode_id,
                "expected_behavior": expected[episode_id],
                "baseline_outcome": outcome,
                "analysis_group": group,
            }
            if accepted is None:
                result.update({signal: None for signal in SIGNALS})
                feature_rows.append(result)
                continue
            query_index = int(accepted["query_index"]) - 1
            query_rows = query_by_episode[episode_id]
            if not 0 <= query_index < len(query_rows):
                raise ValueError(f"{episode_id}: lock query index is outside manifest")
            query_local = _load_xyz(root / query_rows[query_index]["relative_cloud_path"])
            query_world = _voxel_representatives(
                _transform(query_local, _pose_from_hypothesis(_winner(accepted))),
                geometry_scale_m,
            )
            any_support = np.zeros(query_world.shape[0], dtype=bool)
            any_free = np.zeros(query_world.shape[0], dtype=bool)
            support_count = 0
            free_count = 0
            for pose, direction_tree, ranges in map_views:
                support, free_conflict = classify_map_ray_observations(
                    query_world, pose, direction_tree, ranges, geometry_scale_m
                )
                support_count += int(np.count_nonzero(support))
                free_count += int(np.count_nonzero(free_conflict))
                any_support |= support
                any_free |= free_conflict
            query_pose = _pose_from_hypothesis(_winner(accepted))
            nearest_pose, nearest_tree, nearest_ranges = min(
                map_views,
                key=lambda view: float(
                    np.linalg.norm(view[0][:3, 3] - query_pose[:3, 3])
                ),
            )
            nearest_support, nearest_free = classify_map_ray_observations(
                query_world,
                nearest_pose,
                nearest_tree,
                nearest_ranges,
                geometry_scale_m,
            )
            nearest_observed_count = int(
                np.count_nonzero(nearest_support) + np.count_nonzero(nearest_free)
            )
            observed_count = support_count + free_count
            result.update(
                {
                    "observed_support_fraction": (
                        support_count / observed_count if observed_count else None
                    ),
                    "observed_free_conflict_fraction": (
                        free_count / observed_count if observed_count else None
                    ),
                    "query_any_support_fraction": float(np.mean(any_support)),
                    "query_any_free_conflict_fraction": float(np.mean(any_free)),
                    "query_free_without_support_fraction": float(
                        np.mean(any_free & ~any_support)
                    ),
                    "nearest_view_observed_support_fraction": (
                        float(np.count_nonzero(nearest_support)) / nearest_observed_count
                        if nearest_observed_count
                        else None
                    ),
                    "nearest_view_observed_free_conflict_fraction": (
                        float(np.count_nonzero(nearest_free)) / nearest_observed_count
                        if nearest_observed_count
                        else None
                    ),
                    "nearest_view_query_support_fraction": float(
                        np.mean(nearest_support)
                    ),
                    "nearest_view_query_free_conflict_fraction": float(
                        np.mean(nearest_free)
                    ),
                    "nearest_view_origin_distance_m": float(
                        np.linalg.norm(nearest_pose[:3, 3] - query_pose[:3, 3])
                    ),
                }
            )
            feature_rows.append(result)
        label_hashes.append(label_hash)
        benchmark_hashes.append(sha256_file(benchmark_dir / "summary.json"))

    range_rows = [summarize_ranges(feature_rows, signal) for signal in SIGNALS]
    output.mkdir(parents=True)
    features = output / "multiview_free_space_features.csv"
    with features.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    ranges = output / "multiview_free_space_ranges.csv"
    with ranges.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(range_rows[0]))
        writer.writeheader()
        writer.writerows(range_rows)
    summary = {
        "schema_version": 2,
        "evidence_class": "shadow_post_lock_map_origin_free_space",
        "authority": False,
        "geometry_scale_m": geometry_scale_m,
        "positive_correct_lock_count": sum(
            row["analysis_group"] == "positive_correct_lock" for row in feature_rows
        ),
        "hard_negative_false_lock_count": sum(
            row["analysis_group"] == "hard_negative_false_lock" for row in feature_rows
        ),
        "range_disjoint_signal_count": sum(row["range_disjoint"] for row in range_rows),
        "all_ranges_overlap": not any(row["range_disjoint"] for row in range_rows),
        "label_manifest_sha256": label_hashes,
        "benchmark_summary_sha256": benchmark_hashes,
        "multiview_free_space_features_sha256": sha256_file(features),
        "multiview_free_space_ranges_sha256": sha256_file(ranges),
        "boundary": (
            "Only a nearer query endpoint on a map-observed ray is explicit free-space conflict; "
            "occluded and angularly unobserved points remain unknown. Nearest-view signals use "
            "exactly one frozen map scan selected only by sensor-origin distance to the lock pose."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    required = {
        "summary.json",
        "multiview_free_space_features.csv",
        "multiview_free_space_ranges.csv",
    }
    _finalize_hashed_output(output, required)
    _verify_hashed_output(output, required)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-manifest-dir", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-scale-m", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_multiview_free_space(
                args.labeled_manifest_dir,
                args.benchmark_dir,
                args.output,
                geometry_scale_m=args.geometry_scale_m,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
