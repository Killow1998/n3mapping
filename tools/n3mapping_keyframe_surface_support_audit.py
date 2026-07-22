#!/usr/bin/env python3
"""Measure lock-pose query support from individual frozen map scans."""

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
from n3mapping_episode_review_prepare import _matrix_from_xyzw
from n3mapping_runtime_signal_audit import summarize_ranges
from n3mapping_surface_overlap_audit import (
    _load_xyz,
    _pose_resolver,
    _transform,
    _voxel_centers,
)


SIGNALS = (
    "single_frame_coverage_max",
    "single_frame_coverage_p90",
    "single_frame_coverage_median",
    "single_frame_coverage_top3_mean",
    "union_surface_coverage",
)


def _winner(event: dict[str, Any]) -> dict[str, Any]:
    alive = [item for item in event.get("hypotheses", []) if item.get("alive")]
    if not alive:
        raise ValueError("accepted lock event has no alive hypothesis")
    return max(alive, key=lambda item: float(item["cumulative_log_likelihood"]))


def _lock_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in events
            if event.get("record_type") == "relocalize" and event.get("lock_accepted")
        ),
        None,
    )


def _pose_from_hypothesis(hypothesis: dict[str, Any]) -> np.ndarray:
    pose = hypothesis["pose_in_map"]
    return _matrix_from_xyzw(
        (float(pose["x"]), float(pose["y"]), float(pose["z"])),
        (
            float(pose["qx"]),
            float(pose["qy"]),
            float(pose["qz"]),
            float(pose["qw"]),
        ),
    )


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def audit_keyframe_surface_support(
    labeled_manifest_dirs: list[Path],
    benchmark_dirs: list[Path],
    output: Path,
    *,
    voxel_size_m: float = 0.5,
    support_radius_m: float = 0.5,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite keyframe support audit: {output}")
    if not labeled_manifest_dirs or len(labeled_manifest_dirs) != len(benchmark_dirs):
        raise ValueError("provide the same non-zero number of manifests and benchmarks")
    if voxel_size_m <= 0.0 or support_radius_m <= 0.0:
        raise ValueError("voxel size and support radius must be positive")

    feature_rows = []
    label_hashes = []
    benchmark_hashes = []
    for manifest_dir, benchmark_dir in zip(labeled_manifest_dirs, benchmark_dirs):
        manifest, rows = _verify_manifest(manifest_dir)
        _verify_benchmark_output(benchmark_dir)
        summary = json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))
        label_hash = sha256_file(manifest_dir / "dataset_manifest.json")
        if summary.get("manifest_sha256") not in {
            label_hash,
            manifest.get("source_candidate_manifest_sha256"),
        }:
            raise ValueError("benchmark is unrelated to labeled manifest")
        expected = _expected_behaviors(manifest, rows)
        with (benchmark_dir / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            outcomes = {row["episode_id"]: row["outcome"] for row in csv.DictReader(stream)}
        root = Path(manifest["root"])
        pose_for = _pose_resolver(manifest)
        map_trees = []
        map_surfaces = []
        for row in rows:
            if row.get("role") != "map":
                continue
            local = _load_xyz(root / row["relative_cloud_path"])
            world = _transform(
                local, pose_for(manifest["map_sequence"], row["frame_token"])
            )
            surface = _voxel_centers(world, voxel_size_m)
            map_surfaces.append(surface)
            map_trees.append(cKDTree(surface))
        if not map_trees:
            raise ValueError("manifest contains no map surface")
        union_tree = cKDTree(_voxel_centers(np.concatenate(map_surfaces), voxel_size_m))
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
            query_row = query_rows[query_index]
            query_local = _load_xyz(root / query_row["relative_cloud_path"])
            query_world = _voxel_centers(
                _transform(query_local, _pose_from_hypothesis(_winner(accepted))),
                voxel_size_m,
            )
            coverages = [
                float(np.mean(tree.query(query_world, k=1, workers=1)[0] <= support_radius_m))
                for tree in map_trees
            ]
            top = sorted(coverages, reverse=True)
            union_coverage = float(
                np.mean(
                    union_tree.query(query_world, k=1, workers=1)[0]
                    <= support_radius_m
                )
            )
            result.update(
                {
                    "single_frame_coverage_max": max(coverages),
                    "single_frame_coverage_p90": _percentile(coverages, 90.0),
                    "single_frame_coverage_median": _percentile(coverages, 50.0),
                    "single_frame_coverage_top3_mean": sum(top[:3]) / min(3, len(top)),
                    "union_surface_coverage": union_coverage,
                }
            )
            feature_rows.append(result)
        label_hashes.append(label_hash)
        benchmark_hashes.append(sha256_file(benchmark_dir / "summary.json"))

    range_rows = [summarize_ranges(feature_rows, signal) for signal in SIGNALS]
    output.mkdir(parents=True)
    features = output / "keyframe_surface_features.csv"
    with features.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    ranges = output / "keyframe_surface_ranges.csv"
    with ranges.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(range_rows[0]))
        writer.writeheader()
        writer.writerows(range_rows)
    summary = {
        "schema_version": 1,
        "evidence_class": "shadow_post_lock_individual_map_scan_surface_support",
        "authority": False,
        "voxel_size_m": voxel_size_m,
        "support_radius_m": support_radius_m,
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
        "keyframe_surface_features_sha256": sha256_file(features),
        "keyframe_surface_ranges_sha256": sha256_file(ranges),
        "boundary": "Occupied-surface support is not a free-space or visibility test.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    required = {"summary.json", "keyframe_surface_features.csv", "keyframe_surface_ranges.csv"}
    _finalize_hashed_output(output, required)
    _verify_hashed_output(output, required)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-manifest-dir", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.5)
    parser.add_argument("--support-radius-m", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_keyframe_surface_support(
                args.labeled_manifest_dir,
                args.benchmark_dir,
                args.output,
                voxel_size_m=args.voxel_size_m,
                support_radius_m=args.support_radius_m,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
