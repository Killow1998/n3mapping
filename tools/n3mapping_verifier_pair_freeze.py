#!/usr/bin/env python3
"""Freeze candidate-observation pairs for a future shadow verifier."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import (
    _expected_behaviors,
    _finalize_hashed_output,
    _verify_benchmark_output,
    _verify_hashed_output,
    _verify_manifest,
)
from n3mapping_keyframe_surface_support_audit import _pose_from_hypothesis
from n3mapping_surface_overlap_audit import _pose_resolver


def _yaw_deg(transform: np.ndarray) -> float:
    return math.degrees(math.atan2(transform[1, 0], transform[0, 0]))


def _yaw_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = (_yaw_deg(first) - _yaw_deg(second) + 180.0) % 360.0 - 180.0
    return abs(delta)


def classify_pair(
    expected_behavior: str,
    translation_error_m: float,
    yaw_error_deg: float,
    *,
    translation_gate_m: float,
    yaw_gate_deg: float,
) -> str:
    if expected_behavior == "abstain":
        return "hard_negative"
    if expected_behavior != "lock":
        raise ValueError(f"unsupported expected behavior: {expected_behavior}")
    if (
        translation_error_m <= translation_gate_m
        and yaw_error_deg <= yaw_gate_deg
    ):
        return "positive"
    return "hard_negative"


def _keyframe_clouds(
    manifest: dict[str, Any],
    rows: list[dict[str, str]],
    benchmark_dir: Path,
) -> dict[int, tuple[str, str]]:
    ordered_map_rows = [row for row in rows if row.get("role") == "map"]
    map_rows = {
        row["frame_token"]: row
        for row in ordered_map_rows
    }
    keyframes_path = benchmark_dir / "map" / "keyframes_gt.csv"
    if not keyframes_path.is_file():
        raise ValueError(f"missing map keyframe index: {keyframes_path}")
    root = Path(manifest["root"])
    result = {}
    with keyframes_path.open(encoding="utf-8", newline="") as stream:
        for keyframe in csv.DictReader(stream):
            frame_token = keyframe["frame_id"]
            map_row = map_rows.get(frame_token)
            if map_row is None:
                frame_index = int(frame_token)
                if not 0 <= frame_index < len(ordered_map_rows):
                    raise ValueError(
                        f"map keyframe frame is absent from manifest: {frame_token}"
                    )
                map_row = ordered_map_rows[frame_index]
            relative = map_row["relative_cloud_path"]
            result[int(keyframe["keyframe_id"])] = (
                relative,
                sha256_file(root / relative),
            )
    return result


def freeze_verifier_pairs(
    labeled_manifest_dirs: list[Path],
    benchmark_dirs: list[Path],
    output: Path,
    *,
    translation_gate_m: float = 1.0,
    yaw_gate_deg: float = 10.0,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite verifier pair freeze: {output}")
    if not labeled_manifest_dirs or len(labeled_manifest_dirs) != len(benchmark_dirs):
        raise ValueError("provide the same non-zero number of manifests and benchmarks")
    if translation_gate_m <= 0.0 or yaw_gate_deg <= 0.0:
        raise ValueError("pose gates must be positive")

    pair_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for manifest_dir, benchmark_dir in zip(labeled_manifest_dirs, benchmark_dirs):
        manifest, rows = _verify_manifest(manifest_dir)
        _verify_benchmark_output(benchmark_dir)
        benchmark_summary_path = benchmark_dir / "summary.json"
        benchmark_summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(manifest_dir / "dataset_manifest.json")
        if benchmark_summary.get("manifest_sha256") not in {
            manifest_hash,
            manifest.get("source_candidate_manifest_sha256"),
        }:
            raise ValueError("benchmark is unrelated to labeled manifest")

        expected = _expected_behaviors(manifest, rows)
        pose_for = _pose_resolver(manifest)
        root = Path(manifest["root"])
        keyframe_clouds = _keyframe_clouds(manifest, rows, benchmark_dir)
        query_by_episode: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            if row.get("role") == "query":
                query_by_episode.setdefault(row["episode_id"], []).append(row)

        case_id = manifest_dir.name
        case_pair_start = len(pair_rows)
        for episode_id in sorted(expected):
            debug_path = (
                benchmark_dir / "episodes" / episode_id / "relocalization_debug.jsonl"
            )
            events = [
                json.loads(line)
                for line in debug_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            query_rows = query_by_episode[episode_id]
            for event in events:
                hypotheses = event.get("per_basin_best", [])
                if not hypotheses:
                    continue
                query_offset = int(event["query_index"]) - 1
                if not 0 <= query_offset < len(query_rows):
                    raise ValueError(f"{episode_id}: debug query index is outside manifest")
                query_row = query_rows[query_offset]
                query_relative = query_row["relative_cloud_path"]
                query_gt = pose_for(manifest["query_sequence"], query_row["frame_token"])
                query_sha256 = sha256_file(root / query_relative)
                for hypothesis_rank, hypothesis in enumerate(hypotheses):
                    pose = _pose_from_hypothesis(hypothesis)
                    pose_fields = hypothesis["pose_in_map"]
                    translation_error = float(
                        np.linalg.norm(pose[:3, 3] - query_gt[:3, 3])
                    )
                    yaw_error = _yaw_error_deg(pose, query_gt)
                    matched_keyframe_id = int(hypothesis["matched_kf_id"])
                    if matched_keyframe_id not in keyframe_clouds:
                        raise ValueError(
                            f"missing cloud for keyframe {matched_keyframe_id}"
                        )
                    map_relative, map_sha256 = keyframe_clouds[matched_keyframe_id]
                    candidate = hypothesis["candidate"]
                    pair_label = classify_pair(
                        expected[episode_id],
                        translation_error,
                        yaw_error,
                        translation_gate_m=translation_gate_m,
                        yaw_gate_deg=yaw_gate_deg,
                    )
                    pair_rows.append(
                        {
                            "pair_id": (
                                f"{case_id}/{episode_id}/q{query_offset:03d}/"
                                f"h{hypothesis_rank:02d}/kf{matched_keyframe_id}"
                            ),
                            "case_id": case_id,
                            "episode_id": episode_id,
                            "surface_expected_behavior": expected[episode_id],
                            "pair_label": pair_label,
                            "negative_kind": (
                                "none"
                                if pair_label == "positive"
                                else (
                                    "surface_absent"
                                    if expected[episode_id] == "abstain"
                                    else "wrong_pose"
                                )
                            ),
                            "split": "unassigned",
                            "query_index": query_offset,
                            "query_frame_token": query_row["frame_token"],
                            "query_relative_cloud_path": query_relative,
                            "query_cloud_sha256": query_sha256,
                            "matched_keyframe_id": matched_keyframe_id,
                            "map_relative_cloud_path": map_relative,
                            "map_cloud_sha256": map_sha256,
                            "hypothesis_x": pose_fields["x"],
                            "hypothesis_y": pose_fields["y"],
                            "hypothesis_z": pose_fields["z"],
                            "hypothesis_qx": pose_fields["qx"],
                            "hypothesis_qy": pose_fields["qy"],
                            "hypothesis_qz": pose_fields["qz"],
                            "hypothesis_qw": pose_fields["qw"],
                            "translation_error_m": translation_error,
                            "yaw_error_deg": yaw_error,
                            "basin_center_id": hypothesis["basin_center_id"],
                            "candidate_source": candidate["candidate_source"],
                            "rhpd_distance": candidate["rhpd_distance"],
                            "sc_distance": candidate["sc_distance"],
                            "fused_score": candidate["fused_score"],
                            "candidate_yaw_diff_rad": candidate["yaw_diff_rad"],
                            "fitness_score": hypothesis["fitness_score"],
                            "inlier_ratio": hypothesis["inlier_ratio"],
                            "selection_score": hypothesis["selection_score"],
                            "visibility_consistency_ratio": hypothesis[
                                "visibility_consistency_ratio"
                            ],
                            "visibility_observed_coverage": hypothesis[
                                "visibility_observed_coverage"
                            ],
                            "visibility_foreground_conflict_ratio": hypothesis[
                                "visibility_foreground_conflict_ratio"
                            ],
                        }
                    )
        case_rows.append(
            {
                "case_id": case_id,
                "dataset": manifest["dataset"],
                "dataset_root": manifest["root"],
                "map_sequence": manifest["map_sequence"],
                "query_sequence": manifest["query_sequence"],
                "pair_count": len(pair_rows) - case_pair_start,
                "labeled_manifest_sha256": manifest_hash,
                "benchmark_summary_sha256": sha256_file(benchmark_summary_path),
                "labeled_manifest_dir": str(manifest_dir.resolve()),
                "benchmark_dir": str(benchmark_dir.resolve()),
            }
        )

    if not pair_rows:
        raise ValueError("no per-basin candidate-observation pairs were found")
    output.mkdir(parents=True)
    pairs_path = output / "candidate_observation_pairs.csv"
    with pairs_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    cases_path = output / "source_cases.csv"
    with cases_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    positive_count = sum(row["pair_label"] == "positive" for row in pair_rows)
    hard_negative_count = sum(
        row["pair_label"] == "hard_negative" for row in pair_rows
    )
    surface_absent_count = sum(
        row["negative_kind"] == "surface_absent" for row in pair_rows
    )
    wrong_pose_count = sum(row["negative_kind"] == "wrong_pose" for row in pair_rows)
    sequence_groups = {
        (row["dataset"], row["map_sequence"], row["query_sequence"])
        for row in case_rows
    }
    group_by_case = {
        row["case_id"]: (row["dataset"], row["map_sequence"], row["query_sequence"])
        for row in case_rows
    }
    positive_groups = {
        group_by_case[row["case_id"]]
        for row in pair_rows
        if row["pair_label"] == "positive"
    }
    surface_absent_groups = {
        group_by_case[row["case_id"]]
        for row in pair_rows
        if row["negative_kind"] == "surface_absent"
    }
    summary = {
        "schema_version": 1,
        "evidence_class": "oracle_gt_candidate_observation_pair_freeze",
        "authority": False,
        "training_authorized": False,
        "split_assignment": "unassigned",
        "translation_gate_m": translation_gate_m,
        "yaw_gate_deg": yaw_gate_deg,
        "case_count": len(case_rows),
        "sequence_group_count": len(sequence_groups),
        "positive_sequence_group_count": len(positive_groups),
        "surface_absent_sequence_group_count": len(surface_absent_groups),
        "held_out_surface_absent_split_ready": len(surface_absent_groups) >= 2,
        "pair_count": len(pair_rows),
        "positive_pair_count": positive_count,
        "hard_negative_pair_count": hard_negative_count,
        "surface_absent_pair_count": surface_absent_count,
        "wrong_pose_pair_count": wrong_pose_count,
        "candidate_observation_pairs_sha256": sha256_file(pairs_path),
        "source_cases_sha256": sha256_file(cases_path),
        "boundary": (
            "Rows point to immutable raw clouds and post-registration basin hypotheses. "
            "No split is assigned and no model training or lock authority is authorized. "
            "At least two independent surface-absent sequence groups are required before "
            "one can be held out."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    required = {
        "summary.json",
        "candidate_observation_pairs.csv",
        "source_cases.csv",
    }
    _finalize_hashed_output(output, required)
    _verify_hashed_output(output, required)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-manifest-dir", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-gate-m", type=float, default=1.0)
    parser.add_argument("--yaw-gate-deg", type=float, default=10.0)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_verifier_pairs(
                args.labeled_manifest_dir,
                args.benchmark_dir,
                args.output,
                translation_gate_m=args.translation_gate_m,
                yaw_gate_deg=args.yaw_gate_deg,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
