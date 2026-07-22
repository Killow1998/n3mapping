#!/usr/bin/env python3
"""Measure oracle spatial recall of relocalization candidates.

Ground truth is used only after execution to identify which failure stage lost
the correct map neighborhood. It never participates in runtime decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _position(row: dict[str, str]) -> tuple[float, float, float]:
    return (float(row["x"]), float(row["y"]), float(row["z"]))


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _keyframe_positions(
    dataset: str,
    manifest_rows: list[dict[str, str]],
    keyframes_gt: Path,
) -> dict[int, tuple[float, float, float]]:
    map_rows = [row for row in manifest_rows if row["role"] == "map"]
    by_token = {row["frame_token"]: row for row in map_rows}
    positions: dict[int, tuple[float, float, float]] = {}
    for keyframe in _read_csv(keyframes_gt):
        keyframe_id = int(keyframe["keyframe_id"])
        if dataset == "kitti360":
            source = by_token.get(keyframe["frame_id"])
        elif dataset == "m2dgr":
            index = int(keyframe["frame_id"])
            source = map_rows[index] if 0 <= index < len(map_rows) else None
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
        if source is None:
            raise ValueError(
                f"cannot bind keyframe {keyframe_id} frame {keyframe['frame_id']} "
                "to the frozen map manifest"
            )
        positions[keyframe_id] = _position(source)
    if not positions:
        raise ValueError("keyframes_gt.csv contains no bindable keyframes")
    return positions


def _candidate_min_distance(
    candidates: list[dict[str, Any]],
    query_position: tuple[float, float, float],
    keyframes: dict[int, tuple[float, float, float]],
) -> float | None:
    distances = [
        _distance(query_position, keyframes[int(candidate["match_id"])])
        for candidate in candidates
        if int(candidate["match_id"]) in keyframes
    ]
    return min(distances) if distances else None


def diagnose_benchmark(manifest_dir: Path, benchmark_dir: Path) -> dict[str, Any]:
    manifest = json.loads((manifest_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest_rows = _read_csv(manifest_dir / "episode_frames.csv")
    keyframes = _keyframe_positions(
        manifest["dataset"], manifest_rows, benchmark_dir / "map" / "keyframes_gt.csv"
    )
    overlap_radius = float(manifest["overlap_radius_m"])
    query_by_episode: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        if row["role"] == "query":
            query_by_episode.setdefault(row["episode_id"], []).append(row)

    output_rows: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for episode_id in sorted(query_by_episode):
        debug_path = benchmark_dir / "episodes" / episode_id / "relocalization_debug.jsonl"
        events = [
            json.loads(line)
            for line in debug_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_rows = query_by_episode[episode_id]
        if len(events) != len(query_rows):
            raise ValueError(
                f"{episode_id}: {len(events)} debug events for {len(query_rows)} query frames"
            )
        for query_offset, (query, event) in enumerate(zip(query_rows, events)):
            if event.get("record_type") != "relocalize":
                continue
            query_position = _position(query)
            oracle_distance = min(
                _distance(query_position, position) for position in keyframes.values()
            )
            main_candidates = event.get("top_candidates", [])
            motion_candidates = event.get("motion_query_top_candidates", [])
            main_distance = _candidate_min_distance(main_candidates, query_position, keyframes)
            motion_distance = _candidate_min_distance(
                motion_candidates, query_position, keyframes
            )
            top1_distance = _candidate_min_distance(
                main_candidates[:1], query_position, keyframes
            )
            decision = event.get("reject_reason") or event.get("lock_result") or "unknown"
            decisions[decision] += 1
            output_rows.append(
                {
                    "episode_id": episode_id,
                    "query_offset": query_offset,
                    "frame_token": query["frame_token"],
                    "oracle_nearest_keyframe_distance_m": oracle_distance,
                    "oracle_overlap_available": oracle_distance <= overlap_radius,
                    "main_candidate_count": len(main_candidates),
                    "main_top1_distance_m": top1_distance,
                    "main_top1_recall": top1_distance is not None
                    and top1_distance <= overlap_radius,
                    "main_topk_min_distance_m": main_distance,
                    "main_topk_recall": main_distance is not None
                    and main_distance <= overlap_radius,
                    "motion_candidate_count": len(motion_candidates),
                    "motion_topk_min_distance_m": motion_distance,
                    "motion_topk_recall": motion_distance is not None
                    and motion_distance <= overlap_radius,
                    "decision": decision,
                }
            )

    eligible = [row for row in output_rows if row["oracle_overlap_available"]]

    def count(field: str) -> int:
        return sum(bool(row[field]) for row in eligible)

    def rate(field: str) -> float | None:
        return count(field) / len(eligible) if eligible else None

    summary = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "evidence_class": manifest["evidence_class"],
        "overlap_radius_m": overlap_radius,
        "relocalize_frame_count": len(output_rows),
        "oracle_overlap_frame_count": len(eligible),
        "main_top1_recall_count": count("main_top1_recall"),
        "main_top1_recall_rate": rate("main_top1_recall"),
        "main_topk_recall_count": count("main_topk_recall"),
        "main_topk_recall_rate": rate("main_topk_recall"),
        "motion_topk_recall_count": count("motion_topk_recall"),
        "motion_topk_recall_rate": rate("motion_topk_recall"),
        "decision_counts": dict(sorted(decisions.items())),
    }
    benchmark_dir.joinpath("oracle_candidate_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = list(output_rows[0]) if output_rows else ["episode_id"]
    with benchmark_dir.joinpath("oracle_candidate_frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose_benchmark(args.manifest_dir, args.benchmark_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
