#!/usr/bin/env python3
"""Audit whether existing runtime scalars separate correct and false locks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import (
    _expected_behaviors,
    _finalize_hashed_output,
    _verify_benchmark_output,
    _verify_hashed_output,
    _verify_manifest,
)


SIGNALS = (
    "descriptor_fused_score",
    "descriptor_rhpd_distance",
    "descriptor_sc_distance",
    "registration_fitness_score",
    "registration_inlier_ratio",
    "mean_visibility_consistency",
    "mean_visibility_evidence",
    "cumulative_log_likelihood",
    "decision_margin",
    "basin_separation",
    "seed_last_keyframe_id_gap",
    "unique_last_match_count",
    "last_match_id_span",
    "winner_pose_drift_m",
)


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _pose_position(hypothesis: dict[str, Any]) -> tuple[float, float, float]:
    pose = hypothesis["pose_in_map"]
    return (float(pose["x"]), float(pose["y"]), float(pose["z"]))


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _winner(event: dict[str, Any]) -> dict[str, Any]:
    alive = [hypothesis for hypothesis in event.get("hypotheses", []) if hypothesis.get("alive")]
    if not alive:
        raise ValueError("accepted lock event has no alive hypothesis")
    return max(alive, key=lambda hypothesis: float(hypothesis["cumulative_log_likelihood"]))


def _extract_lock_signals(events: list[dict[str, Any]]) -> dict[str, float | int | None]:
    relocalize = [event for event in events if event.get("record_type") == "relocalize"]
    accepted_index = next(
        (index for index, event in enumerate(relocalize) if event.get("lock_accepted")), None
    )
    if accepted_index is None:
        return {signal: None for signal in SIGNALS}
    accepted = relocalize[accepted_index]
    winner = _winner(accepted)
    seed_id = int(winner["seed_match_id"])
    last_id = int(winner["last_match_id"])
    candidate = next(
        (
            item
            for item in accepted.get("top_candidates", [])
            if int(item.get("match_id", -1)) in (last_id, seed_id)
        ),
        {},
    )

    history = []
    probes = []
    for event in relocalize[: accepted_index + 1]:
        hypotheses = [
            item
            for item in event.get("hypotheses", [])
            if item.get("alive") and int(item.get("seed_match_id", -1)) == seed_id
        ]
        if hypotheses:
            history.append(
                max(
                    hypotheses,
                    key=lambda item: float(item["cumulative_log_likelihood"]),
                )
            )
        probes.extend(event.get("per_basin_best", []))
    relevant_probes = [
        probe
        for probe in probes
        if int(probe.get("matched_kf_id", -1)) in (seed_id, last_id)
    ]
    registration = (
        min(
            relevant_probes,
            key=lambda probe: float(probe.get("selection_score", math.inf)),
        )
        if relevant_probes
        else {}
    )
    last_ids = [int(item["last_match_id"]) for item in history]
    positions = [_pose_position(item) for item in history]
    pose_drift = max(
        (_distance(left, right) for left in positions for right in positions), default=0.0
    )
    return {
        "descriptor_fused_score": _finite(candidate.get("fused_score")),
        "descriptor_rhpd_distance": _finite(candidate.get("rhpd_distance")),
        "descriptor_sc_distance": _finite(candidate.get("sc_distance")),
        "registration_fitness_score": _finite(registration.get("fitness_score")),
        "registration_inlier_ratio": _finite(registration.get("inlier_ratio")),
        "mean_visibility_consistency": _finite(winner.get("mean_visibility_consistency")),
        "mean_visibility_evidence": _finite(winner.get("mean_visibility_evidence")),
        "cumulative_log_likelihood": _finite(winner.get("cumulative_log_likelihood")),
        "decision_margin": _finite(accepted.get("margin")),
        "basin_separation": _finite(accepted.get("basin_separation")),
        "seed_last_keyframe_id_gap": abs(seed_id - last_id),
        "unique_last_match_count": len(set(last_ids)),
        "last_match_id_span": max(last_ids) - min(last_ids) if last_ids else 0,
        "winner_pose_drift_m": pose_drift,
    }


def summarize_ranges(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    groups = {}
    for group in ("positive_correct_lock", "hard_negative_false_lock"):
        values = [
            float(row[signal])
            for row in rows
            if row["analysis_group"] == group and _finite(row.get(signal)) is not None
        ]
        groups[group] = values
    positive = groups["positive_correct_lock"]
    negative = groups["hard_negative_false_lock"]
    disjoint = bool(
        positive
        and negative
        and (max(positive) < min(negative) or max(negative) < min(positive))
    )
    return {
        "signal": signal,
        "positive_count": len(positive),
        "positive_min": min(positive) if positive else None,
        "positive_max": max(positive) if positive else None,
        "hard_negative_false_lock_count": len(negative),
        "hard_negative_false_lock_min": min(negative) if negative else None,
        "hard_negative_false_lock_max": max(negative) if negative else None,
        "range_disjoint": disjoint,
    }


def audit_runtime_signals(
    labeled_manifest_dirs: list[Path], benchmark_dirs: list[Path], output: Path
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite runtime-signal audit: {output}")
    if not labeled_manifest_dirs or len(labeled_manifest_dirs) != len(benchmark_dirs):
        raise ValueError("provide the same non-zero number of labeled manifests and benchmarks")

    feature_rows = []
    label_manifest_hashes = []
    benchmark_summary_hashes = []
    for labeled_manifest_dir, benchmark_dir in zip(
        labeled_manifest_dirs, benchmark_dirs
    ):
        manifest, manifest_rows = _verify_manifest(labeled_manifest_dir)
        _verify_benchmark_output(benchmark_dir)
        benchmark_summary = json.loads(
            (benchmark_dir / "summary.json").read_text(encoding="utf-8")
        )
        label_hash = sha256_file(labeled_manifest_dir / "dataset_manifest.json")
        accepted_manifest_hashes = {
            label_hash,
            manifest.get("source_candidate_manifest_sha256"),
        }
        if benchmark_summary.get("manifest_sha256") not in accepted_manifest_hashes:
            raise ValueError(
                "benchmark is not derived from the labeled or source candidate manifest"
            )
        expected = _expected_behaviors(manifest, manifest_rows)
        with (benchmark_dir / "episodes.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            outcomes = {row["episode_id"]: row for row in csv.DictReader(stream)}
        if not set(expected).issubset(outcomes):
            raise ValueError("benchmark output omits labeled episodes")
        label_manifest_hashes.append(label_hash)
        benchmark_summary_hashes.append(sha256_file(benchmark_dir / "summary.json"))
        for episode_id in sorted(expected):
            outcome = outcomes[episode_id]["outcome"]
            if expected[episode_id] == "lock" and outcome == "correct_lock":
                analysis_group = "positive_correct_lock"
            elif expected[episode_id] == "abstain" and outcome == "false_lock":
                analysis_group = "hard_negative_false_lock"
            else:
                analysis_group = "not_locked_or_not_comparable"
            debug_path = (
                benchmark_dir / "episodes" / episode_id / "relocalization_debug.jsonl"
            )
            events = [
                json.loads(line)
                for line in debug_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            feature_rows.append(
                {
                    "case_id": labeled_manifest_dir.name,
                    "episode_id": episode_id,
                    "expected_behavior": expected[episode_id],
                    "baseline_outcome": outcome,
                    "analysis_group": analysis_group,
                    **_extract_lock_signals(events),
                }
            )

    range_rows = [summarize_ranges(feature_rows, signal) for signal in SIGNALS]
    output.mkdir(parents=True)
    features_path = output / "runtime_lock_features.csv"
    with features_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    ranges_path = output / "runtime_signal_ranges.csv"
    with ranges_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(range_rows[0]))
        writer.writeheader()
        writer.writerows(range_rows)
    summary = {
        "schema_version": 1,
        "evidence_class": "shadow_runtime_log_posthoc_labels",
        "authority": False,
        "positive_correct_lock_count": sum(
            row["analysis_group"] == "positive_correct_lock" for row in feature_rows
        ),
        "hard_negative_false_lock_count": sum(
            row["analysis_group"] == "hard_negative_false_lock" for row in feature_rows
        ),
        "signal_count": len(range_rows),
        "range_disjoint_signal_count": sum(row["range_disjoint"] for row in range_rows),
        "all_scalar_ranges_overlap": not any(row["range_disjoint"] for row in range_rows),
        "signals": list(SIGNALS),
        "label_manifest_sha256": label_manifest_hashes,
        "benchmark_summary_sha256": benchmark_summary_hashes,
        "runtime_lock_features_sha256": sha256_file(features_path),
        "runtime_signal_ranges_sha256": sha256_file(ranges_path),
        "interpretation_boundary": (
            "Range overlap falsifies a one-dimensional threshold on these samples; "
            "it does not prove that every future representation is undecidable."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    required = {"summary.json", "runtime_lock_features.csv", "runtime_signal_ranges.csv"}
    _finalize_hashed_output(output, required)
    _verify_hashed_output(output, required)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-manifest-dir", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_runtime_signals(
                args.labeled_manifest_dir, args.benchmark_dir, args.output
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
