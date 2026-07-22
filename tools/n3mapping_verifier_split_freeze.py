#!/usr/bin/env python3
"""Freeze leakage-safe development/test splits for verifier pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import (
    _finalize_hashed_output,
    _verify_hashed_output,
)


PAIR_REQUIRED = {
    "summary.json",
    "candidate_observation_pairs.csv",
    "source_cases.csv",
}


def assign_development_test_splits(
    case_rows: list[dict[str, Any]],
    test_sequences: set[tuple[str, str]],
) -> dict[str, str]:
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    known_nodes = set()
    for row in case_rows:
        map_node = (row["dataset"], row["map_sequence"])
        query_node = (row["dataset"], row["query_sequence"])
        known_nodes.update((map_node, query_node))
        adjacency[map_node].add(query_node)
        adjacency[query_node].add(map_node)
    missing = test_sequences - known_nodes
    if missing:
        raise ValueError(f"test sequences are absent from source cases: {sorted(missing)}")

    reached = set(test_sequences)
    queue = deque(test_sequences)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)

    assignments = {}
    for row in case_rows:
        map_node = (row["dataset"], row["map_sequence"])
        query_node = (row["dataset"], row["query_sequence"])
        map_is_test = map_node in reached
        query_is_test = query_node in reached
        if map_is_test != query_is_test:
            raise ValueError("connected map/query sequences received different splits")
        assignments[row["case_id"]] = "test" if map_is_test else "development"
    return assignments


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def freeze_verifier_split(
    pair_dir: Path,
    output: Path,
    test_sequences: set[tuple[str, str]],
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite verifier split freeze: {output}")
    _verify_hashed_output(pair_dir, PAIR_REQUIRED)
    source_summary = json.loads((pair_dir / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("split_assignment") != "unassigned":
        raise ValueError("source pair freeze must have unassigned splits")
    if source_summary.get("training_authorized") is not False:
        raise ValueError("source pair freeze must not authorize training")

    pairs = _read_csv(pair_dir / "candidate_observation_pairs.csv")
    cases = _read_csv(pair_dir / "source_cases.csv")
    assignments = assign_development_test_splits(cases, test_sequences)
    if set(assignments.values()) != {"development", "test"}:
        raise ValueError("both development and test cases are required")
    for row in cases:
        row["split"] = assignments[row["case_id"]]
    for row in pairs:
        row["split"] = assignments[row["case_id"]]

    counts: dict[str, dict[str, int]] = {
        split: {
            "pair_count": 0,
            "positive_pair_count": 0,
            "wrong_pose_pair_count": 0,
            "surface_absent_pair_count": 0,
        }
        for split in ("development", "test")
    }
    for row in pairs:
        split_counts = counts[row["split"]]
        split_counts["pair_count"] += 1
        if row["pair_label"] == "positive":
            split_counts["positive_pair_count"] += 1
        if row["negative_kind"] == "wrong_pose":
            split_counts["wrong_pose_pair_count"] += 1
        if row["negative_kind"] == "surface_absent":
            split_counts["surface_absent_pair_count"] += 1
    for split, split_counts in counts.items():
        if split_counts["positive_pair_count"] == 0:
            raise ValueError(f"{split} split contains no positive pairs")
        if split_counts["surface_absent_pair_count"] == 0:
            raise ValueError(f"{split} split contains no surface-absent pairs")

    output.mkdir(parents=True)
    pairs_path = output / "candidate_observation_pairs.csv"
    with pairs_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    cases_path = output / "source_cases.csv"
    with cases_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    summary = {
        "schema_version": 1,
        "evidence_class": "oracle_gt_candidate_observation_development_test_split",
        "authority": False,
        "training_authorized": False,
        "split_assignment": "development_test_frozen",
        "test_sequence_nodes": [
            {"dataset": dataset, "sequence": sequence}
            for dataset, sequence in sorted(test_sequences)
        ],
        "development": counts["development"],
        "test": counts["test"],
        "source_pair_summary_sha256": sha256_file(pair_dir / "summary.json"),
        "source_pair_csv_sha256": sha256_file(
            pair_dir / "candidate_observation_pairs.csv"
        ),
        "candidate_observation_pairs_sha256": sha256_file(pairs_path),
        "source_cases_sha256": sha256_file(cases_path),
        "boundary": (
            "Connected sequence identities are atomic across development/test. "
            "The test split is immutable evaluation-only. Train/validation remain "
            "unassigned, so supervised training is not authorized."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _finalize_hashed_output(output, PAIR_REQUIRED)
    _verify_hashed_output(output, PAIR_REQUIRED)
    return summary


def _parse_sequence(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected DATASET=SEQUENCE")
    dataset, sequence = text.split("=", 1)
    if not dataset or not sequence:
        raise argparse.ArgumentTypeError("expected non-empty DATASET=SEQUENCE")
    return dataset, sequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-sequence", type=_parse_sequence, action="append", required=True
    )
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_verifier_split(args.pair_dir, args.output, set(args.test_sequence)),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
