#!/usr/bin/env python3
"""Audit a fixed pretrained ResNet on candidate-conditioned LiDAR BEV pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_benchmark import _finalize_hashed_output, _verify_hashed_output
from n3mapping_episode_review_prepare import _matrix_from_xyzw
from n3mapping_surface_overlap_audit import _load_xyz, _transform
from n3mapping_verifier_split_freeze import PAIR_REQUIRED


def select_zero_false_accept_threshold(rows: list[dict[str, Any]]) -> float:
    negatives = [float(row["score"]) for row in rows if row["pair_label"] != "positive"]
    positives = [float(row["score"]) for row in rows if row["pair_label"] == "positive"]
    if not negatives or not positives:
        raise ValueError("threshold selection requires positive and hard-negative rows")
    return float(np.nextafter(max(negatives), np.inf))


def _pose(prefix: str, row: dict[str, str]) -> np.ndarray:
    return _matrix_from_xyzw(
        tuple(float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")),
        tuple(float(row[f"{prefix}_{axis}"]) for axis in ("qx", "qy", "qz", "qw")),
    )


def _bev(points: np.ndarray, extent_m: float, resolution_m: float) -> np.ndarray:
    size = int(round(2.0 * extent_m / resolution_m))
    if size <= 0:
        raise ValueError("invalid BEV geometry")
    valid = (
        (points[:, 0] >= -extent_m)
        & (points[:, 0] < extent_m)
        & (points[:, 1] >= -extent_m)
        & (points[:, 1] < extent_m)
        & (points[:, 2] >= -3.0)
        & (points[:, 2] < 3.0)
    )
    points = points[valid]
    image = np.zeros((3, size, size), dtype=np.float32)
    if not points.size:
        return image
    columns = np.floor((points[:, 0] + extent_m) / resolution_m).astype(int)
    rows = size - 1 - np.floor((points[:, 1] + extent_m) / resolution_m).astype(int)
    channels = np.clip(np.floor((points[:, 2] + 3.0) / 2.0).astype(int), 0, 2)
    image[channels, rows, columns] = 1.0
    return image


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [row for row in rows if row["pair_label"] == "positive"]
    negatives = [row for row in rows if row["pair_label"] != "positive"]
    accepted_positive = sum(float(row["score"]) >= threshold for row in positives)
    accepted_negative = sum(float(row["score"]) >= threshold for row in negatives)
    return {
        "pair_count": len(rows),
        "positive_count": len(positives),
        "hard_negative_count": len(negatives),
        "positive_accept_count": accepted_positive,
        "hard_negative_false_accept_count": accepted_negative,
        "positive_recall": accepted_positive / len(positives),
        "hard_negative_false_accept_rate": accepted_negative / len(negatives),
        "surface_absent_false_accept_count": sum(
            row["negative_kind"] == "surface_absent"
            and float(row["score"]) >= threshold
            for row in rows
        ),
        "wrong_pose_false_accept_count": sum(
            row["negative_kind"] == "wrong_pose"
            and float(row["score"]) >= threshold
            for row in rows
        ),
    }


def audit_frozen_bev_verifier(
    split_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    extent_m: float = 50.0,
    resolution_m: float = 0.5,
    validation_min_positive_recall: float = 0.8,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite frozen verifier audit: {output}")
    _verify_hashed_output(split_dir, PAIR_REQUIRED)
    split_summary = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))
    if split_summary.get("split_assignment") != "train_validation_test_frozen":
        raise ValueError("a frozen train/validation/test input is required")
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is absent: {checkpoint}")

    with (split_dir / "source_cases.csv").open(encoding="utf-8", newline="") as stream:
        cases = {row["case_id"]: row for row in csv.DictReader(stream)}
    with (split_dir / "candidate_observation_pairs.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        pairs = [row for row in csv.DictReader(stream) if row["split"] != "test"]
    if {row["split"] for row in pairs} != {"train", "validation"}:
        raise ValueError("audit reads train and validation only")

    import torch
    import torch.nn.functional as functional
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    scored = []
    with torch.inference_mode():
        for pair in pairs:
            case = cases[pair["case_id"]]
            root = Path(case["dataset_root"])
            query = _load_xyz(root / pair["query_relative_cloud_path"])
            map_cloud = _load_xyz(root / pair["map_relative_cloud_path"])
            T_world_map = _pose("map_keyframe", pair)
            T_world_query = _pose("hypothesis", pair)
            map_in_query = _transform(map_cloud, np.linalg.inv(T_world_query) @ T_world_map)
            images = np.stack(
                [
                    _bev(query, extent_m, resolution_m),
                    _bev(map_in_query, extent_m, resolution_m),
                ]
            )
            tensor = torch.from_numpy(images)
            tensor = functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
            tensor = (tensor - mean) / std
            embeddings = functional.normalize(model(tensor), dim=1)
            score = float(torch.sum(embeddings[0] * embeddings[1]))
            scored.append(
                {
                    "pair_id": pair["pair_id"],
                    "split": pair["split"],
                    "pair_label": pair["pair_label"],
                    "negative_kind": pair["negative_kind"],
                    "score": score,
                }
            )

    train = [row for row in scored if row["split"] == "train"]
    validation = [row for row in scored if row["split"] == "validation"]
    threshold = select_zero_false_accept_threshold(train)
    train_metrics = _metrics(train, threshold)
    validation_metrics = _metrics(validation, threshold)
    validation_pass = (
        validation_metrics["hard_negative_false_accept_count"] == 0
        and validation_metrics["positive_recall"] >= validation_min_positive_recall
    )

    output.mkdir(parents=True)
    scores_path = output / "frozen_bev_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    summary = {
        "schema_version": 1,
        "evidence_class": "frozen_pretrained_bev_shadow_validation",
        "authority": False,
        "training_performed": False,
        "test_accessed": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_summary_sha256": sha256_file(split_dir / "summary.json"),
        "extent_m": extent_m,
        "resolution_m": resolution_m,
        "height_slices_m": [[-3.0, -1.0], [-1.0, 1.0], [1.0, 3.0]],
        "threshold_selection": "nextafter(max_train_hard_negative_score,+inf)",
        "threshold": threshold,
        "validation_min_positive_recall": validation_min_positive_recall,
        "train": train_metrics,
        "validation": validation_metrics,
        "validation_gate_pass": validation_pass,
        "next_action": "one_shot_test" if validation_pass else "reject_frozen_bev_route",
        "frozen_bev_scores_sha256": sha256_file(scores_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    required = {"summary.json", "frozen_bev_scores.csv"}
    _finalize_hashed_output(output, required)
    _verify_hashed_output(output, required)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extent-m", type=float, default=50.0)
    parser.add_argument("--resolution-m", type=float, default=0.5)
    parser.add_argument("--validation-min-positive-recall", type=float, default=0.8)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_frozen_bev_verifier(
                args.split_dir,
                args.checkpoint,
                args.output,
                extent_m=args.extent_m,
                resolution_m=args.resolution_m,
                validation_min_positive_recall=args.validation_min_positive_recall,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
