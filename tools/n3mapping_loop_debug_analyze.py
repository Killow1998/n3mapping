#!/usr/bin/env python3
"""Label n3mapping loop_debug.jsonl candidates with KITTI360 ground truth."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze loop_debug.jsonl against keyframe ground-truth poses."
    )
    parser.add_argument("--loop_debug", required=True, help="Path to loop_debug.jsonl")
    parser.add_argument("--keyframes_gt", required=True, help="Path to keyframes_gt.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--accepted_loops", default="", help="Optional accepted_loops.csv")
    parser.add_argument("--loop_translation_threshold", type=float, default=5.0)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--min_id_gap", type=int, default=20)
    parser.add_argument("--z_drift_threshold", type=float, default=0.5)
    parser.add_argument("--rpy_drift_threshold_deg", type=float, default=10.0)
    return parser.parse_args()


def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    diff = a - b
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


def finite_float(value, default=float("nan")):
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def load_keyframes_gt(path):
    poses = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"keyframe_id", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing required keyframe GT columns")
        for row in reader:
            keyframe_id = int(row["keyframe_id"])
            x = finite_float(row["x"])
            y = finite_float(row["y"])
            z = finite_float(row["z"])
            qx = finite_float(row["qx"])
            qy = finite_float(row["qy"])
            qz = finite_float(row["qz"])
            qw = finite_float(row["qw"])
            if not all(math.isfinite(v) for v in (x, y, z, qx, qy, qz, qw)):
                raise RuntimeError(f"{path} contains non-finite pose for keyframe {keyframe_id}")
            poses[keyframe_id] = {
                "frame_id": int(row["frame_id"]),
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw_from_quat(qx, qy, qz, qw),
            }
    if not poses:
        raise RuntimeError(f"{path} contains no keyframe ground-truth poses")
    return poses


def load_loop_candidates(path):
    candidates = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if event.get("record_type") != "candidate":
                continue
            if "query_id" not in event or "match_id" not in event:
                raise RuntimeError(f"{path}:{line_number}: candidate is missing query_id or match_id")
            candidates.append(event)
    return candidates


def load_accepted_pairs(path):
    pairs = set()
    if not path:
        return pairs
    if not os.path.exists(path):
        raise RuntimeError(f"accepted loops file does not exist: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"query_id", "match_id"}.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing query_id/match_id columns")
        for row in reader:
            pairs.add((int(row["query_id"]), int(row["match_id"])))
    return pairs


def gt_pair_metrics(poses, query_id, match_id):
    query = poses.get(query_id)
    match = poses.get(match_id)
    if query is None or match is None:
        return float("nan"), float("nan"), False
    dx = query["x"] - match["x"]
    dy = query["y"] - match["y"]
    dz = query["z"] - match["z"]
    translation = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw_deg = abs(angle_diff(query["yaw"], match["yaw"])) * 180.0 / math.pi
    return translation, yaw_deg, True


def is_gt_loop(translation, yaw_deg, translation_threshold, yaw_threshold_deg):
    return (
        math.isfinite(translation)
        and math.isfinite(yaw_deg)
        and translation <= translation_threshold
        and yaw_deg <= yaw_threshold_deg
    )


def enumerate_gt_loop_pairs(poses, translation_threshold, yaw_threshold_deg, min_id_gap):
    ids = sorted(poses.keys())
    pairs = set()
    for i, match_id in enumerate(ids):
        for query_id in ids[i + 1 :]:
            if abs(query_id - match_id) < min_id_gap:
                continue
            translation, yaw_deg, valid = gt_pair_metrics(poses, query_id, match_id)
            if valid and is_gt_loop(translation, yaw_deg, translation_threshold, yaw_threshold_deg):
                pairs.add((query_id, match_id))
    return pairs


def event_float(event, key):
    return finite_float(event.get(key))


def format_float(value):
    if math.isfinite(value):
        return f"{value:.9g}"
    return ""


def analyze(args):
    poses = load_keyframes_gt(args.keyframes_gt)
    candidates = load_loop_candidates(args.loop_debug)
    accepted_pairs = load_accepted_pairs(args.accepted_loops)
    candidate_pairs = {(int(c["query_id"]), int(c["match_id"])) for c in candidates}
    gt_loop_pairs = enumerate_gt_loop_pairs(
        poses,
        args.loop_translation_threshold,
        args.loop_yaw_threshold_deg,
        args.min_id_gap,
    )

    rows = []
    stats = {
        "candidate_count": 0,
        "candidate_with_gt_count": 0,
        "accepted_candidate_count": 0,
        "retrieval_true_positive": 0,
        "retrieval_false_positive": 0,
        "retrieval_miss_estimate": 0,
        "icp_reject_true_loop": 0,
        "accepted_false_loop": 0,
        "accepted_true_loop": 0,
        "z_drift_suspect_count": 0,
    }
    rpy_threshold_rad = args.rpy_drift_threshold_deg * math.pi / 180.0
    accepted_pairs_available = bool(args.accepted_loops)

    for event in candidates:
        query_id = int(event["query_id"])
        match_id = int(event["match_id"])
        translation, yaw_deg, has_gt = gt_pair_metrics(poses, query_id, match_id)
        gt_loop = is_gt_loop(
            translation,
            yaw_deg,
            args.loop_translation_threshold,
            args.loop_yaw_threshold_deg,
        )
        if accepted_pairs_available:
            accepted = (query_id, match_id) in accepted_pairs
        else:
            accepted = event.get("gate_result") == "accepted"

        residual_z = event_float(event, "residual_z")
        residual_roll = event_float(event, "residual_roll")
        residual_pitch = event_float(event, "residual_pitch")
        residual_yaw = event_float(event, "residual_yaw")
        z_drift_suspect = bool(
            accepted
            and (
                (math.isfinite(residual_z) and abs(residual_z) >= args.z_drift_threshold)
                or (math.isfinite(residual_roll) and abs(residual_roll) >= rpy_threshold_rad)
                or (math.isfinite(residual_pitch) and abs(residual_pitch) >= rpy_threshold_rad)
            )
        )

        if accepted and gt_loop:
            category = "accepted_true_loop"
        elif accepted and not gt_loop:
            category = "accepted_false_loop"
        elif gt_loop:
            category = "rejected_true_loop"
        else:
            category = "rejected_false_candidate"

        stats["candidate_count"] += 1
        if has_gt:
            stats["candidate_with_gt_count"] += 1
        if accepted:
            stats["accepted_candidate_count"] += 1
        if gt_loop:
            stats["retrieval_true_positive"] += 1
        elif has_gt:
            stats["retrieval_false_positive"] += 1
        if gt_loop and not accepted:
            stats["icp_reject_true_loop"] += 1
        if accepted and not gt_loop:
            stats["accepted_false_loop"] += 1
        if accepted and gt_loop:
            stats["accepted_true_loop"] += 1
        if z_drift_suspect:
            stats["z_drift_suspect_count"] += 1

        rows.append(
            {
                "query_id": query_id,
                "match_id": match_id,
                "gt_query_match_translation_m": translation,
                "gt_query_match_yaw_deg": yaw_deg,
                "gt_is_loop": gt_loop,
                "candidate_accepted": accepted,
                "candidate_source": event.get("candidate_source", ""),
                "gate_result": event.get("gate_result", ""),
                "reject_reason": event.get("reject_reason", ""),
                "fitness_score": event_float(event, "fitness_score"),
                "inlier_ratio": event_float(event, "inlier_ratio"),
                "residual_z": residual_z,
                "residual_roll": residual_roll,
                "residual_pitch": residual_pitch,
                "residual_yaw": residual_yaw,
                "z_drift_suspect": z_drift_suspect,
                "category": category,
            }
        )

    stats["retrieval_miss_estimate"] = len(gt_loop_pairs - candidate_pairs)
    stats["gt_loop_pair_count"] = len(gt_loop_pairs)
    stats["candidate_unique_pair_count"] = len(candidate_pairs)
    stats["accepted_pairs_source"] = "accepted_loops_csv" if accepted_pairs_available else "loop_debug_gate_result"
    stats["thresholds"] = {
        "loop_translation_threshold_m": args.loop_translation_threshold,
        "loop_yaw_threshold_deg": args.loop_yaw_threshold_deg,
        "min_id_gap": args.min_id_gap,
        "z_drift_threshold_m": args.z_drift_threshold,
        "rpy_drift_threshold_deg": args.rpy_drift_threshold_deg,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "loop_candidates_labeled.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "query_id",
            "match_id",
            "gt_query_match_translation_m",
            "gt_query_match_yaw_deg",
            "gt_is_loop",
            "candidate_accepted",
            "candidate_source",
            "gate_result",
            "reject_reason",
            "fitness_score",
            "inlier_ratio",
            "residual_z",
            "residual_roll",
            "residual_pitch",
            "residual_yaw",
            "z_drift_suspect",
            "category",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in (
                "gt_query_match_translation_m",
                "gt_query_match_yaw_deg",
                "fitness_score",
                "inlier_ratio",
                "residual_z",
                "residual_roll",
                "residual_pitch",
                "residual_yaw",
            ):
                formatted[key] = format_float(row[key])
            writer.writerow(formatted)

    json_path = output / "loop_diagnosis.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"analyzed candidates={stats['candidate_count']} output={output}")
    return 0


def main():
    try:
        return analyze(parse_args())
    except Exception as exc:  # pragma: no cover - surfaced in CLI tests
        print(f"n3mapping_loop_debug_analyze: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
