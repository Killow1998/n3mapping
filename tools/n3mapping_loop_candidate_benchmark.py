#!/usr/bin/env python3
"""Compare loop-candidate sources against keyframe ground truth.

This tool is intentionally evaluation-only.  It does not run ICP, does not
change loop behavior, and does not need a pbstream.  It answers a narrower
question: are we failing because candidates are not retrieved, or because a
retrieved true loop is later rejected / applied poorly?
"""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark n3mapping logged candidates against LIO-SAM-style spatial candidates."
    )
    parser.add_argument("--keyframes_gt", required=True, help="keyframes_gt.csv from offline eval")
    parser.add_argument("--loop_debug", default="", help="Optional loop_debug.jsonl")
    parser.add_argument("--accepted_loops", default="", help="Optional accepted_loops.csv")
    parser.add_argument(
        "--trajectory_est",
        default="",
        help="Optional trajectory_est.txt. If present, spatial candidates use estimated poses; otherwise GT poses.",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--loop_translation_threshold", type=float, default=5.0)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--min_id_gap", type=int, default=20)
    parser.add_argument("--spatial_radius_m", type=float, default=15.0)
    parser.add_argument("--spatial_min_id_gap", type=int, default=50)
    parser.add_argument(
        "--spatial_min_frame_gap",
        type=int,
        default=0,
        help="Optional frame-id separation for LIO-SAM-style spatial candidates.",
    )
    parser.add_argument(
        "--sc_top_k",
        type=int,
        default=5,
        help="Top K by sc_distance within logged candidates. This is not pure SC retrieval.",
    )
    return parser.parse_args()


def finite_float(value, default=float("nan")):
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


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


def pose_distance(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def pose_yaw_diff_deg(a, b):
    return abs(angle_diff(a["yaw"], b["yaw"])) * 180.0 / math.pi


def load_keyframes_gt(path):
    poses = {}
    frame_to_keyframe = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"keyframe_id", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing required keyframe GT columns")
        for row in reader:
            keyframe_id = int(row["keyframe_id"])
            frame_id = int(row["frame_id"])
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
                "keyframe_id": keyframe_id,
                "frame_id": frame_id,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw_from_quat(qx, qy, qz, qw),
            }
            frame_to_keyframe[str(frame_id)] = keyframe_id
    if not poses:
        raise RuntimeError(f"{path} contains no keyframe ground-truth poses")
    return poses, frame_to_keyframe


def load_tum_by_frame(path):
    poses = {}
    if not path:
        return poses
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 8:
                raise RuntimeError(f"{path}:{line_number}: expected 8 fields, got {len(parts)}")
            frame_id = parts[0]
            values = [finite_float(value) for value in parts[1:]]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"{path}:{line_number}: non-finite pose")
            x, y, z, qx, qy, qz, qw = values
            poses[frame_id] = {
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw_from_quat(qx, qy, qz, qw),
            }
    return poses


def load_loop_debug(path):
    if not path:
        return []
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
                raise RuntimeError(f"{path}:{line_number}: candidate missing query_id/match_id")
            candidates.append(event)
    return candidates


def load_accepted_pairs(path):
    pairs = set()
    if not path:
        return pairs
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"query_id", "match_id"}.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing query_id/match_id columns")
        for row in reader:
            pairs.add((int(row["query_id"]), int(row["match_id"])))
    return pairs


def is_gt_loop(gt_poses, query_id, match_id, dist_threshold, yaw_threshold):
    query = gt_poses.get(query_id)
    match = gt_poses.get(match_id)
    if query is None or match is None:
        return False
    return (
        pose_distance(query, match) <= dist_threshold
        and pose_yaw_diff_deg(query, match) <= yaw_threshold
    )


def enumerate_gt_pairs(gt_poses, dist_threshold, yaw_threshold, min_id_gap):
    ids = sorted(gt_poses)
    pairs = set()
    queries = set()
    for i, match_id in enumerate(ids):
        for query_id in ids[i + 1 :]:
            if abs(query_id - match_id) < min_id_gap:
                continue
            if is_gt_loop(gt_poses, query_id, match_id, dist_threshold, yaw_threshold):
                pairs.add((query_id, match_id))
                queries.add(query_id)
    return pairs, queries


def make_logged_candidate_rows(candidates, accepted_pairs):
    rows = []
    for event in candidates:
        query_id = int(event["query_id"])
        match_id = int(event["match_id"])
        accepted = (query_id, match_id) in accepted_pairs if accepted_pairs else event.get("gate_result") == "accepted"
        rows.append(
            {
                "method": "n3mapping_logged",
                "query_id": query_id,
                "match_id": match_id,
                "rank": int(event.get("rank", 0) or 0),
                "candidate_source": event.get("candidate_source", ""),
                "score": finite_float(event.get("fused_score")),
                "rhpd_distance": finite_float(event.get("rhpd_distance")),
                "sc_distance": finite_float(event.get("sc_distance")),
                "spatial_distance_m": float("nan"),
                "accepted_by_n3mapping": accepted,
                "gate_result": event.get("gate_result", ""),
                "reject_reason": event.get("reject_reason", ""),
            }
        )
    return rows


def make_sc_rerank_rows(candidates, accepted_pairs, top_k):
    grouped = {}
    for event in candidates:
        query_id = int(event["query_id"])
        sc = finite_float(event.get("sc_distance"))
        if not math.isfinite(sc):
            continue
        grouped.setdefault(query_id, []).append(event)
    rows = []
    for query_id, events in grouped.items():
        events = sorted(events, key=lambda e: finite_float(e.get("sc_distance")))[: max(1, top_k)]
        for rank, event in enumerate(events, start=1):
            match_id = int(event["match_id"])
            rows.append(
                {
                    "method": "sc_rerank_logged_pool",
                    "query_id": query_id,
                    "match_id": match_id,
                    "rank": rank,
                    "candidate_source": event.get("candidate_source", ""),
                    "score": finite_float(event.get("sc_distance")),
                    "rhpd_distance": finite_float(event.get("rhpd_distance")),
                    "sc_distance": finite_float(event.get("sc_distance")),
                    "spatial_distance_m": float("nan"),
                    "accepted_by_n3mapping": (query_id, match_id) in accepted_pairs if accepted_pairs else event.get("gate_result") == "accepted",
                    "gate_result": event.get("gate_result", ""),
                    "reject_reason": event.get("reject_reason", ""),
                }
            )
    return rows


def build_candidate_pose_source(gt_poses, frame_to_keyframe, trajectory_est):
    if not trajectory_est:
        return dict(gt_poses), "gt"
    poses = {}
    for frame_id, keyframe_id in frame_to_keyframe.items():
        pose = trajectory_est.get(frame_id)
        if pose is None:
            continue
        poses[keyframe_id] = {
            "keyframe_id": keyframe_id,
            "frame_id": int(frame_id),
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
            "yaw": pose["yaw"],
        }
    if not poses:
        raise RuntimeError("trajectory_est did not contain any keyframe frame ids")
    return poses, "trajectory_est"


def make_spatial_rows(candidate_poses, gt_poses, args):
    rows_all = []
    rows_nearest = []
    ids = sorted(candidate_poses)
    for query_id in ids:
        query_pose = candidate_poses[query_id]
        query_gt = gt_poses.get(query_id)
        per_query = []
        for match_id in ids:
            if match_id >= query_id:
                continue
            if abs(query_id - match_id) < args.spatial_min_id_gap:
                continue
            match_pose = candidate_poses[match_id]
            if args.spatial_min_frame_gap > 0:
                if query_gt is None or gt_poses.get(match_id) is None:
                    continue
                if abs(query_gt["frame_id"] - gt_poses[match_id]["frame_id"]) < args.spatial_min_frame_gap:
                    continue
            distance = pose_distance(query_pose, match_pose)
            if distance > args.spatial_radius_m:
                continue
            per_query.append((distance, match_id))
        per_query.sort()
        for rank, (distance, match_id) in enumerate(per_query, start=1):
            row = {
                "method": "liosam_spatial_all",
                "query_id": query_id,
                "match_id": match_id,
                "rank": rank,
                "candidate_source": "spatial_radius",
                "score": distance,
                "rhpd_distance": float("nan"),
                "sc_distance": float("nan"),
                "spatial_distance_m": distance,
                "accepted_by_n3mapping": False,
                "gate_result": "",
                "reject_reason": "",
            }
            rows_all.append(row)
            if rank == 1:
                nearest = dict(row)
                nearest["method"] = "liosam_spatial_nearest"
                rows_nearest.append(nearest)
    return rows_all + rows_nearest


def annotate_rows(rows, gt_poses, args):
    annotated = []
    for row in rows:
        query_id = row["query_id"]
        match_id = row["match_id"]
        query = gt_poses.get(query_id)
        match = gt_poses.get(match_id)
        gt_distance = pose_distance(query, match) if query and match else float("nan")
        gt_yaw = pose_yaw_diff_deg(query, match) if query and match else float("nan")
        gt_loop = is_gt_loop(
            gt_poses,
            query_id,
            match_id,
            args.loop_translation_threshold,
            args.loop_yaw_threshold_deg,
        )
        annotated.append(
            {
                **row,
                "gt_distance_m": gt_distance,
                "gt_yaw_diff_deg": gt_yaw,
                "gt_is_loop": gt_loop,
            }
        )
    return annotated


def method_summary(rows, gt_pairs, gt_queries):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    summaries = []
    for method in sorted(grouped):
        method_rows = grouped[method]
        pairs = {(row["query_id"], row["match_id"]) for row in method_rows}
        true_pairs = {(row["query_id"], row["match_id"]) for row in method_rows if row["gt_is_loop"]}
        false_pairs = pairs - true_pairs
        queries = {row["query_id"] for row in method_rows}
        true_queries = {row["query_id"] for row in method_rows if row["gt_is_loop"]}
        accepted_rows = [row for row in method_rows if row["accepted_by_n3mapping"]]
        accepted_true = [row for row in accepted_rows if row["gt_is_loop"]]
        accepted_false = [row for row in accepted_rows if not row["gt_is_loop"]]
        summaries.append(
            {
                "method": method,
                "candidate_pair_count": len(pairs),
                "candidate_query_count": len(queries),
                "true_candidate_pair_count": len(true_pairs),
                "false_candidate_pair_count": len(false_pairs),
                "true_candidate_query_count": len(true_queries),
                "gt_opportunity_pair_count": len(gt_pairs),
                "gt_opportunity_query_count": len(gt_queries),
                "pair_precision": len(true_pairs) / len(pairs) if pairs else float("nan"),
                "pair_recall": len(true_pairs) / len(gt_pairs) if gt_pairs else float("nan"),
                "query_recall": len(true_queries & gt_queries) / len(gt_queries) if gt_queries else float("nan"),
                "false_pair_rate": len(false_pairs) / len(pairs) if pairs else float("nan"),
                "accepted_count": len(accepted_rows),
                "accepted_true_count": len(accepted_true),
                "accepted_false_count": len(accepted_false),
                "accepted_precision": len(accepted_true) / len(accepted_rows) if accepted_rows else float("nan"),
                "true_candidates_rejected_by_processing": sum(
                    1 for row in method_rows
                    if row["gt_is_loop"] and not row["accepted_by_n3mapping"] and row["method"] != "liosam_spatial_all" and row["method"] != "liosam_spatial_nearest"
                ),
            }
        )
    return summaries


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_number(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gt_poses, frame_to_keyframe = load_keyframes_gt(args.keyframes_gt)
    gt_pairs, gt_queries = enumerate_gt_pairs(
        gt_poses,
        args.loop_translation_threshold,
        args.loop_yaw_threshold_deg,
        args.min_id_gap,
    )
    accepted_pairs = load_accepted_pairs(args.accepted_loops)
    loop_candidates = load_loop_debug(args.loop_debug)
    trajectory_est = load_tum_by_frame(args.trajectory_est)
    candidate_poses, spatial_pose_source = build_candidate_pose_source(
        gt_poses,
        frame_to_keyframe,
        trajectory_est,
    )

    rows = []
    if loop_candidates:
        rows.extend(make_logged_candidate_rows(loop_candidates, accepted_pairs))
        rows.extend(make_sc_rerank_rows(loop_candidates, accepted_pairs, args.sc_top_k))
    rows.extend(make_spatial_rows(candidate_poses, gt_poses, args))
    rows = annotate_rows(rows, gt_poses, args)
    summaries = method_summary(rows, gt_pairs, gt_queries)

    candidate_fields = [
        "method",
        "query_id",
        "match_id",
        "rank",
        "candidate_source",
        "score",
        "rhpd_distance",
        "sc_distance",
        "spatial_distance_m",
        "accepted_by_n3mapping",
        "gate_result",
        "reject_reason",
        "gt_distance_m",
        "gt_yaw_diff_deg",
        "gt_is_loop",
    ]
    summary_fields = [
        "method",
        "candidate_pair_count",
        "candidate_query_count",
        "true_candidate_pair_count",
        "false_candidate_pair_count",
        "true_candidate_query_count",
        "gt_opportunity_pair_count",
        "gt_opportunity_query_count",
        "pair_precision",
        "pair_recall",
        "query_recall",
        "false_pair_rate",
        "accepted_count",
        "accepted_true_count",
        "accepted_false_count",
        "accepted_precision",
        "true_candidates_rejected_by_processing",
    ]
    write_csv(output / "candidate_benchmark_pairs.csv", rows, candidate_fields)
    write_csv(output / "candidate_benchmark_summary.csv", summaries, summary_fields)

    report = {
        "keyframes_gt": str(Path(args.keyframes_gt)),
        "loop_debug": str(Path(args.loop_debug)) if args.loop_debug else "",
        "accepted_loops": str(Path(args.accepted_loops)) if args.accepted_loops else "",
        "trajectory_est": str(Path(args.trajectory_est)) if args.trajectory_est else "",
        "spatial_pose_source": spatial_pose_source,
        "thresholds": {
            "loop_translation_threshold_m": args.loop_translation_threshold,
            "loop_yaw_threshold_deg": args.loop_yaw_threshold_deg,
            "min_id_gap": args.min_id_gap,
            "spatial_radius_m": args.spatial_radius_m,
            "spatial_min_id_gap": args.spatial_min_id_gap,
            "spatial_min_frame_gap": args.spatial_min_frame_gap,
            "sc_top_k": args.sc_top_k,
        },
        "method_summaries": [
            {key: json_number(value) for key, value in row.items()} for row in summaries
        ],
        "notes": [
            "n3mapping_logged is the current loop_debug candidate set, normally RHPD-primary with SC auxiliary scores.",
            "sc_rerank_logged_pool ranks only within logged candidates; it is not pure SC global retrieval.",
            "liosam_spatial_* is a LIO-SAM-style spatial/time-gap candidate oracle over keyframe poses; it does not run ICP.",
        ],
    }
    with open(output / "candidate_benchmark_summary.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
