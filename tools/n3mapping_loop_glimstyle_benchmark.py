#!/usr/bin/env python3
"""Offline GLIM-style submap overlap / registration-cost benchmark.

This is not GLIM.  It is the cheapest useful test of GLIM's relevant idea for
n3mapping: submap pairs should be judged by overlap and registration cost, not
only descriptor rank.  It does not change runtime loop closure or relocalization.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import n3mapping_loop_stdlite_benchmark as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti_root", required=False, default="")
    parser.add_argument("--sequence", required=False, default="")
    parser.add_argument("--keyframes_gt", required=False, default="")
    parser.add_argument("--loop_debug", required=False, default="")
    parser.add_argument("--accepted_loops", default="")
    parser.add_argument("--output", required=False, default="")
    parser.add_argument("--max_pairs", type=int, default=80)
    parser.add_argument("--submap_keyframe_radius", type=int, default=2)
    parser.add_argument("--range_min", type=float, default=2.0)
    parser.add_argument("--range_max", type=float, default=60.0)
    parser.add_argument("--voxel_size", type=float, default=0.8)
    parser.add_argument("--max_points", type=int, default=2500)
    parser.add_argument("--inlier_threshold", type=float, default=1.25)
    parser.add_argument("--loop_translation_threshold", type=float, default=5.0)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def transform_from_event(event, prefix):
    names = [f"{prefix}_{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    values = [common.finite_float(event.get(name)) for name in names]
    if not all(math.isfinite(value) for value in values):
        return None
    if np.linalg.norm(values[:3]) < 1e-9 and abs(values[5]) < 1e-9 and prefix != "pred_match_query":
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_matrix(values[3], values[4], values[5])
    T[:3, 3] = values[:3]
    return T


def cap_points(points, max_points):
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices]


def cost_proxy(T_target_source, source, target, threshold):
    if len(source) == 0 or len(target) == 0:
        return {"inlier_ratio": 0.0, "rmse": None, "median": None, "p90": None}
    transformed = common.transform_points(T_target_source, source)
    diff = transformed[:, None, :] - target[None, :, :]
    dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    inliers = dist <= threshold
    if not np.any(inliers):
        return {
            "inlier_ratio": 0.0,
            "rmse": float(np.sqrt(np.mean(dist * dist))),
            "median": float(np.median(dist)),
            "p90": float(np.percentile(dist, 90)),
        }
    inlier_dist = dist[inliers]
    return {
        "inlier_ratio": float(np.mean(inliers)),
        "rmse": float(np.sqrt(np.mean(inlier_dist * inlier_dist))),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
    }


def load_submap(gt_poses, sorted_ids, keyframe_id, args):
    points = common.load_submap_points(args.kitti_root, args.sequence, gt_poses, sorted_ids, keyframe_id, args)
    points = common.voxel_downsample(points, args.voxel_size)
    return cap_points(points, args.max_points)


def summarize(rows, field, predicate):
    values = [row[field] for row in rows if predicate(row) and isinstance(row[field], (float, int)) and math.isfinite(row[field])]
    return sum(values) / len(values) if values else None


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_clean(v) for key, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def run_self_check():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(300, 3))
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.2, -0.1]
    target = common.transform_points(T, source)
    good = cost_proxy(T, source, target, 0.2)
    bad = cost_proxy(np.eye(4), source, target, 0.2)
    if good["inlier_ratio"] < 0.99 or bad["inlier_ratio"] > 0.5:
        raise RuntimeError(f"self-check failed: good={good} bad={bad}")
    print("GLIM-style proxy self-check passed")


def main():
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.output:
        raise RuntimeError("--output is required unless --self_check is used")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gt_poses = common.load_keyframes_gt(args.keyframes_gt)
    sorted_ids = sorted(gt_poses)
    accepted_pairs = common.load_accepted_pairs(args.accepted_loops)
    candidates = common.select_candidate_pairs(
        common.load_loop_debug_candidates(args.loop_debug),
        accepted_pairs,
        args.max_pairs,
    )

    rows = []
    cloud_cache = {}
    for rank, event in enumerate(candidates, start=1):
        query_id = int(event["query_id"])
        match_id = int(event["match_id"])
        if query_id not in gt_poses or match_id not in gt_poses:
            continue
        for keyframe_id in (query_id, match_id):
            if keyframe_id not in cloud_cache:
                cloud_cache[keyframe_id] = load_submap(gt_poses, sorted_ids, keyframe_id, args)
        source = cloud_cache[query_id]
        target = cloud_cache[match_id]
        gt_is_loop, gt_distance, gt_yaw, T_gt = common.gt_loop_label(
            gt_poses,
            query_id,
            match_id,
            args.loop_translation_threshold,
            args.loop_yaw_threshold_deg,
        )
        T_pred = transform_from_event(event, "pred_match_query")
        T_measured = transform_from_event(event, "measured_match_query")
        gt_cost = cost_proxy(T_gt, source, target, args.inlier_threshold)
        pred_cost = cost_proxy(T_pred, source, target, args.inlier_threshold) if T_pred is not None else {}
        measured_cost = cost_proxy(T_measured, source, target, args.inlier_threshold) if T_measured is not None else {}
        rows.append(
            {
                "rank": rank,
                "query_id": query_id,
                "match_id": match_id,
                "candidate_source": event.get("candidate_source", ""),
                "gate_result": event.get("gate_result", ""),
                "reject_reason": event.get("reject_reason", ""),
                "accepted_by_n3mapping": (query_id, match_id) in accepted_pairs or event.get("gate_result") == "accepted",
                "gt_is_loop": gt_is_loop,
                "gt_distance_m": gt_distance,
                "gt_yaw_diff_deg": gt_yaw,
                "source_points": len(source),
                "target_points": len(target),
                "gt_inlier_ratio": gt_cost.get("inlier_ratio"),
                "gt_rmse": gt_cost.get("rmse"),
                "gt_p90": gt_cost.get("p90"),
                "pred_inlier_ratio": pred_cost.get("inlier_ratio"),
                "pred_rmse": pred_cost.get("rmse"),
                "pred_p90": pred_cost.get("p90"),
                "measured_inlier_ratio": measured_cost.get("inlier_ratio"),
                "measured_rmse": measured_cost.get("rmse"),
                "measured_p90": measured_cost.get("p90"),
            }
        )
        print(
            f"[{rank}/{len(candidates)}] {query_id}->{match_id} gt_loop={gt_is_loop} "
            f"gt_inlier={gt_cost.get('inlier_ratio'):.3f} "
            f"pred_inlier={pred_cost.get('inlier_ratio', float('nan'))}"
        )

    write_csv(output / "glimstyle_pairs.csv", rows)
    report = {
        "tested_pair_count": len(rows),
        "gt_loop_pair_count": sum(1 for row in rows if row["gt_is_loop"]),
        "non_loop_pair_count": sum(1 for row in rows if not row["gt_is_loop"]),
        "gt_inlier_mean_true_loop": summarize(rows, "gt_inlier_ratio", lambda row: row["gt_is_loop"]),
        "gt_inlier_mean_non_loop": summarize(rows, "gt_inlier_ratio", lambda row: not row["gt_is_loop"]),
        "pred_inlier_mean_true_loop": summarize(rows, "pred_inlier_ratio", lambda row: row["gt_is_loop"]),
        "pred_inlier_mean_non_loop": summarize(rows, "pred_inlier_ratio", lambda row: not row["gt_is_loop"]),
        "measured_inlier_mean_true_loop": summarize(rows, "measured_inlier_ratio", lambda row: row["gt_is_loop"]),
        "measured_inlier_mean_non_loop": summarize(rows, "measured_inlier_ratio", lambda row: not row["gt_is_loop"]),
        "notes": [
            "This is an offline GLIM-style overlap/cost proxy, not a GLIM factor implementation.",
            "gt_* uses ground truth transform and measures upper-bound physical overlap.",
            "pred_* and measured_* use logged n3mapping transforms when available.",
            "No runtime loop closure or relocalization behavior is changed.",
        ],
    }
    with open(output / "glimstyle_summary.json", "w") as f:
        json.dump(json_clean(report), f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
