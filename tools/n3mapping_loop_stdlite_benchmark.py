#!/usr/bin/env python3
"""Offline STD-like loop verifier benchmark.

This tool does not change n3mapping runtime behavior.  It answers one narrow
question: for logged loop candidates, can a lightweight correspondence-bearing
triangle descriptor recover a usable relative pose from the raw KITTI360 lidar
frames?

It is intentionally "STD-lite", not a production STD implementation.  The
output is evidence for whether a correspondence verifier is worth integrating.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti_root", default="", help="KITTI360 root directory")
    parser.add_argument("--sequence", default="", help="KITTI360 sequence, e.g. 2013_05_28_drive_0005_sync")
    parser.add_argument("--keyframes_gt", default="", help="keyframes_gt.csv from offline eval")
    parser.add_argument("--loop_debug", default="", help="loop_debug.jsonl from offline eval")
    parser.add_argument("--accepted_loops", default="", help="accepted_loops.csv from offline eval")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--max_pairs", type=int, default=80)
    parser.add_argument("--range_min", type=float, default=2.0)
    parser.add_argument("--range_max", type=float, default=60.0)
    parser.add_argument("--voxel_size", type=float, default=0.8)
    parser.add_argument("--max_keypoints", type=int, default=80)
    parser.add_argument("--keypoint_mode", choices=["uniform", "farthest"], default="uniform")
    parser.add_argument("--max_triangles", type=int, default=3000)
    parser.add_argument(
        "--submap_keyframe_radius",
        type=int,
        default=0,
        help="Aggregate +/- N neighboring keyframe scans into the center lidar frame before STD-lite.",
    )
    parser.add_argument("--side_bin", type=float, default=0.75)
    parser.add_argument("--vote_distance", type=float, default=1.25)
    parser.add_argument("--min_votes", type=int, default=12)
    parser.add_argument("--loop_translation_threshold", type=float, default=5.0)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def finite_float(value, default=float("nan")):
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def quat_to_rot(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0 or not math.isfinite(norm):
        raise RuntimeError("invalid zero quaternion")
    q /= norm
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(x, y, z, qx, qy, qz, qw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


def yaw_from_rot(R):
    return math.atan2(R[1, 0], R[0, 0])


def angle_diff(a, b):
    diff = a - b
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


def transform_points(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def load_keyframes_gt(path):
    poses = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"keyframe_id", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing required keyframe GT columns")
        for row in reader:
            keyframe_id = int(row["keyframe_id"])
            frame_id = int(row["frame_id"])
            values = [finite_float(row[name]) for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"{path} has non-finite pose for keyframe {keyframe_id}")
            T = pose_matrix(*values)
            poses[keyframe_id] = {
                "keyframe_id": keyframe_id,
                "frame_id": frame_id,
                "T_world_lidar": T,
                "yaw": yaw_from_rot(T[:3, :3]),
            }
    if not poses:
        raise RuntimeError(f"{path} contains no keyframe poses")
    return poses


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


def load_loop_debug_candidates(path):
    rows = []
    if not path:
        return rows
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("record_type") != "candidate":
                continue
            if "query_id" not in event or "match_id" not in event:
                raise RuntimeError(f"{path}:{line_number}: candidate missing query_id/match_id")
            rows.append(event)
    return rows


def candidate_sort_key(event, accepted_pairs):
    pair = (int(event["query_id"]), int(event["match_id"]))
    accepted = pair in accepted_pairs or event.get("gate_result") == "accepted"
    converged = bool(event.get("icp_converged", False))
    score = finite_float(event.get("fused_score"), float("inf"))
    return (0 if accepted else 1, 0 if converged else 1, score, pair[0], pair[1])


def select_candidate_pairs(events, accepted_pairs, max_pairs):
    by_pair = {}
    for event in events:
        pair = (int(event["query_id"]), int(event["match_id"]))
        if pair not in by_pair or candidate_sort_key(event, accepted_pairs) < candidate_sort_key(by_pair[pair], accepted_pairs):
            by_pair[pair] = event
    return sorted(by_pair.values(), key=lambda event: candidate_sort_key(event, accepted_pairs))[:max_pairs]


def kitti_bin_path(kitti_root, sequence, frame_id):
    return (
        Path(kitti_root)
        / "data_3d_raw"
        / sequence
        / "velodyne_points"
        / "data"
        / f"{frame_id:010d}.bin"
    )


def load_kitti_points(path, range_min, range_max):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise RuntimeError(f"{path} does not contain XYZI float32 records")
    points = raw.reshape((-1, 4))[:, :3].astype(np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    radius = np.linalg.norm(points[:, :2], axis=1)
    mask = (radius >= range_min) & (radius <= range_max)
    return points[mask]


def load_submap_points(kitti_root, sequence, gt_poses, sorted_ids, center_id, args):
    center_pose = gt_poses[center_id]
    center_index = sorted_ids.index(center_id)
    start = max(0, center_index - args.submap_keyframe_radius)
    stop = min(len(sorted_ids), center_index + args.submap_keyframe_radius + 1)
    T_center_world = np.linalg.inv(center_pose["T_world_lidar"])
    chunks = []
    for keyframe_id in sorted_ids[start:stop]:
        pose = gt_poses[keyframe_id]
        path = kitti_bin_path(kitti_root, sequence, pose["frame_id"])
        if not path.exists():
            raise RuntimeError(f"missing KITTI360 bin for keyframe {keyframe_id}: {path}")
        points = load_kitti_points(path, args.range_min, args.range_max)
        T_center_scan = T_center_world @ pose["T_world_lidar"]
        chunks.append(transform_points(T_center_scan, points))
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(chunks)


def voxel_downsample(points, voxel_size):
    if points.size == 0:
        return points.reshape((0, 3))
    keys = np.floor(points / voxel_size).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def farthest_keypoints(points, max_keypoints):
    if len(points) <= max_keypoints:
        return points.copy()
    # Cap the FPS input deterministically so large scans stay cheap.
    if len(points) > 4000:
        order = np.argsort(np.linalg.norm(points[:, :2], axis=1))
        indices = np.linspace(0, len(order) - 1, 4000).astype(np.int64)
        points = points[order[indices]]
    selected = np.empty((max_keypoints, 3), dtype=np.float64)
    first = int(np.argmax(np.linalg.norm(points[:, :2], axis=1)))
    selected[0] = points[first]
    min_dist2 = np.sum((points - selected[0]) ** 2, axis=1)
    for i in range(1, max_keypoints):
        idx = int(np.argmax(min_dist2))
        selected[i] = points[idx]
        dist2 = np.sum((points - selected[i]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected


def uniform_keypoints(points, max_keypoints):
    if len(points) <= max_keypoints:
        return points.copy()
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    indices = np.linspace(0, len(order) - 1, max_keypoints).astype(np.int64)
    return points[order[indices]]


def triangle_area(points):
    a, b, c = points
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def build_triangle_index(points, max_triangles, side_bin):
    if len(points) < 3:
        return {}
    rng = np.random.default_rng(17 + len(points))
    index = {}
    attempts = max(max_triangles * 4, 100)
    generated = 0
    seen = set()
    for _ in range(attempts):
        tri = tuple(sorted(rng.choice(len(points), size=3, replace=False).tolist()))
        if tri in seen:
            continue
        seen.add(tri)
        tri_points = points[list(tri)]
        sides = np.array(
            [
                np.linalg.norm(tri_points[0] - tri_points[1]),
                np.linalg.norm(tri_points[0] - tri_points[2]),
                np.linalg.norm(tri_points[1] - tri_points[2]),
            ],
            dtype=np.float64,
        )
        if np.min(sides) < 2.0 or np.max(sides) > 55.0 or triangle_area(tri_points) < 1.0:
            continue
        key = tuple(np.round(np.sort(sides) / side_bin).astype(np.int32).tolist())
        bucket = index.setdefault(key, [])
        if len(bucket) < 8:
            bucket.append(tri)
        generated += 1
        if generated >= max_triangles:
            break
    return index


def estimate_rigid(source, target):
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    H = (source - source_centroid).T @ (target - target_centroid)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = target_centroid - R @ source_centroid
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def vote_transform(T_target_source, source_points, target_points, vote_distance):
    transformed = transform_points(T_target_source, source_points)
    diff = transformed[:, None, :] - target_points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.sqrt(np.min(dist2, axis=1))
    inliers = nearest <= vote_distance
    if not np.any(inliers):
        return 0, float("inf")
    return int(np.sum(inliers)), float(np.sqrt(np.mean(nearest[inliers] ** 2)))


def stdlite_verify(source_points, target_points, args):
    source_index = build_triangle_index(source_points, args.max_triangles, args.side_bin)
    target_index = build_triangle_index(target_points, args.max_triangles, args.side_bin)
    shared_keys = sorted(set(source_index) & set(target_index), key=lambda k: len(source_index[k]) * len(target_index[k]), reverse=True)
    best = {
        "pose_found": False,
        "vote_count": 0,
        "vote_ratio": 0.0,
        "rmse": float("inf"),
        "T_target_source": np.eye(4, dtype=np.float64),
        "shared_descriptor_bins": len(shared_keys),
        "hypothesis_count": 0,
    }
    for key in shared_keys[:80]:
        for source_tri in source_index[key]:
            source_tri_points = source_points[list(source_tri)]
            for target_tri in target_index[key]:
                target_tri_points_base = target_points[list(target_tri)]
                for perm in itertools.permutations(range(3)):
                    target_tri_points = target_tri_points_base[list(perm)]
                    T = estimate_rigid(source_tri_points, target_tri_points)
                    votes, rmse = vote_transform(T, source_points, target_points, args.vote_distance)
                    best["hypothesis_count"] += 1
                    if votes > best["vote_count"] or (votes == best["vote_count"] and rmse < best["rmse"]):
                        best.update(
                            {
                                "pose_found": votes >= args.min_votes,
                                "vote_count": votes,
                                "vote_ratio": votes / len(source_points) if len(source_points) else 0.0,
                                "rmse": rmse,
                                "T_target_source": T,
                            }
                        )
                    if best["hypothesis_count"] >= 600:
                        return best
    return best


def relative_pose_error(T_gt, T_est):
    T_err = np.linalg.inv(T_gt) @ T_est
    trans = float(np.linalg.norm(T_err[:3, 3]))
    z = float(T_err[2, 3])
    yaw = abs(angle_diff(yaw_from_rot(T_est[:3, :3]), yaw_from_rot(T_gt[:3, :3]))) * 180.0 / math.pi
    return trans, abs(z), yaw


def gt_loop_label(gt_poses, query_id, match_id, trans_threshold, yaw_threshold_deg):
    T_m_q = np.linalg.inv(gt_poses[match_id]["T_world_lidar"]) @ gt_poses[query_id]["T_world_lidar"]
    trans = float(np.linalg.norm(T_m_q[:3, 3]))
    yaw = abs(angle_diff(gt_poses[query_id]["yaw"], gt_poses[match_id]["yaw"])) * 180.0 / math.pi
    return trans <= trans_threshold and yaw <= yaw_threshold_deg, trans, yaw, T_m_q


def prepare_cloud(points, args):
    downsampled = voxel_downsample(points, args.voxel_size)
    if args.keypoint_mode == "farthest":
        return farthest_keypoints(downsampled, args.max_keypoints)
    return uniform_keypoints(downsampled, args.max_keypoints)


def json_number(value):
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def run_self_check():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(70, 3))
    base[:, 0] *= 12.0
    base[:, 1] *= 8.0
    base[:, 2] *= 2.0
    yaw = 0.35
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T[:3, 3] = [3.0, -1.5, 0.7]
    target = transform_points(T, base)

    class Args:
        max_triangles = 2500
        side_bin = 0.25
        vote_distance = 0.5
        min_votes = 30

    result = stdlite_verify(base, target, Args())
    trans_err, z_err, yaw_err = relative_pose_error(T, result["T_target_source"])
    if not result["pose_found"] or trans_err > 0.5 or yaw_err > 2.0:
        raise RuntimeError(
            f"STD-lite self-check failed: found={result['pose_found']} "
            f"votes={result['vote_count']} trans_err={trans_err:.3f} yaw_err={yaw_err:.3f}"
        )
    print("STD-lite self-check passed")


def main():
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.output:
        raise RuntimeError("--output is required unless --self_check is used")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    gt_poses = load_keyframes_gt(args.keyframes_gt)
    sorted_keyframe_ids = sorted(gt_poses)
    accepted_pairs = load_accepted_pairs(args.accepted_loops)
    candidates = select_candidate_pairs(load_loop_debug_candidates(args.loop_debug), accepted_pairs, args.max_pairs)

    rows = []
    for index, event in enumerate(candidates, start=1):
        query_id = int(event["query_id"])
        match_id = int(event["match_id"])
        if query_id not in gt_poses or match_id not in gt_poses:
            continue
        gt_is_loop, gt_dist, gt_yaw, T_match_query_gt = gt_loop_label(
            gt_poses,
            query_id,
            match_id,
            args.loop_translation_threshold,
            args.loop_yaw_threshold_deg,
        )
        query_points = prepare_cloud(
            load_submap_points(args.kitti_root, args.sequence, gt_poses, sorted_keyframe_ids, query_id, args),
            args,
        )
        match_points = prepare_cloud(
            load_submap_points(args.kitti_root, args.sequence, gt_poses, sorted_keyframe_ids, match_id, args),
            args,
        )
        oracle_votes, oracle_rmse = vote_transform(T_match_query_gt, query_points, match_points, args.vote_distance)
        result = stdlite_verify(query_points, match_points, args)
        trans_err, z_err, yaw_err = relative_pose_error(T_match_query_gt, result["T_target_source"])
        pose_success = (
            result["pose_found"]
            and trans_err <= args.loop_translation_threshold
            and yaw_err <= args.loop_yaw_threshold_deg
        )
        rows.append(
            {
                "rank": index,
                "query_id": query_id,
                "match_id": match_id,
                "query_frame_id": gt_poses[query_id]["frame_id"],
                "match_frame_id": gt_poses[match_id]["frame_id"],
                "candidate_source": event.get("candidate_source", ""),
                "gate_result": event.get("gate_result", ""),
                "reject_reason": event.get("reject_reason", ""),
                "accepted_by_n3mapping": (query_id, match_id) in accepted_pairs or event.get("gate_result") == "accepted",
                "gt_is_loop": gt_is_loop,
                "gt_distance_m": gt_dist,
                "gt_yaw_diff_deg": gt_yaw,
                "query_keypoints": len(query_points),
                "match_keypoints": len(match_points),
                "submap_keyframe_radius": args.submap_keyframe_radius,
                "oracle_overlap_votes": oracle_votes,
                "oracle_overlap_ratio": oracle_votes / len(query_points) if len(query_points) else 0.0,
                "oracle_overlap_rmse": oracle_rmse,
                "stdlite_pose_found": result["pose_found"],
                "stdlite_pose_gt_success": pose_success,
                "stdlite_vote_count": result["vote_count"],
                "stdlite_vote_ratio": result["vote_ratio"],
                "stdlite_rmse": result["rmse"],
                "stdlite_shared_descriptor_bins": result["shared_descriptor_bins"],
                "stdlite_hypothesis_count": result["hypothesis_count"],
                "stdlite_translation_error_to_gt_m": trans_err,
                "stdlite_z_error_to_gt_m": z_err,
                "stdlite_yaw_error_to_gt_deg": yaw_err,
            }
        )
        print(
            f"[{index}/{len(candidates)}] {query_id}->{match_id} gt_loop={gt_is_loop} "
            f"oracle={oracle_votes}/{len(query_points)} std_votes={result['vote_count']} "
            f"err=({trans_err:.2f}m,{yaw_err:.1f}deg)"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        with open(output / "stdlite_pairs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    gt_loop_rows = [row for row in rows if row["gt_is_loop"]]
    non_loop_rows = [row for row in rows if not row["gt_is_loop"]]
    success_rows = [row for row in rows if row["stdlite_pose_gt_success"]]
    report = {
        "kitti_root": args.kitti_root,
        "sequence": args.sequence,
        "keyframes_gt": args.keyframes_gt,
        "loop_debug": args.loop_debug,
        "accepted_loops": args.accepted_loops,
        "tested_pair_count": len(rows),
        "submap_keyframe_radius": args.submap_keyframe_radius,
        "keypoint_mode": args.keypoint_mode,
        "gt_loop_pair_count": len(gt_loop_rows),
        "non_loop_pair_count": len(non_loop_rows),
        "stdlite_pose_found_count": sum(1 for row in rows if row["stdlite_pose_found"]),
        "stdlite_pose_gt_success_count": len(success_rows),
        "stdlite_gt_loop_success_rate": (
            sum(1 for row in gt_loop_rows if row["stdlite_pose_gt_success"]) / len(gt_loop_rows)
            if gt_loop_rows
            else None
        ),
        "stdlite_false_pose_success_count": sum(1 for row in non_loop_rows if row["stdlite_pose_gt_success"]),
        "oracle_overlap_mean_gt_loop": (
            sum(row["oracle_overlap_ratio"] for row in gt_loop_rows) / len(gt_loop_rows)
            if gt_loop_rows
            else None
        ),
        "oracle_overlap_mean_non_loop": (
            sum(row["oracle_overlap_ratio"] for row in non_loop_rows) / len(non_loop_rows)
            if non_loop_rows
            else None
        ),
        "notes": [
            "STD-lite is a diagnostic triangle-correspondence verifier, not production STD.",
            "oracle_overlap uses ground-truth relative pose and is an upper-bound overlap signal, not runtime evidence.",
            "No n3mapping runtime loop or relocalization behavior is changed by this tool.",
        ],
    }
    with open(output / "stdlite_summary.json", "w") as f:
        json.dump({k: json_number(v) for k, v in report.items()}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
