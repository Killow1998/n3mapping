#!/usr/bin/env python3
"""Find ground-truth loop opportunities in a TUM trajectory file."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect loop-opportunity pairs in timestamp x y z qx qy qz qw trajectories."
    )
    parser.add_argument("--trajectory", required=True, help="TUM trajectory file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sequence", default="", help="Sequence name for reports")
    parser.add_argument("--distance_threshold_m", type=float, default=3.0)
    parser.add_argument("--yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--min_time_gap_s", type=float, default=20.0)
    parser.add_argument("--min_index_gap", type=int, default=50)
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
        help="Use every Nth pose for pair enumeration. Default: 1.",
    )
    parser.add_argument("--max_pairs", type=int, default=5000)
    return parser.parse_args()


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(value)
    return parsed


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def load_tum(path):
    poses = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 8:
                raise RuntimeError(f"{path}:{line_number}: expected 8 fields, got {len(parts)}")
            try:
                t, x, y, z, qx, qy, qz, qw = [finite_float(v) for v in parts]
            except ValueError as exc:
                raise RuntimeError(f"{path}:{line_number}: non-finite value {exc}") from exc
            poses.append(
                {
                    "index": len(poses),
                    "timestamp": t,
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": quat_to_yaw(qx, qy, qz, qw),
                }
            )
    if not poses:
        raise RuntimeError(f"{path}: no poses")
    return poses


def percentile(values, q):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    idx = max(0.0, min(1.0, q)) * (len(values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[int(lo)]
    t = idx - lo
    return values[int(lo)] * (1.0 - t) + values[int(hi)] * t


def path_length(poses):
    total = 0.0
    for a, b in zip(poses, poses[1:]):
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return total


def detect_pairs(poses, args):
    sampled = poses[:: max(1, args.sample_stride)]
    yaw_threshold = args.yaw_threshold_deg * math.pi / 180.0
    pairs = []
    query_has_loop = set()
    for i, a in enumerate(sampled):
        for b in sampled[:i]:
            index_gap = a["index"] - b["index"]
            time_gap = abs(a["timestamp"] - b["timestamp"])
            if index_gap < args.min_index_gap or time_gap < args.min_time_gap_s:
                continue
            dx = a["x"] - b["x"]
            dy = a["y"] - b["y"]
            dz = a["z"] - b["z"]
            xy = math.hypot(dx, dy)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            yaw = abs(angle_diff(a["yaw"], b["yaw"]))
            if dist <= args.distance_threshold_m and yaw <= yaw_threshold:
                query_has_loop.add(a["index"])
                pairs.append(
                    {
                        "query_index": a["index"],
                        "match_index": b["index"],
                        "query_timestamp": a["timestamp"],
                        "match_timestamp": b["timestamp"],
                        "time_gap_s": time_gap,
                        "index_gap": index_gap,
                        "distance_m": dist,
                        "xy_distance_m": xy,
                        "z_delta_m": dz,
                        "yaw_diff_deg": yaw * 180.0 / math.pi,
                    }
                )
    pairs.sort(key=lambda row: (row["query_index"], row["distance_m"], -row["time_gap_s"]))
    return pairs, query_has_loop


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_number(value):
    return value if math.isfinite(value) else None


def main():
    args = parse_args()
    trajectory = Path(args.trajectory)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    poses = load_tum(trajectory)
    pairs, query_has_loop = detect_pairs(poses, args)

    write_csv(
        output / "loop_opportunities.csv",
        pairs[: max(0, args.max_pairs)],
        [
            "query_index",
            "match_index",
            "query_timestamp",
            "match_timestamp",
            "time_gap_s",
            "index_gap",
            "distance_m",
            "xy_distance_m",
            "z_delta_m",
            "yaw_diff_deg",
        ],
    )
    trajectory_rows = [
        {
            "index": pose["index"],
            "timestamp": pose["timestamp"],
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
            "yaw_rad": pose["yaw"],
            "has_loop_opportunity": pose["index"] in query_has_loop,
        }
        for pose in poses
    ]
    write_csv(
        output / "trajectory_xy.csv",
        trajectory_rows,
        ["index", "timestamp", "x", "y", "z", "yaw_rad", "has_loop_opportunity"],
    )

    dists = [row["distance_m"] for row in pairs]
    time_gaps = [row["time_gap_s"] for row in pairs]
    summary = {
        "sequence": args.sequence or trajectory.stem,
        "trajectory": str(trajectory),
        "pose_count": len(poses),
        "duration_s": poses[-1]["timestamp"] - poses[0]["timestamp"],
        "path_length_xy_m": path_length(poses),
        "distance_threshold_m": args.distance_threshold_m,
        "yaw_threshold_deg": args.yaw_threshold_deg,
        "min_time_gap_s": args.min_time_gap_s,
        "min_index_gap": args.min_index_gap,
        "sample_stride": max(1, args.sample_stride),
        "loop_opportunity_pair_count": len(pairs),
        "query_with_loop_opportunity_count": len(query_has_loop),
        "query_with_loop_opportunity_ratio": len(query_has_loop) / len(poses),
        "loop_distance_median_m": json_number(percentile(dists, 0.5)),
        "loop_distance_p95_m": json_number(percentile(dists, 0.95)),
        "loop_time_gap_median_s": json_number(percentile(time_gaps, 0.5)),
        "loop_time_gap_p95_s": json_number(percentile(time_gaps, 0.95)),
    }
    with open(output / "loop_opportunity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
