#!/usr/bin/env python3
"""Summarize n3mapping offline evaluation artifacts into a compact matrix."""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_FIELDS = [
    "run",
    "run_dir",
    "mode",
    "sequence",
    "frames_processed",
    "query_count",
    "accepted_keyframes",
    "accepted_loop_count",
    "dense_trajectory_count",
    "lock_success_rate",
    "pose_success_rate",
    "trajectory_pair_count",
    "trajectory_translation_mean_m",
    "trajectory_translation_median_m",
    "trajectory_translation_p95_m",
    "trajectory_translation_max_m",
    "trajectory_xy_p95_m",
    "trajectory_z_p95_m",
    "trajectory_yaw_p95_deg",
    "loop_candidate_count",
    "loop_gt_pair_count",
    "loop_accepted_candidate_count",
    "loop_accepted_true_loop",
    "loop_accepted_true_loop_good",
    "loop_accepted_true_loop_bad_z",
    "loop_accepted_true_loop_bad_z_measurement",
    "loop_accepted_true_loop_bad_z_after",
    "loop_accepted_true_loop_corrected_z",
    "loop_accepted_true_loop_bad_roll_pitch",
    "loop_accepted_false_loop",
    "loop_accepted_full6dof",
    "loop_accepted_planar_xy_yaw",
    "loop_precision",
    "loop_gt_pair_coverage",
    "loop_icp_reject_true_loop",
    "loop_verification_reject_true_loop",
    "loop_true_loop_not_selected",
    "loop_retrieval_false_positive",
    "loop_retrieval_miss_estimate",
    "loop_z_drift_suspect_count",
    "optimization_summary_count",
    "optimization_high_residual_z_after_count",
    "optimization_max_residual_z_after_m",
    "meets_loop_precision_80",
    "meets_reloc_pose_success_80",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a CSV/JSON leaderboard from n3mapping eval output directories."
    )
    parser.add_argument("--output", required=True, help="Output directory for matrix_summary.csv/json")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Evaluation run to summarize. May be repeated.",
    )
    parser.add_argument("--analyzer", default="", help="Path to n3mapping_loop_debug_analyze.py")
    parser.add_argument("--loop_translation_threshold", type=float, default=5.0)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--min_id_gap", type=int, default=20)
    parser.add_argument("--z_drift_threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip_loop_analyzer",
        action="store_true",
        help="Do not synthesize missing loop_diagnosis.json from loop_debug/keyframes_gt.",
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


def json_number(value):
    return value if math.isfinite(value) else None


def format_cell(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9g}" if math.isfinite(value) else ""
    if value is None:
        return ""
    return str(value)


def percentile(values, q):
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    idx = max(0.0, min(1.0, q)) * (len(finite) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return finite[int(lo)]
    t = idx - lo
    return finite[int(lo)] * (1.0 - t) + finite[int(hi)] * t


def mean(values):
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


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


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_trajectory(path):
    poses = {}
    if not path.exists():
        return poses
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 8:
                raise RuntimeError(f"{path}:{line_number}: expected 8 trajectory fields")
            frame_id = parts[0]
            values = [finite_float(v) for v in parts[1:]]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"{path}:{line_number}: non-finite trajectory pose")
            x, y, z, qx, qy, qz, qw = values
            poses[frame_id] = {
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw_from_quat(qx, qy, qz, qw),
            }
    return poses


def trajectory_stats(run_dir):
    est = load_trajectory(run_dir / "trajectory_est.txt")
    gt = load_trajectory(run_dir / "trajectory_gt.txt")
    common = sorted(set(est.keys()) & set(gt.keys()))
    trans = []
    xy = []
    z_abs = []
    yaw_deg = []
    for frame_id in common:
        e = est[frame_id]
        g = gt[frame_id]
        dx = e["x"] - g["x"]
        dy = e["y"] - g["y"]
        dz = e["z"] - g["z"]
        trans.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        xy.append(math.sqrt(dx * dx + dy * dy))
        z_abs.append(abs(dz))
        yaw_deg.append(abs(angle_diff(e["yaw"], g["yaw"])) * 180.0 / math.pi)
    return {
        "trajectory_pair_count": len(common),
        "trajectory_translation_mean_m": mean(trans),
        "trajectory_translation_median_m": percentile(trans, 0.5),
        "trajectory_translation_p95_m": percentile(trans, 0.95),
        "trajectory_translation_max_m": max(trans) if trans else float("nan"),
        "trajectory_xy_p95_m": percentile(xy, 0.95),
        "trajectory_z_p95_m": percentile(z_abs, 0.95),
        "trajectory_yaw_p95_deg": percentile(yaw_deg, 0.95),
    }


def parse_run_spec(spec):
    if "=" not in spec:
        raise RuntimeError(f"--run must be NAME=DIR, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise RuntimeError(f"--run has empty name: {spec}")
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.exists():
        raise RuntimeError(f"run directory does not exist: {run_dir}")
    return name, run_dir


def safe_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "run"


def default_analyzer_path():
    return Path(__file__).resolve().with_name("n3mapping_loop_debug_analyze.py")


def find_existing_loop_diagnosis(run_dir):
    candidates = [
        run_dir / "loop_gt_analysis_v2" / "loop_diagnosis.json",
        run_dir / "loop_gt_analysis" / "loop_diagnosis.json",
        run_dir / "loop_diagnosis.json",
    ]
    candidates.extend(sorted(run_dir.glob("loop_gt_analysis*/loop_diagnosis.json")))
    existing = []
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        existing.append(candidate)
    if not existing:
        return None
    for candidate in existing:
        try:
            diagnosis = load_json(candidate)
        except Exception:
            continue
        if "optimization_summary_count" in diagnosis:
            return candidate
    return existing[0]


def loop_diagnosis_has_current_schema(diagnosis):
    required = {
        "optimization_summary_count",
        "accepted_true_loop_good",
        "accepted_true_loop_bad_z",
        "accepted_true_loop_bad_z_measurement",
        "accepted_true_loop_bad_z_after",
        "accepted_true_loop_corrected_z",
        "accepted_true_loop_bad_roll_pitch",
        "verification_reject_true_loop",
        "failure_class_counts",
        "edge_mode_counts",
        "accepted_full6dof",
        "accepted_planar_xy_yaw",
    }
    return required.issubset(set(diagnosis.keys()))


def ensure_loop_diagnosis(args, run_name, run_dir, matrix_output):
    existing = find_existing_loop_diagnosis(run_dir)
    if existing:
        try:
            diagnosis = load_json(existing)
        except Exception:
            diagnosis = {}
        if loop_diagnosis_has_current_schema(diagnosis) or args.skip_loop_analyzer:
            return existing
    if args.skip_loop_analyzer:
        return None
    required = [run_dir / "loop_debug.jsonl", run_dir / "keyframes_gt.csv"]
    if not all(path.exists() for path in required):
        return None
    analyzer = Path(args.analyzer).expanduser() if args.analyzer else default_analyzer_path()
    if not analyzer.exists():
        return None
    analysis_dir = matrix_output / "loop_analysis" / safe_name(run_name)
    command = [
        "python3",
        str(analyzer),
        "--loop_debug",
        str(run_dir / "loop_debug.jsonl"),
        "--keyframes_gt",
        str(run_dir / "keyframes_gt.csv"),
        "--output",
        str(analysis_dir),
        "--loop_translation_threshold",
        str(args.loop_translation_threshold),
        "--loop_yaw_threshold_deg",
        str(args.loop_yaw_threshold_deg),
        "--min_id_gap",
        str(args.min_id_gap),
        "--z_drift_threshold",
        str(args.z_drift_threshold),
    ]
    accepted_loops = run_dir / "accepted_loops.csv"
    if accepted_loops.exists():
        command.extend(["--accepted_loops", str(accepted_loops)])
    subprocess.run(command, check=True)
    diagnosis = analysis_dir / "loop_diagnosis.json"
    return diagnosis if diagnosis.exists() else None


def summarize_run(args, run_name, run_dir, matrix_output):
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"{run_dir} is missing metrics.json")
    metrics = load_json(metrics_path)
    row = {field: "" for field in SUMMARY_FIELDS}
    row["run"] = run_name
    row["run_dir"] = str(run_dir)
    row["mode"] = metrics.get("mode", "")
    row["sequence"] = metrics.get("sequence", "")

    for key in (
        "frames_processed",
        "query_count",
        "accepted_keyframes",
        "accepted_loop_count",
        "dense_trajectory_count",
        "lock_success_rate",
        "pose_success_rate",
    ):
        if key in metrics:
            row[key] = metrics[key]

    row.update(trajectory_stats(run_dir))

    diagnosis_path = ensure_loop_diagnosis(args, run_name, run_dir, matrix_output)
    if diagnosis_path:
        diagnosis = load_json(diagnosis_path)
        mapping = {
            "candidate_count": "loop_candidate_count",
            "gt_loop_pair_count": "loop_gt_pair_count",
            "accepted_candidate_count": "loop_accepted_candidate_count",
            "accepted_true_loop": "loop_accepted_true_loop",
            "accepted_true_loop_good": "loop_accepted_true_loop_good",
            "accepted_true_loop_bad_z": "loop_accepted_true_loop_bad_z",
            "accepted_true_loop_bad_z_measurement": "loop_accepted_true_loop_bad_z_measurement",
            "accepted_true_loop_bad_z_after": "loop_accepted_true_loop_bad_z_after",
            "accepted_true_loop_corrected_z": "loop_accepted_true_loop_corrected_z",
            "accepted_true_loop_bad_roll_pitch": "loop_accepted_true_loop_bad_roll_pitch",
            "accepted_false_loop": "loop_accepted_false_loop",
            "accepted_full6dof": "loop_accepted_full6dof",
            "accepted_planar_xy_yaw": "loop_accepted_planar_xy_yaw",
            "icp_reject_true_loop": "loop_icp_reject_true_loop",
            "verification_reject_true_loop": "loop_verification_reject_true_loop",
            "true_loop_not_selected": "loop_true_loop_not_selected",
            "retrieval_false_positive": "loop_retrieval_false_positive",
            "retrieval_miss_estimate": "loop_retrieval_miss_estimate",
            "z_drift_suspect_count": "loop_z_drift_suspect_count",
            "optimization_summary_count": "optimization_summary_count",
            "optimization_high_residual_z_after_count": "optimization_high_residual_z_after_count",
            "optimization_max_residual_z_after": "optimization_max_residual_z_after_m",
        }
        for source_key, dest_key in mapping.items():
            if source_key in diagnosis:
                row[dest_key] = diagnosis[source_key]

    accepted = finite_float(row.get("loop_accepted_candidate_count"))
    accepted_true = finite_float(row.get("loop_accepted_true_loop"))
    gt_pairs = finite_float(row.get("loop_gt_pair_count"))
    loop_precision = accepted_true / accepted if accepted > 0 else float("nan")
    loop_coverage = accepted_true / gt_pairs if gt_pairs > 0 else float("nan")
    row["loop_precision"] = loop_precision
    row["loop_gt_pair_coverage"] = loop_coverage
    row["meets_loop_precision_80"] = bool(math.isfinite(loop_precision) and loop_precision >= 0.8)
    pose_success = finite_float(row.get("pose_success_rate"))
    row["meets_reloc_pose_success_80"] = bool(math.isfinite(pose_success) and pose_success >= 0.8)
    return row


def write_outputs(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "matrix_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, "")) for field in SUMMARY_FIELDS})

    json_rows = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, float):
                converted[key] = json_number(value)
            else:
                converted[key] = value
        json_rows.append(converted)
    with open(output / "matrix_summary.json", "w") as f:
        json.dump({"runs": json_rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"matrix runs={len(rows)} output={output}")


def main():
    try:
        args = parse_args()
        output = Path(args.output).expanduser().resolve()
        rows = [
            summarize_run(args, run_name, run_dir, output)
            for run_name, run_dir in (parse_run_spec(spec) for spec in args.run)
        ]
        write_outputs(rows, output)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"n3mapping_eval_matrix: analyzer failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - surfaced in CLI tests
        print(f"n3mapping_eval_matrix: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
