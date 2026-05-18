#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def find_eval_executable():
    executable_name = "n3mapping_synthetic_relocalization_eval"
    for base in (Path(sys.argv[0]), Path(sys.argv[0]).resolve()):
        sibling = base.with_name(executable_name)
        if sibling.exists():
            return str(sibling)
    return "n3mapping_synthetic_relocalization_eval"


def main():
    parser = argparse.ArgumentParser(
        description="Run a dropout/noise/yaw matrix for n3mapping synthetic relocalization eval."
    )
    parser.add_argument("--map", required=True, help="Path to n3map.pbstream")
    parser.add_argument("--output", required=True, help="Output directory for matrix results")
    parser.add_argument("--max_queries", type=int, default=100)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--dropouts", default="0.0,0.3,0.5,0.7")
    parser.add_argument("--noise_sigmas", default="0.0,0.02,0.05")
    parser.add_argument("--fake_yaws", default="0,45,90,180")
    parser.add_argument("--query_source", default="same_keyframe", choices=["same_keyframe", "local_submap", "global_map"])
    parser.add_argument("--range_max", type=float, default=30.0)
    parser.add_argument("--query_submap_radius", type=int, default=2)
    parser.add_argument("--query_voxel_size", type=float, default=0.12)
    parser.add_argument("--pose_translation_threshold", type=float, default=1.0)
    parser.add_argument("--pose_yaw_threshold_deg", type=float, default=10.0)
    parser.add_argument("--pose_roll_pitch_threshold_deg", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    eval_exe = find_eval_executable()

    rows = []
    failures = 0
    for dropout in parse_float_list(args.dropouts):
        for noise_sigma in parse_float_list(args.noise_sigmas):
            for yaw in parse_float_list(args.fake_yaws):
                name = f"dropout_{dropout:g}_noise_{noise_sigma:g}_yaw_{yaw:g}".replace(".", "p")
                run_dir = output_root / name
                cmd = [
                    eval_exe,
                    "--map", args.map,
                    "--output", str(run_dir),
                    "--max_queries", str(args.max_queries),
                    "--dropout", str(dropout),
                    "--noise_sigma", str(noise_sigma),
                    "--fake_odom_yaw_deg", str(yaw),
                    "--query_source", args.query_source,
                    "--range_max", str(args.range_max),
                    "--query_submap_radius", str(args.query_submap_radius),
                    "--query_voxel_size", str(args.query_voxel_size),
                    "--pose_translation_threshold", str(args.pose_translation_threshold),
                    "--pose_yaw_threshold_deg", str(args.pose_yaw_threshold_deg),
                    "--pose_roll_pitch_threshold_deg", str(args.pose_roll_pitch_threshold_deg),
                ]
                if args.stride > 0:
                    cmd.extend(["--stride", str(args.stride)])
                if args.strict:
                    cmd.append("--strict")

                print("running", " ".join(cmd))
                completed = subprocess.run(cmd, check=False)
                if completed.returncode != 0:
                    failures += 1

                summary_path = run_dir / "summary.json"
                row = {
                    "dropout": dropout,
                    "noise_sigma": noise_sigma,
                    "fake_yaw_deg": yaw,
                    "query_source": args.query_source,
                    "returncode": completed.returncode,
                    "output_dir": str(run_dir),
                }
                if summary_path.exists():
                    with summary_path.open() as f:
                        summary = json.load(f)
                    for key in [
                        "tested",
                        "lock_success",
                        "pose_success",
                        "self_matches",
                        "lock_success_rate",
                        "pose_success_rate",
                        "median_translation_error_m",
                        "p95_translation_error_m",
                        "median_yaw_error_deg",
                        "p95_yaw_error_deg",
                        "median_roll_pitch_error_deg",
                        "p95_roll_pitch_error_deg",
                    ]:
                        row[key] = summary.get(key)
                rows.append(row)

    matrix_csv = output_root / "matrix_summary.csv"
    fieldnames = [
        "dropout",
        "noise_sigma",
        "fake_yaw_deg",
        "returncode",
        "query_source",
        "tested",
        "lock_success",
        "pose_success",
        "self_matches",
        "lock_success_rate",
        "pose_success_rate",
        "median_translation_error_m",
        "p95_translation_error_m",
        "median_yaw_error_deg",
        "p95_yaw_error_deg",
        "median_roll_pitch_error_deg",
        "p95_roll_pitch_error_deg",
        "output_dir",
    ]
    with matrix_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {matrix_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
