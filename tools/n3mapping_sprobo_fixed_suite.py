#!/usr/bin/env python3
"""Acquire independent fixed-pose queries; preserve closed bags and exact pairs.

Run on the authorized host with the ROS workspace sourced. Scene/position choices
are frozen in the plan before any localization evaluation. Existing output is
never replaced, and a failed acquisition stops the batch without deleting it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from n3mapping_sprobo_score import score


def main():
    tools = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=tools / "fixtures/sprobo_fixed14.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--sim-executable", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=205)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_bytes(plan_bytes)
    results = []
    for position in plan["positions"]:
        directory = args.output / position["id"]
        command = [sys.executable, str(tools / "n3mapping_sprobo_smoke.py"),
                   "--workspace", str(args.workspace), "--sim-root", str(args.sim_root),
                   "--sim-executable", str(args.sim_executable), "--adapter", str(args.adapter),
                   "--output", str(directory), "--domain-id", str(args.domain_id),
                   "--input-only", "--route", plan["route"], "--variant", plan["variant"],
                   "--duration", str(plan["duration_s"]), "--yaw", str(position["yaw_deg"]),
                   "--spawn", *map(str, position["spawn"])]
        print("Recording " + position["id"], flush=True)
        with (args.output / (position["id"] + ".log")).open("x") as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        # The recorder has closed before any database reader is started.
        with (args.output / (position["id"] + ".extract.log")).open("x") as log:
            subprocess.run([sys.executable, str(tools / "extract_ros1_relocalization_manifest.py"),
                            "--bag", str(directory / "bag"), "--output", str(directory / "manifest"),
                            "--episode", position["id"]], stdout=log, stderr=log, check=True)
        acquisition = json.loads((directory / "summary.json").read_text())
        frontend = score(directory, directory, "/Odometry")
        (directory / "frontend_quality.json").write_text(json.dumps(frontend, indent=2) + "\n")
        result = dict(id=position["id"], acquisition_passed=acquisition["passed"],
                      simulation_seconds=acquisition["simulation_seconds"],
                      counts=acquisition["counts"],
                      frontend_position_error_m=frontend["position_error_m"],
                      frontend_rotation_error_deg=frontend["rotation_error_deg"])
        results.append(result)
        (args.output / "acquisitions.json").write_text(json.dumps(dict(
            plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), results=results), indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
