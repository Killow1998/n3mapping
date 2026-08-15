#!/usr/bin/env python3
"""Run frozen FA-02B correlated-odometry loop-on/off pairs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

from n3mapping_fa02_freeze import freeze, load_json, sha256_file
from n3mapping_fa02_run import episode_command, fingerprint, write_checksums
from n3mapping_fa02b_gate import (
    CONTRACT_SCHEMA,
    EXIT_CODES,
    RUN_SCHEMA,
    drift_config,
    evaluate,
)


SUMMARY_SCHEMA = "n3mapping_fa02b_run_summary_v1"
REQUIRED_ARTIFACTS = (
    "metrics.json",
    "trajectory_gt.txt",
    "trajectory_odom.txt",
    "trajectory_est.txt",
    "trajectory_optimized.txt",
    "keyframes_gt.csv",
    "accepted_loops.csv",
    "loop_debug.jsonl",
    "n3map.pbstream",
)


def timestamp(timezone: ZoneInfo) -> str:
    return datetime.now(timezone).isoformat()


def drift_arguments(config: dict[str, Any]) -> list[str]:
    return [
        "--enable_correlated_odom_drift",
        "--odom_drift_seed", str(config["seed"]),
        "--odom_translation_scale_error",
        str(config["translation_scale_error_fraction"]),
        "--odom_yaw_bias_deg_per_meter", str(config["yaw_bias_deg_per_meter"]),
        "--odom_translation_rw_std_m_per_sqrt_meter",
        str(config["translation_rw_std_m_per_sqrt_meter"]),
        "--odom_rotation_rw_std_deg_per_sqrt_meter",
        str(config["rotation_rw_std_deg_per_sqrt_meter"]),
    ]


def run(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve(strict=True)
    source_repo = args.source_repo.resolve(strict=True)
    output = args.output.resolve()
    kitti_evaluator = args.kitti_evaluator.resolve(strict=True)
    m2dgr_evaluator = args.m2dgr_evaluator.resolve(strict=True)
    proto = args.proto.resolve(strict=True)
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("FA-02B contract schema mismatch")
    section = contract.get("fa02b")
    if not isinstance(section, dict):
        raise ValueError("contract has no fa02b section")
    timezone = ZoneInfo(str(contract.get("recorded_timezone", "America/Los_Angeles")))
    print(f"FA-02B freeze start output={output}", flush=True)
    input_manifest = freeze(
        contract_path,
        output,
        source_repo,
        contract_schema=CONTRACT_SCHEMA,
        section="fa02b",
    )
    if input_manifest["source_commit"] != args.expected_commit:
        raise ValueError(
            f"source commit {input_manifest['source_commit']} != expected {args.expected_commit}"
        )

    executable_root = output / "executables"
    executable_root.mkdir()
    copied_kitti = executable_root / kitti_evaluator.name
    copied_m2dgr = executable_root / m2dgr_evaluator.name
    shutil.copy2(kitti_evaluator, copied_kitti)
    shutil.copy2(m2dgr_evaluator, copied_m2dgr)
    copied_kitti.chmod(copied_kitti.stat().st_mode | 0o111)
    copied_m2dgr.chmod(copied_m2dgr.stat().st_mode | 0o111)
    copied_kitti = copied_kitti.resolve(strict=True)
    copied_m2dgr = copied_m2dgr.resolve(strict=True)

    frozen_by_id = {str(item["id"]): item for item in input_manifest["episodes"]}
    episodes = {str(item["id"]): item for item in section["episodes"]}
    profiles = {str(item["id"]): item for item in section["drift_profiles"]}
    frame_manifest = output / "episode_frames.csv"
    input_manifest_hash = sha256_file(output / "input_manifest.json")
    frame_manifest_hash = sha256_file(frame_manifest)
    runs_root = output / "runs"
    runs_root.mkdir()
    run_summaries: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for case in section["cases"]:
        case_id = str(case["id"])
        if case_id in seen_cases:
            raise ValueError(f"duplicate FA-02B case id: {case_id}")
        seen_cases.add(case_id)
        episode = episodes[str(case["episode_id"])]
        profile = profiles[str(case["profile_id"])]
        config = drift_config(profile, int(case["seed"]))
        evaluator_fingerprint: dict[str, Any] | None = None
        for condition in ("loop_off", "loop_on"):
            run_dir = runs_root / case_id / condition
            evaluator, command = episode_command(
                episode,
                frozen_by_id[str(episode["id"])],
                contract,
                frame_manifest,
                run_dir,
                source_repo,
                copied_kitti,
                copied_m2dgr,
                section="fa02b",
            )
            command.extend(drift_arguments(config))
            if condition == "loop_off":
                command.append("--disable_loop_closure")
            current_fingerprint = fingerprint(evaluator)
            if evaluator_fingerprint is None:
                evaluator_fingerprint = current_fingerprint
            elif current_fingerprint != evaluator_fingerprint:
                raise ValueError(f"{case_id}: evaluator changed between paired runs")
            print(f"FA-02B run start case={case_id} condition={condition}", flush=True)
            run_dir.mkdir(parents=True)
            started_at = timestamp(timezone)
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
            status = "COMPLETE" if process.returncode == 0 and not missing else "FAILED"
            manifest = {
                "schema": RUN_SCHEMA,
                "status": status,
                "case_id": case_id,
                "condition": condition,
                "episode_id": str(episode["id"]),
                "dataset": str(episode["dataset"]),
                "sequence": str(episode["sequence"]),
                "source_commit": input_manifest["source_commit"],
                "input_manifest_sha256": input_manifest_hash,
                "episode_frames_sha256": frame_manifest_hash,
                "evaluator": current_fingerprint,
                "drift_config": config,
                "command": command,
                "started_at": started_at,
                "finished_at": timestamp(timezone),
                "process_return_code": process.returncode,
                "missing_artifacts": missing,
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            run_summaries.append(manifest)
            print(
                f"FA-02B run done case={case_id} condition={condition} "
                f"rc={process.returncode} missing={len(missing)}",
                flush=True,
            )
            if status != "COMPLETE":
                summary = {
                    "schema": SUMMARY_SCHEMA,
                    "status": "RUN_FAILED",
                    "source_commit": input_manifest["source_commit"],
                    "runs": run_summaries,
                }
                (output / "run_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                write_checksums(output)
                return 2

    print("FA-02B gate start", flush=True)
    verdict = evaluate(contract_path, output, proto)
    (output / "fa02b_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "COMPLETE",
        "verdict": verdict["status"],
        "source_commit": input_manifest["source_commit"],
        "runs": run_summaries,
        "aggregate": verdict["aggregate"],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = write_checksums(output)
    (output / "COMPLETE").write_text(
        f"status={verdict['status']}\nchecksums_sha256={sha256_file(checksums)}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": verdict["status"],
        "aggregate": verdict["aggregate"],
        "output": str(output),
    }, sort_keys=True), flush=True)
    return EXIT_CODES[verdict["status"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--kitti-evaluator", type=Path, required=True)
    parser.add_argument("--m2dgr-evaluator", type=Path, required=True)
    parser.add_argument("--proto", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"n3mapping_fa02b_run: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
