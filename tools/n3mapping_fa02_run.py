#!/usr/bin/env python3
"""Run the complete frozen FA-02 dataset loop-closure acceptance matrix."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

from n3mapping_fa02_freeze import freeze, load_json, sha256_file
from n3mapping_fa02_gate import EXIT_CODES, evaluate


RUN_SCHEMA = "n3mapping_fa02_episode_run_v1"
SUMMARY_SCHEMA = "n3mapping_fa02_run_summary_v1"
REQUIRED_ARTIFACTS = (
    "metrics.json",
    "trajectory_gt.txt",
    "trajectory_est.txt",
    "trajectory_optimized.txt",
    "keyframes_gt.csv",
    "accepted_loops.csv",
    "loop_debug.jsonl",
    "n3map.pbstream",
)


def fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def timestamp(timezone: ZoneInfo) -> str:
    return datetime.now(timezone).isoformat()


def episode_command(
    episode: dict[str, Any], frozen: dict[str, Any], contract: dict[str, Any],
    frames: Path, output: Path, source_repo: Path,
    kitti_evaluator: Path, m2dgr_evaluator: Path,
    *, section: str = "fa02",
) -> tuple[Path, list[str]]:
    common = [
        "--sequence", str(episode["sequence"]),
        "--mode", "mapping_loop",
        "--frame_manifest", str(frames),
        "--episode_id", str(episode["id"]),
        "--input_voxel_size", str(episode["input_voxel_size_m"]),
        "--output", str(output),
    ]
    root = Path(contract[section]["dataset_roots"][episode["dataset"]]).resolve()
    if episode["dataset"] == "kitti360":
        return kitti_evaluator, [
            str(kitti_evaluator), "--kitti_root", str(root),
            "--calib_mode", str(episode["calibration_mode"]), *common,
        ]
    gt_path = frozen["inputs"]["gt"]["path"]
    calibration = source_repo / str(episode["calibration_file"])
    command = [
        str(m2dgr_evaluator), "--m2dgr_root", str(root),
        "--gt", str(gt_path),
        "--gt_sensor_frame", str(episode["gt_sensor_frame"]),
        "--calibration_file", str(calibration),
        "--max_time_diff", str(episode["max_time_diff_s"]),
        *common,
    ]
    if episode.get("normalize_gt_origin"):
        command.append("--normalize_gt_origin")
    return m2dgr_evaluator, command


def write_checksums(root: Path) -> Path:
    checksum_path = root / "checksums.sha256"
    excluded = {checksum_path, root / "COMPLETE"}
    lines: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path in excluded:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def run(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve(strict=True)
    source_repo = args.source_repo.resolve(strict=True)
    output = args.output.resolve()
    kitti_evaluator = args.kitti_evaluator.resolve(strict=True)
    m2dgr_evaluator = args.m2dgr_evaluator.resolve(strict=True)
    proto = args.proto.resolve(strict=True)
    contract = load_json(contract_path)
    timezone = ZoneInfo(str(contract.get("recorded_timezone", "America/Los_Angeles")))
    print(f"FA-02 freeze start output={output}", flush=True)
    input_manifest = freeze(contract_path, output, source_repo)
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
    kitti_evaluator = copied_kitti.resolve(strict=True)
    m2dgr_evaluator = copied_m2dgr.resolve(strict=True)
    frozen_by_id = {episode["id"]: episode for episode in input_manifest["episodes"]}
    frames = output / "episode_frames.csv"
    runs_root = output / "runs"
    runs_root.mkdir()
    episode_summaries: list[dict[str, Any]] = []
    for episode in contract["fa02"]["episodes"]:
        episode_id = str(episode["id"])
        run_dir = runs_root / episode_id
        evaluator, command = episode_command(
            episode, frozen_by_id[episode_id], contract, frames, run_dir,
            source_repo, kitti_evaluator, m2dgr_evaluator,
        )
        print(f"FA-02 episode start id={episode_id}", flush=True)
        started_at = timestamp(timezone)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        run_dir.mkdir()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
        status = "COMPLETE" if process.returncode == 0 and not missing else "FAILED"
        run_manifest = {
            "schema": RUN_SCHEMA,
            "status": status,
            "episode_id": episode_id,
            "dataset": episode["dataset"],
            "sequence": episode["sequence"],
            "source_commit": input_manifest["source_commit"],
            "input_manifest_sha256": sha256_file(output / "input_manifest.json"),
            "episode_frames_sha256": sha256_file(frames),
            "evaluator": fingerprint(evaluator),
            "command": command,
            "started_at": started_at,
            "finished_at": timestamp(timezone),
            "process_return_code": process.returncode,
            "missing_artifacts": missing,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        episode_summaries.append(run_manifest)
        print(
            f"FA-02 episode done id={episode_id} rc={process.returncode} missing={len(missing)}",
            flush=True,
        )
        if status != "COMPLETE":
            failure = {
                "schema": SUMMARY_SCHEMA,
                "status": "RUN_FAILED",
                "source_commit": input_manifest["source_commit"],
                "episodes": episode_summaries,
            }
            (output / "run_summary.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            write_checksums(output)
            return 2

    print("FA-02 gate start", flush=True)
    verdict = evaluate(contract_path, output, proto)
    verdict_path = output / "fa02_verdict.json"
    verdict_path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "COMPLETE",
        "verdict": verdict["status"],
        "source_commit": input_manifest["source_commit"],
        "episodes": episode_summaries,
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
    print(
        json.dumps(
            {"status": verdict["status"], "aggregate": verdict["aggregate"], "output": str(output)},
            sort_keys=True,
        ),
        flush=True,
    )
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
        print(f"n3mapping_fa02_run: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
