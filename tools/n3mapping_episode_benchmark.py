#!/usr/bin/env python3
"""Run a frozen relocalization manifest as independent map/query episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

from n3mapping_dataset_readiness import sha256_file
from n3mapping_episode_freeze import KITTI360_OFFICIAL_CALIBRATION_FILES


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finalize_hashed_output(output: Path, required_relative_paths: set[str]) -> None:
    checksum_path = output / "checksums.sha256"
    complete_path = output / "COMPLETE"
    if checksum_path.exists() or complete_path.exists():
        raise ValueError(f"benchmark output is already finalized: {output}")
    payloads = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"benchmark output contains a symlink: {path}")
        if path.is_file():
            payloads.append(path)
    required = {output / relative for relative in required_relative_paths}
    if not required.issubset(payloads):
        raise ValueError(
            "artifact output is missing required payloads: "
            + ", ".join(sorted(required_relative_paths))
        )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in payloads
    ]
    with checksum_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    complete_tmp = output / ".COMPLETE.tmp"
    with complete_tmp.open("x", encoding="utf-8") as stream:
        stream.write(
            "status=complete\n"
            f"checksums_sha256={sha256_file(checksum_path)}\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(complete_tmp, complete_path)


def _verify_hashed_output(output: Path, required_relative_paths: set[str]) -> None:
    complete_path = output / "COMPLETE"
    checksum_path = output / "checksums.sha256"
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError(f"benchmark output is incomplete: {output}")
    complete_fields = {}
    for line in complete_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in complete_fields:
            raise ValueError(f"malformed benchmark COMPLETE file: {complete_path}")
        complete_fields[key] = value
    if complete_fields.get("status") != "complete":
        raise ValueError(f"benchmark COMPLETE status is not complete: {complete_path}")
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ValueError(f"benchmark checksums are missing: {checksum_path}")
    if sha256_file(checksum_path) != complete_fields.get("checksums_sha256"):
        raise ValueError(f"benchmark checksum manifest hash mismatch: {checksum_path}")
    seen = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, separator, relative_text = line.partition("  ")
        relative = PurePosixPath(relative_text)
        if (
            not separator
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise ValueError(f"malformed benchmark checksum line: {line!r}")
        seen.add(relative_text)
        path = output.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"benchmark payload is missing or not regular: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"benchmark payload hash mismatch: {path}")
    if not required_relative_paths.issubset(seen):
        raise ValueError(
            "artifact checksums omit required payloads: "
            + ", ".join(sorted(required_relative_paths))
        )


def _finalize_benchmark_output(output: Path) -> None:
    _finalize_hashed_output(output, {"summary.json", "episodes.csv"})


def _verify_benchmark_output(output: Path) -> None:
    _verify_hashed_output(output, {"summary.json", "episodes.csv"})


def _verify_manifest(
    manifest_dir: Path,
    *,
    allow_legacy_unhashed_calibration: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = manifest_dir / "dataset_manifest.json"
    frames_path = manifest_dir / "episode_frames.csv"
    manifest = _read_json(manifest_path)
    if sha256_file(frames_path) != manifest["episode_frames_sha256"]:
        raise ValueError("episode_frames.csv hash does not match dataset_manifest.json")
    with frames_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if "expected_behavior" in manifest:
        expected_behavior = manifest["expected_behavior"]
        if expected_behavior not in {"lock", "abstain"}:
            raise ValueError(f"unsupported expected behavior: {expected_behavior}")
        overlap_radius_m = float(manifest["overlap_radius_m"])
        if not math.isfinite(overlap_radius_m) or overlap_radius_m <= 0.0:
            raise ValueError("manifest overlap_radius_m must be finite and positive")

        def positions(role: str) -> list[tuple[float, float, float]]:
            result = []
            for row in rows:
                if row.get("role") != role:
                    continue
                try:
                    position = tuple(float(row[axis]) for axis in ("x", "y", "z"))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"manifest row has invalid {role} position") from error
                if not all(math.isfinite(value) for value in position):
                    raise ValueError(f"manifest row has non-finite {role} position")
                result.append(position)
            return result

        map_positions = positions("map")
        query_positions = positions("query")
        if not map_positions or not query_positions:
            raise ValueError("manifest must contain map and query frames")
        radius_sq = overlap_radius_m * overlap_radius_m
        frozen_covered_count = sum(
            any(
                sum((query[axis] - mapped[axis]) ** 2 for axis in range(3))
                <= radius_sq
                for mapped in map_positions
            )
            for query in query_positions
        )
        declared_covered_count = manifest.get("frozen_covered_query_frame_count")
        if (
            declared_covered_count is not None
            and int(declared_covered_count) != frozen_covered_count
        ):
            raise ValueError("manifest frozen coverage count does not match episode frames")
        if expected_behavior == "lock" and frozen_covered_count != len(query_positions):
            raise ValueError("lock manifest contains query frames outside frozen map coverage")
        if expected_behavior == "abstain" and frozen_covered_count != 0:
            raise ValueError("abstain manifest contains query frames covered by the frozen map")
    root = Path(manifest["root"])
    for role in ("map", "query"):
        gt_path = Path(manifest[f"{role}_gt_path"])
        if sha256_file(gt_path) != manifest[f"{role}_gt_sha256"]:
            raise ValueError(f"{role} ground-truth hash mismatch: {gt_path}")
    for row in rows:
        cloud = root / row["relative_cloud_path"]
        if sha256_file(cloud) != row["cloud_sha256"]:
            raise ValueError(f"point-cloud hash mismatch: {cloud}")
    if not manifest.get("map_query_disjoint", False):
        raise ValueError("manifest does not prove map/query disjointness")
    if manifest.get("dataset") == "kitti360":
        calibration = manifest.get("kitti360_calibration")
        if calibration is None:
            if not allow_legacy_unhashed_calibration:
                raise ValueError(
                    "KITTI-360 manifest has no frozen official calibration; "
                    "re-freeze it or explicitly allow legacy unhashed calibration"
                )
        else:
            if calibration.get("mode") != "official":
                raise ValueError("KITTI-360 calibration mode must be official")
            files = calibration.get("files")
            if not isinstance(files, list):
                raise ValueError("KITTI-360 calibration files must be a list")
            by_relative_path = {
                entry.get("relative_path"): entry
                for entry in files
                if isinstance(entry, dict)
            }
            expected = set(KITTI360_OFFICIAL_CALIBRATION_FILES)
            if set(by_relative_path) != expected:
                raise ValueError(
                    "KITTI-360 calibration contract must contain exactly: "
                    + ", ".join(KITTI360_OFFICIAL_CALIBRATION_FILES)
                )
            for relative in KITTI360_OFFICIAL_CALIBRATION_FILES:
                calibration_path = root / relative
                if sha256_file(calibration_path) != by_relative_path[relative].get(
                    "sha256"
                ):
                    raise ValueError(
                        f"KITTI-360 calibration hash mismatch: {calibration_path}"
                    )
    return manifest, rows


def _run(command: list[str], output_dir: Path) -> tuple[float, str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    start = time.monotonic()
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    (output_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"evaluator failed with code {process.returncode}: {shlex.join(command)}\n"
            f"{process.stderr[-2000:]}"
        )
    return elapsed, process.stdout, process.stderr


def classify_episode(metrics: dict[str, Any]) -> str:
    if int(metrics.get("false_lock_count", 0)) > 0:
        return "false_lock"
    if int(metrics.get("correct_lock_count", 0)) > 0:
        return "correct_lock"
    return "no_lock"


def classify_contract(outcome: str, expected_behavior: str) -> str:
    if expected_behavior == "lock":
        return "pass" if outcome == "correct_lock" else "fail"
    if expected_behavior == "abstain":
        return "pass" if outcome == "no_lock" else "fail"
    raise ValueError(f"unsupported expected behavior: {expected_behavior}")


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    ratio = index - low
    return ordered[low] * (1.0 - ratio) + ordered[high] * ratio


def run_benchmark(
    *,
    evaluator: Path,
    atlas_compiler: Path,
    manifest_dir: Path,
    output: Path,
    fake_x_m: float,
    fake_y_m: float,
    fake_yaw_deg: float,
    input_voxel_size_m: float,
    m2dgr_max_time_diff_s: float | None,
    allow_legacy_unhashed_calibration: bool = False,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output}")
    manifest, rows = _verify_manifest(
        manifest_dir,
        allow_legacy_unhashed_calibration=allow_legacy_unhashed_calibration,
    )
    output.mkdir(parents=True, exist_ok=True)
    dataset = manifest["dataset"]
    root = manifest["root"]
    frame_manifest = manifest_dir / "episode_frames.csv"
    effective_m2dgr_max_time_diff_s = m2dgr_max_time_diff_s
    if dataset == "m2dgr":
        frozen_tolerance = manifest.get("m2dgr_max_time_diff_s")
        if frozen_tolerance is None and effective_m2dgr_max_time_diff_s is None:
            raise ValueError(
                "legacy M2DGR manifest has no m2dgr_max_time_diff_s; "
                "provide --m2dgr-max-time-diff-s explicitly"
            )
        if frozen_tolerance is not None:
            frozen_tolerance = float(frozen_tolerance)
            if (
                effective_m2dgr_max_time_diff_s is not None
                and not math.isclose(
                    effective_m2dgr_max_time_diff_s,
                    frozen_tolerance,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "--m2dgr-max-time-diff-s differs from the frozen manifest"
                )
            effective_m2dgr_max_time_diff_s = frozen_tolerance

    map_output = output / "map"
    if dataset == "kitti360":
        map_command = [
            str(evaluator), "--kitti_root", root,
            "--sequence", manifest["map_sequence"],
            "--calib_mode", "official",
            "--mode", "mapping_loop",
            "--frame_manifest", str(frame_manifest),
            "--episode_id", "map",
            "--input_voxel_size", str(input_voxel_size_m),
            "--output", str(map_output),
        ]
    elif dataset == "m2dgr":
        map_command = [
            str(evaluator), "--m2dgr_root", root,
            "--sequence", manifest["map_sequence"],
            "--gt", manifest["map_gt_path"],
            "--max_time_diff", str(effective_m2dgr_max_time_diff_s),
            "--normalize_gt_origin",
            "--mode", "mapping_loop",
            "--frame_manifest", str(frame_manifest),
            "--episode_id", "map",
            "--input_voxel_size", str(input_voxel_size_m),
            "--output", str(map_output),
        ]
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    map_seconds, _, _ = _run(map_command, map_output)
    map_path = map_output / "n3map.pbstream"
    if not map_path.is_file():
        raise RuntimeError("mapping evaluator did not write n3map.pbstream")
    atlas_path = map_path.with_name(map_path.name + ".localization_atlas.pb")
    atlas_command = [
        str(atlas_compiler), "--map", str(map_path), "--output", str(atlas_path)
    ]
    (map_output / "atlas_command.txt").write_text(
        shlex.join(atlas_command) + "\n", encoding="utf-8"
    )
    atlas_process = subprocess.run(atlas_command, check=False, capture_output=True, text=True)
    (map_output / "atlas_stdout.log").write_text(atlas_process.stdout, encoding="utf-8")
    (map_output / "atlas_stderr.log").write_text(atlas_process.stderr, encoding="utf-8")
    if atlas_process.returncode != 0 or not atlas_path.is_file():
        raise RuntimeError(
            f"Atlas compilation failed with code {atlas_process.returncode}: "
            f"{atlas_process.stderr[-2000:]}"
        )

    episode_ids = sorted({row["episode_id"] for row in rows if row["role"] == "query"})
    expected_behavior = manifest.get("expected_behavior", "lock")
    if expected_behavior not in {"lock", "abstain"}:
        raise ValueError(f"unsupported expected behavior: {expected_behavior}")
    episode_rows = []
    translation_errors = []
    yaw_errors = []
    for episode_id in episode_ids:
        episode_output = output / "episodes" / episode_id
        common = [
            "--sequence", manifest["query_sequence"],
            "--mode", "relocalization",
            "--map", str(map_path),
            "--atlas", str(atlas_path),
            "--frame_manifest", str(frame_manifest),
            "--episode_id", episode_id,
            "--fake_x", str(fake_x_m),
            "--fake_y", str(fake_y_m),
            "--fake_yaw", str(fake_yaw_deg),
            "--input_voxel_size", str(input_voxel_size_m),
            "--output", str(episode_output),
        ]
        if dataset == "kitti360":
            command = [
                str(evaluator), "--kitti_root", root,
                "--calib_mode", "official",
                *common,
            ]
        else:
            command = [
                str(evaluator), "--m2dgr_root", root,
                "--gt", manifest["query_gt_path"],
                "--max_time_diff", str(effective_m2dgr_max_time_diff_s),
                "--normalize_gt_origin",
                *common,
            ]
        elapsed, _, _ = _run(command, episode_output)
        metrics = _read_json(episode_output / "metrics.json")
        outcome = classify_episode(metrics)
        contract_outcome = classify_contract(outcome, expected_behavior)
        with (episode_output / "relocalization_queries.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            queries = list(csv.DictReader(stream))
        for query in queries:
            if query.get("lock", "").lower() != "true":
                continue
            translation = _finite(query.get("translation_error_m"))
            yaw = _finite(query.get("yaw_error_deg"))
            if translation is not None:
                translation_errors.append(translation)
            if yaw is not None:
                yaw_errors.append(yaw)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "outcome": outcome,
                "expected_behavior": expected_behavior,
                "contract_outcome": contract_outcome,
                "wall_seconds": elapsed,
                "query_count": metrics.get("query_count", 0),
                "correct_lock_count": metrics.get("correct_lock_count", 0),
                "false_lock_count": metrics.get("false_lock_count", 0),
                "first_lock_frame": metrics.get("first_lock_frame", -1),
                "pose_error_at_lock_p95_m": metrics.get("pose_error_at_lock_p95_m"),
                "yaw_error_at_lock_p95_deg": metrics.get("yaw_error_at_lock_p95_deg"),
            }
        )

    episodes_csv = output / "episodes.csv"
    with episodes_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    correct = sum(row["outcome"] == "correct_lock" for row in episode_rows)
    false = sum(row["outcome"] == "false_lock" for row in episode_rows)
    no_lock = sum(row["outcome"] == "no_lock" for row in episode_rows)
    contract_pass = sum(row["contract_outcome"] == "pass" for row in episode_rows)
    unexpected_lock = sum(
        expected_behavior == "abstain" and row["outcome"] != "no_lock"
        for row in episode_rows
    )
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "evidence_class": manifest["evidence_class"],
        "map_sequence": manifest["map_sequence"],
        "query_sequence": manifest["query_sequence"],
        "map_frame_count": manifest["map_frame_count"],
        "attempt_count": len(episode_rows),
        "correct_attempt_count": correct,
        "false_attempt_count": false,
        "no_lock_attempt_count": no_lock,
        "expected_behavior": expected_behavior,
        "contract_pass_count": contract_pass,
        "contract_pass_rate": contract_pass / len(episode_rows),
        "unexpected_lock_count": unexpected_lock,
        "unexpected_lock_rate": unexpected_lock / len(episode_rows),
        "correct_attempt_rate": correct / len(episode_rows),
        "false_attempt_rate": false / len(episode_rows),
        "map_build_seconds": map_seconds,
        "episode_wall_seconds_p50": _percentile(
            [float(row["wall_seconds"]) for row in episode_rows], 0.5
        ),
        "episode_wall_seconds_p95": _percentile(
            [float(row["wall_seconds"]) for row in episode_rows], 0.95
        ),
        "translation_error_at_lock_p50_m": _percentile(translation_errors, 0.5),
        "translation_error_at_lock_p95_m": _percentile(translation_errors, 0.95),
        "yaw_error_at_lock_p50_deg": _percentile(yaw_errors, 0.5),
        "yaw_error_at_lock_p95_deg": _percentile(yaw_errors, 0.95),
        "fake_map_to_odom": {"x_m": fake_x_m, "y_m": fake_y_m, "yaw_deg": fake_yaw_deg},
        "input_voxel_size_m": input_voxel_size_m,
        "normalize_gt_origin": dataset == "m2dgr",
        "m2dgr_max_time_diff_s": (
            effective_m2dgr_max_time_diff_s if dataset == "m2dgr" else None
        ),
        "kitti360_calibration": manifest.get("kitti360_calibration"),
        "manifest_sha256": sha256_file(manifest_dir / "dataset_manifest.json"),
        "map_sha256": sha256_file(map_path),
        "atlas_sha256": sha256_file(atlas_path),
        "atlas_enabled": True,
        "formal_gate_ready": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _finalize_benchmark_output(output)
    _verify_benchmark_output(output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--atlas-compiler", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fake-x", type=float, default=20.0)
    parser.add_argument("--fake-y", type=float, default=-15.0)
    parser.add_argument("--fake-yaw", type=float, default=90.0)
    parser.add_argument("--input-voxel-size", type=float, default=0.2)
    parser.add_argument("--m2dgr-max-time-diff-s", type=float)
    parser.add_argument(
        "--allow-legacy-unhashed-calibration",
        action="store_true",
        help="permit old KITTI-360 manifests that do not freeze calibration hashes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_benchmark(
        evaluator=args.evaluator,
        atlas_compiler=args.atlas_compiler,
        manifest_dir=args.manifest_dir,
        output=args.output,
        fake_x_m=args.fake_x,
        fake_y_m=args.fake_y,
        fake_yaw_deg=args.fake_yaw,
        input_voxel_size_m=args.input_voxel_size,
        m2dgr_max_time_diff_s=args.m2dgr_max_time_diff_s,
        allow_legacy_unhashed_calibration=args.allow_legacy_unhashed_calibration,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
