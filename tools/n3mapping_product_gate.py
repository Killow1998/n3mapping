#!/usr/bin/env python3
"""Run the fail-closed n3mapping CPU relocalization Product Gate V1.

The gate deliberately owns evaluation orchestration and acceptance policy only.
It does not import research evaluators, expose matcher thresholds, or alter the
product bundle.  Exit codes are:

* 0: the complete gate passed;
* 2: the complete gate ran, but one or more product requirements failed;
* 1: the run is incomplete because an input, verifier, evaluator, or artifact
  violated its contract.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable

from n3mapping_product_identity import (
    ProductIdentityError,
    read_product_build_identity,
)


DATASET_SCHEMA = "n3mapping_product_gate_dataset_v1"
CANDIDATE_RUN_SCHEMA = "n3mapping_product_gate_candidate_run_v1"
EVALUATOR_RESULT_SCHEMA = "n3mapping_product_gate_case_result_v1"
GATE_RESULT_SCHEMA = "n3mapping_product_gate_result_v1"
ACTIVE_RUNTIME_AUTHORITY_SCHEMA = (
    "n3mapping_product_active_runtime_authority_preflight_v1"
)

STRICT_P95_LIMIT_MS = 1000.0
ACQUISITION_TO_LOCK_LIMIT_S = 5.0
PEAK_RSS_LIMIT_KIB = 600 * 1024
TRANSLATION_LIMIT_M = 0.5
YAW_LIMIT_DEG = 3.0
ROLL_PITCH_LIMIT_DEG = 2.0

CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POSE_KEYS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
CASE_KEYS = {
    "id",
    "source_id",
    "map_role",
    "manifest",
    "class",
    "raw_first_stamp_ns",
    "baseline",
    "reference_trajectory_csv",
    "reference_evidence_json",
    "reference_trajectory_sha256",
    "reference_evidence_sha256",
    "raw_lio_replay_evidence_json",
    "raw_lio_replay_evidence_sha256",
    "input_manifest_sha256",
    "input_pcd_set_sha256",
    "negative_kind",
}
CASE_CLASSES = {"required_lock", "ambiguity_allowed", "must_reject"}
NEGATIVE_KINDS = {"no_overlap", "wrong_map"}
REQUIRED_FLOOR7_POSITIVES = {
    "floor7_inside_710_room",
    "floor7_outside_707_room",
    "floor7_outside_710_room",
    "floor7_outside_713_room",
    "floor7_710toMeeting_static_70_75s",
}
REQUIRED_NEGATIVE_CASES = {
    "b22_query_on_floor7_wrong_map": {
        "source_id": "b22_query",
        "map_role": "floor7",
        "negative_kind": "wrong_map",
    },
    "f7tof9_query_on_b22_no_overlap": {
        "source_id": "f7tof9_query",
        "map_role": "b22",
        "negative_kind": "no_overlap",
    },
}
REQUIRED_MAP_ROLES = {"floor7", "b22"}
MAP_ARTIFACT_KEYS = {
    "map",
    "atlas",
    "atlas_format_version",
    "product_profile_sha256",
    "calibration_files",
}
FILE_IDENTITY_KEYS = {"bytes", "sha256"}
CALIBRATION_IDENTITY_KEYS = {"path", "bytes", "sha256"}
BASELINE_KEYS = {
    "reported_map_body_pose",
    "lock_frame_index",
    "relocalization_seed_keyframe_id",
    "relocalization_support_keyframe_id",
}
MANIFEST_HEADER = (
    "episode_id",
    "frame_index",
    "stamp_ns",
    "pcd_path",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
)
FRAME_STATUS_REQUIRED_HEADERS = {
    "frame_index",
    "stamp_ns",
    "success",
    "relocalization_lock_edge",
    "authoritative_full",
    "seed_keyframe_id",
    "support_keyframe_id",
    "matched_keyframe_id",
    "relocalization_state",
    "pose_source",
    "decision",
    "preprocessing_ms",
    "backend_ms",
    "strict_ms",
    "map_tx",
    "map_ty",
    "map_tz",
    "map_qx",
    "map_qy",
    "map_qz",
    "map_qw",
}
STATE_POSE_SOURCE = {
    "SEARCHING": "NONE",
    "REGION_HYPOTHESIS": "NONE",
    "FULL_6DOF_LOCKED": "GEOMETRICALLY_CORRECTED",
    "DEGRADED_TRACKING": "ODOM_PREDICTED",
}
SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
FORBIDDEN_INPUT_KEY_FRAGMENTS = (
    "threshold",
    "fitness",
    "inlier",
    "overlap",
    "scalar_score",
)


class GateError(RuntimeError):
    """An error that makes the gate run incomplete."""


def run_evaluator_with_resources(
    command: list[str],
) -> tuple[subprocess.CompletedProcess[str], dict[str, int]]:
    """Run one evaluator and collect Linux per-process resource evidence."""
    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
            )
            _, wait_status, usage = os.wait4(process.pid, 0)
            if os.WIFEXITED(wait_status):
                return_code = os.WEXITSTATUS(wait_status)
            elif os.WIFSIGNALED(wait_status):
                return_code = -os.WTERMSIG(wait_status)
            else:
                raise GateError(
                    "evaluator wait4 returned no terminal exit status"
                )
            process.returncode = return_code
            stdout_file.seek(0)
            stderr_file.seek(0)
            completed = subprocess.CompletedProcess(
                command,
                return_code,
                stdout_file.read().decode("utf-8", errors="replace"),
                stderr_file.read().decode("utf-8", errors="replace"),
            )
    except OSError as error:
        raise GateError(f"cannot start evaluator: {error}") from error
    return completed, {
        "peak_rss_kib": int(usage.ru_maxrss),
        "swap_operations": int(usage.ru_nswap),
    }


_SHA256_CACHE: dict[tuple[Any, ...], str] = {}


def now_shanghai() -> str:
    return datetime.now(SHANGHAI_TZ).isoformat(timespec="seconds")


def reject_nonfinite_json(token: str) -> None:
    raise GateError(f"non-finite JSON number is forbidden: {token}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    require_regular_file(path, "JSON input")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                parse_constant=reject_nonfinite_json,
                object_pairs_hook=reject_duplicate_keys,
            )
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot parse JSON {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    resolved = path.resolve()
    before = resolved.stat()
    key = (
        str(resolved),
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise GateError(f"input changed while hashing: {resolved}")
    result = digest.hexdigest()
    _SHA256_CACHE[key] = result
    return result


def require_regular_file(
    path: Path, label: str, *, nonempty: bool = True
) -> None:
    if path.is_symlink() or not path.is_file():
        raise GateError(f"{label} is not a regular file: {path}")
    if nonempty and path.stat().st_size == 0:
        raise GateError(f"{label} is empty: {path}")


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise GateError(f"{label} must be a finite number")
    return converted


def finite_csv_number(value: Any, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{label} must be a finite CSV number")
    try:
        converted = float(value)
    except ValueError as error:
        raise GateError(f"{label} must be a finite CSV number") from error
    if not math.isfinite(converted):
        raise GateError(f"{label} must be a finite CSV number")
    return converted


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"{label} must be an integer")
    return value


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise GateError(f"{label} must be a boolean")
    return value


def sha256_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise GateError(f"{label} must be a lowercase SHA-256 string")
    return value


def product_library_hashes(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    parsed: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name.startswith("libn3mapping"):
            raise GateError(f"{label} contains an invalid library name")
        parsed[name] = sha256_string(digest, f"{label}.{name}")
    return dict(sorted(parsed.items()))


def csv_integer(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise GateError(f"{label} must be an integer") from error


def csv_boolean(value: Any, label: str) -> bool:
    parsed = csv_integer(value, label)
    if parsed not in (0, 1):
        raise GateError(f"{label} must be 0 or 1")
    return bool(parsed)


def exact_keys(
    value: dict[str, Any], allowed: set[str], label: str, required: set[str]
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise GateError(f"{label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise GateError(f"{label} has unsupported keys: {', '.join(unknown)}")


def scan_for_algorithm_controls(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = key.lower()
            if any(part in normalized for part in FORBIDDEN_INPUT_KEY_FRAGMENTS):
                raise GateError(
                    f"algorithm scoring/threshold key is forbidden at "
                    f"{location}.{key}"
                )
            scan_for_algorithm_controls(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            scan_for_algorithm_controls(nested, f"{location}[{index}]")


def resolve_input_path(base: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise GateError(f"{label} must be a non-empty path string")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def fingerprint_input_manifest(path: Path) -> dict[str, Any]:
    """Freeze exactly the PCD bytes addressed by the evaluator manifest.

    The aggregate contract is shared with the extractor.  For every manifest
    row, in row order, it hashes:

      pcd_path + NUL + decimal byte size + NUL + file SHA-256 + LF
    """

    require_regular_file(path, "input frames.csv")
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as error:
        raise GateError(f"cannot open input frames.csv: {error}") from error
    aggregate = hashlib.sha256()
    pcd_files: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    manifest_base = path.parent.resolve()
    episode_id: str | None = None
    previous_stamp: int | None = None
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_HEADER):
            raise GateError(
                "input frames.csv header does not match the evaluator schema"
            )
        for row_number, row in enumerate(reader, start=2):
            frame_index = csv_integer(
                row.get("frame_index"),
                f"input frames.csv row {row_number}.frame_index",
            )
            if frame_index != len(pcd_files):
                raise GateError(
                    "input frames.csv frame_index must be contiguous and "
                    "start at zero"
                )
            stamp_ns = csv_integer(
                row.get("stamp_ns"),
                f"input frames.csv row {row_number}.stamp_ns",
            )
            if stamp_ns <= 0 or (
                previous_stamp is not None and stamp_ns <= previous_stamp
            ):
                raise GateError(
                    "input frames.csv stamp_ns must be positive and strictly "
                    "increasing"
                )
            current_episode = row.get("episode_id")
            if not isinstance(current_episode, str) or not current_episode:
                raise GateError(
                    f"input frames.csv row {row_number}.episode_id is empty"
                )
            if episode_id is None:
                episode_id = current_episode
            elif current_episode != episode_id:
                raise GateError(
                    "input frames.csv must contain exactly one episode_id"
                )
            raw_pcd_path = row.get("pcd_path")
            if (
                not isinstance(raw_pcd_path, str)
                or not raw_pcd_path
                or "\x00" in raw_pcd_path
                or "\n" in raw_pcd_path
                or "\r" in raw_pcd_path
            ):
                raise GateError(
                    f"input frames.csv row {row_number}.pcd_path is invalid"
                )
            relative = Path(raw_pcd_path)
            if (
                relative.is_absolute()
                or relative == Path(".")
                or any(part in {".", ".."} for part in relative.parts)
            ):
                raise GateError(
                    "input frames.csv pcd_path must be a normalized relative "
                    f"path (row {row_number})"
                )
            unresolved = manifest_base / relative
            cursor = manifest_base
            for part in relative.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise GateError(f"input PCD must not be a symlink: {cursor}")
            pcd_path = unresolved.resolve()
            try:
                pcd_path.relative_to(manifest_base)
                inside_manifest_directory = True
            except ValueError:
                inside_manifest_directory = False
            if (
                not inside_manifest_directory
                or not pcd_path.is_file()
                or pcd_path.stat().st_size == 0
            ):
                raise GateError(
                    f"input PCD is not a non-empty regular file: {pcd_path}"
                )
            if pcd_path in seen_paths:
                raise GateError(
                    f"input frames.csv reuses a PCD path: {raw_pcd_path}"
                )
            seen_paths.add(pcd_path)
            byte_size = pcd_path.stat().st_size
            pcd_sha256 = sha256_file(pcd_path)
            aggregate.update(raw_pcd_path.encode("utf-8"))
            aggregate.update(b"\x00")
            aggregate.update(str(byte_size).encode("ascii"))
            aggregate.update(b"\x00")
            aggregate.update(pcd_sha256.encode("ascii"))
            aggregate.update(b"\n")
            pcd_files.append(
                {
                    "frame_index": frame_index,
                    "stamp_ns": stamp_ns,
                    "manifest_path": raw_pcd_path,
                    "path": pcd_path,
                    "bytes": byte_size,
                    "sha256": pcd_sha256,
                }
            )
            previous_stamp = stamp_ns
    if not pcd_files:
        raise GateError("input frames.csv contains no frames")
    return {
        "manifest_sha256": sha256_file(path),
        "pcd_set_sha256": aggregate.hexdigest(),
        "pcd_files": pcd_files,
        "frame_count": len(pcd_files),
        "episode_id": episode_id,
        "first_stamp_ns": pcd_files[0]["stamp_ns"],
        "last_stamp_ns": pcd_files[-1]["stamp_ns"],
    }


def parse_pose(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    exact_keys(value, set(POSE_KEYS), label, set(POSE_KEYS))
    pose = {key: finite_number(value[key], f"{label}.{key}") for key in POSE_KEYS}
    norm = math.sqrt(sum(pose[key] ** 2 for key in ("qx", "qy", "qz", "qw")))
    if norm < 1e-12 or abs(norm - 1.0) > 1e-3:
        raise GateError(f"{label} quaternion must be unit length")
    for key in ("qx", "qy", "qz", "qw"):
        pose[key] /= norm
    return pose


def quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def relative_euler_errors_deg(
    reference: dict[str, float], actual: dict[str, float]
) -> tuple[float, float, float]:
    reference_inverse = (
        -reference["qx"],
        -reference["qy"],
        -reference["qz"],
        reference["qw"],
    )
    actual_q = (
        actual["qx"],
        actual["qy"],
        actual["qz"],
        actual["qw"],
    )
    x, y, z, w = quaternion_multiply(reference_inverse, actual_q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(abs(math.degrees(angle)) for angle in (roll, pitch, yaw))


def score_pose(
    actual: dict[str, float],
    reference: dict[str, float],
    source: str,
) -> dict[str, Any]:
    translation = math.sqrt(
        sum((actual[key] - reference[key]) ** 2 for key in ("tx", "ty", "tz"))
    )
    roll, pitch, yaw = relative_euler_errors_deg(reference, actual)
    passed = (
        translation <= TRANSLATION_LIMIT_M
        and yaw <= YAW_LIMIT_DEG
        and roll <= ROLL_PITCH_LIMIT_DEG
        and pitch <= ROLL_PITCH_LIMIT_DEG
    )
    return {
        "source": source,
        "translation_error_m": translation,
        "roll_error_deg": roll,
        "pitch_error_deg": pitch,
        "yaw_error_deg": yaw,
        "pass": passed,
    }


def quaternion_slerp(
    first: dict[str, float], second: dict[str, float], fraction: float
) -> tuple[float, float, float, float]:
    q0 = tuple(first[key] for key in ("qx", "qy", "qz", "qw"))
    q1 = tuple(second[key] for key in ("qx", "qy", "qz", "qw"))
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        interpolated = tuple(
            a + fraction * (b - a) for a, b in zip(q0, q1)
        )
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        left = math.sin((1.0 - fraction) * theta) / sin_theta
        right = math.sin(fraction * theta) / sin_theta
        interpolated = tuple(left * a + right * b for a, b in zip(q0, q1))
    norm = math.sqrt(sum(value * value for value in interpolated))
    return tuple(value / norm for value in interpolated)


def read_reference_trajectory(path: Path) -> list[tuple[int, dict[str, float]]]:
    require_regular_file(path, "reference trajectory CSV")
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as error:
        raise GateError(f"cannot open reference trajectory {path}: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise GateError(
                f"reference trajectory has missing or duplicate CSV headers: {path}"
            )
        plain = set(POSE_KEYS)
        mapped = {f"map_{key}" for key in POSE_KEYS}
        headers = set(reader.fieldnames)
        if "stamp_ns" not in headers:
            raise GateError("reference trajectory requires stamp_ns")
        if plain.issubset(headers):
            pose_columns = {key: key for key in POSE_KEYS}
        elif mapped.issubset(headers):
            pose_columns = {key: f"map_{key}" for key in POSE_KEYS}
        else:
            raise GateError(
                "reference trajectory requires tx..qw or map_tx..map_qw"
            )
        samples: list[tuple[int, dict[str, float]]] = []
        previous_stamp: int | None = None
        for row_number, row in enumerate(reader, start=2):
            try:
                stamp = int(row["stamp_ns"])
            except (TypeError, ValueError) as error:
                raise GateError(
                    f"invalid reference stamp_ns at row {row_number}"
                ) from error
            if stamp <= 0 or (
                previous_stamp is not None and stamp <= previous_stamp
            ):
                raise GateError(
                    "reference trajectory stamps must be positive and strictly "
                    f"increasing (row {row_number})"
                )
            pose = parse_pose(
                {
                    key: finite_csv_number(
                        row[column],
                        f"reference trajectory row {row_number}.{column}",
                    )
                    for key, column in pose_columns.items()
                },
                f"reference trajectory row {row_number}",
            )
            samples.append((stamp, pose))
            previous_stamp = stamp
    if not samples:
        raise GateError("reference trajectory contains no samples")
    return samples


def validate_reference_evidence(
    evidence_path: Path,
    reference_path: Path,
    manifest_path: Path,
    manifest_fingerprint: dict[str, Any],
    map_path: Path,
    reference_pose_count: int,
) -> dict[str, Any]:
    evidence = load_json(evidence_path)
    if not isinstance(evidence, dict):
        raise GateError("reference evidence JSON must be an object")
    if evidence.get("schema") != "n3mapping_same_frame_reference_evidence_v1":
        raise GateError("reference evidence schema mismatch")
    if (
        evidence.get("reference_class")
        != "same_run_dense_optimized_reference"
    ):
        raise GateError(
            "reference evidence is not a same-run dense optimized reference"
        )
    if evidence.get("pose_convention") != "T_world_body":
        raise GateError("reference evidence pose convention mismatch")
    if evidence.get("pose_source") != "native pbstream dense trajectory":
        raise GateError("reference evidence pose source mismatch")

    recorded_manifest = evidence.get("lio_frames")
    if not isinstance(recorded_manifest, str) or not recorded_manifest:
        raise GateError("reference evidence is missing lio_frames")
    if Path(recorded_manifest).resolve() != manifest_path:
        raise GateError(
            "reference evidence lio_frames points to a different manifest"
        )
    manifest_sha256 = sha256_string(
        evidence.get("lio_frames_sha256"),
        "reference evidence lio_frames_sha256",
    )
    if manifest_sha256 != manifest_fingerprint["manifest_sha256"]:
        raise GateError(
            "reference evidence lio_frames SHA-256 mismatch"
        )
    if evidence.get("lio_manifest_count") != manifest_fingerprint["frame_count"]:
        raise GateError("reference evidence LIO manifest count mismatch")
    if evidence.get("reference_pose_count") != reference_pose_count:
        raise GateError("reference evidence pose count mismatch")

    output = evidence.get("output")
    if not isinstance(output, dict):
        raise GateError("reference evidence output must be an object")
    if output.get("reference_trajectory") != reference_path.name:
        raise GateError(
            "reference evidence names a different reference trajectory"
        )
    reference_sha256 = sha256_string(
        output.get("reference_trajectory_sha256"),
        "reference evidence output.reference_trajectory_sha256",
    )
    if reference_sha256 != sha256_file(reference_path):
        raise GateError("reference trajectory SHA-256 disagrees with evidence")

    raw_first_stamp_ns = integer(
        evidence.get("raw_first_header_stamp_ns"),
        "reference evidence raw_first_header_stamp_ns",
    )
    raw_lidar_count = integer(
        evidence.get("raw_lidar_count"),
        "reference evidence raw_lidar_count",
    )
    raw_last_stamp_ns = integer(
        evidence.get("raw_last_header_stamp_ns"),
        "reference evidence raw_last_header_stamp_ns",
    )
    if raw_first_stamp_ns <= 0 or raw_lidar_count < manifest_fingerprint[
        "frame_count"
    ] or raw_last_stamp_ns < raw_first_stamp_ns:
        raise GateError("reference evidence raw acquisition fields are invalid")
    if (
        raw_first_stamp_ns > manifest_fingerprint["first_stamp_ns"]
        or raw_last_stamp_ns < manifest_fingerprint["last_stamp_ns"]
    ):
        raise GateError(
            "reference evidence raw acquisition does not cover the LIO "
            "manifest time range"
        )
    if evidence.get("skipped_raw_lidar_count") != (
        raw_lidar_count - manifest_fingerprint["frame_count"]
    ):
        raise GateError("reference evidence skipped raw count mismatch")
    if evidence.get("lio_to_raw_coverage_ratio") != (
        manifest_fingerprint["frame_count"] / raw_lidar_count
    ):
        raise GateError("reference evidence LIO/raw coverage ratio mismatch")

    overlap = evidence.get("inventory_payload_overlap")
    if not isinstance(overlap, dict) or not (
        overlap.get("matched_payload_ratio") == 1.0
        and overlap.get("unique_payload_ratio") == 1.0
        and overlap.get("lidar_message_count") == raw_lidar_count
    ):
        raise GateError(
            "reference evidence does not prove unique complete raw overlap"
        )

    def recorded_path(field: str) -> Path:
        raw = evidence.get(field)
        if not isinstance(raw, str) or not raw:
            raise GateError(f"reference evidence is missing {field}")
        path = Path(raw)
        if not path.is_absolute():
            raise GateError(f"reference evidence {field} must be absolute")
        require_regular_file(path, f"reference evidence {field}")
        return path.resolve()

    query_path = recorded_path("query_bag")
    query_sha256 = sha256_string(
        evidence.get("query_bag_sha256"),
        "reference evidence query_bag_sha256",
    )
    if sha256_file(query_path) != query_sha256:
        raise GateError("reference evidence query bag SHA-256 mismatch")
    if query_sha256 != sha256_string(
        evidence.get("inventory_query_bag_sha256"),
        "reference evidence inventory_query_bag_sha256",
    ):
        raise GateError("reference evidence query bag SHA-256 mismatch")
    full_path = recorded_path("full_bag")
    full_sha256 = sha256_string(
        evidence.get("full_bag_sha256"),
        "reference evidence full_bag_sha256",
    )
    if sha256_file(full_path) != full_sha256:
        raise GateError("reference evidence full bag SHA-256 mismatch")

    inventory_path = recorded_path("inventory")
    inventory_sha256 = sha256_string(
        evidence.get("inventory_sha256"),
        "reference evidence inventory_sha256",
    )
    if sha256_file(inventory_path) != inventory_sha256:
        raise GateError("reference evidence inventory SHA-256 mismatch")
    inventory = load_json(inventory_path)
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != "n3mapping_ros1_raw_bag_inventory_v1"
        or inventory.get("file_hashes_included") is not True
    ):
        raise GateError("reference evidence inventory contract mismatch")
    input_directory = inventory.get("input_directory")
    if (
        not isinstance(input_directory, str)
        or Path(input_directory).resolve() != query_path.parent
        or full_path.parent != query_path.parent
    ):
        raise GateError("reference evidence inventory directory mismatch")
    inventory_full = inventory.get("full_bag")
    if not isinstance(inventory_full, dict) or (
        inventory_full.get("file") != full_path.name
        or inventory_full.get("sha256") != full_sha256
    ):
        raise GateError("reference evidence full bag disagrees with inventory")
    inventory_queries = inventory.get("query_bags")
    if not isinstance(inventory_queries, list):
        raise GateError("reference evidence inventory query list is invalid")
    query_entries = [
        entry
        for entry in inventory_queries
        if isinstance(entry, dict) and entry.get("file") == query_path.name
    ]
    if len(query_entries) != 1:
        raise GateError("reference evidence query bag is not unique in inventory")
    query_entry = query_entries[0]
    entry_overlap = query_entry.get("payload_overlap")
    if (
        query_entry.get("sha256") != query_sha256
        or not isinstance(entry_overlap, dict)
        or entry_overlap.get("reference_class") != "same_frame_reference"
        or entry_overlap.get("matched_payload_ratio")
        != overlap["matched_payload_ratio"]
        or entry_overlap.get("unique_payload_ratio")
        != overlap["unique_payload_ratio"]
        or entry_overlap.get("lidar_message_count")
        != overlap["lidar_message_count"]
    ):
        raise GateError("reference evidence query bag disagrees with inventory")

    dense_trajectory_path = recorded_path("dense_trajectory")
    dense_trajectory_sha256 = sha256_string(
        evidence.get("dense_trajectory_sha256"),
        "reference evidence dense_trajectory_sha256",
    )
    if sha256_file(dense_trajectory_path) != dense_trajectory_sha256:
        raise GateError("reference evidence dense trajectory SHA-256 mismatch")
    dense_evidence_path = recorded_path("dense_evidence")
    dense_evidence_sha256 = sha256_string(
        evidence.get("dense_evidence_sha256"),
        "reference evidence dense_evidence_sha256",
    )
    if sha256_file(dense_evidence_path) != dense_evidence_sha256:
        raise GateError("reference evidence dense sidecar SHA-256 mismatch")
    dense_evidence = load_json(dense_evidence_path)
    dense_count = integer(
        evidence.get("dense_trajectory_count"),
        "reference evidence dense_trajectory_count",
    )
    if (
        dense_count < reference_pose_count
        or not isinstance(dense_evidence, dict)
        or dense_evidence.get("schema")
        != "n3mapping_dense_trajectory_evidence_v1"
        or dense_evidence.get("source") != "native"
        or dense_evidence.get("degraded") is not False
        or dense_evidence.get("pose_convention") != "T_world_body"
        or dense_evidence.get("row_count") != dense_count
        or Path(str(dense_evidence.get("trajectory_csv", ""))).resolve()
        != dense_trajectory_path
        or dense_evidence.get("trajectory_csv_sha256")
        != dense_trajectory_sha256
    ):
        raise GateError("reference evidence dense trajectory contract mismatch")
    with dense_trajectory_path.open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        dense_reader = csv.DictReader(stream)
        if dense_reader.fieldnames != [
            "stamp_ns",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
            "seq",
        ]:
            raise GateError("dense trajectory CSV header mismatch")
        if sum(1 for _ in dense_reader) != dense_count:
            raise GateError("dense trajectory CSV row count mismatch")

    recorded_pbstream = recorded_path("pbstream")
    if recorded_pbstream != map_path:
        raise GateError(
            "reference evidence pbstream path is not the Product Bundle map"
        )
    pbstream_sha256 = sha256_string(
        evidence.get("pbstream_sha256"),
        "reference evidence pbstream_sha256",
    )
    if pbstream_sha256 != sha256_file(map_path):
        raise GateError(
            "reference evidence pbstream is not the verified Product Bundle map"
        )
    if (
        Path(str(dense_evidence.get("pbstream", ""))).resolve() != map_path
        or dense_evidence.get("pbstream_sha256") != pbstream_sha256
    ):
        raise GateError("dense sidecar pbstream provenance mismatch")

    timestamp_alignment = evidence.get("timestamp_alignment")
    if not isinstance(timestamp_alignment, dict) or not (
        timestamp_alignment.get("raw_to_lio_frame_index_inferred") is False
        and timestamp_alignment.get("fitted_time_offset") is False
        and timestamp_alignment.get("join_tolerance_ns") == 512
        and timestamp_alignment.get("acquisition_start_definition")
        == "first query CustomMsg header.stamp_ns"
    ):
        raise GateError(
            "reference evidence timestamp alignment contract mismatch"
        )
    residual = timestamp_alignment.get("manifest_minus_dense_ns")
    if not isinstance(residual, dict):
        raise GateError("reference evidence timestamp residual is missing")
    residual_min = integer(
        residual.get("min"),
        "reference evidence timestamp residual min",
    )
    residual_max = integer(
        residual.get("max"),
        "reference evidence timestamp residual max",
    )
    if (
        residual_min > residual_max
        or residual_min < -512
        or residual_max > 512
    ):
        raise GateError(
            "reference evidence timestamp residual exceeds join tolerance"
        )
    return {
        "raw_first_stamp_ns": raw_first_stamp_ns,
        "raw_last_stamp_ns": raw_last_stamp_ns,
        "raw_lidar_count": raw_lidar_count,
        "sha256": sha256_file(evidence_path),
        "reference_trajectory_sha256": reference_sha256,
        "pbstream_sha256": pbstream_sha256,
        "query_bag_sha256": query_sha256,
        "full_bag_sha256": full_sha256,
        "inventory_sha256": inventory_sha256,
        "dense_trajectory_sha256": dense_trajectory_sha256,
        "dense_evidence_sha256": dense_evidence_sha256,
        "source_files": {
            "query_bag": str(query_path),
            "full_bag": str(full_path),
            "inventory": str(inventory_path),
            "dense_trajectory": str(dense_trajectory_path),
            "dense_evidence": str(dense_evidence_path),
            "pbstream": str(recorded_pbstream),
        },
    }


def validate_raw_lio_replay_evidence(
    evidence_path: Path,
    manifest_path: Path,
    manifest_fingerprint: dict[str, Any],
    reference_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        from n3mapping_raw_lio_replay_evidence import (
            ReplayEvidenceError,
            verify_evidence,
        )
    except ImportError as error:
        raise GateError(
            "raw-to-LIO replay evidence verifier is unavailable"
        ) from error
    try:
        verified = verify_evidence(evidence_path)
    except ReplayEvidenceError as error:
        raise GateError(
            f"raw-to-LIO replay evidence verification failed: {error}"
        ) from error
    document = load_json(evidence_path)
    spec = document.get("spec")
    bags = document.get("bags")
    extractor = document.get("extractor")
    if (
        not isinstance(spec, dict)
        or not isinstance(bags, dict)
        or not isinstance(extractor, dict)
    ):
        raise GateError("raw-to-LIO replay evidence structure is invalid")
    if reference_evidence is not None:
        if (
            Path(str(spec.get("source_ros1_bag", ""))).resolve()
            != Path(reference_evidence["source_files"]["query_bag"])
        ):
            raise GateError(
                "raw-to-LIO replay evidence source is not the reference "
                "query bag"
            )
    frames_identity = extractor.get("frames_csv")
    if (
        not isinstance(frames_identity, dict)
        or Path(str(frames_identity.get("path", ""))).resolve()
        != manifest_path
        or frames_identity.get("sha256")
        != manifest_fingerprint["manifest_sha256"]
        or extractor.get("pcd_set_sha256")
        != manifest_fingerprint["pcd_set_sha256"]
        or extractor.get("frame_count")
        != manifest_fingerprint["frame_count"]
    ):
        raise GateError(
            "raw-to-LIO replay evidence extractor output is not the Gate input"
        )
    source = bags.get("source_ros1")
    output = bags.get("lio_output_ros2")
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise GateError("raw-to-LIO replay evidence bag summaries are invalid")
    source_topics = source.get("topics")
    output_topics = output.get("topics")
    if not isinstance(source_topics, dict) or not isinstance(
        output_topics, dict
    ):
        raise GateError("raw-to-LIO replay evidence topic summaries are invalid")
    raw_lidar = source_topics.get(spec.get("source_lidar_topic"))
    output_cloud = output_topics.get(spec.get("cloud_topic"))
    if (
        not isinstance(raw_lidar, dict)
        or not isinstance(output_cloud, dict)
        or output_cloud.get("message_count")
        != manifest_fingerprint["frame_count"]
        or output_cloud.get("first_header_stamp_ns")
        != manifest_fingerprint["first_stamp_ns"]
        or output_cloud.get("last_header_stamp_ns")
        != manifest_fingerprint["last_stamp_ns"]
    ):
        raise GateError(
            "raw-to-LIO replay evidence output stamp/count chain disagrees "
            "with the Gate input"
        )
    raw_count = raw_lidar.get("message_count")
    raw_first_stamp_ns = raw_lidar.get("first_header_stamp_ns")
    raw_last_stamp_ns = raw_lidar.get("last_header_stamp_ns")
    if (
        isinstance(raw_count, bool)
        or not isinstance(raw_count, int)
        or raw_count < manifest_fingerprint["frame_count"]
        or isinstance(raw_first_stamp_ns, bool)
        or not isinstance(raw_first_stamp_ns, int)
        or isinstance(raw_last_stamp_ns, bool)
        or not isinstance(raw_last_stamp_ns, int)
        or raw_first_stamp_ns <= 0
        or raw_first_stamp_ns > manifest_fingerprint["first_stamp_ns"]
        or raw_last_stamp_ns < manifest_fingerprint["last_stamp_ns"]
    ):
        raise GateError(
            "raw-to-LIO replay evidence raw acquisition does not cover the "
            "Gate input"
        )
    if reference_evidence is not None and (
        raw_count != reference_evidence["raw_lidar_count"]
        or raw_first_stamp_ns != reference_evidence["raw_first_stamp_ns"]
        or raw_last_stamp_ns != reference_evidence["raw_last_stamp_ns"]
    ):
        raise GateError(
            "raw-to-LIO replay evidence raw acquisition disagrees with the "
            "reference evidence"
        )
    return {
        "sha256": verified["evidence_sha256"],
        "verdict": verified["verdict"],
        "source_ros1_bag": spec["source_ros1_bag"],
        "converted_ros2_bag": spec["converted_ros2_bag"],
        "lio_output_ros2_bag": spec["lio_output_ros2_bag"],
        "extractor_output": spec["extractor_output"],
        "raw_lidar_count": raw_count,
        "raw_first_stamp_ns": raw_first_stamp_ns,
        "raw_last_stamp_ns": raw_last_stamp_ns,
    }


def interpolate_reference_pose(
    samples: list[tuple[int, dict[str, float]]], stamp_ns: int
) -> tuple[dict[str, float], dict[str, Any]]:
    if stamp_ns < samples[0][0] or stamp_ns > samples[-1][0]:
        raise GateError(
            "lock_stamp_ns is outside the reference trajectory time range"
        )
    for index, (stamp, pose) in enumerate(samples):
        if stamp == stamp_ns:
            return dict(pose), {
                "lower_stamp_ns": stamp,
                "upper_stamp_ns": stamp,
                "interpolation_fraction": 0.0,
            }
        if stamp > stamp_ns:
            lower_stamp, lower = samples[index - 1]
            fraction = (stamp_ns - lower_stamp) / (stamp - lower_stamp)
            quaternion = quaternion_slerp(lower, pose, fraction)
            result = {
                key: lower[key] + fraction * (pose[key] - lower[key])
                for key in ("tx", "ty", "tz")
            }
            result.update(dict(zip(("qx", "qy", "qz", "qw"), quaternion)))
            return result, {
                "lower_stamp_ns": lower_stamp,
                "upper_stamp_ns": stamp,
                "interpolation_fraction": fraction,
            }
    raise GateError("cannot interpolate reference trajectory")


def percentile95(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise GateError("cannot compute strict p95 from no frame samples")
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def parse_frame_history(
    path: Path,
    expected_rows: int,
    manifest_fingerprint: dict[str, Any],
    result: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    require_regular_file(path, "frame_status.csv")
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as error:
        raise GateError(f"cannot open frame_status.csv: {error}") from error
    rows: list[dict[str, Any]] = []
    with stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not FRAME_STATUS_REQUIRED_HEADERS.issubset(
                set(reader.fieldnames)
            )
        ):
            raise GateError(
                "frame_status.csv does not match the required state-history "
                "schema"
            )
        previous_stamp: int | None = None
        for row_number, raw in enumerate(reader, start=2):
            label = f"frame_status row {row_number}"
            frame_index = csv_integer(
                raw.get("frame_index"), f"{label}.frame_index"
            )
            if frame_index != len(rows):
                raise GateError(
                    "frame_status.csv frame_index must be contiguous and "
                    "start at zero"
                )
            stamp_ns = csv_integer(raw.get("stamp_ns"), f"{label}.stamp_ns")
            if stamp_ns <= 0 or (
                previous_stamp is not None and stamp_ns <= previous_stamp
            ):
                raise GateError(
                    "frame_status.csv stamp_ns must be positive and strictly "
                    "increasing"
                )
            if frame_index >= manifest_fingerprint["frame_count"]:
                raise GateError(
                    "frame_status.csv has more rows than input frames.csv"
                )
            manifest_frame = manifest_fingerprint["pcd_files"][frame_index]
            if (
                frame_index != manifest_frame["frame_index"]
                or stamp_ns != manifest_frame["stamp_ns"]
            ):
                raise GateError(
                    "frame_status.csv frame identity disagrees with input "
                    "frames.csv"
                )
            state = raw.get("relocalization_state")
            pose_source = raw.get("pose_source")
            if state not in STATE_POSE_SOURCE:
                raise GateError(f"{label}.relocalization_state is invalid")
            if pose_source != STATE_POSE_SOURCE[state]:
                raise GateError(
                    f"{label} has an illegal state/pose_source combination"
                )
            success = csv_boolean(raw.get("success"), f"{label}.success")
            if state == "FULL_6DOF_LOCKED" and not success:
                raise GateError(
                    f"{label}.success disagrees with state/pose_source"
                )
            if state in {"SEARCHING", "REGION_HYPOTHESIS"} and success:
                raise GateError(
                    f"{label}.success disagrees with state/pose_source"
                )
            lock_edge = csv_boolean(
                raw.get("relocalization_lock_edge"),
                f"{label}.relocalization_lock_edge",
            )
            authoritative = csv_boolean(
                raw.get("authoritative_full"),
                f"{label}.authoritative_full",
            )
            expected_authoritative = state == "FULL_6DOF_LOCKED"
            if authoritative != expected_authoritative:
                raise GateError(
                    f"{label}.authoritative_full disagrees with state"
                )
            if lock_edge and not authoritative:
                raise GateError(
                    f"{label} lock edge is not an authoritative FULL lock"
                )
            preprocessing_ms = finite_csv_number(
                raw.get("preprocessing_ms"), f"{label}.preprocessing_ms"
            )
            backend_ms = finite_csv_number(
                raw.get("backend_ms"), f"{label}.backend_ms"
            )
            strict_ms = finite_csv_number(
                raw.get("strict_ms"), f"{label}.strict_ms"
            )
            if min(preprocessing_ms, backend_ms, strict_ms) < 0.0:
                raise GateError(
                    f"{label} timing values must be non-negative"
                )
            if not math.isclose(
                strict_ms,
                preprocessing_ms + backend_ms,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise GateError(
                    f"{label}.strict_ms is not preprocessing_ms + backend_ms"
                )
            pose = parse_pose(
                {
                    key: finite_csv_number(
                        raw.get(f"map_{key}"), f"{label}.map_{key}"
                    )
                    for key in POSE_KEYS
                },
                f"{label}.map_body_pose",
            )
            rows.append(
                {
                    "frame_index": frame_index,
                    "stamp_ns": stamp_ns,
                    "success": success,
                    "lock_edge": lock_edge,
                    "authoritative_full": authoritative,
                    "state": state,
                    "pose_source": pose_source,
                    "seed": csv_integer(
                        raw.get("seed_keyframe_id"),
                        f"{label}.seed_keyframe_id",
                    ),
                    "support": csv_integer(
                        raw.get("support_keyframe_id"),
                        f"{label}.support_keyframe_id",
                    ),
                    "matched": csv_integer(
                        raw.get("matched_keyframe_id"),
                        f"{label}.matched_keyframe_id",
                    ),
                    "strict_ms": strict_ms,
                    "pose": pose,
                }
            )
            previous_stamp = stamp_ns
    if len(rows) != expected_rows:
        raise GateError(
            "frame_status.csv row count does not match processed_frame_count"
        )
    if len(rows) != manifest_fingerprint["frame_count"]:
        raise GateError(
            "frame_status.csv does not cover every frozen input frame"
        )
    final = rows[-1]
    if (
        final["state"] != result["final_state"]
        or final["pose_source"] != result["final_pose_source"]
    ):
        raise GateError(
            "evaluator final state/pose_source disagrees with frame history"
        )
    authoritative_rows = [row for row in rows if row["authoritative_full"]]
    ever_authoritative = bool(authoritative_rows)
    if validated["locked"] != ever_authoritative:
        raise GateError(
            "algorithm_lock disagrees with authoritative FULL lock history"
        )
    first_authoritative = authoritative_rows[0] if authoritative_rows else None
    if first_authoritative is not None:
        if not first_authoritative["lock_edge"]:
            raise GateError("first authoritative FULL row is not a lock edge")
        if (
            first_authoritative["frame_index"] != validated["lock_frame"]
            or first_authoritative["stamp_ns"] != validated["lock_stamp"]
            or first_authoritative["seed"] != validated["seed"]
            or first_authoritative["support"] != validated["support"]
        ):
            raise GateError(
                "reported first lock fields disagree with frame history"
            )
        pose_delta = score_pose(
            validated["pose"],
            first_authoritative["pose"],
            "frame_status_first_authoritative",
        )
        if (
            pose_delta["translation_error_m"] > 1e-8
            or pose_delta["roll_error_deg"] > 1e-6
            or pose_delta["pitch_error_deg"] > 1e-6
            or pose_delta["yaw_error_deg"] > 1e-6
        ):
            raise GateError(
                "reported first lock pose disagrees with frame history"
            )
    return {
        "rows": rows,
        "strict_times": [row["strict_ms"] for row in rows],
        "ever_authoritative_lock": ever_authoritative,
        "first_authoritative": first_authoritative,
        "lock_edge_count": sum(row["lock_edge"] for row in rows),
    }


def validate_evaluator_result(
    result: Any,
    exit_code: int,
    map_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise GateError("evaluator result.json must be an object")
    required = {
        "schema",
        "episode_id",
        "algorithm_lock",
        "alignment_pose_authoritative",
        "final_state",
        "final_pose_source",
        "lock_frame_index",
        "lock_stamp_ns",
        "acquisition_start_stamp_ns",
        "acquisition_to_lock_s",
        "relocalization_seed_keyframe_id",
        "relocalization_support_keyframe_id",
        "reported_map_body_pose",
        "performance",
        "input_frame_count",
        "processed_frame_count",
        "map_path",
        "manifest_path",
        "alignment_review_pcd",
    }
    missing = sorted(required - set(result))
    if missing:
        raise GateError(
            "evaluator result.json is missing keys: " + ", ".join(missing)
        )
    if result["schema"] != EVALUATOR_RESULT_SCHEMA:
        raise GateError("evaluator result schema mismatch")
    if not isinstance(result["episode_id"], str) or not result["episode_id"]:
        raise GateError("result.episode_id must be a non-empty string")
    locked = boolean(result["algorithm_lock"], "result.algorithm_lock")
    authoritative = boolean(
        result["alignment_pose_authoritative"],
        "result.alignment_pose_authoritative",
    )
    if authoritative != locked:
        raise GateError(
            "alignment_pose_authoritative must equal algorithm_lock"
        )
    if exit_code != (0 if locked else 2):
        raise GateError("evaluator exit code disagrees with algorithm_lock")
    if not isinstance(result["final_state"], str) or result["final_state"] not in {
        "SEARCHING",
        "REGION_HYPOTHESIS",
        "FULL_6DOF_LOCKED",
        "DEGRADED_TRACKING",
    }:
        raise GateError("result.final_state is invalid")
    if (
        not isinstance(result["final_pose_source"], str)
        or result["final_pose_source"] not in {
        "NONE",
        "ODOM_PREDICTED",
        "GEOMETRICALLY_CORRECTED",
        }
    ):
        raise GateError("result.final_pose_source is invalid")
    if not isinstance(result["map_path"], str) or not result["map_path"]:
        raise GateError("result.map_path must be a non-empty string")
    if not isinstance(result["manifest_path"], str) or not result["manifest_path"]:
        raise GateError("result.manifest_path must be a non-empty string")
    if Path(result["map_path"]).resolve() != map_path:
        raise GateError("evaluator result map_path mismatch")
    if Path(result["manifest_path"]).resolve() != manifest_path:
        raise GateError("evaluator result manifest_path mismatch")
    if result["alignment_review_pcd"] != "alignment.pcd":
        raise GateError("evaluator must name the canonical PCD alignment.pcd")
    input_count = integer(
        result["input_frame_count"], "result.input_frame_count"
    )
    processed_count = integer(
        result["processed_frame_count"], "result.processed_frame_count"
    )
    if (
        input_count <= 0
        or processed_count <= 0
        or processed_count != input_count
    ):
        raise GateError("evaluator frame counts are invalid")
    lock_frame = integer(result["lock_frame_index"], "result.lock_frame_index")
    lock_stamp = integer(result["lock_stamp_ns"], "result.lock_stamp_ns")
    acquisition_start_stamp_ns = integer(
        result["acquisition_start_stamp_ns"],
        "result.acquisition_start_stamp_ns",
    )
    if acquisition_start_stamp_ns <= 0:
        raise GateError("result.acquisition_start_stamp_ns must be positive")
    acquisition = finite_number(
        result["acquisition_to_lock_s"],
        "result.acquisition_to_lock_s",
    )
    seed = integer(
        result["relocalization_seed_keyframe_id"],
        "result.relocalization_seed_keyframe_id",
    )
    support = integer(
        result["relocalization_support_keyframe_id"],
        "result.relocalization_support_keyframe_id",
    )
    if locked:
        if (
            lock_frame < 0
            or lock_stamp <= 0
            or acquisition < 0.0
            or seed < 0
            or support < 0
        ):
            raise GateError("locked evaluator result has invalid lock fields")
    elif not (
        lock_frame == -1
        and lock_stamp == 0
        and acquisition == -1.0
        and seed == -1
        and support == -1
    ):
        raise GateError("no-lock evaluator result must use sentinel lock fields")
    pose = parse_pose(
        result["reported_map_body_pose"], "result.reported_map_body_pose"
    )
    performance = result["performance"]
    if not isinstance(performance, dict):
        raise GateError("result.performance must be an object")
    for field in (
        "strict_p95_ms",
        "search_phase_p95_ms",
        "runtime_peak_rss_kib",
        "runtime_swap_operations",
    ):
        if field not in performance:
            raise GateError(f"result.performance.{field} is missing")
    strict_p95 = finite_number(
        performance["strict_p95_ms"], "result.performance.strict_p95_ms"
    )
    search_phase_p95 = finite_number(
        performance["search_phase_p95_ms"],
        "result.performance.search_phase_p95_ms",
    )
    strict_max = (
        finite_number(
            performance["strict_max_ms"], "result.performance.strict_max_ms"
        )
        if "strict_max_ms" in performance
        else None
    )
    rss = integer(
        performance["runtime_peak_rss_kib"],
        "result.performance.runtime_peak_rss_kib",
    )
    swap = integer(
        performance["runtime_swap_operations"],
        "result.performance.runtime_swap_operations",
    )
    if (
        strict_p95 < 0.0
        or search_phase_p95 < 0.0
        or (strict_max is not None and strict_max < 0.0)
        or rss < 0
        or swap < 0
    ):
        raise GateError("result performance values must be non-negative")
    if "manual_alignment_correct" in result and result[
        "manual_alignment_correct"
    ] is not None:
        raise GateError(
            "evaluator must not self-label manual_alignment_correct"
        )
    return {
        "locked": locked,
        "episode_id": result["episode_id"],
        "pose": pose,
        "input_count": input_count,
        "processed_count": processed_count,
        "lock_frame": lock_frame,
        "lock_stamp": lock_stamp,
        "acquisition_start_stamp_ns": acquisition_start_stamp_ns,
        "reported_acquisition_s": acquisition,
        "seed": seed,
        "support": support,
        "strict_p95_ms": strict_p95,
        "search_phase_p95_ms": search_phase_p95,
        "strict_max_ms": strict_max,
        "rss_kib": rss,
        "swap_operations": swap,
    }


def parse_baseline(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    exact_keys(value, BASELINE_KEYS, label, set())
    if not value:
        raise GateError(f"{label} must not be empty")
    baseline: dict[str, Any] = {}
    if "reported_map_body_pose" in value:
        baseline["reported_map_body_pose"] = parse_pose(
            value["reported_map_body_pose"],
            f"{label}.reported_map_body_pose",
        )
    for key in (
        "lock_frame_index",
        "relocalization_seed_keyframe_id",
        "relocalization_support_keyframe_id",
    ):
        if key in value:
            parsed = integer(value[key], f"{label}.{key}")
            if parsed < 0:
                raise GateError(f"{label}.{key} must be non-negative")
            baseline[key] = parsed
    return baseline


def parse_file_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    exact_keys(value, FILE_IDENTITY_KEYS, label, FILE_IDENTITY_KEYS)
    size = integer(value["bytes"], f"{label}.bytes")
    if size <= 0:
        raise GateError(f"{label}.bytes must be positive")
    return {
        "bytes": size,
        "sha256": sha256_string(value["sha256"], f"{label}.sha256"),
    }


def parse_map_artifacts(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise GateError("dataset map_artifacts must be a non-empty object")
    parsed: dict[str, dict[str, Any]] = {}
    for role, raw in value.items():
        label = f"dataset map_artifacts.{role}"
        if (
            not isinstance(role, str)
            or not CASE_ID_RE.fullmatch(role)
            or role in {".", ".."}
        ):
            raise GateError("dataset map_artifacts contains an invalid role")
        if not isinstance(raw, dict):
            raise GateError(f"{label} must be an object")
        exact_keys(raw, MAP_ARTIFACT_KEYS, label, MAP_ARTIFACT_KEYS)
        atlas_format = raw["atlas_format_version"]
        if not isinstance(atlas_format, str) or not atlas_format.strip():
            raise GateError(f"{label}.atlas_format_version must be non-empty")
        calibration = raw["calibration_files"]
        if not isinstance(calibration, list) or not calibration:
            raise GateError(
                f"{label}.calibration_files must be a non-empty array"
            )
        calibration_files: list[dict[str, Any]] = []
        calibration_paths: set[str] = set()
        for index, item in enumerate(calibration):
            item_label = f"{label}.calibration_files[{index}]"
            if not isinstance(item, dict):
                raise GateError(f"{item_label} must be an object")
            exact_keys(
                item,
                CALIBRATION_IDENTITY_KEYS,
                item_label,
                CALIBRATION_IDENTITY_KEYS,
            )
            relative = item["path"]
            if not isinstance(relative, str) or not relative:
                raise GateError(f"{item_label}.path must be non-empty")
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or relative_path.as_posix() != relative
                or ".." in relative_path.parts
                or "." in relative_path.parts
            ):
                raise GateError(
                    f"{item_label}.path must be a normalized relative path"
                )
            if relative in calibration_paths:
                raise GateError(
                    f"{label}.calibration_files contains duplicate path "
                    f"{relative}"
                )
            calibration_paths.add(relative)
            size = integer(item["bytes"], f"{item_label}.bytes")
            if size <= 0:
                raise GateError(f"{item_label}.bytes must be positive")
            calibration_files.append(
                {
                    "path": relative,
                    "bytes": size,
                    "sha256": sha256_string(
                        item["sha256"], f"{item_label}.sha256"
                    ),
                }
            )
        parsed[role] = {
            "map": parse_file_identity(raw["map"], f"{label}.map"),
            "atlas": parse_file_identity(raw["atlas"], f"{label}.atlas"),
            "atlas_format_version": atlas_format,
            "product_profile_sha256": sha256_string(
                raw["product_profile_sha256"],
                f"{label}.product_profile_sha256",
            ),
            "calibration_files": sorted(
                calibration_files, key=lambda item: item["path"]
            ),
        }
    missing = sorted(REQUIRED_MAP_ROLES - set(parsed))
    if missing:
        raise GateError(
            "dataset is missing required map roles: " + ", ".join(missing)
        )
    if (
        parsed["floor7"]["map"]["sha256"]
        == parsed["b22"]["map"]["sha256"]
    ):
        raise GateError("floor7 and b22 map roles must identify different maps")
    return parsed


def parse_candidate_run(path: Path) -> dict[str, Any]:
    document = load_json(path)
    scan_for_algorithm_controls(document)
    if not isinstance(document, dict):
        raise GateError("candidate run JSON must be an object")
    keys = {
        "schema",
        "candidate_commit",
        "evaluator_sha256",
        "evaluator_linked_product_libraries",
        "bundle_tool_sha256",
        "preflight_tool_sha256",
        "runtime_preflight_json",
        "runtime_preflight_sha256",
        "authority_preflight_json",
        "authority_preflight_sha256",
        "bundles",
    }
    exact_keys(document, keys, "candidate run JSON", keys)
    if document["schema"] != CANDIDATE_RUN_SCHEMA:
        raise GateError(
            f"candidate run JSON schema must be {CANDIDATE_RUN_SCHEMA}"
        )
    candidate_commit = document["candidate_commit"]
    if (
        not isinstance(candidate_commit, str)
        or not COMMIT_RE.fullmatch(candidate_commit)
    ):
        raise GateError(
            "candidate run candidate_commit must be a lowercase "
            "40-character Git SHA"
        )
    raw_bundles = document["bundles"]
    if not isinstance(raw_bundles, dict) or not raw_bundles:
        raise GateError("candidate run bundles must be a non-empty object")
    bundles: dict[str, Path] = {}
    for role, raw_path in raw_bundles.items():
        if (
            not isinstance(role, str)
            or not CASE_ID_RE.fullmatch(role)
            or role in {".", ".."}
        ):
            raise GateError("candidate run bundles contains an invalid role")
        bundles[role] = resolve_input_path(
            path.parent, raw_path, f"candidate run bundles.{role}"
        )
    return {
        "candidate_commit": candidate_commit,
        "evaluator_sha256": sha256_string(
            document["evaluator_sha256"],
            "candidate run evaluator_sha256",
        ),
        "evaluator_linked_product_libraries": product_library_hashes(
            document["evaluator_linked_product_libraries"],
            "candidate run evaluator_linked_product_libraries",
        ),
        "bundle_tool_sha256": sha256_string(
            document["bundle_tool_sha256"],
            "candidate run bundle_tool_sha256",
        ),
        "preflight_tool_sha256": sha256_string(
            document["preflight_tool_sha256"],
            "candidate run preflight_tool_sha256",
        ),
        "runtime_preflight_json": resolve_input_path(
            path.parent,
            document["runtime_preflight_json"],
            "candidate run runtime_preflight_json",
        ),
        "runtime_preflight_sha256": sha256_string(
            document["runtime_preflight_sha256"],
            "candidate run runtime_preflight_sha256",
        ),
        "authority_preflight_json": resolve_input_path(
            path.parent,
            document["authority_preflight_json"],
            "candidate run authority_preflight_json",
        ),
        "authority_preflight_sha256": sha256_string(
            document["authority_preflight_sha256"],
            "candidate run authority_preflight_sha256",
        ),
        "bundles": bundles,
    }


def parse_gate_inputs(
    candidate_run_path: Path,
    dataset_path: Path,
    expected_dataset_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = parse_candidate_run(candidate_run_path)
    expected_digest = sha256_string(
        expected_dataset_sha256, "--expected-dataset-sha256"
    )
    actual_digest = sha256_file(dataset_path)
    if actual_digest != expected_digest:
        raise GateError(
            "dataset contract SHA-256 does not match the externally supplied "
            "trusted digest"
        )
    document = load_json(dataset_path)
    scan_for_algorithm_controls(document)
    if not isinstance(document, dict):
        raise GateError("dataset contract must be an object")
    dataset_keys = {
        "schema",
        "schema_version",
        "dataset_id",
        "dataset_revision",
        "frozen_at_shanghai",
        "map_artifacts",
        "cases",
    }
    exact_keys(document, dataset_keys, "dataset contract", dataset_keys)
    if document["schema"] != DATASET_SCHEMA:
        raise GateError(
            f"dataset contract schema must be {DATASET_SCHEMA}"
        )
    if document["schema_version"] != 1:
        raise GateError("dataset contract schema_version must be 1")
    dataset_id = document["dataset_id"]
    if (
        not isinstance(dataset_id, str)
        or not CASE_ID_RE.fullmatch(dataset_id)
        or dataset_id in {".", ".."}
    ):
        raise GateError("dataset contract dataset_id is invalid")
    revision = integer(
        document["dataset_revision"], "dataset contract dataset_revision"
    )
    if revision <= 0:
        raise GateError("dataset contract dataset_revision must be positive")
    frozen_at = document["frozen_at_shanghai"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("+08:00"):
        raise GateError(
            "dataset contract frozen_at_shanghai must be an ISO-8601 "
            "timestamp with +08:00 offset"
        )
    try:
        datetime.fromisoformat(frozen_at)
    except ValueError as error:
        raise GateError(
            "dataset contract frozen_at_shanghai is invalid"
        ) from error
    map_artifacts = parse_map_artifacts(document["map_artifacts"])
    if set(identity["bundles"]) != set(map_artifacts):
        raise GateError(
            "candidate run bundle roles must exactly match dataset map roles"
        )
    identity.update(
        {
            "dataset_contract": str(dataset_path.resolve()),
            "dataset_contract_sha256": actual_digest,
            "dataset_id": dataset_id,
            "dataset_revision": revision,
            "map_artifacts": map_artifacts,
        }
    )
    if not isinstance(document["cases"], list) or not document["cases"]:
        raise GateError("dataset contract requires a non-empty cases array")
    base = dataset_path.parent
    parsed_cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(document["cases"]):
        label = f"dataset cases[{index}]"
        if not isinstance(raw, dict):
            raise GateError(f"{label} must be an object")
        exact_keys(
            raw,
            CASE_KEYS,
            label,
            {
                "id",
                "source_id",
                "map_role",
                "manifest",
                "class",
                "input_manifest_sha256",
                "input_pcd_set_sha256",
            },
        )
        case_id = raw["id"]
        if (
            not isinstance(case_id, str)
            or not CASE_ID_RE.fullmatch(case_id)
            or case_id in {".", ".."}
        ):
            raise GateError(f"{label}.id is not a safe case identifier")
        if case_id in ids:
            raise GateError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        source_id = raw["source_id"]
        if (
            not isinstance(source_id, str)
            or not CASE_ID_RE.fullmatch(source_id)
            or source_id in {".", ".."}
        ):
            raise GateError(f"{label}.source_id is invalid")
        map_role = raw["map_role"]
        if map_role not in map_artifacts:
            raise GateError(f"{label}.map_role is not defined by the dataset")
        case_class = raw["class"]
        if case_class not in CASE_CLASSES:
            raise GateError(
                f"{label}.class must be one of: "
                + ", ".join(sorted(CASE_CLASSES))
            )
        intent = {
            name: case_class == name for name in sorted(CASE_CLASSES)
        }
        parsed: dict[str, Any] = {
            "id": case_id,
            "source_id": source_id,
            "map_role": map_role,
            "class": case_class,
            **intent,
            "bundle": identity["bundles"][map_role],
            "map_artifacts": map_artifacts[map_role],
            "manifest": resolve_input_path(
                base, raw["manifest"], f"{label}.manifest"
            ),
            "input_manifest_sha256": sha256_string(
                raw["input_manifest_sha256"],
                f"{label}.input_manifest_sha256",
            ),
            "input_pcd_set_sha256": sha256_string(
                raw["input_pcd_set_sha256"],
                f"{label}.input_pcd_set_sha256",
            ),
        }
        if "raw_first_stamp_ns" in raw:
            raw_first_stamp_ns = integer(
                raw["raw_first_stamp_ns"], f"{label}.raw_first_stamp_ns"
            )
            if raw_first_stamp_ns <= 0:
                raise GateError(
                    f"{label}.raw_first_stamp_ns must be positive"
                )
            parsed["raw_first_stamp_ns"] = raw_first_stamp_ns
        if "baseline" in raw:
            parsed["baseline"] = parse_baseline(
                raw["baseline"], f"{label}.baseline"
            )
        if "reference_trajectory_csv" in raw:
            parsed["reference_trajectory_csv"] = resolve_input_path(
                base,
                raw["reference_trajectory_csv"],
                f"{label}.reference_trajectory_csv",
            )
        if "reference_evidence_json" in raw:
            parsed["reference_evidence_json"] = resolve_input_path(
                base,
                raw["reference_evidence_json"],
                f"{label}.reference_evidence_json",
            )
        if "reference_trajectory_sha256" in raw:
            parsed["reference_trajectory_sha256"] = sha256_string(
                raw["reference_trajectory_sha256"],
                f"{label}.reference_trajectory_sha256",
            )
        if "reference_evidence_sha256" in raw:
            parsed["reference_evidence_sha256"] = sha256_string(
                raw["reference_evidence_sha256"],
                f"{label}.reference_evidence_sha256",
            )
        if "raw_lio_replay_evidence_json" in raw:
            parsed["raw_lio_replay_evidence_json"] = resolve_input_path(
                base,
                raw["raw_lio_replay_evidence_json"],
                f"{label}.raw_lio_replay_evidence_json",
            )
        if "raw_lio_replay_evidence_sha256" in raw:
            parsed["raw_lio_replay_evidence_sha256"] = sha256_string(
                raw["raw_lio_replay_evidence_sha256"],
                f"{label}.raw_lio_replay_evidence_sha256",
            )
        if (
            ("reference_trajectory_csv" in parsed)
            != ("reference_evidence_json" in parsed)
        ):
            raise GateError(
                f"{label} reference trajectory and evidence must be provided "
                "together"
            )
        reference_paths_present = "reference_trajectory_csv" in parsed
        reference_hashes_present = (
            "reference_trajectory_sha256" in parsed
            and "reference_evidence_sha256" in parsed
        )
        if reference_paths_present != reference_hashes_present:
            raise GateError(
                f"{label} reference paths and frozen SHA-256 values must be "
                "provided together"
            )
        if (
            ("reference_trajectory_sha256" in parsed)
            != ("reference_evidence_sha256" in parsed)
        ):
            raise GateError(
                f"{label} reference SHA-256 values must be provided together"
            )
        replay_path_present = "raw_lio_replay_evidence_json" in parsed
        replay_hash_present = "raw_lio_replay_evidence_sha256" in parsed
        if replay_path_present != replay_hash_present:
            raise GateError(
                f"{label} raw-to-LIO replay evidence path and SHA-256 must be "
                "provided together"
            )
        if case_class == "must_reject":
            negative_kind = raw.get("negative_kind")
            if negative_kind not in NEGATIVE_KINDS:
                raise GateError(
                    f"{label}.negative_kind must be one of: "
                    + ", ".join(sorted(NEGATIVE_KINDS))
                )
            parsed["negative_kind"] = negative_kind
        elif "negative_kind" in raw:
            raise GateError(
                f"{label}.negative_kind is only valid for must_reject cases"
            )
        if parsed["required_lock"] and not (
            parsed.get("baseline", {}).get("reported_map_body_pose")
            or "reference_trajectory_csv" in parsed
        ):
            raise GateError(
                f"{label} required-lock case needs an accepted baseline pose "
                "or reference trajectory"
            )
        if parsed["required_lock"] and not (
            "raw_first_stamp_ns" in parsed
            or "reference_evidence_json" in parsed
        ):
            raise GateError(
                f"{label} required-lock case needs raw acquisition start "
                "evidence"
            )
        parsed_cases.append(parsed)

    by_id = {case["id"]: case for case in parsed_cases}
    positive_ids = {
        case["id"] for case in parsed_cases if case["required_lock"]
    }
    missing_floor7 = sorted(REQUIRED_FLOOR7_POSITIVES - positive_ids)
    if missing_floor7:
        raise GateError(
            "product Gate is missing required floor7 positive cases: "
            + ", ".join(missing_floor7)
        )
    for case_id in REQUIRED_FLOOR7_POSITIVES:
        case = by_id[case_id]
        if case["source_id"] != case_id or case["map_role"] != "floor7":
            raise GateError(
                f"required floor7 case {case_id} must bind its exact source "
                "id to the floor7 map role"
            )
    missing_negatives = sorted(set(REQUIRED_NEGATIVE_CASES) - set(by_id))
    if missing_negatives:
        raise GateError(
            "product Gate is missing required exact negative cases: "
            + ", ".join(missing_negatives)
        )
    for case_id, expected in REQUIRED_NEGATIVE_CASES.items():
        case = by_id[case_id]
        if (
            not case["must_reject"]
            or case["source_id"] != expected["source_id"]
            or case["map_role"] != expected["map_role"]
            or case["negative_kind"] != expected["negative_kind"]
        ):
            raise GateError(
                f"required negative case {case_id} source, map role, class, "
                "or negative kind does not match the frozen product contract"
            )
    return identity, parsed_cases


def command_for_python_or_binary(path: Path) -> list[str]:
    if path.suffix == ".py":
        return [sys.executable, "-B", str(path)]
    return [str(path)]


def run_bundle_verifier(bundle_tool: Path, bundle: Path) -> dict[str, Any]:
    command = command_for_python_or_binary(bundle_tool)
    command.extend(["verify", "--bundle", str(bundle)])
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except OSError as error:
        raise GateError(f"cannot start Product Bundle verifier: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError(
            f"Product Bundle verification failed for {bundle}: {detail}"
        )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_evaluator_log(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def bundle_checksum_files(bundle: Path) -> list[Path]:
    files = [
        bundle / "map.pbstream",
        bundle / "map.pbstream.localization_atlas.pb",
        bundle / "product_v1.yaml",
        bundle / "manifest.json",
    ]
    calibration = bundle / "calibration"
    if calibration.is_symlink() or not calibration.is_dir():
        raise GateError(
            f"verified bundle calibration is not a directory: {calibration}"
        )
    calibration_files: list[Path] = []
    for candidate in sorted(calibration.rglob("*")):
        if candidate.is_symlink():
            raise GateError(
                f"verified bundle calibration contains symlink: {candidate}"
            )
        if candidate.is_file():
            require_regular_file(candidate, "verified bundle calibration")
            calibration_files.append(candidate)
    if not calibration_files:
        raise GateError("verified bundle calibration contains no files")
    for path in files:
        require_regular_file(path, "verified bundle input")
    return [path.resolve() for path in files + calibration_files]


def write_case_checksums(
    path: Path,
    files: Iterable[Path],
) -> tuple[dict[str, str], str]:
    by_path: dict[str, str] = {}
    for candidate in files:
        resolved = candidate.resolve()
        require_regular_file(
            resolved,
            f"checksums.sha256 input {resolved}",
            nonempty=False,
        )
        label = str(resolved)
        if "\n" in label or "\r" in label or "\x00" in label:
            raise GateError("checksums.sha256 path contains control characters")
        by_path[label] = sha256_file(resolved)
    path.write_text(
        "".join(
            f"{digest}  {label}\n"
            for label, digest in sorted(by_path.items())
        ),
        encoding="utf-8",
    )
    require_regular_file(path, "case checksums.sha256")
    return by_path, sha256_file(path)


def baseline_score(
    baseline: dict[str, Any] | None,
    validated: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if baseline is None:
        return None, None
    score: dict[str, Any] = {}
    pose_score = None
    if "reported_map_body_pose" in baseline and validated["locked"]:
        pose_score = score_pose(
            validated["pose"],
            baseline["reported_map_body_pose"],
            "frozen_baseline",
        )
        score["pose"] = pose_score
    for baseline_key, actual_key, report_key in (
        ("lock_frame_index", "lock_frame", "lock_frame"),
        (
            "relocalization_seed_keyframe_id",
            "seed",
            "relocalization_seed_keyframe",
        ),
        (
            "relocalization_support_keyframe_id",
            "support",
            "relocalization_support_keyframe",
        ),
    ):
        if baseline_key in baseline:
            expected = baseline[baseline_key]
            actual = validated[actual_key]
            item: dict[str, Any] = {
                "expected": expected,
                "actual": actual,
                "exact_match": actual == expected,
            }
            if report_key == "lock_frame":
                item["delta_frames"] = actual - expected
            score[report_key] = item
    return score, pose_score


def verify_dataset_map_artifacts(
    case_id: str,
    bundle: Path,
    bundle_document: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for role, relative in (
        ("map", "map.pbstream"),
        ("atlas", "map.pbstream.localization_atlas.pb"),
    ):
        path = bundle / relative
        require_regular_file(path, f"case {case_id} dataset {role}")
        actual = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual != expected[role]:
            raise GateError(
                f"case {case_id} {role} bytes do not match the externally "
                "frozen dataset contract"
            )
        observed[role] = actual

    product_profile = bundle / "product_v1.yaml"
    require_regular_file(
        product_profile, f"case {case_id} dataset product profile"
    )
    profile_sha256 = sha256_file(product_profile)
    if profile_sha256 != expected["product_profile_sha256"]:
        raise GateError(
            f"case {case_id} product profile does not match the externally "
            "frozen dataset contract"
        )
    observed["product_profile_sha256"] = profile_sha256

    if (
        bundle_document.get("atlas_format_version")
        != expected["atlas_format_version"]
    ):
        raise GateError(
            f"case {case_id} atlas format does not match the externally "
            "frozen dataset contract"
        )
    observed["atlas_format_version"] = expected["atlas_format_version"]

    calibration_root = bundle / "calibration"
    if calibration_root.is_symlink() or not calibration_root.is_dir():
        raise GateError(
            f"case {case_id} calibration directory is missing or a symlink"
        )
    actual_calibration: list[dict[str, Any]] = []
    for path in sorted(calibration_root.rglob("*")):
        if path.is_symlink():
            raise GateError(
                f"case {case_id} calibration contains a symlink: {path}"
            )
        if not path.is_file():
            continue
        require_regular_file(path, f"case {case_id} calibration file")
        actual_calibration.append(
            {
                "path": path.relative_to(calibration_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if actual_calibration != expected["calibration_files"]:
        raise GateError(
            f"case {case_id} calibration files do not match the externally "
            "frozen dataset contract"
        )
    observed["calibration_files"] = actual_calibration
    return observed


def run_case(
    case: dict[str, Any],
    evaluator: Path,
    bundle_tool: Path,
    output_root: Path,
    candidate_commit: str,
    candidate_profile_sha256: str,
    evaluator_product_libraries: dict[str, str],
) -> tuple[dict[str, Any], list[float], dict[str, Any] | None]:
    bundle = case["bundle"]
    if bundle.is_symlink() or not bundle.is_dir():
        raise GateError(f"case {case['id']} bundle is not a directory: {bundle}")
    manifest = case["manifest"]
    require_regular_file(manifest, f"case {case['id']} manifest")
    manifest_fingerprint = fingerprint_input_manifest(manifest)
    if manifest_fingerprint["episode_id"] != case["id"]:
        raise GateError(
            f"case {case['id']} id disagrees with input manifest episode_id"
        )
    if (
        manifest_fingerprint["manifest_sha256"]
        != case["input_manifest_sha256"]
    ):
        raise GateError(
            f"case {case['id']} input frames.csv SHA-256 mismatch"
        )
    if (
        manifest_fingerprint["pcd_set_sha256"]
        != case["input_pcd_set_sha256"]
    ):
        raise GateError(f"case {case['id']} input PCD set SHA-256 mismatch")
    bundle_verification = run_bundle_verifier(bundle_tool, bundle)
    map_path = (bundle / "map.pbstream").resolve()
    atlas_path = (bundle / "map.pbstream.localization_atlas.pb").resolve()
    bundle_manifest = (bundle / "manifest.json").resolve()
    require_regular_file(map_path, "verified bundle map")
    require_regular_file(atlas_path, "verified bundle atlas")
    require_regular_file(bundle_manifest, "verified bundle manifest")
    bundle_document = load_json(bundle_manifest)
    if (
        not isinstance(bundle_document, dict)
        or bundle_document.get("n3mapping_commit") != candidate_commit
    ):
        raise GateError(
            f"case {case['id']} Product Bundle commit does not match "
            "the frozen candidate"
        )
    if (
        bundle_document.get("product_profile_sha256")
        != candidate_profile_sha256
    ):
        raise GateError(
            f"case {case['id']} Product Bundle profile does not match "
            "the frozen candidate"
        )
    if (
        candidate_profile_sha256
        != case["map_artifacts"]["product_profile_sha256"]
    ):
        raise GateError(
            f"case {case['id']} candidate profile does not match the "
            "externally frozen dataset contract"
        )
    dataset_map_artifacts = verify_dataset_map_artifacts(
        case["id"], bundle, bundle_document, case["map_artifacts"]
    )
    bundle_product_libraries = bundle_document.get(
        "runtime_node_linked_product_libraries", {}
    )
    bundle_core_sha256 = (
        bundle_product_libraries.get("libn3mapping_core.so")
        if isinstance(bundle_product_libraries, dict)
        else None
    )
    evaluator_core_sha256 = evaluator_product_libraries.get(
        "libn3mapping_core.so"
    )
    if (
        bundle_core_sha256 is None
        or evaluator_core_sha256 is None
        or bundle_core_sha256 != evaluator_core_sha256
    ):
        raise GateError(
            f"case {case['id']} evaluator and runtime node do not load the "
            "same libn3mapping_core.so"
        )
    frozen_bundle_files = bundle_checksum_files(bundle)

    reference_samples = None
    reference_evidence = None
    raw_lio_replay_evidence = None
    reference_path = case.get("reference_trajectory_csv")
    if reference_path is not None:
        if (
            sha256_file(reference_path)
            != case["reference_trajectory_sha256"]
        ):
            raise GateError(
                f"case {case['id']} reference trajectory SHA-256 mismatch"
            )
        if (
            sha256_file(case["reference_evidence_json"])
            != case["reference_evidence_sha256"]
        ):
            raise GateError(
                f"case {case['id']} reference evidence SHA-256 mismatch"
            )
        reference_samples = read_reference_trajectory(reference_path)
        reference_evidence = validate_reference_evidence(
            case["reference_evidence_json"],
            reference_path,
            manifest,
            manifest_fingerprint,
            map_path,
            len(reference_samples),
        )
    replay_evidence_path = case.get("raw_lio_replay_evidence_json")
    if replay_evidence_path is not None:
        if (
            sha256_file(replay_evidence_path)
            != case["raw_lio_replay_evidence_sha256"]
        ):
            raise GateError(
                f"case {case['id']} raw-to-LIO replay evidence SHA-256 "
                "mismatch"
            )
        raw_lio_replay_evidence = validate_raw_lio_replay_evidence(
            replay_evidence_path,
            manifest,
            manifest_fingerprint,
            reference_evidence,
        )
    effective_raw_first_stamp_ns = case.get("raw_first_stamp_ns")
    if reference_evidence is not None:
        evidence_raw_first = reference_evidence["raw_first_stamp_ns"]
        if (
            effective_raw_first_stamp_ns is not None
            and effective_raw_first_stamp_ns != evidence_raw_first
        ):
            raise GateError(
                f"case {case['id']} raw_first_stamp_ns disagrees with "
                "reference evidence"
            )
        effective_raw_first_stamp_ns = evidence_raw_first
    if raw_lio_replay_evidence is not None:
        replay_raw_first = raw_lio_replay_evidence["raw_first_stamp_ns"]
        if (
            effective_raw_first_stamp_ns is not None
            and effective_raw_first_stamp_ns != replay_raw_first
        ):
            raise GateError(
                f"case {case['id']} raw_first_stamp_ns disagrees with "
                "raw-to-LIO replay evidence"
            )
        effective_raw_first_stamp_ns = replay_raw_first
    if (
        effective_raw_first_stamp_ns is not None
        and effective_raw_first_stamp_ns
        > manifest_fingerprint["first_stamp_ns"]
    ):
        raise GateError(
            f"case {case['id']} raw acquisition start is later than the "
            "first LIO manifest frame"
        )

    case_output = output_root / "cases" / case["id"]
    if case_output.exists():
        raise GateError(f"case output unexpectedly exists: {case_output}")
    command = command_for_python_or_binary(evaluator)
    command.extend(
        [
            "--map",
            str(map_path),
            "--atlas",
            str(atlas_path),
            "--manifest",
            str(manifest),
            "--output",
            str(case_output),
        ]
    )
    if effective_raw_first_stamp_ns is not None:
        command.extend(
            ["--raw-first-stamp-ns", str(effective_raw_first_stamp_ns)]
        )
    try:
        completed, measured_resources = run_evaluator_with_resources(command)
    except GateError as error:
        raise GateError(
            f"cannot start evaluator for case {case['id']}: {error}"
        ) from error
    case_output.mkdir(parents=True, exist_ok=True)
    stdout_log = case_output / "evaluator.stdout.log"
    stderr_log = case_output / "evaluator.stderr.log"
    write_evaluator_log(stdout_log, completed.stdout)
    write_evaluator_log(stderr_log, completed.stderr)
    if completed.returncode not in (0, 2):
        raise GateError(
            f"evaluator failed for case {case['id']} with exit "
            f"{completed.returncode}; see {stderr_log}"
        )

    alignment = case_output / "alignment.pcd"
    frame_status = case_output / "frame_status.csv"
    result_path = case_output / "result.json"
    require_regular_file(alignment, "canonical alignment.pcd")
    require_regular_file(frame_status, "canonical frame_status.csv")
    require_regular_file(result_path, "canonical result.json")
    result = load_json(result_path)
    validated = validate_evaluator_result(
        result, completed.returncode, map_path, manifest
    )
    if validated["episode_id"] != manifest_fingerprint["episode_id"]:
        raise GateError(
            f"case {case['id']} evaluator episode_id mismatch"
        )
    if (
        validated["input_count"] != manifest_fingerprint["frame_count"]
        or validated["processed_count"] != manifest_fingerprint["frame_count"]
    ):
        raise GateError(
            f"case {case['id']} evaluator did not cover the frozen manifest"
        )
    expected_acquisition_start = (
        effective_raw_first_stamp_ns
        if effective_raw_first_stamp_ns is not None
        else manifest_fingerprint["first_stamp_ns"]
    )
    if (
        validated["acquisition_start_stamp_ns"]
        != expected_acquisition_start
    ):
        raise GateError(
            f"case {case['id']} evaluator acquisition start mismatch"
        )
    if validated["locked"]:
        acquisition_ns = (
            validated["lock_stamp"] - expected_acquisition_start
        )
        if acquisition_ns < 0:
            raise GateError(
                f"case {case['id']} lock stamp precedes trusted acquisition "
                "start"
            )
        reported_acquisition_ns = int(
            round(validated["reported_acquisition_s"] * 1_000_000_000)
        )
        if reported_acquisition_ns != acquisition_ns:
            raise GateError(
                f"case {case['id']} evaluator acquisition_to_lock_s "
                "disagrees with trusted integer-nanosecond timestamps"
            )
        recomputed_acquisition_s = acquisition_ns / 1_000_000_000.0
    else:
        if validated["reported_acquisition_s"] != -1.0:
            raise GateError(
                f"case {case['id']} no-lock acquisition duration must use "
                "the -1 sentinel"
            )
        recomputed_acquisition_s = -1.0
    validated["acquisition_s"] = recomputed_acquisition_s
    frame_history = parse_frame_history(
        frame_status,
        validated["processed_count"],
        manifest_fingerprint,
        result,
        validated,
    )
    strict_times = frame_history["strict_times"]
    recomputed_p95 = percentile95(strict_times)
    recomputed_max = max(strict_times)
    if not math.isclose(
        recomputed_p95,
        validated["strict_p95_ms"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise GateError(
            f"case {case['id']} strict p95 disagrees with frame_status.csv"
        )

    baseline, baseline_pose_score = baseline_score(
        case.get("baseline"), validated
    )
    pose_checks: list[dict[str, Any]] = []
    if baseline_pose_score is not None:
        pose_checks.append(baseline_pose_score)
    reference_score = None
    if reference_samples is not None and validated["locked"]:
        reference_pose, interpolation = interpolate_reference_pose(
            reference_samples, validated["lock_stamp"]
        )
        reference_pose_score = score_pose(
            validated["pose"], reference_pose, "reference_trajectory"
        )
        reference_score = {
            **interpolation,
            "reference_pose": reference_pose,
            "pose": reference_pose_score,
        }
        pose_checks.append(reference_pose_score)

    semantic_reasons: list[str] = []
    if case["required_lock"]:
        if not validated["locked"]:
            semantic_reasons.append("required_lock_not_obtained")
        elif not pose_checks:
            semantic_reasons.append("lock_has_no_pose_correctness_evidence")
        elif not all(score["pass"] for score in pose_checks):
            semantic_reasons.append("locked_pose_outside_accuracy_contract")
        if baseline is not None and any(
            not item["exact_match"]
            for key, item in baseline.items()
            if key in {
                "lock_frame",
                "relocalization_seed_keyframe",
                "relocalization_support_keyframe",
            }
        ):
            semantic_reasons.append("frozen_baseline_identity_mismatch")
    elif case["must_reject"]:
        if frame_history["ever_authoritative_lock"]:
            semantic_reasons.append("false_lock_on_must_reject_case")
    elif validated["locked"]:
        if not pose_checks:
            semantic_reasons.append("ambiguous_case_lock_is_unverified")
        elif not all(score["pass"] for score in pose_checks):
            semantic_reasons.append("ambiguous_case_locked_to_wrong_pose")

    acquisition_pass = (
        validated["acquisition_s"] < ACQUISITION_TO_LOCK_LIMIT_S
        if validated["locked"]
        else None
    )
    resource = {
        "strict_p95_under_1000_ms": recomputed_p95 < STRICT_P95_LIMIT_MS,
        "acquisition_to_lock_under_5_s": acquisition_pass,
        "peak_rss_under_600_mib": measured_resources["peak_rss_kib"]
        < PEAK_RSS_LIMIT_KIB,
        "swap_operations_zero": measured_resources["swap_operations"] == 0,
    }
    semantic_pass = not semantic_reasons
    case_pass = semantic_pass and all(
        value for value in resource.values() if value is not None
    )

    output_checksums = {
        "alignment.pcd": sha256_file(alignment),
        "frame_status.csv": sha256_file(frame_status),
        "result.json": sha256_file(result_path),
        "evaluator.stdout.log": sha256_file(stdout_log),
        "evaluator.stderr.log": sha256_file(stderr_log),
    }
    checksums = {
        "bundle": {
            str(path.relative_to(bundle)): sha256_file(path)
            for path in frozen_bundle_files
        },
        "input_manifest_sha256": manifest_fingerprint["manifest_sha256"],
        "input_pcd_set_sha256": manifest_fingerprint["pcd_set_sha256"],
        "input_pcd_files": [
            {
                key: item[key]
                for key in (
                    "frame_index",
                    "stamp_ns",
                    "manifest_path",
                    "bytes",
                    "sha256",
                )
            }
            for item in manifest_fingerprint["pcd_files"]
        ],
        "evaluator_sha256": sha256_file(evaluator),
        "bundle_tool_sha256": sha256_file(bundle_tool),
        "outputs": output_checksums,
    }
    if reference_path is not None:
        checksums["reference_trajectory_csv_sha256"] = sha256_file(
            reference_path
        )
        checksums["reference_evidence_json_sha256"] = reference_evidence[
            "sha256"
        ]
    if raw_lio_replay_evidence is not None:
        checksums["raw_lio_replay_evidence_json_sha256"] = (
            raw_lio_replay_evidence["sha256"]
        )

    checksum_inputs = [
        *frozen_bundle_files,
        manifest,
        *(item["path"] for item in manifest_fingerprint["pcd_files"]),
        evaluator,
        bundle_tool,
        alignment,
        frame_status,
        result_path,
        stdout_log,
        stderr_log,
    ]
    if reference_path is not None:
        checksum_inputs.extend(
            [
                reference_path,
                case["reference_evidence_json"],
                *(
                    Path(path)
                    for path in reference_evidence["source_files"].values()
                ),
            ]
        )
    if raw_lio_replay_evidence is not None:
        checksum_inputs.append(case["raw_lio_replay_evidence_json"])
    checksums_path = case_output / "checksums.sha256"
    _, checksums_file_sha256 = write_case_checksums(
        checksums_path, checksum_inputs
    )
    checksums["outputs"]["checksums.sha256"] = checksums_file_sha256

    primary_accuracy = None
    if reference_score is not None:
        primary_accuracy = reference_score["pose"]
    elif baseline_pose_score is not None:
        primary_accuracy = baseline_pose_score
    report = {
        "id": case["id"],
        "source_id": case["source_id"],
        "map_role": case["map_role"],
        "class": case["class"],
        "intent": {
            key: case[key]
            for key in ("required_lock", "ambiguity_allowed", "must_reject")
        },
        "bundle": str(bundle),
        "dataset_map_artifacts": dataset_map_artifacts,
        "manifest": str(manifest),
        "raw_first_stamp_ns": effective_raw_first_stamp_ns,
        "negative_kind": case.get("negative_kind"),
        "output": str(case_output),
        "bundle_verification": bundle_verification,
        "evaluator_command": command,
        "evaluator_exit_code": completed.returncode,
        "algorithm_lock": validated["locked"],
        "history_ever_authoritative_lock": frame_history[
            "ever_authoritative_lock"
        ],
        "lock_edge_count": frame_history["lock_edge_count"],
        "final_state": result["final_state"],
        "pose_source": result["final_pose_source"],
        "pose": validated["pose"],
        "lock_frame_index": validated["lock_frame"],
        "lock_stamp_ns": validated["lock_stamp"],
        "seed_keyframe_id": validated["seed"],
        "support_keyframe_id": validated["support"],
        "performance": {
            "strict_p95_ms": recomputed_p95,
            "strict_max_ms": recomputed_max,
            "search_phase_p95_ms": validated["search_phase_p95_ms"],
            "acquisition_to_lock_s": validated["acquisition_s"],
            "runtime_peak_rss_kib": measured_resources["peak_rss_kib"],
            "runtime_swap_operations": measured_resources[
                "swap_operations"
            ],
            "evaluator_reported_peak_rss_kib": validated["rss_kib"],
            "evaluator_reported_swap_operations": validated[
                "swap_operations"
            ],
        },
        "baseline_score": baseline,
        "reference_trajectory_score": reference_score,
        "reference_evidence": reference_evidence,
        "raw_lio_replay_evidence": raw_lio_replay_evidence,
        "semantic_pass": semantic_pass,
        "semantic_failure_reasons": semantic_reasons,
        "resource_gates": resource,
        "pass": case_pass,
        "checksums": checksums,
    }
    return report, strict_times, primary_accuracy


def aggregate(
    case_reports: list[dict[str, Any]],
    strict_times: list[float],
    primary_accuracy: list[dict[str, Any]],
) -> dict[str, Any]:
    positives = [
        case for case in case_reports if case["intent"]["required_lock"]
    ]
    negatives = [
        case for case in case_reports if case["intent"]["must_reject"]
    ]
    ambiguous = [
        case for case in case_reports if case["intent"]["ambiguity_allowed"]
    ]
    correct_positives = sum(case["semantic_pass"] for case in positives)
    false_locks = sum(
        case["history_ever_authoritative_lock"] for case in negatives
    )
    global_p95 = percentile95(strict_times)
    peak_rss = max(
        case["performance"]["runtime_peak_rss_kib"] for case in case_reports
    )
    swap_operations = sum(
        case["performance"]["runtime_swap_operations"] for case in case_reports
    )
    locked_positive_acquisition = [
        case["performance"]["acquisition_to_lock_s"]
        for case in positives
        if case["algorithm_lock"]
    ]
    all_positives_locked = len(locked_positive_acquisition) == len(positives)
    maximum_acquisition = (
        max(locked_positive_acquisition) if locked_positive_acquisition else None
    )
    acquisition_gate = (
        all_positives_locked
        and maximum_acquisition is not None
        and maximum_acquisition < ACQUISITION_TO_LOCK_LIMIT_S
    )
    accuracy = {
        "evaluated_positive_cases": len(primary_accuracy),
        "translation_rmse_m": None,
        "translation_max_m": None,
        "yaw_mean_deg": None,
        "yaw_max_deg": None,
        "roll_max_deg": None,
        "pitch_max_deg": None,
    }
    if primary_accuracy:
        translations = [
            score["translation_error_m"] for score in primary_accuracy
        ]
        yaws = [score["yaw_error_deg"] for score in primary_accuracy]
        accuracy.update(
            {
                "translation_rmse_m": math.sqrt(
                    statistics.fmean(value * value for value in translations)
                ),
                "translation_max_m": max(translations),
                "yaw_mean_deg": statistics.fmean(yaws),
                "yaw_max_deg": max(yaws),
                "roll_max_deg": max(
                    score["roll_error_deg"] for score in primary_accuracy
                ),
                "pitch_max_deg": max(
                    score["pitch_error_deg"] for score in primary_accuracy
                ),
            }
        )
    gates = {
        "all_case_semantics_pass": all(
            case["semantic_pass"] for case in case_reports
        ),
        "all_case_resource_gates_pass": all(
            case["pass"] for case in case_reports
        ),
        "positive_cases_have_accuracy_evidence": len(primary_accuracy)
        == len(positives),
        "cpu_strict_p95_under_1000_ms": global_p95 < STRICT_P95_LIMIT_MS,
        "required_lock_acquisition_under_5_s": acquisition_gate,
        "peak_rss_under_600_mib": peak_rss < PEAK_RSS_LIMIT_KIB,
        "swap_operations_zero": swap_operations == 0,
        "negative_false_locks_zero": false_locks == 0,
    }
    return {
        "case_counts": {
            "total": len(case_reports),
            "positive": len(positives),
            "negative": len(negatives),
            "ambiguous": len(ambiguous),
            "correct_positive": correct_positives,
            "negative_false_locks": false_locks,
        },
        "product_positive_success_rate": correct_positives / len(positives)
        if positives
        else None,
        "negative_false_lock_rate": false_locks / len(negatives)
        if negatives
        else None,
        "accuracy": accuracy,
        "performance": {
            "cpu_strict_p95_ms": global_p95,
            "required_lock_acquisition_max_s": maximum_acquisition,
            "peak_rss_kib": peak_rss,
            "swap_operations": swap_operations,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def initialize_output(path: Path) -> None:
    if path.is_symlink():
        raise GateError(f"output must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise GateError(f"output exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise GateError(f"output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)
    (path / "cases").mkdir()


def run_preflight_verifier(
    preflight_tool: Path, action: str, artifact: Path
) -> dict[str, Any]:
    require_regular_file(artifact, f"{action} preflight artifact")
    command = command_for_python_or_binary(preflight_tool)
    command.extend([action, "--artifact", str(artifact)])
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except OSError as error:
        raise GateError(
            f"cannot start Product V1 preflight verifier: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError(
            f"{action} Product V1 preflight verification failed: {detail}"
        )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def verify_product_preflights(
    identity: dict[str, Any],
    evaluator: Path,
    evaluator_identity: dict[str, Any],
    preflight_tool: Path,
) -> dict[str, Any]:
    require_regular_file(preflight_tool, "Product V1 preflight verifier")
    if sha256_file(preflight_tool) != identity["preflight_tool_sha256"]:
        raise GateError(
            "Product V1 preflight verifier SHA-256 does not match candidate "
            "run JSON"
        )
    runtime_path = identity["runtime_preflight_json"]
    authority_path = identity["authority_preflight_json"]
    for path, expected, label in (
        (
            runtime_path,
            identity["runtime_preflight_sha256"],
            "runtime preflight",
        ),
        (
            authority_path,
            identity["authority_preflight_sha256"],
            "authority preflight",
        ),
    ):
        require_regular_file(path, label)
        if sha256_file(path) != expected:
            raise GateError(f"{label} SHA-256 mismatch")

    runtime_verification = run_preflight_verifier(
        preflight_tool, "runtime-verify", runtime_path
    )
    authority_verification = run_preflight_verifier(
        preflight_tool, "authority-verify", authority_path
    )
    runtime = load_json(runtime_path)
    authority = load_json(authority_path)
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema")
        != "n3mapping_product_runtime_preflight_v1"
        or runtime.get("status") != "PASS"
        or runtime.get("candidate_commit") != identity["candidate_commit"]
        or runtime.get("product_profile_sha256")
        != evaluator_identity["product_profile_sha256"]
    ):
        raise GateError(
            "runtime preflight identity does not match the frozen candidate"
        )
    runtime_node = runtime.get("node")
    runtime_evaluator = runtime.get("evaluator")
    if not isinstance(runtime_node, dict) or not isinstance(
        runtime_evaluator, dict
    ):
        raise GateError("runtime preflight binary evidence is invalid")
    node_libraries = runtime_node.get("product_libraries")
    evaluator_libraries = runtime_evaluator.get("product_libraries")
    if (
        runtime_evaluator.get("path") != str(evaluator.resolve())
        or runtime_evaluator.get("sha256") != sha256_file(evaluator)
        or evaluator_libraries
        != evaluator_identity["linked_product_libraries"]
        or not isinstance(node_libraries, dict)
        or node_libraries.get("libn3mapping_core.so")
        != evaluator_libraries.get("libn3mapping_core.so")
    ):
        raise GateError(
            "runtime preflight evaluator/core identity does not match the "
            "candidate Gate evaluator"
        )
    runtime_bundles = runtime.get("bundles")
    if not isinstance(runtime_bundles, list):
        raise GateError("runtime preflight Bundle evidence is invalid")
    audited_bundle_paths = {
        str(Path(str(bundle.get("path", ""))).resolve())
        for bundle in runtime_bundles
        if isinstance(bundle, dict)
    }
    expected_bundle_paths = {
        str(path.resolve()) for path in identity["bundles"].values()
    }
    if (
        len(audited_bundle_paths) != len(runtime_bundles)
        or audited_bundle_paths != expected_bundle_paths
    ):
        raise GateError(
            "runtime preflight does not audit exactly the candidate Bundle set"
        )
    for bundle in runtime_bundles:
        bundle_path = Path(bundle["path"]).resolve()
        manifest_path = bundle_path / "manifest.json"
        if (
            bundle.get("manifest_sha256") != sha256_file(manifest_path)
            or load_json(manifest_path).get("runtime_node_sha256")
            != runtime_node.get("sha256")
        ):
            raise GateError(
                "runtime preflight Bundle/runtime node identity mismatch"
            )

    if (
        not isinstance(authority, dict)
        or authority.get("schema")
        != ACTIVE_RUNTIME_AUTHORITY_SCHEMA
        or authority.get("evidence_kind") != "active_runtime"
        or authority.get("status") != "PASS"
        or authority.get("candidate_commit") != identity["candidate_commit"]
        or authority.get("product_profile_sha256")
        != evaluator_identity["product_profile_sha256"]
    ):
        raise GateError(
            "active-runtime authority preflight is missing or does not match "
            "the frozen candidate"
        )
    source_files = authority.get("source_files")
    authority_node = (
        source_files.get("runtime_node")
        if isinstance(source_files, dict)
        else None
    )
    if (
        not isinstance(authority_node, dict)
        or authority_node.get("sha256") != runtime_node.get("sha256")
    ):
        raise GateError(
            "authority preflight did not observe the audited runtime node"
        )
    wrapper_by_distro = {
        "humble": "libn3mapping_humble_wrapper.so",
        "noetic": "libn3mapping_noetic_wrapper.so",
    }
    distro = authority.get("distro")
    if (
        distro not in wrapper_by_distro
        or wrapper_by_distro[distro] not in node_libraries
    ):
        raise GateError(
            "authority preflight distro does not match the audited runtime "
            "node wrapper"
        )
    return {
        "status": "PASS",
        "tool_sha256": identity["preflight_tool_sha256"],
        "runtime": {
            "path": str(runtime_path),
            "sha256": identity["runtime_preflight_sha256"],
            "verification": runtime_verification,
        },
        "authority": {
            "path": str(authority_path),
            "sha256": identity["authority_preflight_sha256"],
            "distro": distro,
            "verification": authority_verification,
        },
    }


def run_gate(
    candidate_run_path: Path,
    dataset_path: Path,
    expected_dataset_sha256: str,
    evaluator: Path,
    bundle_tool: Path,
    preflight_tool: Path,
    output: Path,
) -> bool:
    initialize_output(output)
    report_path = output / "gate_result.json"
    started_at = now_shanghai()
    progress: dict[str, Any] = {
        "schema": GATE_RESULT_SCHEMA,
        "status": "INCOMPLETE",
        "pass": False,
        "product_verdict": "INCOMPLETE",
        "started_at": started_at,
        "finished_at": None,
        "invocation_argv": [sys.executable, *sys.argv],
        "candidate_run": str(candidate_run_path),
        "dataset_contract": str(dataset_path),
        "expected_dataset_sha256": expected_dataset_sha256,
        "evaluator": str(evaluator),
        "bundle_tool": str(bundle_tool),
        "preflight_tool": str(preflight_tool),
        "completed_cases": [],
        "error": None,
    }
    write_json(report_path, progress)
    try:
        require_regular_file(evaluator, "explicit evaluator")
        require_regular_file(bundle_tool, "Product Bundle verifier")
        require_regular_file(preflight_tool, "Product V1 preflight verifier")
        identity, cases = parse_gate_inputs(
            candidate_run_path,
            dataset_path,
            expected_dataset_sha256,
        )
        evaluator_sha256 = sha256_file(evaluator)
        bundle_tool_sha256 = sha256_file(bundle_tool)
        if evaluator_sha256 != identity["evaluator_sha256"]:
            raise GateError(
                "explicit evaluator SHA-256 does not match candidate run JSON"
            )
        if bundle_tool_sha256 != identity["bundle_tool_sha256"]:
            raise GateError(
                "Product Bundle verifier SHA-256 does not match candidate "
                "run JSON"
            )
        try:
            evaluator_identity = read_product_build_identity(evaluator)
        except ProductIdentityError as error:
            raise GateError(str(error)) from error
        if evaluator_identity["commit"] != identity["candidate_commit"]:
            raise GateError(
                "evaluator build commit does not match candidate run JSON"
            )
        if (
            evaluator_identity["linked_product_libraries"]
            != identity["evaluator_linked_product_libraries"]
        ):
            raise GateError(
                "evaluator linked product libraries do not match candidate "
                "run JSON"
            )
        preflights = verify_product_preflights(
            identity,
            evaluator,
            evaluator_identity,
            preflight_tool,
        )
        input_checksums = {
            "candidate_run_sha256": sha256_file(candidate_run_path),
            "dataset_contract_sha256": identity[
                "dataset_contract_sha256"
            ],
            "externally_expected_dataset_sha256": expected_dataset_sha256,
            "gate_tool_sha256": sha256_file(Path(__file__).resolve()),
            "candidate_commit": identity["candidate_commit"],
            "evaluator_sha256": evaluator_sha256,
            "bundle_tool_sha256": bundle_tool_sha256,
            "preflight_tool_sha256": sha256_file(preflight_tool),
            "runtime_preflight_sha256": identity[
                "runtime_preflight_sha256"
            ],
            "authority_preflight_sha256": identity[
                "authority_preflight_sha256"
            ],
            "product_profile_sha256": evaluator_identity[
                "product_profile_sha256"
            ],
            "evaluator_build_identity": evaluator_identity,
        }
        all_strict_times: list[float] = []
        primary_accuracy: list[dict[str, Any]] = []
        for case in cases:
            case_report, strict_times, case_accuracy = run_case(
                case,
                evaluator,
                bundle_tool,
                output,
                identity["candidate_commit"],
                evaluator_identity["product_profile_sha256"],
                evaluator_identity["linked_product_libraries"],
            )
            progress["completed_cases"].append(case_report)
            all_strict_times.extend(strict_times)
            if case["required_lock"] and case_accuracy is not None:
                primary_accuracy.append(case_accuracy)
            write_json(report_path, progress)
        summary = aggregate(
            progress["completed_cases"], all_strict_times, primary_accuracy
        )
        complete = {
            "schema": GATE_RESULT_SCHEMA,
            "status": "COMPLETE",
            "pass": summary["pass"],
            "started_at": started_at,
            "finished_at": now_shanghai(),
            "invocation_argv": progress["invocation_argv"],
            "candidate_run": str(candidate_run_path),
            "dataset_contract": str(dataset_path),
            "dataset_id": identity["dataset_id"],
            "dataset_revision": identity["dataset_revision"],
            "evaluator": str(evaluator),
            "bundle_tool": str(bundle_tool),
            "preflight_tool": str(preflight_tool),
            "preflights": preflights,
            "acceptance_limits": {
                "translation_m_inclusive": TRANSLATION_LIMIT_M,
                "yaw_deg_inclusive": YAW_LIMIT_DEG,
                "roll_pitch_deg_inclusive": ROLL_PITCH_LIMIT_DEG,
                "cpu_strict_p95_ms_exclusive": STRICT_P95_LIMIT_MS,
                "acquisition_to_lock_s_exclusive": ACQUISITION_TO_LOCK_LIMIT_S,
                "peak_rss_kib_exclusive": PEAK_RSS_LIMIT_KIB,
                "swap_operations": 0,
            },
            "input_checksums": input_checksums,
            "cases": progress["completed_cases"],
            "summary": summary,
            "product_verdict": "V1_PASS" if summary["pass"] else "V1_FAIL",
            "error": None,
        }
        write_json(report_path, complete)
        return summary["pass"]
    except (GateError, OSError) as error:
        progress["error"] = str(error)
        progress["finished_at"] = now_shanghai()
        write_json(report_path, progress)
        raise


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Run the n3mapping CPU relocalization Product Gate V1."
    )
    command.add_argument("--candidate-run", type=Path, required=True)
    command.add_argument("--dataset-contract", type=Path, required=True)
    command.add_argument("--expected-dataset-sha256", required=True)
    command.add_argument("--evaluator", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--bundle-tool",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "n3mapping_product_bundle.py"
        ),
        help="Product Bundle verifier (override is intended for tests only)",
    )
    command.add_argument(
        "--preflight-tool",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "n3mapping_product_preflight.py"
        ),
        help="Product V1 preflight verifier (override is for tests only)",
    )
    return command


def main() -> int:
    args = parser().parse_args()
    candidate_run = args.candidate_run.resolve()
    dataset_contract = args.dataset_contract.resolve()
    evaluator = args.evaluator.resolve()
    bundle_tool = args.bundle_tool.resolve()
    preflight_tool = args.preflight_tool.resolve()
    output = args.output.resolve()
    try:
        passed = run_gate(
            candidate_run,
            dataset_contract,
            args.expected_dataset_sha256,
            evaluator,
            bundle_tool,
            preflight_tool,
            output,
        )
    except (GateError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    result = {
        "output": str(output),
        "status": "passed" if passed else "failed",
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
