#!/usr/bin/env python3
"""Validate an n3mapping evaluation run against the version-2 artifact contract.

The frozen contract uses JSON syntax in a ``.yaml`` file. JSON is a strict YAML
subset and keeps this tool dependency-free and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = 2
ODOM_SOURCES = {
    "oracle_gt_odom",
    "synthetic_drift_odom",
    "recorded_lio_odom",
    "robot_odom",
}
QUERY_SOURCES = {
    "global_map_render",
    "local_submap_render",
    "same_keyframe",
    "independent_real_scan",
}
EVIDENCE_CLASSES = {
    "synthetic_map_render",
    "oracle_dataset",
    "recorded_session",
    "recorded_cross_session",
}
QUERY_OUTCOMES = {
    "RETRIEVAL_EMPTY",
    "GT_BASIN_NOT_IN_TOPK",
    "WRONG_TOP_BASIN",
    "ICP_NOT_CONVERGED",
    "ICP_QUALITY_REJECT",
    "OBSERVABILITY_DEFER",
    "TEMPORAL_DEFER",
    "CORRECT_REGION_ONLY",
    "CORRECT_FULL_LOCK",
    "FALSE_FULL_LOCK",
    "TRACKING_DEGRADED",
    "TRACKING_LOST",
    "INTERNAL_ERROR",
}
ATTEMPT_OUTCOMES = {"correct", "false", "timeout"}
BASE_REQUIRED_ARTIFACTS = {
    "manifest.json",
    "resolved_config.yaml",
    "dataset_manifest.json",
    "frozen_contract.yaml",
    "command.txt",
    "environment.json",
    "stdout.log",
    "summary/metrics.json",
    "checksums.sha256",
    "COMPLETE",
}
RELOCALIZATION_REQUIRED_ARTIFACTS = {
    "raw/relocalization_queries.csv",
    "raw/relocalization_attempts.csv",
    "summary/failure_counts.json",
}
HASHED_MANIFEST_ARTIFACTS = {
    "resolved_config_sha256": "resolved_config.yaml",
    "dataset_manifest_sha256": "dataset_manifest.json",
    "frozen_contract_sha256": "frozen_contract.yaml",
}
NONFINITE_TOKENS = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
RUNTIME_ARTIFACTS = {
    "raw/runtime_events.jsonl",
    "raw/relocalization_debug.jsonl",
    "raw/state_transitions.jsonl",
    "raw/stage_timing.csv",
}
FORBIDDEN_RUNTIME_KEYS = {
    "query_outcome",
    "pose_success",
    "lock_correct",
    "false_lock",
    "translation_error_m",
    "yaw_error_deg",
    "gt_pose",
    "ground_truth",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_RE = re.compile(r"^([0-9A-Fa-f]{64}) [ *](.+)$")
FINGERPRINT_ALGORITHM = "fnv1a64_xyz_intensity_float32_le_v1"
PRODUCT_EVAL_PROFILE = "product_default"
FIXED_SYNTHETIC_FRAME_PERIOD_S = 1.0
REQUIRED_TEMPORAL_WINDOW_SIZE = 5
REQUIRED_MINIMUM_QUERY_POINTS = 100
REQUIRED_MINIMUM_OCCUPIED_RAY_BINS = 100


class DuplicateKeyError(ValueError):
    pass


class ValidationReport:
    def __init__(self, run_dir: Path, strict: bool, pre_complete: bool = False) -> None:
        self.run_dir = run_dir
        self.strict = strict
        self.pre_complete = pre_complete
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.checks: list[str] = []

    def error(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})

    def warning(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})

    def checked(self, name: str) -> None:
        self.checks.append(name)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "strict": self.strict,
            "pre_complete": self.pre_complete,
            "validation_phase": "PAYLOAD_PREFLIGHT" if self.pre_complete else "COMPLETE_RUN",
            "valid": self.valid,
            "verdict": "VALID" if self.valid else "INVALID",
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": sorted(set(self.checks)),
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def load_json_strict(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )


def load_contract(path: Path) -> dict[str, Any]:
    try:
        value = load_json_strict(path)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must use JSON-compatible YAML syntax; parse error: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _require_key(report: ValidationReport, obj: dict[str, Any], key: str, owner: str) -> Any:
    if key not in obj:
        report.error("missing_field", f"{owner} is missing required field {key!r}")
        return None
    return obj[key]


def _as_nonnegative_int(report: ValidationReport, value: Any, name: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        report.error("invalid_integer", f"{name} must be a non-negative integer, got {value!r}")
        return None
    return value


def _as_optional_rate(report: ValidationReport, value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        report.error("invalid_rate", f"{name} must be a finite number or null, got {value!r}")
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        report.error("invalid_rate", f"{name} must be within [0, 1], got {value!r}")
        return None
    return number


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _walk_json(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else key
            yield child_path, child
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{prefix}[{index}]"
            yield child_path, child
            yield from _walk_json(child, child_path)


def _validate_no_nonfinite_json(report: ValidationReport, value: Any, owner: str) -> None:
    for field, child in _walk_json(value):
        if isinstance(child, float) and not math.isfinite(child):
            report.error("nonfinite_json", f"{owner}:{field} is non-finite")


def _load_csv(report: ValidationReport, path: Path, relative: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                report.error("csv_header", f"{relative} has no CSV header")
                return [], []
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                report.error("csv_header", f"{relative} has duplicate CSV columns")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        report.error("csv_parse", f"failed to parse {relative}: {exc}")
        return [], []
    for row_index, row in enumerate(rows, start=2):
        if None in row:
            report.error("csv_shape", f"{relative}:{row_index} has extra columns")
        for column, raw in row.items():
            if raw is not None and raw.strip().lower() in NONFINITE_TOKENS:
                report.error(
                    "nonfinite_csv",
                    f"{relative}:{row_index}:{column} contains non-finite token {raw!r}",
                )
    return list(reader.fieldnames or []), rows


def _load_required_json(
    report: ValidationReport,
    run_dir: Path,
    relative: str,
) -> Any | None:
    path = run_dir / relative
    try:
        value = load_json_strict(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.error("json_parse", f"failed to parse {relative}: {exc}")
        return None
    _validate_no_nonfinite_json(report, value, relative)
    return value


def _validate_manifest(
    report: ValidationReport,
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[set[str], str | None]:
    required_fields = (
        "schema_version",
        "run_id",
        "mode",
        "repo_sha",
        "branch",
        "dirty",
        "start_time_asia_shanghai",
        "end_time_asia_shanghai",
        "command",
        "exact_argv",
        "exit_code",
        "dataset_id",
        "odom_id",
        "gt_id",
        "odom_source",
        "query_source",
        "gt_runtime_access",
        "evidence_class",
        "candidate_budget",
        "icp_budget",
        "cpu_model",
        "thread_count",
        "memory_max_bytes",
        "memory_swap_max_bytes",
        "evaluator_version",
        "analyzer_version",
        "label_thresholds",
        "feature_flags",
        "resolved_config_sha256",
        "dataset_manifest_sha256",
        "frozen_contract_sha256",
        "required_artifacts",
        "complete",
    )
    for field in required_fields:
        _require_key(report, manifest, field, "manifest.json")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        report.error(
            "schema_version",
            f"manifest schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}",
        )
    if contract.get("schema_version") != SCHEMA_VERSION:
        report.error(
            "schema_version",
            f"frozen contract schema_version must be {SCHEMA_VERSION}, got {contract.get('schema_version')!r}",
        )
    if manifest.get("complete") is not True:
        report.error("incomplete_manifest", "manifest complete must be true")
    if manifest.get("exit_code") != 0:
        report.error("exit_code", f"manifest exit_code must be 0, got {manifest.get('exit_code')!r}")
    if not isinstance(manifest.get("exact_argv"), list) or not all(
        isinstance(item, str) for item in manifest.get("exact_argv", [])
    ):
        report.error("argv", "manifest exact_argv must be a list of strings")
    if not isinstance(manifest.get("dirty"), bool):
        report.error("dirty", "manifest dirty must be boolean")
    for field in ("thread_count", "memory_max_bytes", "memory_swap_max_bytes"):
        _as_nonnegative_int(report, manifest.get(field), f"manifest.{field}")
    for field in ("candidate_budget", "icp_budget", "label_thresholds", "feature_flags"):
        if not isinstance(manifest.get(field), dict):
            report.error("manifest_type", f"manifest {field} must be an object")
    if not isinstance(manifest.get("command"), str) or not manifest.get("command", "").strip():
        report.error("command", "manifest command must be a non-empty string")
    for field in ("start_time_asia_shanghai", "end_time_asia_shanghai"):
        value = manifest.get(field)
        if not isinstance(value, str) or not (value.endswith("+08:00") or value.endswith("+0800")):
            report.error("timezone", f"manifest {field} must explicitly use UTC+8")
    odom_source = manifest.get("odom_source")
    if odom_source not in ODOM_SOURCES:
        report.error("odom_source", f"unsupported odom_source {odom_source!r}")
        odom_source = None
    if manifest.get("query_source") not in QUERY_SOURCES:
        report.error("query_source", f"unsupported query_source {manifest.get('query_source')!r}")
    evidence_class = manifest.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        report.error("evidence_class", f"unsupported evidence_class {evidence_class!r}")
    if odom_source == "recorded_lio_odom" and manifest.get("gt_runtime_access") is not False:
        report.error(
            "gt_isolation",
            "recorded_lio_odom requires gt_runtime_access=false",
        )
    if not isinstance(manifest.get("gt_runtime_access"), bool):
        report.error("gt_isolation", "manifest gt_runtime_access must be boolean")
    if report.strict and evidence_class == "recorded_cross_session":
        if (
            manifest.get("odom_source") != "recorded_lio_odom"
            or manifest.get("query_source") != "independent_real_scan"
            or manifest.get("gt_runtime_access") is not False
        ):
            report.error(
                "formal_evidence_contract",
                "recorded_cross_session requires recorded_lio_odom, independent_real_scan, "
                "and gt_runtime_access=false",
            )
    for field in HASHED_MANIFEST_ARTIFACTS:
        value = manifest.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            report.error("manifest_hash", f"manifest {field} must be a lowercase SHA256")

    manifest_required = manifest.get("required_artifacts")
    contract_required = contract.get("required_artifacts")
    required: set[str] = set()
    if not isinstance(manifest_required, list):
        report.error("required_artifacts", "manifest required_artifacts must be a list")
        manifest_required = []
    if not isinstance(contract_required, list):
        report.error("required_artifacts", "frozen contract required_artifacts must be a list")
        contract_required = []
    for owner, values in (("manifest", manifest_required), ("contract", contract_required)):
        safe_values: list[str] = []
        for value in values:
            safe = _safe_relative_path(value)
            if safe is None:
                report.error("unsafe_artifact_path", f"{owner} has unsafe artifact path {value!r}")
            else:
                safe_values.append(safe)
        if len(safe_values) != len(set(safe_values)):
            report.error("duplicate_artifact", f"{owner} required_artifacts contains duplicates")
        if owner == "manifest":
            required = set(safe_values)
        elif required != set(safe_values):
            report.error(
                "required_artifacts_mismatch",
                "manifest and frozen contract required_artifacts must match exactly",
            )
    missing_base = sorted(BASE_REQUIRED_ARTIFACTS - required)
    if missing_base:
        report.error("required_artifacts", f"required_artifacts omits base files: {missing_base}")
    if manifest.get("mode") == "relocalization":
        missing_reloc = sorted(RELOCALIZATION_REQUIRED_ARTIFACTS - required)
        if missing_reloc:
            report.error(
                "required_artifacts",
                f"relocalization required_artifacts omits: {missing_reloc}",
            )
    return required, odom_source


def _validate_artifact_files(
    report: ValidationReport,
    run_dir: Path,
    required: set[str],
) -> None:
    root = run_dir.resolve()
    for relative in sorted(required):
        path = run_dir / relative
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            report.error("artifact_path", f"cannot resolve {relative}: {exc}")
            continue
        if root != resolved and root not in resolved.parents:
            report.error("artifact_escape", f"artifact escapes run directory: {relative}")
            continue
        if path.is_symlink():
            report.error("artifact_symlink", f"required artifact must not be a symlink: {relative}")
        if relative == "COMPLETE" and report.pre_complete:
            if path.exists():
                report.error(
                    "precomplete_has_complete",
                    "payload preflight requires COMPLETE to be absent",
                )
            continue
        if not path.is_file():
            report.error("missing_artifact", f"missing required artifact: {relative}")
    complete = run_dir / "COMPLETE"
    if complete.is_file() and complete.stat().st_size > 4096:
        report.error("complete_sentinel", "COMPLETE sentinel is unexpectedly large")


def _validate_checksums(
    report: ValidationReport,
    run_dir: Path,
    required: set[str],
    manifest: dict[str, Any],
) -> None:
    checksum_path = run_dir / "checksums.sha256"
    if not checksum_path.is_file():
        return
    entries: dict[str, str] = {}
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        report.error("checksums_read", f"failed to read checksums.sha256: {exc}")
        return
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            report.error("checksums_format", f"checksums.sha256:{line_number} is malformed")
            continue
        digest, raw_relative = match.groups()
        relative = _safe_relative_path(raw_relative)
        if relative is None:
            report.error(
                "unsafe_checksum_path",
                f"checksums.sha256:{line_number} has unsafe path {raw_relative!r}",
            )
            continue
        if relative in entries:
            report.error("duplicate_checksum", f"duplicate checksum entry for {relative}")
            continue
        entries[relative] = digest.lower()
    expected_entries = required - {"checksums.sha256", "COMPLETE"}
    missing = sorted(expected_entries - set(entries))
    if missing:
        report.error("missing_checksum", f"required artifacts missing checksum entries: {missing}")
    unexpected = sorted(set(entries) - expected_entries)
    if unexpected:
        report.error("unexpected_checksum", f"checksum file contains undeclared artifacts: {unexpected}")
    for relative, expected in sorted(entries.items()):
        path = run_dir / relative
        if not path.is_file() or path.is_symlink():
            continue
        actual = sha256_file(path)
        if actual != expected:
            report.error(
                "checksum_mismatch",
                f"checksum mismatch for {relative}: expected {expected}, got {actual}",
            )
    for manifest_field, relative in HASHED_MANIFEST_ARTIFACTS.items():
        path = run_dir / relative
        expected = manifest.get(manifest_field)
        if path.is_file() and isinstance(expected, str) and SHA256_RE.fullmatch(expected):
            actual = sha256_file(path)
            if actual != expected:
                report.error(
                    "manifest_hash_mismatch",
                    f"manifest {manifest_field} does not match {relative}",
                )
            checksum_value = entries.get(relative)
            if checksum_value is not None and checksum_value != expected:
                report.error(
                    "manifest_checksum_disagree",
                    f"manifest and checksums.sha256 disagree for {relative}",
                )


def _validate_declared_contents(
    report: ValidationReport,
    run_dir: Path,
    required: set[str],
) -> dict[str, tuple[list[str], list[dict[str, str]]]]:
    csv_data: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for relative in sorted(required):
        path = run_dir / relative
        if not path.is_file() or path.is_symlink():
            continue
        if relative.endswith(".json"):
            _load_required_json(report, run_dir, relative)
        elif relative.endswith(".jsonl"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                report.error("jsonl_read", f"failed to read {relative}: {exc}")
                continue
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_json_constant,
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    report.error("jsonl_parse", f"{relative}:{line_number}: {exc}")
                    continue
                _validate_no_nonfinite_json(report, value, f"{relative}:{line_number}")
        elif relative.endswith(".csv"):
            csv_data[relative] = _load_csv(report, path, relative)
    return csv_data


def _validate_runtime_gt_isolation(
    report: ValidationReport,
    run_dir: Path,
    required: set[str],
    csv_data: dict[str, tuple[list[str], list[dict[str, str]]]],
) -> None:
    for relative in sorted(required & RUNTIME_ARTIFACTS):
        path = run_dir / relative
        if not path.is_file():
            continue
        if relative.endswith(".csv"):
            headers = csv_data.get(relative, ([], []))[0]
            for header in headers:
                lowered = header.lower()
                if lowered in FORBIDDEN_RUNTIME_KEYS or lowered.startswith("gt_"):
                    report.error(
                        "runtime_gt_leak",
                        f"runtime artifact {relative} contains forbidden column {header!r}",
                    )
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (ValueError, json.JSONDecodeError):
                continue
            for key_path, _ in _walk_json(value):
                leaf = re.split(r"[.\[]", key_path)[-1].rstrip("]").lower()
                if leaf in FORBIDDEN_RUNTIME_KEYS or leaf.startswith("gt_"):
                    report.error(
                        "runtime_gt_leak",
                        f"runtime artifact {relative}:{line_number} contains forbidden key {key_path!r}",
                    )


def _validate_stage_timing(
    report: ValidationReport,
    csv_data: dict[str, tuple[list[str], list[dict[str, str]]]],
) -> None:
    relative = "raw/stage_timing.csv"
    if relative not in csv_data:
        return
    headers, rows = csv_data[relative]
    duration_columns = [name for name in headers if name.endswith("_ms")]
    for row_index, row in enumerate(rows, start=2):
        for column in duration_columns:
            raw = (row.get(column) or "").strip()
            if raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                report.error("stage_timing", f"{relative}:{row_index}:{column} is not numeric")
                continue
            if not math.isfinite(value) or value < 0.0:
                report.error(
                    "stage_timing",
                    f"{relative}:{row_index}:{column} must be finite and non-negative",
                )
        start_raw = (row.get("start_ns") or "").strip()
        end_raw = (row.get("end_ns") or "").strip()
        if start_raw and end_raw:
            try:
                if int(end_raw) < int(start_raw):
                    report.error("stage_timing", f"{relative}:{row_index} end_ns precedes start_ns")
            except ValueError:
                report.error("stage_timing", f"{relative}:{row_index} has invalid start_ns/end_ns")


def _contract_number(
    report: ValidationReport,
    value: Any,
    owner: str,
    *,
    nonnegative: bool = True,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        report.error("synthetic_contract", f"{owner} must be numeric")
        return None
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        report.error("synthetic_contract", f"{owner} must be finite and non-negative")
        return None
    return number


def _csv_nonnegative_int(raw: str, owner: str, report: ValidationReport) -> int | None:
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        report.error("synthetic_visibility", f"{owner} must be an integer")
        return None
    if value < 0:
        report.error("synthetic_visibility", f"{owner} must be non-negative")
        return None
    return value


def _validate_synthetic_strict_contract(
    report: ValidationReport,
    run_dir: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    required: set[str],
    csv_data: dict[str, tuple[list[str], list[dict[str, str]]]],
) -> None:
    if not report.strict or manifest.get("evidence_class") != "synthetic_map_render":
        return

    if manifest.get("gt_runtime_access") is not True:
        report.error("synthetic_honesty", "synthetic_map_render requires gt_runtime_access=true")
    if manifest.get("verdict_ceiling") != "SHADOW_ONLY":
        report.error("synthetic_honesty", "synthetic_map_render requires verdict_ceiling=SHADOW_ONLY")
    feature_flags = manifest.get("feature_flags")
    if not isinstance(feature_flags, dict):
        feature_flags = {}
    expected_flags = {
        "odom_derived_from_gt": True,
        "synthetic_from_target_map": True,
        "gt_passed_to_localizer": False,
    }
    for field, expected in expected_flags.items():
        if feature_flags.get(field) is not expected:
            report.error(
                "synthetic_honesty",
                f"synthetic_map_render requires feature_flags.{field}={str(expected).lower()}",
            )

    required_profile = contract.get("required_eval_profile")
    if required_profile != PRODUCT_EVAL_PROFILE:
        report.error(
            "synthetic_eval_profile",
            "synthetic frozen contract must freeze required_eval_profile=product_default",
        )
    required_period = _contract_number(
        report,
        contract.get("required_frame_period_s"),
        "frozen_contract.required_frame_period_s",
    )
    if required_period is not None and not math.isclose(
        required_period,
        FIXED_SYNTHETIC_FRAME_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        report.error(
            "synthetic_frame_period",
            "synthetic frozen contract required_frame_period_s must equal 1.0",
        )
    required_temporal_window = contract.get("required_temporal_window_size")
    if (
        isinstance(required_temporal_window, bool)
        or not isinstance(required_temporal_window, int)
        or required_temporal_window != REQUIRED_TEMPORAL_WINDOW_SIZE
    ):
        report.error(
            "synthetic_temporal_window",
            "synthetic frozen contract required_temporal_window_size must equal integer 5",
        )
    visibility_thresholds = {
        "minimum_query_points": contract.get("minimum_query_points"),
        "minimum_occupied_ray_bins": contract.get("minimum_occupied_ray_bins"),
    }
    expected_visibility_thresholds = {
        "minimum_query_points": REQUIRED_MINIMUM_QUERY_POINTS,
        "minimum_occupied_ray_bins": REQUIRED_MINIMUM_OCCUPIED_RAY_BINS,
    }
    for field, expected in expected_visibility_thresholds.items():
        value = visibility_thresholds[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            report.error(
                "synthetic_visibility_threshold",
                f"synthetic frozen contract {field} must equal integer {expected}",
            )

    config: dict[str, Any] = {}
    try:
        config_value = load_contract(run_dir / "resolved_config.yaml")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.error("resolved_config_parse", f"failed to parse resolved_config.yaml: {exc}")
    else:
        config = config_value
    if config:
        if config.get("eval_profile") != required_profile:
            report.error(
                "synthetic_eval_profile",
                "resolved_config eval_profile does not match frozen required_eval_profile",
            )
        if config.get("reloc_temporal_window_size") != required_temporal_window:
            report.error(
                "synthetic_temporal_window",
                "resolved_config reloc_temporal_window_size differs from the frozen window",
            )
        if config.get("required_temporal_window_size") != required_temporal_window:
            report.error(
                "synthetic_temporal_window",
                "resolved_config required_temporal_window_size differs from the frozen window",
            )
        for field, expected in expected_visibility_thresholds.items():
            if config.get(field) != expected:
                report.error(
                    "synthetic_visibility_threshold",
                    f"resolved_config {field} differs from the frozen value {expected}",
                )
        for owner, value in (
            ("manifest.synthetic_frame_period_s", manifest.get("synthetic_frame_period_s")),
            ("resolved_config.synthetic_frame_period_s", config.get("synthetic_frame_period_s")),
        ):
            number = _contract_number(report, value, owner)
            if number is not None and not math.isclose(
                number,
                FIXED_SYNTHETIC_FRAME_PERIOD_S,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                report.error("synthetic_frame_period", f"{owner} must equal 1.0")
        config_expectations = {
            "evidence_class": "synthetic_map_render",
            "odom_source": "oracle_gt_odom",
            "gt_runtime_access": True,
            "verdict": "SHADOW_ONLY",
        }
        for field, expected in config_expectations.items():
            if config.get(field) != expected:
                report.error(
                    "synthetic_honesty",
                    f"resolved_config.{field} must be {expected!r}",
                )

    candidate_budget = manifest.get("candidate_budget")
    if not isinstance(candidate_budget, dict) or (
        candidate_budget.get("reloc_temporal_window_size") != required_temporal_window
    ):
        report.error(
            "synthetic_temporal_window",
            "manifest candidate_budget.reloc_temporal_window_size differs from the frozen window",
        )
    if manifest.get("required_temporal_window_size") != required_temporal_window:
        report.error(
            "synthetic_temporal_window",
            "manifest required_temporal_window_size differs from the frozen window",
        )

    threshold_fields = {
        "pose_translation_threshold_m",
        "pose_yaw_threshold_deg",
        "pose_roll_pitch_threshold_deg",
    }
    required_thresholds = contract.get("required_label_thresholds")
    if not isinstance(required_thresholds, dict) or set(required_thresholds) != threshold_fields:
        report.error(
            "synthetic_label_thresholds",
            "synthetic contract must freeze the three pose label thresholds exactly",
        )
        required_thresholds = {}
    manifest_thresholds = manifest.get("label_thresholds")
    for field in sorted(threshold_fields):
        frozen = _contract_number(
            report,
            required_thresholds.get(field),
            f"frozen_contract.required_label_thresholds.{field}",
        )
        if frozen is None:
            continue
        for owner, container in (
            ("manifest.label_thresholds", manifest_thresholds),
            ("resolved_config", config),
        ):
            value = container.get(field) if isinstance(container, dict) else None
            observed = _contract_number(report, value, f"{owner}.{field}")
            if observed is not None and not math.isclose(
                observed,
                frozen,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                report.error(
                    "synthetic_label_thresholds",
                    f"{owner}.{field} differs from the frozen threshold",
                )

    synthetic_requirements = contract.get("synthetic_requirements", {})
    if not isinstance(synthetic_requirements, dict):
        report.error("synthetic_contract", "synthetic_requirements must be an object")
        synthetic_requirements = {}
    require_fingerprint = synthetic_requirements.get(
        "require_query_cloud_fingerprint_fnv1a64",
        False,
    )
    require_raycast = synthetic_requirements.get("require_raycast_visibility", False)
    for field, value in (
        ("require_query_cloud_fingerprint_fnv1a64", require_fingerprint),
        ("require_raycast_visibility", require_raycast),
    ):
        if not isinstance(value, bool):
            report.error("synthetic_contract", f"synthetic_requirements.{field} must be boolean")

    resolved_fingerprints: dict[str, tuple[bool, str, str]] = {}
    resolved_relative = "raw/resolved_queries.csv"
    if require_fingerprint is True:
        if resolved_relative not in required:
            report.error(
                "synthetic_fingerprint",
                "contract-required FNV evidence must declare raw/resolved_queries.csv",
            )
        headers, rows = csv_data.get(resolved_relative, ([], []))
        fingerprint_fields = {
            "query_id",
            "query_cloud_fingerprint_available",
            "query_cloud_fingerprint_algorithm",
            "query_cloud_fingerprint_fnv1a64",
        }
        missing = sorted(fingerprint_fields - set(headers))
        if missing:
            report.error(
                "synthetic_fingerprint",
                f"raw/resolved_queries.csv is missing fingerprint fields: {missing}",
            )
        else:
            for row_index, row in enumerate(rows, start=2):
                query_id = row.get("query_id", "").strip()
                available = _parse_bool(row.get("query_cloud_fingerprint_available", ""))
                algorithm = row.get("query_cloud_fingerprint_algorithm", "").strip()
                fingerprint = row.get("query_cloud_fingerprint_fnv1a64", "").strip()
                if not query_id or query_id in resolved_fingerprints:
                    report.error(
                        "synthetic_fingerprint",
                        f"{resolved_relative}:{row_index} has empty or duplicate query_id",
                    )
                    continue
                if available is None:
                    report.error(
                        "synthetic_fingerprint",
                        f"{resolved_relative}:{row_index} fingerprint availability must be boolean",
                    )
                if algorithm != FINGERPRINT_ALGORITHM:
                    report.error(
                        "synthetic_fingerprint",
                        f"{resolved_relative}:{row_index} has the wrong fingerprint algorithm",
                    )
                if re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None:
                    report.error(
                        "synthetic_fingerprint",
                        f"{resolved_relative}:{row_index} fingerprint must be 16 lowercase hex",
                    )
                if available is not None:
                    resolved_fingerprints[query_id] = (available, algorithm, fingerprint)
        if manifest.get("query_identity_capability") != "per_query_fnv1a64":
            report.error(
                "synthetic_fingerprint",
                "synthetic FNV contract requires query_identity_capability=per_query_fnv1a64",
            )

    visibility_relative = "raw/renderer_visibility.csv"
    if require_raycast is True or require_fingerprint is True:
        if visibility_relative not in required:
            report.error(
                "synthetic_visibility",
                "synthetic visibility requirements must declare raw/renderer_visibility.csv",
            )
        headers, rows = csv_data.get(visibility_relative, ([], []))
        count_fields = {
            "input_map_points",
            "finite_points",
            "range_eligible_points",
            "fov_eligible_points",
            "azimuth_bins",
            "vertical_bins",
            "occupied_ray_bins",
            "same_ray_occluded_points",
            "dropout_suppressed_points",
            "occlusion_suppressed_bins",
            "query_points_before_voxel",
            "num_query_points",
        }
        required_fields = {"query_id", "generation_valid", "raycast_enabled"} | count_fields
        if require_fingerprint is True:
            required_fields |= {
                "query_cloud_fingerprint_available",
                "query_cloud_fingerprint_algorithm",
                "query_cloud_fingerprint_fnv1a64",
            }
        missing = sorted(required_fields - set(headers))
        if missing:
            report.error(
                "synthetic_visibility",
                f"raw/renderer_visibility.csv is missing fields: {missing}",
            )
        else:
            visibility_fingerprints: dict[str, tuple[bool, str, str]] = {}
            visibility_validity: dict[str, bool] = {}
            for row_index, row in enumerate(rows, start=2):
                owner = f"{visibility_relative}:{row_index}"
                query_id = row.get("query_id", "").strip()
                generation_valid = _parse_bool(row.get("generation_valid", ""))
                raycast_enabled = _parse_bool(row.get("raycast_enabled", ""))
                if generation_valid is None or raycast_enabled is None:
                    report.error(
                        "synthetic_visibility",
                        f"{owner} generation_valid/raycast_enabled must be boolean",
                    )
                    continue
                if not query_id or query_id in visibility_validity:
                    report.error(
                        "synthetic_visibility",
                        f"{owner} has empty or duplicate query_id",
                    )
                else:
                    visibility_validity[query_id] = generation_valid
                if require_raycast is True and generation_valid and not raycast_enabled:
                    report.error(
                        "synthetic_visibility",
                        f"{owner} valid Gate query must have raycast_enabled=true",
                    )
                counts = {
                    field: _csv_nonnegative_int(row.get(field, ""), f"{owner}.{field}", report)
                    for field in count_fields
                }
                if any(value is None for value in counts.values()):
                    continue
                values = {field: int(value) for field, value in counts.items()}
                if generation_valid:
                    if values["num_query_points"] < REQUIRED_MINIMUM_QUERY_POINTS:
                        report.error(
                            "synthetic_visibility_threshold",
                            f"{owner} valid query has fewer than 100 query points",
                        )
                    if (
                        values["occupied_ray_bins"]
                        < REQUIRED_MINIMUM_OCCUPIED_RAY_BINS
                    ):
                        report.error(
                            "synthetic_visibility_threshold",
                            f"{owner} valid query has fewer than 100 occupied ray bins",
                        )
                if not (
                    values["input_map_points"]
                    >= values["finite_points"]
                    >= values["range_eligible_points"]
                    >= values["fov_eligible_points"]
                ):
                    report.error("synthetic_visibility", f"{owner} visibility counts are not monotonic")
                if raycast_enabled:
                    expected_occupied = (
                        values["fov_eligible_points"] - values["same_ray_occluded_points"]
                    )
                    expected_prevoxel = (
                        values["occupied_ray_bins"]
                        - values["dropout_suppressed_points"]
                        - values["occlusion_suppressed_bins"]
                    )
                    if (
                        values["azimuth_bins"] <= 0
                        or values["vertical_bins"] <= 0
                        or values["occupied_ray_bins"]
                        > values["azimuth_bins"] * values["vertical_bins"]
                        or values["occupied_ray_bins"] != expected_occupied
                        or values["query_points_before_voxel"] != expected_prevoxel
                    ):
                        report.error(
                            "synthetic_visibility",
                            f"{owner} raycast visibility counts are not conserved",
                        )
                if values["num_query_points"] > values["query_points_before_voxel"]:
                    report.error(
                        "synthetic_visibility",
                        f"{owner} voxel output exceeds pre-voxel points",
                    )
                if require_fingerprint is True:
                    available = _parse_bool(row["query_cloud_fingerprint_available"])
                    fingerprint_tuple = (
                        available,
                        row["query_cloud_fingerprint_algorithm"].strip(),
                        row["query_cloud_fingerprint_fnv1a64"].strip(),
                    )
                    if not query_id or query_id in visibility_fingerprints or available is None:
                        report.error(
                            "synthetic_fingerprint",
                            f"{owner} has invalid or duplicate fingerprint identity",
                        )
                    else:
                        visibility_fingerprints[query_id] = fingerprint_tuple  # type: ignore[assignment]
            if require_fingerprint is True and visibility_fingerprints != resolved_fingerprints:
                report.error(
                    "synthetic_fingerprint",
                    "renderer_visibility and resolved_queries per-query FNV evidence differ",
                )
            query_rows = csv_data.get("raw/relocalization_queries.csv", ([], []))[1]
            invalid_eligible_ids = sorted(
                {
                    row.get("query_id", "").strip()
                    for row in query_rows
                    if visibility_validity.get(row.get("query_id", "").strip()) is not True
                }
            )
            if invalid_eligible_ids:
                report.error(
                    "synthetic_visibility_threshold",
                    "generation-invalid or unknown queries entered the eligible denominator: "
                    f"{invalid_eligible_ids}",
                )
            renderer_metrics_value = _load_required_json(
                report,
                run_dir,
                "summary/renderer_metrics.json",
            )
            if isinstance(renderer_metrics_value, dict):
                expected_counts = {
                    "planned_render_query_count": len(visibility_validity),
                    "valid_render_query_count": sum(visibility_validity.values()),
                    "invalid_render_query_count": len(visibility_validity)
                    - sum(visibility_validity.values()),
                    "eligible_query_count": len(query_rows),
                }
                for field, expected in expected_counts.items():
                    if renderer_metrics_value.get(field) != expected:
                        report.error(
                            "synthetic_visibility_threshold",
                            f"renderer_metrics.{field} does not match raw artifacts",
                        )

    if require_raycast is True and config:
        if config.get("visibility_model") != "map_conditioned_point_zbuffer_v1":
            report.error(
                "synthetic_visibility",
                "resolved_config visibility_model must be map_conditioned_point_zbuffer_v1",
            )
        for field in ("raycast_azimuth_resolution_deg", "raycast_vertical_resolution_deg"):
            number = _contract_number(report, config.get(field), f"resolved_config.{field}")
            if number is not None and number <= 0.0:
                report.error("synthetic_visibility", f"resolved_config.{field} must be positive")


def _validate_cross_artifact_semantics(
    report: ValidationReport,
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if not report.strict:
        return
    try:
        config = load_contract(run_dir / "resolved_config.yaml")
    except (OSError, ValueError, json.JSONDecodeError):
        return
    dataset_value = _load_required_json(report, run_dir, "dataset_manifest.json")
    dataset = dataset_value if isinstance(dataset_value, dict) else {}
    semantic_fields = (
        "odom_source",
        "query_source",
        "evidence_class",
        "gt_runtime_access",
    )
    for field in semantic_fields:
        expected = manifest.get(field)
        for owner, artifact in (("metrics", metrics), ("resolved_config", config), ("dataset", dataset)):
            if field not in artifact:
                report.error(
                    "cross_artifact_semantics",
                    f"{owner} is missing required semantic field {field}",
                )
            elif artifact.get(field) != expected:
                report.error(
                    "cross_artifact_semantics",
                    f"{owner}.{field} differs from manifest.{field}",
                )
    if dataset.get("dataset_id") != manifest.get("dataset_id"):
        report.error(
            "cross_artifact_semantics",
            "dataset_manifest.dataset_id differs from manifest.dataset_id",
        )
    if manifest.get("evidence_class") == "recorded_cross_session":
        feature_flags = manifest.get("feature_flags")
        expected_flags = {
            "gt_passed_to_localizer": False,
            "odom_derived_from_gt": False,
            "synthetic_from_target_map": False,
        }
        if not isinstance(feature_flags, dict):
            feature_flags = {}
        for field, expected in expected_flags.items():
            if feature_flags.get(field) is not expected:
                report.error(
                    "formal_evidence_contract",
                    f"formal evidence requires feature_flags.{field}={str(expected).lower()}",
                )


def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    return None


def _validate_relocalization_accounting(
    report: ValidationReport,
    metrics: dict[str, Any],
    failure_counts: dict[str, Any] | None,
    csv_data: dict[str, tuple[list[str], list[dict[str, str]]]],
) -> None:
    query_relative = "raw/relocalization_queries.csv"
    attempt_relative = "raw/relocalization_attempts.csv"
    if query_relative not in csv_data or attempt_relative not in csv_data:
        return
    query_headers, query_rows = csv_data[query_relative]
    attempt_headers, attempt_rows = csv_data[attempt_relative]
    for column in ("query_id", "attempt_id", "query_outcome", "correct_lock_within_10s"):
        if column not in query_headers:
            report.error("query_schema", f"{query_relative} is missing {column}")
    for column in ("attempt_id", "attempt_outcome", "ever_false_lock"):
        if column not in attempt_headers:
            report.error("attempt_schema", f"{attempt_relative} is missing {column}")
    if report.errors and any(item["code"] in {"query_schema", "attempt_schema"} for item in report.errors):
        return

    query_ids = [row.get("query_id", "").strip() for row in query_rows]
    attempt_ids = [row.get("attempt_id", "").strip() for row in attempt_rows]
    if any(not value for value in query_ids):
        report.error("query_id", "query_id must be non-empty")
    if any(not value for value in attempt_ids):
        report.error("attempt_id", "attempt_id must be non-empty")
    if len(query_ids) != len(set(query_ids)):
        report.error("duplicate_query_id", "relocalization query_id values must be unique")
    if len(attempt_ids) != len(set(attempt_ids)):
        report.error("duplicate_attempt_id", "relocalization attempt_id values must be unique")
    attempt_id_set = set(attempt_ids)
    unknown_attempts = sorted(
        {row.get("attempt_id", "").strip() for row in query_rows} - attempt_id_set
    )
    if unknown_attempts:
        report.error("unknown_attempt_id", f"queries reference unknown attempts: {unknown_attempts}")

    query_counts = Counter(row.get("query_outcome", "").strip() for row in query_rows)
    unknown_outcomes = sorted(set(query_counts) - QUERY_OUTCOMES)
    if unknown_outcomes:
        report.error("query_outcome", f"unknown query outcomes: {unknown_outcomes}")
    per_attempt_query_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_attempt_timely_correct: Counter[str] = Counter()
    for row_index, row in enumerate(query_rows, start=2):
        attempt_id = row.get("attempt_id", "").strip()
        query_outcome = row.get("query_outcome", "").strip()
        within_deadline = _parse_bool(row.get("correct_lock_within_10s", ""))
        if within_deadline is None:
            report.error(
                "query_schema",
                f"{query_relative}:{row_index} correct_lock_within_10s must be boolean",
            )
            within_deadline = False
        if within_deadline and query_outcome != "CORRECT_FULL_LOCK":
            report.error(
                "attempt_evidence",
                f"{query_relative}:{row_index} only CORRECT_FULL_LOCK may be within deadline",
            )
        per_attempt_query_counts[attempt_id][query_outcome] += 1
        if within_deadline:
            per_attempt_timely_correct[attempt_id] += 1

    attempt_counts: Counter[str] = Counter()
    expected_timely_correct_attempts = 0
    for row_index, row in enumerate(attempt_rows, start=2):
        attempt_id = row.get("attempt_id", "").strip()
        outcome = row.get("attempt_outcome", "").strip().lower()
        if outcome not in ATTEMPT_OUTCOMES:
            report.error(
                "attempt_outcome",
                f"{attempt_relative}:{row_index} has unknown attempt_outcome {outcome!r}",
            )
        else:
            attempt_counts[outcome] += 1
        ever_false = _parse_bool(row.get("ever_false_lock", ""))
        if ever_false is None:
            report.error(
                "attempt_schema",
                f"{attempt_relative}:{row_index} ever_false_lock must be boolean",
            )
        evidence = per_attempt_query_counts[attempt_id]
        expected_ever_false = evidence["FALSE_FULL_LOCK"] > 0
        if expected_ever_false:
            expected_outcome = "false"
        elif per_attempt_timely_correct[attempt_id] > 0:
            expected_outcome = "correct"
            expected_timely_correct_attempts += 1
        else:
            expected_outcome = "timeout"
        if outcome in ATTEMPT_OUTCOMES and outcome != expected_outcome:
            report.error(
                "attempt_evidence",
                f"{attempt_relative}:{row_index} outcome {outcome!r} disagrees with query evidence "
                f"{expected_outcome!r}",
            )
        if ever_false is not None and ever_false != expected_ever_false:
            report.error(
                "false_lock_sticky",
                f"{attempt_relative}:{row_index} ever_false_lock disagrees with query evidence",
            )

    metric_counts: dict[str, int | None] = {}
    for field in (
        "attempt_count",
        "correct_attempt_count",
        "false_attempt_count",
        "timeout_attempt_count",
        "eligible_query_count",
        "correct_lock_event_count",
        "false_lock_event_count",
        "correct_lock_by_10s_count",
    ):
        metric_counts[field] = _as_nonnegative_int(report, metrics.get(field), f"metrics.{field}")
    attempt_count = metric_counts["attempt_count"]
    correct_attempts = metric_counts["correct_attempt_count"]
    false_attempts = metric_counts["false_attempt_count"]
    timeout_attempts = metric_counts["timeout_attempt_count"]
    eligible_queries = metric_counts["eligible_query_count"]
    if None not in (attempt_count, correct_attempts, false_attempts, timeout_attempts):
        if attempt_count != correct_attempts + false_attempts + timeout_attempts:
            report.error(
                "attempt_invariant",
                "attempt_count must equal correct + false + timeout attempts",
            )
        if attempt_count != len(attempt_rows):
            report.error("attempt_count", "metrics attempt_count does not match attempt CSV rows")
        expected_attempt_counts = {
            "correct": correct_attempts,
            "false": false_attempts,
            "timeout": timeout_attempts,
        }
        if any(attempt_counts[name] != expected for name, expected in expected_attempt_counts.items()):
            report.error("attempt_count", "attempt CSV outcomes do not match metrics counts")
    if eligible_queries is not None and eligible_queries != len(query_rows):
        report.error("query_count", "metrics eligible_query_count does not match query CSV rows")

    declared_query_counts = metrics.get("query_outcome_counts")
    if not isinstance(declared_query_counts, dict):
        report.error("query_counts", "metrics query_outcome_counts must be an object")
    else:
        normalized: dict[str, int] = {}
        for outcome, value in declared_query_counts.items():
            count = _as_nonnegative_int(report, value, f"metrics.query_outcome_counts.{outcome}")
            if count is not None:
                normalized[outcome] = count
        if set(normalized) - QUERY_OUTCOMES:
            report.error("query_outcome", "metrics contains unknown query_outcome keys")
        if normalized != dict(query_counts):
            report.error("query_counts", "metrics query_outcome_counts does not match query CSV")
        if eligible_queries is not None and sum(normalized.values()) != eligible_queries:
            report.error(
                "query_invariant",
                "eligible_query_count must equal the sum of mutually exclusive query outcomes",
            )
        if failure_counts is not None:
            failure_outcomes = failure_counts.get("query_outcome_counts")
            if failure_outcomes != declared_query_counts:
                report.error(
                    "failure_counts",
                    "failure_counts.json query_outcome_counts must match metrics",
                )

    correct_events = metric_counts["correct_lock_event_count"]
    false_events = metric_counts["false_lock_event_count"]
    if correct_events is not None and correct_events != query_counts["CORRECT_FULL_LOCK"]:
        report.error(
            "lock_event_count",
            "metrics correct_lock_event_count does not match CORRECT_FULL_LOCK query rows",
        )
    if false_events is not None and false_events != query_counts["FALSE_FULL_LOCK"]:
        report.error(
            "lock_event_count",
            "metrics false_lock_event_count does not match FALSE_FULL_LOCK query rows",
        )
    lock_precision = _as_optional_rate(report, metrics.get("lock_precision"), "metrics.lock_precision")
    false_given_lock = _as_optional_rate(
        report,
        metrics.get("false_lock_given_lock"),
        "metrics.false_lock_given_lock",
    )
    if correct_events is not None and false_events is not None:
        event_count = correct_events + false_events
        if event_count == 0:
            if lock_precision is not None or false_given_lock is not None:
                report.error(
                    "zero_denominator",
                    "lock precision metrics must be null when there are no lock events",
                )
        else:
            expected_precision = correct_events / event_count
            expected_false = false_events / event_count
            if lock_precision is None or not _close(lock_precision, expected_precision):
                report.error("lock_precision", "lock_precision does not match lock event counts")
            if false_given_lock is None or not _close(false_given_lock, expected_false):
                report.error(
                    "false_lock_given_lock",
                    "false_lock_given_lock does not match lock event counts",
                )

    correct_by_deadline_count = metric_counts["correct_lock_by_10s_count"]
    correct_by_deadline = _as_optional_rate(
        report,
        metrics.get("correct_lock_by_10s"),
        "metrics.correct_lock_by_10s",
    )
    false_attempt_rate = _as_optional_rate(
        report,
        metrics.get("false_lock_attempt_rate"),
        "metrics.false_lock_attempt_rate",
    )
    if attempt_count is not None and attempt_count > 0:
        if correct_by_deadline_count is not None:
            if correct_by_deadline_count != expected_timely_correct_attempts:
                report.error(
                    "correct_lock_by_10s",
                    "metrics correct_lock_by_10s_count does not match deadline query evidence",
                )
            expected = correct_by_deadline_count / attempt_count
            if correct_by_deadline is None or not _close(correct_by_deadline, expected):
                report.error("correct_lock_by_10s", "correct_lock_by_10s has the wrong denominator")
        if false_attempts is not None:
            expected = false_attempts / attempt_count
            if false_attempt_rate is None or not _close(false_attempt_rate, expected):
                report.error(
                    "false_lock_attempt_rate",
                    "false_lock_attempt_rate has the wrong denominator",
                )
        if correct_events == 0 and false_events == 0 and correct_by_deadline != 0.0:
            report.error(
                "all_reject_utility",
                "an all-reject run must report correct_lock_by_10s=0",
            )
    elif attempt_count == 0:
        if correct_by_deadline is not None or false_attempt_rate is not None:
            report.error("zero_denominator", "attempt rates must be null when attempt_count is zero")


def validate_run(
    run: Path | str,
    strict: bool = False,
    pre_complete: bool = False,
) -> ValidationReport:
    run_dir = Path(run).expanduser().resolve()
    report = ValidationReport(run_dir, strict, pre_complete=pre_complete)
    if not run_dir.is_dir():
        report.error("run_dir", f"run directory does not exist: {run_dir}")
        return report

    manifest_value = _load_required_json(report, run_dir, "manifest.json")
    if not isinstance(manifest_value, dict):
        if manifest_value is not None:
            report.error("manifest_type", "manifest.json must contain an object")
        return report
    manifest = manifest_value
    try:
        contract = load_contract(run_dir / "frozen_contract.yaml")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.error("contract_parse", f"failed to parse frozen_contract.yaml: {exc}")
        return report

    required, odom_source = _validate_manifest(report, manifest, contract)
    _validate_artifact_files(report, run_dir, required)
    _validate_checksums(report, run_dir, required, manifest)
    csv_data = _validate_declared_contents(report, run_dir, required)
    _validate_runtime_gt_isolation(report, run_dir, required, csv_data)
    _validate_stage_timing(report, csv_data)
    _validate_synthetic_strict_contract(
        report,
        run_dir,
        manifest,
        contract,
        required,
        csv_data,
    )

    metrics_value = _load_required_json(report, run_dir, "summary/metrics.json")
    metrics = metrics_value if isinstance(metrics_value, dict) else None
    if metrics_value is not None and metrics is None:
        report.error("metrics_type", "summary/metrics.json must contain an object")
    failure_value: Any | None = None
    if "summary/failure_counts.json" in required and (run_dir / "summary/failure_counts.json").is_file():
        failure_value = _load_required_json(report, run_dir, "summary/failure_counts.json")
        if failure_value is not None and not isinstance(failure_value, dict):
            report.error("failure_counts_type", "summary/failure_counts.json must contain an object")
            failure_value = None
    if metrics is not None:
        if metrics.get("schema_version") != SCHEMA_VERSION:
            report.error("schema_version", "metrics schema_version must be 2")
        if metrics.get("odom_source") != odom_source:
            report.error("odom_source_mismatch", "manifest and metrics odom_source differ")
        if metrics.get("query_source") != manifest.get("query_source"):
            report.error("query_source_mismatch", "manifest and metrics query_source differ")
        if metrics.get("evidence_class") != manifest.get("evidence_class"):
            report.error("evidence_class_mismatch", "manifest and metrics evidence_class differ")
        if metrics.get("mode") != manifest.get("mode"):
            report.error("mode_mismatch", "manifest and metrics mode differ")
        _validate_cross_artifact_semantics(report, run_dir, manifest, metrics)
        if manifest.get("mode") == "relocalization":
            _validate_relocalization_accounting(
                report,
                metrics,
                failure_value if isinstance(failure_value, dict) else None,
                csv_data,
            )

    command_path = run_dir / "command.txt"
    if command_path.is_file() and isinstance(manifest.get("command"), str):
        try:
            command_text = command_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            report.error("command_read", f"failed to read command.txt: {exc}")
        else:
            if command_text != manifest["command"].strip():
                report.error("command_mismatch", "command.txt does not match manifest command")

    if strict:
        report.checked("strict_schema_v2")
    if pre_complete:
        report.checked("payload_preflight_without_complete")
    report.checked("required_artifacts")
    report.checked("checksums")
    report.checked("finite_json_csv")
    report.checked("odom_gt_isolation")
    report.checked("relocalization_accounting")
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Evaluation run directory")
    parser.add_argument("--strict", action="store_true", help="Enable the complete v2 contract")
    parser.add_argument(
        "--pre-complete",
        "--payload",
        dest="pre_complete",
        action="store_true",
        help="Validate the complete payload before the atomic COMPLETE sentinel is created",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report destination")
    args = parser.parse_args(argv)

    if args.pre_complete and not args.strict:
        parser.error("--pre-complete/--payload requires --strict")
    result = validate_run(
        args.run,
        strict=args.strict,
        pre_complete=args.pre_complete,
    ).to_dict()
    if args.report:
        _write_report(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
