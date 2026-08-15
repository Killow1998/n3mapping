#!/usr/bin/env python3
"""Fail-closed FA-01 tri-mode runtime and quality gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable


CONTRACT_SCHEMA = "n3mapping_final_acceptance_contract_v1"
QUALITY_SCHEMA = "n3mapping_fa01_quality_reference_v1"
RUN_SCHEMA = "n3mapping_runtime_performance_run_v1"
SUMMARY_SCHEMA = "n3mapping_runtime_performance_summary_v2"
REPORT_SCHEMA = "n3mapping_fa01_acceptance_v1"
PASS = "PASS"
FAIL_PERFORMANCE = "FAIL_FA01_PERFORMANCE"
FAIL_QUALITY = "FAIL_FA01_QUALITY"
FAIL_MULTIPLE = "FAIL_FA01_MULTIPLE"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {
    PASS: 0,
    FAIL_PERFORMANCE: 1,
    FAIL_QUALITY: 2,
    FAIL_MULTIPLE: 2,
    INVALID: 3,
}
MODES = ("mapping", "localization", "map_extension")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STEADY_SELECTION = {
    "mapping": "all_valid_synchronized_frame_callbacks",
    "localization": (
        "tracking_callbacks_after_first_FULL_6DOF_LOCKED_"
        "no_additional_warmup"
    ),
    "map_extension": (
        "tracking_callbacks_after_first_FULL_6DOF_LOCKED_"
        "no_additional_warmup"
    ),
}
MAX_ISSUES = 100


class DuplicateKeyError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []
        self.total = 0
        self.by_category: Counter[str] = Counter()

    def add(self, category: str, code: str, message: str) -> None:
        self.total += 1
        self.by_category[category] += 1
        if len(self.items) < MAX_ISSUES:
            self.items.append(
                {"category": category, "code": code, "message": message}
            )

    def count(self, category: str) -> int:
        return self.by_category[category]


def load_runtime_states(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
            if not isinstance(value, dict) or not isinstance(
                value.get("record_type"), str
            ):
                raise ValueError(f"invalid runtime record on line {line_number}")
            if value["record_type"] != "runtime_frame":
                continue
            state = value.get("relocalization_state")
            if not isinstance(state, str):
                raise ValueError(
                    f"runtime state missing on line {line_number}"
                )
            counts[state] += 1
    return counts


def load_proto_module(proto_path: Path) -> Any:
    with tempfile.TemporaryDirectory(prefix="n3mapping_fa01_proto_") as temporary:
        generated = Path(temporary)
        process = subprocess.run(
            [
                "protoc",
                "-I",
                str(proto_path.parent),
                f"--python_out={generated}",
                str(proto_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"protoc failed: {detail}")
        module_path = generated / f"{proto_path.stem}_pb2.py"
        spec = importlib.util.spec_from_file_location(
            f"n3mapping_fa01_{hashlib.sha256(str(proto_path).encode()).hexdigest()[:12]}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load generated module {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def parse_map(path: Path, module: Any) -> Any:
    proto = module.N3Map()
    proto.ParseFromString(path.read_bytes())
    return proto


def map_structure(map_proto: Any, module: Any) -> dict[str, Any]:
    odometry = sum(
        edge.type == module.EdgeProto.ODOMETRY for edge in map_proto.edges
    )
    loops = sum(edge.type == module.EdgeProto.LOOP for edge in map_proto.edges)
    anchors = sum(
        edge.type == module.EdgeProto.SESSION_ANCHOR
        for edge in map_proto.edges
    )
    keyframe_ids = {keyframe.id for keyframe in map_proto.keyframes}
    duplicate = len(map_proto.keyframes) - len(keyframe_ids)
    dangling = sum(
        edge.from_id not in keyframe_ids or edge.to_id not in keyframe_ids
        for edge in map_proto.edges
    )
    metadata = map_proto.metadata
    metadata_match = (
        metadata.num_keyframes == len(map_proto.keyframes)
        and metadata.num_odometry_edges == odometry
        and metadata.num_loop_edges == loops
        and metadata.num_session_anchor_edges == anchors
    )
    return {
        "version": metadata.version,
        "keyframes": len(map_proto.keyframes),
        "edges": len(map_proto.edges),
        "odometry": odometry,
        "loop": loops,
        "session_anchor": anchors,
        "unknown_edges": len(map_proto.edges) - odometry - loops - anchors,
        "dense_trajectory": len(map_proto.dense_optimized_trajectory),
        "metadata_match": metadata_match,
        "duplicate_keyframes": duplicate,
        "dangling_edges": dangling,
        "session_anchors": [
            [edge.from_id, edge.to_id]
            for edge in map_proto.edges
            if edge.type == module.EdgeProto.SESSION_ANCHOR
        ],
    }


def pose_delta(reference: Any, candidate: Any) -> tuple[float, float]:
    translation = math.sqrt(
        (reference.tx - candidate.tx) ** 2
        + (reference.ty - candidate.ty) ** 2
        + (reference.tz - candidate.tz) ** 2
    )
    lhs = (reference.qx, reference.qy, reference.qz, reference.qw)
    rhs = (candidate.qx, candidate.qy, candidate.qz, candidate.qw)
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return translation, math.inf
    dot = abs(sum(a * b for a, b in zip(lhs, rhs)) / (lhs_norm * rhs_norm))
    rotation = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
    return translation, rotation


def default_map_inspector(
    proto_path: Path,
    reference_map: Path,
    mapping_map: Path,
    extension_map: Path,
) -> dict[str, Any]:
    module = load_proto_module(proto_path)
    reference = parse_map(reference_map, module)
    mapping = parse_map(mapping_map, module)
    extension = parse_map(extension_map, module)
    reference_poses = {
        keyframe.id: keyframe.pose_optimized for keyframe in reference.keyframes
    }
    extension_poses = {
        keyframe.id: keyframe.pose_optimized for keyframe in extension.keyframes
    }
    translations: list[float] = []
    rotations: list[float] = []
    missing_ids: list[int] = []
    for keyframe_id, pose in reference_poses.items():
        candidate = extension_poses.get(keyframe_id)
        if candidate is None:
            missing_ids.append(keyframe_id)
            continue
        translation, rotation = pose_delta(pose, candidate)
        translations.append(translation)
        rotations.append(rotation)
    return {
        "reference_map_sha256": sha256_file(reference_map),
        "mapping": {
            "output_map_sha256": sha256_file(mapping_map),
            "structure": map_structure(mapping, module),
        },
        "map_extension": {
            "output_map_sha256": sha256_file(extension_map),
            "structure": map_structure(extension, module),
            "loaded_pose_delta": {
                "compared_keyframes": len(translations),
                "missing_keyframe_ids": missing_ids,
                "translation_m": {
                    "p50": percentile(translations, 0.50),
                    "p95": percentile(translations, 0.95),
                    "max": max(translations, default=None),
                },
                "rotation_rad": {
                    "p50": percentile(rotations, 0.50),
                    "p95": percentile(rotations, 0.95),
                    "max": max(rotations, default=None),
                },
            },
        },
    }


MapInspector = Callable[[Path, Path, Path, Path], dict[str, Any]]


def compare_structure(
    mode: str,
    observed: Any,
    expected: Any,
    issues: Issues,
) -> None:
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        issues.add("evidence", "invalid_map_structure", f"{mode} map structure missing")
        return
    for field, value in expected.items():
        if observed.get(field) != value:
            issues.add(
                "quality",
                "map_structure_mismatch",
                f"{mode}.{field}={observed.get(field)!r}, expected={value!r}",
            )


def evaluate(
    contract: dict[str, Any],
    quality: dict[str, Any],
    run_dirs: dict[str, Path],
    reference_map: Path,
    proto_path: Path,
    map_inspector: MapInspector = default_map_inspector,
) -> dict[str, Any]:
    issues = Issues()
    if contract.get("schema") != CONTRACT_SCHEMA:
        issues.add("evidence", "contract_schema", "acceptance contract schema mismatch")
    if quality.get("schema") != QUALITY_SCHEMA:
        issues.add("evidence", "quality_schema", "quality reference schema mismatch")
    fa01 = contract.get("fa01")
    if not isinstance(fa01, dict) or fa01.get("modes") != list(MODES):
        issues.add("evidence", "contract_modes", "FA-01 modes are not frozen tri-mode order")
        fa01 = {}
    thresholds = fa01.get("steady_state_thresholds", {})
    inputs = fa01.get("inputs", {})

    runs: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        directory = run_dirs.get(mode)
        if directory is None:
            issues.add("evidence", "missing_run", f"{mode} run directory missing")
            continue
        try:
            manifest_path = directory / "run_manifest.json"
            summary_path = directory / "performance_summary.json"
            runtime_path = directory / "runtime_performance_debug.jsonl"
            if not (directory / "COMPLETE").is_file():
                raise ValueError("COMPLETE marker missing")
            manifest = load_json(manifest_path)
            summary = load_json(summary_path)
            states = load_runtime_states(runtime_path)
            runs[mode] = {
                "directory": directory,
                "manifest_path": manifest_path,
                "summary_path": summary_path,
                "manifest": manifest,
                "summary": summary,
                "states": states,
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.add("evidence", "run_read", f"{mode}: {error}")

    common_commit: str | None = None
    common_profile: str | None = None
    common_node: Any = None
    common_bag: Any = None
    common_config: Any = None
    common_rate: Any = None
    common_offset: Any = None
    loaded_map_fingerprint: Any = None
    mode_reports: dict[str, Any] = {}

    for mode in MODES:
        if mode not in runs:
            continue
        run = runs[mode]
        manifest = run["manifest"]
        summary = run["summary"]
        if manifest.get("schema") != RUN_SCHEMA:
            issues.add("evidence", "run_schema", f"{mode} run schema mismatch")
        if manifest.get("status") != "COMPLETE":
            issues.add("evidence", "run_status", f"{mode} run is not COMPLETE")
        if manifest.get("mode") != mode or summary.get("mode") != mode:
            issues.add("evidence", "mode_mismatch", f"{mode} evidence mode mismatch")
        if manifest.get("performance_status") != "PROFILE_READY" or summary.get("status") != "PROFILE_READY":
            issues.add("evidence", "profile_status", f"{mode} profile is not ready")
        if summary.get("schema") != SUMMARY_SCHEMA:
            issues.add("evidence", "summary_schema", f"{mode} summary schema mismatch")
        return_codes = manifest.get("process_return_codes")
        if not isinstance(return_codes, dict) or any(value != 0 for value in return_codes.values()):
            issues.add("evidence", "process_return_code", f"{mode} has non-zero process return code")

        identity = manifest.get("node_build_identity")
        if not isinstance(identity, dict):
            issues.add("evidence", "build_identity", f"{mode} build identity missing")
            identity = {}
        commit = identity.get("commit")
        profile = identity.get("product_profile_sha256")
        if (
            not isinstance(commit, str)
            or not COMMIT_RE.fullmatch(commit)
            or manifest.get("expected_commit") != commit
            or identity.get("build_type") != "Release"
            or identity.get("research_tools") != "OFF"
            or identity.get("verified") is not True
        ):
            issues.add("evidence", "build_identity", f"{mode} build identity is not exact verified Release")
        if not isinstance(profile, str) or not SHA256_RE.fullmatch(profile):
            issues.add("evidence", "profile_identity", f"{mode} product profile identity invalid")
        if common_commit is None:
            common_commit = commit
            common_profile = profile
            common_node = manifest.get("node_executable")
            common_bag = manifest.get("bag")
            common_config = manifest.get("config")
            common_rate = manifest.get("rate")
            common_offset = manifest.get("start_offset_s")
        else:
            for code, observed, expected in (
                ("commit_mismatch", commit, common_commit),
                ("profile_mismatch", profile, common_profile),
                ("node_mismatch", manifest.get("node_executable"), common_node),
                ("bag_mismatch", manifest.get("bag"), common_bag),
                ("config_mismatch", manifest.get("config"), common_config),
                ("rate_mismatch", manifest.get("rate"), common_rate),
                ("offset_mismatch", manifest.get("start_offset_s"), common_offset),
            ):
                if observed != expected:
                    issues.add("evidence", code, f"{mode} does not match the common run identity")
        if manifest.get("bag", {}).get("path") != inputs.get("derived_lio_bag"):
            issues.add("evidence", "bag_contract", f"{mode} bag path differs from contract")
        if manifest.get("rate") != inputs.get("replay_rate"):
            issues.add("evidence", "rate_contract", f"{mode} replay rate differs from contract")
        command_text = json.dumps(manifest.get("commands", {}), sort_keys=True)
        if inputs.get("headless") is True and "rviz" in command_text.lower():
            issues.add("evidence", "headless_contract", f"{mode} command includes RViz")
        if mode == "mapping":
            if manifest.get("map") is not None:
                issues.add("evidence", "mapping_map_input", "mapping run unexpectedly has a loaded map")
        else:
            fingerprint = manifest.get("map")
            if loaded_map_fingerprint is None:
                loaded_map_fingerprint = fingerprint
            elif fingerprint != loaded_map_fingerprint:
                issues.add("evidence", "map_mismatch", f"{mode} loaded map identity differs")

        counts = summary.get("counts", {})
        steady = summary.get("steady_state", {})
        callback = summary.get("steady_state_runtime_timing_ms", {}).get("callback_total_ms", {})
        steady_frames = counts.get("steady_state_runtime_frames")
        expected_frames = counts.get("expected_frame_count")
        processed_rate = counts.get("processed_input_rate")
        over_count = counts.get("steady_state_callback_over_sensor_budget")
        over_rate = (
            float(over_count) / float(steady_frames)
            if isinstance(over_count, int) and isinstance(steady_frames, int) and steady_frames > 0
            else None
        )
        max_streak = counts.get("steady_state_max_consecutive_over_sensor_budget")
        callback_p95 = callback.get("p95")
        performance_checks = {
            "minimum_runtime_frames": isinstance(steady_frames, int) and steady_frames >= thresholds.get("minimum_runtime_frames", math.inf),
            "minimum_processed_input_rate": finite(processed_rate) and float(processed_rate) >= thresholds.get("minimum_processed_input_rate", math.inf),
            "callback_total_p95": finite(callback_p95) and float(callback_p95) <= thresholds.get("callback_total_p95_ms_max", -math.inf),
            "callback_over_sensor_budget_rate": over_rate is not None and over_rate <= thresholds.get("callback_over_sensor_budget_rate_max", -math.inf),
            "max_consecutive_over_sensor_budget": isinstance(max_streak, int) and max_streak <= thresholds.get("max_consecutive_over_sensor_budget", -1),
        }
        if steady.get("selection") != STEADY_SELECTION[mode]:
            issues.add("evidence", "steady_window", f"{mode} steady-state selection mismatch")
        if steady.get("sensor_period_ms") != inputs.get("sensor_period_ms"):
            issues.add("evidence", "sensor_period", f"{mode} sensor period differs from contract")
        if not isinstance(expected_frames, int) or expected_frames <= 0:
            issues.add("evidence", "expected_frames", f"{mode} expected frame count invalid")
        if manifest.get("observed_runtime_frames") != counts.get("runtime_frames"):
            issues.add("evidence", "runtime_frame_count", f"{mode} manifest/summary frame count mismatch")
        if sum(run["states"].values()) != counts.get("runtime_frames"):
            issues.add("evidence", "runtime_state_count", f"{mode} runtime JSONL frame count mismatch")

        events = manifest.get("runtime_events", {})
        event_totals: dict[str, int] = {}
        for name in (
            "queue_overflow",
            "fatal",
            "oom",
            "nonfinite",
            "tracking_failed",
            "message_filter_drop",
        ):
            event_totals[name] = 0
            for owner_name in ("node", "bag"):
                owner = events.get(owner_name) if isinstance(events, dict) else None
                value = owner.get(name) if isinstance(owner, dict) else None
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    issues.add(
                        "evidence",
                        "runtime_event",
                        f"{mode}.{owner_name}.{name} is invalid",
                    )
                    continue
                event_totals[name] += value
        lost_count = run["states"].get("LOST", 0)
        adverse_checks = {
            "queue_overflow": event_totals["queue_overflow"] <= thresholds.get("queue_overflow_count_max", -1),
            "fatal": event_totals["fatal"] <= thresholds.get("fatal_count_max", -1),
            "oom": event_totals["oom"] <= thresholds.get("oom_count_max", -1),
            "nonfinite_pose": event_totals["nonfinite"] <= thresholds.get("nonfinite_pose_count_max", -1),
            "unexpected_tracking_loss": lost_count <= thresholds.get("unexpected_tracking_loss_count_max", -1),
        }
        performance_checks.update(adverse_checks)
        loop_report = None
        if mode == "mapping":
            loop_stats = summary.get("loop_work_cycle_timing_ms", {}).get("total_ms", {})
            loop_report = loop_stats
            performance_checks["loop_cycle_p95"] = finite(loop_stats.get("p95")) and float(loop_stats["p95"]) <= thresholds.get("loop_cycle_p95_ms_max", -math.inf)
            performance_checks["loop_cycle_max"] = finite(loop_stats.get("max")) and float(loop_stats["max"]) <= thresholds.get("loop_cycle_max_ms_max", -math.inf)

        for check, passed in performance_checks.items():
            if not passed:
                issues.add("performance", check, f"{mode} failed {check}")
        resources = summary.get("steady_state_process_resources", {})
        resource_report = {
            "cpu_cores": summary.get("steady_state_process_cpu_cores"),
            "host_capacity_percent": summary.get("steady_state_process_host_capacity_percent"),
            "rss_mib": resources.get("rss_mib"),
            "vm_hwm_mib": resources.get("vm_hwm_mib"),
            "thread_count": resources.get("thread_count"),
        }
        for name, stats in resource_report.items():
            if not isinstance(stats, dict) or not finite(stats.get("p95")):
                issues.add("evidence", "resource_metric", f"{mode}.{name} p95 missing")

        mode_reports[mode] = {
            "evidence_dir": str(run["directory"]),
            "run_manifest_sha256": sha256_file(run["manifest_path"]),
            "performance_summary_sha256": sha256_file(run["summary_path"]),
            "runtime_frames": counts.get("runtime_frames"),
            "steady_state_frames": steady_frames,
            "expected_frames": expected_frames,
            "processed_input_rate": processed_rate,
            "callback_total_ms": callback,
            "over_sensor_budget": {
                "count": over_count,
                "rate": over_rate,
                "max_consecutive": max_streak,
            },
            "loop_work_total_ms": loop_report,
            "runtime_events": event_totals,
            "relocalization_states": dict(sorted(run["states"].items())),
            "controlled_recently_lost_frames": run["states"].get("RECENTLY_LOST", 0),
            "unexpected_tracking_loss_frames": lost_count,
            "resources_report_only": resource_report,
            "performance_checks": performance_checks,
            "performance_pass": all(performance_checks.values()),
        }

    if isinstance(loaded_map_fingerprint, dict):
        if loaded_map_fingerprint.get("sha256") != quality.get("reference_map_sha256"):
            issues.add("evidence", "reference_map_hash", "loaded map SHA-256 differs from quality reference")
        try:
            if sha256_file(reference_map) != loaded_map_fingerprint.get("sha256"):
                issues.add("evidence", "reference_map_file", "reference map file differs from run manifest")
        except OSError as error:
            issues.add("evidence", "reference_map_read", str(error))

    map_quality: dict[str, Any] = {}
    if all(mode in runs for mode in MODES):
        try:
            map_quality = map_inspector(
                proto_path,
                reference_map,
                runs["mapping"]["directory"] / "n3map.pbstream",
                runs["map_extension"]["directory"] / "n3map.pbstream",
            )
        except Exception as error:  # Evidence parsers must fail closed.
            issues.add("evidence", "map_inspection", str(error))

    if map_quality:
        if map_quality.get("reference_map_sha256") != quality.get("reference_map_sha256"):
            issues.add("evidence", "map_inspector_reference", "map inspector reference SHA-256 mismatch")
        compare_structure(
            "mapping",
            map_quality.get("mapping", {}).get("structure"),
            quality.get("mapping", {}).get("output_map"),
            issues,
        )
        compare_structure(
            "map_extension",
            map_quality.get("map_extension", {}).get("structure"),
            quality.get("map_extension", {}).get("output_map"),
            issues,
        )
        delta = map_quality.get("map_extension", {}).get("loaded_pose_delta", {})
        limits = quality.get("map_extension", {}).get("loaded_pose_delta_max", {})
        missing = delta.get("missing_keyframe_ids")
        translation_max = delta.get("translation_m", {}).get("max")
        rotation_max = delta.get("rotation_rad", {}).get("max")
        if missing != []:
            issues.add("quality", "loaded_keyframe_missing", "map extension output lost loaded keyframes")
        if not finite(translation_max) or float(translation_max) > limits.get("translation_m", -math.inf):
            issues.add("quality", "loaded_pose_translation", "loaded-map translation revision exceeds reference")
        if not finite(rotation_max) or float(rotation_max) > limits.get("rotation_rad", -math.inf):
            issues.add("quality", "loaded_pose_rotation", "loaded-map rotation revision exceeds reference")

    if "mapping" in runs:
        counts = runs["mapping"]["summary"].get("counts", {})
        expected = quality.get("mapping", {})
        for field, observed, wanted in (
            ("accepted_keyframes", counts.get("accepted_keyframes"), expected.get("accepted_keyframes")),
            ("accepted_loops", counts.get("loop_accepted"), expected.get("accepted_loops")),
        ):
            if observed != wanted:
                issues.add("quality", "mapping_non_regression", f"mapping {field}={observed}, expected={wanted}")
    for mode in ("localization", "map_extension"):
        if mode not in runs:
            continue
        expected = quality.get(mode, {})
        counts = runs[mode]["summary"].get("counts", {})
        states = runs[mode]["states"]
        if runs[mode]["summary"].get("steady_state", {}).get("lock_frame_index") != expected.get("lock_frame_index"):
            issues.add("quality", "lock_frame", f"{mode} lock frame changed")
        if states.get("LOST", 0) > expected.get("lost_frames_max", -1):
            issues.add("quality", "tracking_lost", f"{mode} contains LOST frames")
        if states.get("RECENTLY_LOST", 0) > expected.get("recently_lost_frames_max", -1):
            issues.add("quality", "tracking_fallback", f"{mode} RECENTLY_LOST count regressed")
        if mode == "localization":
            if counts.get("tracking_failure") > expected.get("tracking_failure_max", -1):
                issues.add("quality", "tracking_failure", "localization tracking failure count regressed")
            if counts.get("tracking_records") != counts.get("steady_state_runtime_frames") or counts.get("tracking_success") != counts.get("steady_state_runtime_frames"):
                issues.add("quality", "tracking_coverage", "localization tracking coverage is incomplete")
        else:
            if counts.get("accepted_keyframes") != expected.get("accepted_keyframes"):
                issues.add("quality", "extension_keyframes", "map extension keyframe count changed")
            if counts.get("strict_tracking_failure") > expected.get("strict_tracking_failure_max", -1):
                issues.add("quality", "strict_tracking_failure", "map extension strict tracking failed")
            if counts.get("strict_tracking_records") != counts.get("steady_state_runtime_frames") or counts.get("strict_tracking_success") != counts.get("steady_state_runtime_frames"):
                issues.add("quality", "strict_tracking_coverage", "map extension strict tracking coverage is incomplete")

    evidence_count = issues.count("evidence")
    performance_count = issues.count("performance")
    quality_count = issues.count("quality")
    if evidence_count:
        classification = INVALID
    elif performance_count and quality_count:
        classification = FAIL_MULTIPLE
    elif performance_count:
        classification = FAIL_PERFORMANCE
    elif quality_count:
        classification = FAIL_QUALITY
    else:
        classification = PASS
    return {
        "schema": REPORT_SCHEMA,
        "classification": classification,
        "contract_schema": contract.get("schema"),
        "quality_reference_schema": quality.get("schema"),
        "source_commit": common_commit,
        "product_profile_sha256": common_profile,
        "common_node_executable": common_node,
        "common_bag": common_bag,
        "common_config": common_config,
        "common_replay_rate": common_rate,
        "common_start_offset_s": common_offset,
        "modes": mode_reports,
        "map_quality": map_quality,
        "issue_counts": {
            "evidence": evidence_count,
            "performance": performance_count,
            "quality": quality_count,
            "total": issues.total,
        },
        "issues": issues.items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--quality-reference", required=True, type=Path)
    parser.add_argument("--mapping-dir", required=True, type=Path)
    parser.add_argument("--localization-dir", required=True, type=Path)
    parser.add_argument("--map-extension-dir", required=True, type=Path)
    parser.add_argument("--reference-map", required=True, type=Path)
    parser.add_argument("--proto", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = evaluate(
            load_json(args.contract),
            load_json(args.quality_reference),
            {
                "mapping": args.mapping_dir,
                "localization": args.localization_dir,
                "map_extension": args.map_extension_dir,
            },
            args.reference_map,
            args.proto,
        )
    except Exception as error:  # The CLI must emit INVALID_EVIDENCE, not partial PASS.
        report = {
            "schema": REPORT_SCHEMA,
            "classification": INVALID,
            "issue_counts": {"evidence": 1, "performance": 0, "quality": 0, "total": 1},
            "issues": [{"category": "evidence", "code": "top_level_read", "message": str(error)}],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_CODES[report["classification"]]


if __name__ == "__main__":
    sys.exit(main())
