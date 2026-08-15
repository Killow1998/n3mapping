#!/usr/bin/env python3
"""Summarize tri-mode n3mapping runtime evidence without changing behavior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


RUNTIME_SCHEMA_V1 = "n3mapping_runtime_performance_v1"
RUNTIME_SCHEMA_V2 = "n3mapping_runtime_performance_v2"
RESOURCE_SCHEMA = "n3mapping_process_resource_v1"
REPORT_SCHEMA = "n3mapping_runtime_performance_summary_v2"
PROFILE_READY = "PROFILE_READY"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {PROFILE_READY: 0, INSUFFICIENT: 2, INVALID: 3}
MODES = {"mapping", "localization", "map_extension"}

RUNTIME_TIMINGS = (
    "callback_lock_wait_ms",
    "ros_conversion_ms",
    "core_frame_ms",
    "initial_relocalization_ms",
    "loaded_map_tracking_ms",
    "keyframe_gate_ms",
    "keyframe_commit_ms",
    "graph_update_ms",
    "descriptor_update_ms",
    "post_commit_refresh_ms",
    "authority_publish_ms",
    "odometry_path_publish_ms",
    "callback_locked_ms",
    "cloud_publish_ms",
    "callback_total_ms",
)
LOOP_TIMINGS = ("lock_wait_ms", "core_ms", "publish_ms", "total_ms")
TRACKING_TIMINGS = (
    "tracking_total_ms",
    "nearest_keyframe_ms",
    "loaded_map_cache_ms",
    "submap_build_ms",
    "target_prepare_ms",
    "source_prepare_ms",
    "registration_ms",
    "retry_registration_ms",
    "visibility_ms",
)
TRACKING_TARGET_CACHE_FIELDS = (
    "loaded_map_target_cache_hit",
    "loaded_map_target_cache_miss",
)
TRACKING_LOCALIZATION_TARGET_CACHE_FIELDS = (
    "localization_target_cache_enabled",
    "localization_target_cache_hit",
    "localization_target_cache_miss",
    "localization_target_cache_entry_bytes",
    "localization_target_cache_total_bytes",
    "localization_target_cache_entries",
)
TRACKING_VISIBILITY_SHADOW_FIELDS = (
    "visibility_prediction_ms",
    "visibility_registration_ms",
    "visibility_prediction_valid",
    "visibility_prediction_consistency_ratio",
    "visibility_prediction_evidence_log_odds",
    "visibility_registration_valid",
    "visibility_registration_consistency_ratio",
    "visibility_registration_evidence_log_odds",
    "visibility_selected_pose_source",
    "visibility_registration_delta_translation_m",
    "visibility_registration_delta_rotation_rad",
    "visibility_registration_would_accept",
)
RUNTIME_REQUIRED = {
    "schema",
    "record_type",
    "processing_time",
    "frame_index",
    "sensor_timestamp",
    "sensor_delta_ms",
    "callback_interarrival_ms",
    "input_points",
    "core_success",
    "accepted_keyframe",
    "keyframe_id",
    "matched_keyframe_id",
    "relocalization_state",
    "pose_source",
    "relocalization_decision",
    "callback_skipped",
    "published_global_pose",
    "published_body_cloud",
    "published_world_cloud",
    *RUNTIME_TIMINGS,
}
RUNTIME_V2_REQUIRED = {"mode", "relocalization_locked", "tracking_attempted"}
LOOP_REQUIRED = {
    "schema",
    "record_type",
    "mode",
    "processing_time",
    "cycle_index",
    "queued_keyframe_count",
    "detected_candidate_count",
    "place_candidate_count",
    "accepted_loop_count",
    "edge_count",
    "optimized",
    *LOOP_TIMINGS,
}
TRACKING_REQUIRED = {
    "record_type",
    "query_index",
    "strict_loaded_map",
    "nearest_kf_id",
    "retry_used",
    "result_success",
    "reject_reason",
    *TRACKING_TIMINGS,
}
RESOURCE_REQUIRED = {
    "schema",
    "sample_index",
    "processing_time",
    "monotonic_time",
    "pid",
    "process_start_ticks",
    "host_logical_cpus",
    "interval_s",
    "process_cpu_percent",
    "rss_mib",
    "vm_hwm_mib",
    "thread_count",
}


class DuplicateKeyError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: Iterable[Any], total: int) -> dict[str, Any]:
    finite_values = [float(value) for value in values if finite(value)]
    result: dict[str, Any] = {
        "count": len(finite_values),
        "missing": total - len(finite_values),
    }
    if not finite_values:
        result.update({"mean": None, "p50": None, "p95": None, "max": None})
        return result
    result.update(
        {
            "mean": statistics.fmean(finite_values),
            "p50": percentile(finite_values, 0.50),
            "p95": percentile(finite_values, 0.95),
            "max": max(finite_values),
        }
    )
    return result


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], [f"missing file: {path}"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(
                    raw,
                    object_pairs_hook=strict_object,
                    parse_constant=reject_constant,
                )
            except (ValueError, json.JSONDecodeError) as error:
                errors.append(f"{path}:{line_number}: {error}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path}:{line_number}: record must be an object")
                continue
            records.append(value)
    return records, errors


def runtime_mode(record: dict[str, Any]) -> str | None:
    if (
        record.get("schema") == RUNTIME_SCHEMA_V1
        and record.get("record_type") == "map_extension_frame"
    ):
        return "map_extension"
    mode = record.get("mode")
    return mode if isinstance(mode, str) else None


def split_runtime_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    frames: list[dict[str, Any]] = []
    loops: list[dict[str, Any]] = []
    errors: list[str] = []
    for line, record in enumerate(records, 1):
        record_type = record.get("record_type")
        if record_type in ("map_extension_frame", "runtime_frame"):
            frames.append(record)
        elif record_type == "loop_cycle":
            loops.append(record)
        else:
            errors.append(f"runtime stream record {line}: unexpected record_type")
    return frames, loops, errors


def validate_runtime(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    previous_index = 0
    for line, record in enumerate(records, 1):
        missing = sorted(RUNTIME_REQUIRED - record.keys())
        if record.get("schema") == RUNTIME_SCHEMA_V2:
            missing.extend(sorted(RUNTIME_V2_REQUIRED - record.keys()))
        if missing:
            errors.append(f"runtime record {line}: missing {','.join(missing)}")
            continue
        schema = record["schema"]
        if schema == RUNTIME_SCHEMA_V1:
            if record["record_type"] != "map_extension_frame":
                errors.append(f"runtime record {line}: invalid v1 record_type")
        elif schema == RUNTIME_SCHEMA_V2:
            if record["record_type"] != "runtime_frame":
                errors.append(f"runtime record {line}: invalid v2 record_type")
            if record["mode"] not in MODES:
                errors.append(f"runtime record {line}: invalid mode")
            for field in ("relocalization_locked", "tracking_attempted"):
                if not isinstance(record[field], bool):
                    errors.append(f"runtime record {line}: {field} must be bool")
        else:
            errors.append(f"runtime record {line}: unexpected schema")
        index = record["frame_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index != previous_index + 1
        ):
            errors.append(f"runtime record {line}: frame_index is not contiguous")
        else:
            previous_index = index
        if not nonnegative_integer(record["input_points"]) or record["input_points"] == 0:
            errors.append(f"runtime record {line}: input_points must be positive")
        for field in ("processing_time", "sensor_timestamp", "callback_total_ms"):
            if not finite(record[field]):
                errors.append(f"runtime record {line}: {field} must be finite")
        for field in RUNTIME_TIMINGS:
            if record[field] is not None and (
                not finite(record[field]) or float(record[field]) < 0.0
            ):
                errors.append(f"runtime record {line}: invalid {field}")
        for field in (
            "core_success",
            "accepted_keyframe",
            "callback_skipped",
            "published_global_pose",
            "published_body_cloud",
            "published_world_cloud",
        ):
            if not isinstance(record[field], bool):
                errors.append(f"runtime record {line}: {field} must be bool")
    return errors


def validate_loops(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    previous_index = 0
    for line, record in enumerate(records, 1):
        missing = sorted(LOOP_REQUIRED - record.keys())
        if missing:
            errors.append(f"loop record {line}: missing {','.join(missing)}")
            continue
        if record["schema"] != RUNTIME_SCHEMA_V2:
            errors.append(f"loop record {line}: unexpected schema")
        if record["record_type"] != "loop_cycle" or record["mode"] != "mapping":
            errors.append(f"loop record {line}: invalid type or mode")
        index = record["cycle_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index != previous_index + 1
        ):
            errors.append(f"loop record {line}: cycle_index is not contiguous")
        else:
            previous_index = index
        for field in (
            "queued_keyframe_count",
            "detected_candidate_count",
            "place_candidate_count",
            "accepted_loop_count",
            "edge_count",
        ):
            if not nonnegative_integer(record[field]):
                errors.append(f"loop record {line}: invalid {field}")
        if not isinstance(record["optimized"], bool):
            errors.append(f"loop record {line}: optimized must be bool")
        if not finite(record["processing_time"]):
            errors.append(f"loop record {line}: processing_time must be finite")
        for field in LOOP_TIMINGS:
            if not finite(record[field]) or float(record[field]) < 0.0:
                errors.append(f"loop record {line}: invalid {field}")
    return errors


def validate_tracking(records: list[dict[str, Any]], mode: str) -> list[str]:
    errors: list[str] = []
    previous_index = 0
    strict_expected = mode == "map_extension"
    for line, record in enumerate(records, 1):
        missing = sorted(TRACKING_REQUIRED - record.keys())
        if missing:
            errors.append(f"tracking record {line}: missing {','.join(missing)}")
            continue
        if (
            record["record_type"] != "tracking"
            or record["strict_loaded_map"] is not strict_expected
        ):
            errors.append(f"tracking record {line}: mode mismatch")
        index = record["query_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index <= previous_index
        ):
            errors.append(f"tracking record {line}: query_index is not increasing")
        else:
            previous_index = index
        for field in TRACKING_TIMINGS:
            if record[field] is not None and (
                not finite(record[field]) or float(record[field]) < 0.0
            ):
                errors.append(f"tracking record {line}: invalid {field}")
        for field in ("strict_loaded_map", "retry_used", "result_success"):
            if not isinstance(record[field], bool):
                errors.append(f"tracking record {line}: {field} must be bool")

        if strict_expected:
            cache_fields_present = [
                field in record for field in TRACKING_TARGET_CACHE_FIELDS
            ]
            if any(cache_fields_present) and not all(cache_fields_present):
                errors.append(
                    f"tracking record {line}: incomplete loaded-map target cache outcome"
                )
            elif all(cache_fields_present):
                cache_hit = record["loaded_map_target_cache_hit"]
                cache_miss = record["loaded_map_target_cache_miss"]
                if not isinstance(cache_hit, bool) or not isinstance(cache_miss, bool):
                    errors.append(
                        f"tracking record {line}: target cache outcomes must be bool"
                    )
                elif cache_hit and cache_miss:
                    errors.append(
                        f"tracking record {line}: target cache hit and miss both true"
                    )
                elif finite(record["target_prepare_ms"]) and cache_hit == cache_miss:
                    errors.append(
                        f"tracking record {line}: prepared target lacks one cache outcome"
                    )
                elif (cache_hit or cache_miss) and not finite(
                    record["target_prepare_ms"]
                ):
                    errors.append(
                        f"tracking record {line}: cache outcome lacks target timing"
                    )

        localization_cache_fields_present = [
            field in record for field in TRACKING_LOCALIZATION_TARGET_CACHE_FIELDS
        ]
        if any(localization_cache_fields_present) and not all(
            localization_cache_fields_present
        ):
            errors.append(
                f"tracking record {line}: incomplete localization target cache metrics"
            )
        elif all(localization_cache_fields_present):
            enabled = record["localization_target_cache_enabled"]
            hit = record["localization_target_cache_hit"]
            miss = record["localization_target_cache_miss"]
            if not all(isinstance(value, bool) for value in (enabled, hit, miss)):
                errors.append(
                    f"tracking record {line}: localization cache outcomes must be bool"
                )
            sizes = [
                record["localization_target_cache_entry_bytes"],
                record["localization_target_cache_total_bytes"],
                record["localization_target_cache_entries"],
            ]
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in sizes
            ):
                errors.append(
                    f"tracking record {line}: invalid localization cache size metrics"
                )
            if enabled is True and strict_expected:
                errors.append(
                    f"tracking record {line}: strict tracking used ordinary localization cache"
                )
            elif enabled is True and finite(record["target_prepare_ms"]) and hit == miss:
                errors.append(
                    f"tracking record {line}: enabled localization cache lacks one outcome"
                )
            elif enabled is not True and (hit is True or miss is True):
                errors.append(
                    f"tracking record {line}: disabled localization cache has an outcome"
                )
            elif hit is True and (sizes[0] == 0 or sizes[2] == 0):
                errors.append(
                    f"tracking record {line}: localization cache hit lacks a retained entry"
                )

        visibility_shadow_fields_present = [
            field in record for field in TRACKING_VISIBILITY_SHADOW_FIELDS
        ]
        if any(visibility_shadow_fields_present) and not all(
            visibility_shadow_fields_present
        ):
            errors.append(
                f"tracking record {line}: incomplete visibility shadow metrics"
            )
        elif all(visibility_shadow_fields_present):
            prediction_valid = record["visibility_prediction_valid"]
            registration_valid = record["visibility_registration_valid"]
            endpoint_accept = record["visibility_registration_would_accept"]
            if not all(
                isinstance(value, bool)
                for value in (prediction_valid, registration_valid, endpoint_accept)
            ):
                errors.append(
                    f"tracking record {line}: visibility shadow outcomes must be bool"
                )
            for field in (
                "visibility_prediction_ms",
                "visibility_registration_ms",
                "visibility_registration_delta_translation_m",
                "visibility_registration_delta_rotation_rad",
            ):
                if record[field] is not None and (
                    not finite(record[field]) or float(record[field]) < 0.0
                ):
                    errors.append(
                        f"tracking record {line}: invalid visibility shadow {field}"
                    )
            for prefix, valid in (
                ("prediction", prediction_valid),
                ("registration", registration_valid),
            ):
                for suffix in ("consistency_ratio", "evidence_log_odds"):
                    field = f"visibility_{prefix}_{suffix}"
                    if record[field] is not None and not finite(record[field]):
                        errors.append(
                            f"tracking record {line}: invalid visibility shadow {field}"
                        )
                    elif valid is True and not finite(record[field]):
                        errors.append(
                            f"tracking record {line}: valid visibility lacks {field}"
                        )
            source = record["visibility_selected_pose_source"]
            if source not in {
                "not_evaluated",
                "registration_endpoint",
                "motion_prediction",
            }:
                errors.append(
                    f"tracking record {line}: invalid visibility selected pose source"
                )
            elif source != "not_evaluated":
                if not strict_expected:
                    errors.append(
                        f"tracking record {line}: ordinary tracking evaluated strict visibility"
                    )
                for field in (
                    "visibility_prediction_ms",
                    "visibility_registration_ms",
                    "visibility_registration_delta_translation_m",
                    "visibility_registration_delta_rotation_rad",
                ):
                    if not finite(record[field]):
                        errors.append(
                            f"tracking record {line}: evaluated visibility lacks {field}"
                        )

        geometric_success = (
            record["result_success"] is True and record["reject_reason"] == ""
        )
        if geometric_success:
            required_success = [
                "tracking_total_ms",
                "nearest_keyframe_ms",
                "submap_build_ms",
                "target_prepare_ms",
                "source_prepare_ms",
                "registration_ms",
            ]
            if strict_expected:
                required_success.extend(["loaded_map_cache_ms", "visibility_ms"])
            for field in required_success:
                if not finite(record[field]):
                    errors.append(f"tracking record {line}: success missing {field}")
    return errors


def validate_resources(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    previous_index = 0
    identity: tuple[int, int] | None = None
    for line, record in enumerate(records, 1):
        missing = sorted(RESOURCE_REQUIRED - record.keys())
        if missing:
            errors.append(f"resource record {line}: missing {','.join(missing)}")
            continue
        if record["schema"] != RESOURCE_SCHEMA:
            errors.append(f"resource record {line}: unexpected schema")
        index = record["sample_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index != previous_index + 1
        ):
            errors.append(f"resource record {line}: sample_index is not contiguous")
        else:
            previous_index = index
        current_identity = (record["pid"], record["process_start_ticks"])
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in current_identity
        ):
            errors.append(f"resource record {line}: invalid process identity")
        elif identity is None:
            identity = current_identity
        elif current_identity != identity:
            errors.append(f"resource record {line}: process identity changed")
        for field in ("host_logical_cpus", "thread_count"):
            if (
                not isinstance(record[field], int)
                or isinstance(record[field], bool)
                or record[field] <= 0
            ):
                errors.append(f"resource record {line}: invalid {field}")
        for field in (
            "processing_time",
            "monotonic_time",
            "interval_s",
            "process_cpu_percent",
            "rss_mib",
            "vm_hwm_mib",
        ):
            if not finite(record[field]) or float(record[field]) < 0.0:
                errors.append(f"resource record {line}: invalid {field}")
    return errors


def dominant_stage(summary: dict[str, dict[str, Any]], fields: Iterable[str]) -> str | None:
    candidates = [
        (summary[field]["p95"], field)
        for field in fields
        if summary[field]["p95"] is not None
    ]
    return max(candidates)[1] if candidates else None


def select_runtime_resources(
    runtime: list[dict[str, Any]], resources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    if not runtime:
        return [], None, None
    runtime_start = min(float(row["processing_time"]) for row in runtime)
    runtime_end = max(
        float(row["processing_time"])
        + (
            float(row["callback_total_ms"]) / 1000.0
            if finite(row.get("callback_total_ms"))
            else 0.0
        )
        for row in runtime
    )
    selected = [
        row
        for row in resources
        if float(row["processing_time"]) > runtime_start
        and float(row["processing_time"]) - float(row["interval_s"]) < runtime_end
    ]
    return selected, runtime_start, runtime_end


def steady_state_runtime(
    runtime: list[dict[str, Any]], mode: str
) -> tuple[list[dict[str, Any]], int | None, str]:
    if mode == "mapping":
        return runtime, None, "all_valid_synchronized_frame_callbacks"
    if runtime and runtime[0].get("schema") == RUNTIME_SCHEMA_V1:
        return runtime, None, "legacy_v1_all_recorded_frames"
    lock_position = next(
        (
            position
            for position, row in enumerate(runtime)
            if row.get("relocalization_locked") is True
            and row.get("relocalization_state") == "FULL_6DOF_LOCKED"
        ),
        None,
    )
    if lock_position is None:
        return [], None, "missing_FULL_6DOF_LOCKED_transition"
    selected = [
        row
        for row in runtime[lock_position + 1 :]
        if row.get("tracking_attempted") is True
    ]
    return (
        selected,
        int(runtime[lock_position]["frame_index"]),
        "tracking_callbacks_after_first_FULL_6DOF_LOCKED_no_additional_warmup",
    )


def max_consecutive_over_budget(
    runtime: list[dict[str, Any]], sensor_period_ms: float
) -> int:
    maximum = 0
    current = 0
    for row in runtime:
        if (
            finite(row.get("callback_total_ms"))
            and float(row["callback_total_ms"]) > sensor_period_ms
        ):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def infer_expected_frames(
    runtime: list[dict[str, Any]], sensor_period_ms: float
) -> int | None:
    if len(runtime) < 2:
        return len(runtime) if runtime else None
    first = runtime[0].get("sensor_timestamp")
    last = runtime[-1].get("sensor_timestamp")
    if not finite(first) or not finite(last) or float(last) < float(first):
        return None
    duration_ms = (float(last) - float(first)) * 1000.0
    return max(1, int(round(duration_ms / sensor_period_ms)) + 1)


def resource_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        field: stats((row.get(field) for row in records), len(records))
        for field in (
            "process_cpu_percent",
            "rss_mib",
            "vm_hwm_mib",
            "thread_count",
        )
    }


def cpu_summaries(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu_cores = [
        float(row["process_cpu_percent"]) / 100.0
        for row in records
        if finite(row.get("process_cpu_percent"))
    ]
    host_capacity = [
        float(row["process_cpu_percent"]) / float(row["host_logical_cpus"])
        for row in records
        if finite(row.get("process_cpu_percent"))
        and isinstance(row.get("host_logical_cpus"), int)
        and row["host_logical_cpus"] > 0
    ]
    return stats(cpu_cores, len(records)), stats(host_capacity, len(records))


def analyze(
    runtime_path: Path,
    tracking_path: Path,
    resource_path: Path | None = None,
    *,
    expected_frames: int | None = None,
    sensor_period_ms: float = 100.0,
) -> tuple[dict[str, Any], int]:
    runtime_stream, runtime_read_errors = read_jsonl(runtime_path)
    runtime, loop_cycles, split_errors = split_runtime_records(runtime_stream)
    modes = {runtime_mode(record) for record in runtime}
    modes.update(runtime_mode(record) for record in loop_cycles)
    modes.discard(None)
    mode = next(iter(modes)) if len(modes) == 1 else None

    if tracking_path.is_file():
        all_reloc, tracking_read_errors = read_jsonl(tracking_path)
    elif mode == "mapping":
        all_reloc, tracking_read_errors = [], []
    else:
        all_reloc, tracking_read_errors = read_jsonl(tracking_path)
    if resource_path is not None and resource_path.is_file():
        resources, resource_read_errors = read_jsonl(resource_path)
    else:
        resources, resource_read_errors = [], []

    all_tracking = [
        record for record in all_reloc if record.get("record_type") == "tracking"
    ]
    if mode == "map_extension":
        tracking = [row for row in all_tracking if row.get("strict_loaded_map") is True]
    elif mode == "localization":
        tracking = [row for row in all_tracking if row.get("strict_loaded_map") is False]
    else:
        tracking = []

    errors = (
        runtime_read_errors
        + split_errors
        + tracking_read_errors
        + resource_read_errors
    )
    if len(modes) > 1:
        errors.append(f"runtime stream contains multiple modes: {sorted(modes)}")
    elif runtime and mode not in MODES:
        errors.append("runtime mode is missing or invalid")
    if runtime:
        errors.extend(validate_runtime(runtime))
    if loop_cycles:
        errors.extend(validate_loops(loop_cycles))
    if mode in ("localization", "map_extension"):
        if len(tracking) != len(all_tracking):
            errors.append("tracking stream contains records for another runtime mode")
        if tracking:
            errors.extend(validate_tracking(tracking, mode))
    elif mode == "mapping" and all_tracking:
        errors.append("mapping evidence unexpectedly contains tracking records")
    if resources:
        errors.extend(validate_resources(resources))

    steady_runtime, lock_frame_index, steady_selection = steady_state_runtime(
        runtime, mode or ""
    )
    runtime_resources: list[dict[str, Any]] = []
    runtime_start: float | None = None
    runtime_end: float | None = None
    steady_resources: list[dict[str, Any]] = []
    steady_start: float | None = None
    steady_end: float | None = None
    if runtime and resources and not errors:
        runtime_resources, runtime_start, runtime_end = select_runtime_resources(
            runtime, resources
        )
        steady_resources, steady_start, steady_end = select_runtime_resources(
            steady_runtime, resources
        )

    loaded_runtime = [
        row for row in runtime if finite(row.get("loaded_map_tracking_ms"))
    ]
    if mode == "map_extension" and runtime and tracking:
        comparable_runtime = (
            loaded_runtime
            if runtime[0].get("schema") == RUNTIME_SCHEMA_V1
            else steady_runtime
        )
        if len(comparable_runtime) != len(tracking):
            errors.append(
                "loaded-map tracking count mismatch: "
                f"runtime={len(comparable_runtime)} tracking={len(tracking)}"
            )
    if mode == "localization" and runtime and tracking:
        if len(steady_runtime) != len(tracking):
            errors.append(
                "ordinary tracking count mismatch: "
                f"runtime={len(steady_runtime)} tracking={len(tracking)}"
            )

    work_loop_cycles = [
        row for row in loop_cycles if row.get("queued_keyframe_count", 0) > 0
    ]
    if errors:
        status = INVALID
    else:
        ready = bool(runtime and steady_runtime and steady_resources)
        if mode in ("localization", "map_extension"):
            legacy_v1 = runtime[0].get("schema") == RUNTIME_SCHEMA_V1
            ready = ready and bool(tracking) and (
                legacy_v1 or lock_frame_index is not None
            )
        if mode == "mapping":
            ready = ready and bool(loop_cycles) and bool(work_loop_cycles)
        status = PROFILE_READY if ready else INSUFFICIENT

    runtime_stats = {
        field: stats((row.get(field) for row in runtime), len(runtime))
        for field in RUNTIME_TIMINGS
    }
    steady_runtime_stats = {
        field: stats((row.get(field) for row in steady_runtime), len(steady_runtime))
        for field in RUNTIME_TIMINGS
    }
    tracking_stats = {
        field: stats((row.get(field) for row in tracking), len(tracking))
        for field in TRACKING_TIMINGS
    }
    loop_stats = {
        field: stats((row.get(field) for row in loop_cycles), len(loop_cycles))
        for field in LOOP_TIMINGS
    }
    work_loop_stats = {
        field: stats((row.get(field) for row in work_loop_cycles), len(work_loop_cycles))
        for field in LOOP_TIMINGS
    }
    budget_overrun = [
        float(row["callback_total_ms"]) - sensor_period_ms
        for row in runtime
        if finite(row.get("callback_total_ms"))
    ]
    steady_budget_overrun = [
        float(row["callback_total_ms"]) - sensor_period_ms
        for row in steady_runtime
        if finite(row.get("callback_total_ms"))
    ]
    schedule_lag = [
        float(row["callback_interarrival_ms"]) - float(row["sensor_delta_ms"])
        for row in runtime
        if finite(row.get("callback_interarrival_ms"))
        and finite(row.get("sensor_delta_ms"))
        and float(row["sensor_delta_ms"]) > 0.0
    ]
    steady_schedule_lag = [
        float(row["callback_interarrival_ms"]) - float(row["sensor_delta_ms"])
        for row in steady_runtime
        if finite(row.get("callback_interarrival_ms"))
        and finite(row.get("sensor_delta_ms"))
        and float(row["sensor_delta_ms"]) > 0.0
    ]
    accepted = [row for row in runtime if row.get("accepted_keyframe") is True]
    cpu_cores, host_capacity = cpu_summaries(runtime_resources)
    steady_cpu_cores, steady_host_capacity = cpu_summaries(steady_resources)
    inferred_expected_frames = infer_expected_frames(runtime, sensor_period_ms)
    effective_expected_frames = (
        expected_frames if expected_frames is not None else inferred_expected_frames
    )
    expected_frame_source = (
        "explicit"
        if expected_frames is not None
        else "observed_sensor_timestamp_span"
        if inferred_expected_frames is not None
        else None
    )
    processed_input_rate = (
        len(runtime) / effective_expected_frames
        if isinstance(effective_expected_frames, int)
        and effective_expected_frames > 0
        else None
    )
    cache_observed = sum(
        row.get("loaded_map_target_cache_hit") is True
        or row.get("loaded_map_target_cache_miss") is True
        for row in tracking
    )
    localization_cache_observed = sum(
        row.get("localization_target_cache_enabled") is True for row in tracking
    )
    localization_cache_peak_bytes = max(
        (
            row.get("localization_target_cache_total_bytes", 0)
            for row in tracking
            if isinstance(row.get("localization_target_cache_total_bytes"), int)
            and not isinstance(
                row.get("localization_target_cache_total_bytes"), bool
            )
        ),
        default=0,
    )
    localization_cache_peak_entries = max(
        (
            row.get("localization_target_cache_entries", 0)
            for row in tracking
            if isinstance(row.get("localization_target_cache_entries"), int)
            and not isinstance(row.get("localization_target_cache_entries"), bool)
        ),
        default=0,
    )
    visibility_shadow = [
        row
        for row in tracking
        if row.get("visibility_selected_pose_source")
        in {"registration_endpoint", "motion_prediction"}
    ]
    visibility_shadow_stats = {
        field: stats(
            (row.get(field) for row in visibility_shadow), len(visibility_shadow)
        )
        for field in ("visibility_prediction_ms", "visibility_registration_ms")
    }

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "scope": "measurement_only_no_optimization_or_quality_verdict",
        "mode": mode,
        "runtime_jsonl": str(runtime_path),
        "tracking_jsonl": str(tracking_path),
        "resource_jsonl": str(resource_path) if resource_path is not None else None,
        "errors": errors,
        "steady_state": {
            "selection": steady_selection,
            "lock_frame_index": lock_frame_index,
            "sensor_period_ms": sensor_period_ms,
        },
        "counts": {
            "runtime_frames": len(runtime),
            "steady_state_runtime_frames": len(steady_runtime),
            "expected_frame_count": effective_expected_frames,
            "expected_frame_count_source": expected_frame_source,
            "processed_input_rate": processed_input_rate,
            "loaded_map_tracking_frames": len(loaded_runtime),
            "tracking_records": len(tracking),
            "tracking_success": sum(row.get("result_success") is True for row in tracking),
            "tracking_failure": sum(row.get("result_success") is False for row in tracking),
            "strict_tracking_records": sum(
                row.get("strict_loaded_map") is True for row in tracking
            ),
            "strict_tracking_success": sum(
                row.get("strict_loaded_map") is True
                and row.get("result_success") is True
                for row in tracking
            ),
            "strict_tracking_failure": sum(
                row.get("strict_loaded_map") is True
                and row.get("result_success") is False
                for row in tracking
            ),
            "ordinary_tracking_records": sum(
                row.get("strict_loaded_map") is False for row in tracking
            ),
            "tracking_retry": sum(row.get("retry_used") is True for row in tracking),
            "loaded_map_target_cache_observed": cache_observed,
            "loaded_map_target_cache_hit": sum(
                row.get("loaded_map_target_cache_hit") is True for row in tracking
            ),
            "loaded_map_target_cache_miss": sum(
                row.get("loaded_map_target_cache_miss") is True for row in tracking
            ),
            "localization_target_cache_observed": localization_cache_observed,
            "localization_target_cache_hit": sum(
                row.get("localization_target_cache_hit") is True for row in tracking
            ),
            "localization_target_cache_miss": sum(
                row.get("localization_target_cache_miss") is True for row in tracking
            ),
            "localization_target_cache_peak_bytes": localization_cache_peak_bytes,
            "localization_target_cache_peak_entries": localization_cache_peak_entries,
            "visibility_shadow_observed": len(visibility_shadow),
            "visibility_registration_selected": sum(
                row.get("visibility_selected_pose_source")
                == "registration_endpoint"
                for row in visibility_shadow
            ),
            "visibility_prediction_selected": sum(
                row.get("visibility_selected_pose_source") == "motion_prediction"
                for row in visibility_shadow
            ),
            "visibility_registration_would_accept": sum(
                row.get("visibility_registration_would_accept") is True
                for row in visibility_shadow
            ),
            "visibility_registration_decision_mismatch": sum(
                row.get("visibility_registration_would_accept")
                != row.get("result_success")
                for row in visibility_shadow
            ),
            "core_failure": sum(row.get("core_success") is False for row in runtime),
            "callback_skipped": sum(row.get("callback_skipped") is True for row in runtime),
            "accepted_keyframes": len(accepted),
            "callback_over_sensor_budget": sum(value > 0.0 for value in budget_overrun),
            "steady_state_callback_over_sensor_budget": sum(
                value > 0.0 for value in steady_budget_overrun
            ),
            "max_consecutive_over_sensor_budget": max_consecutive_over_budget(
                runtime, sensor_period_ms
            ),
            "steady_state_max_consecutive_over_sensor_budget": max_consecutive_over_budget(
                steady_runtime, sensor_period_ms
            ),
            "callback_interarrival_slower_than_sensor": sum(
                value > 0.0 for value in schedule_lag
            ),
            "loop_cycles": len(loop_cycles),
            "loop_work_cycles": len(work_loop_cycles),
            "loop_queued_keyframes": sum(
                row.get("queued_keyframe_count", 0) for row in loop_cycles
            ),
            "loop_detected_candidates": sum(
                row.get("detected_candidate_count", 0) for row in loop_cycles
            ),
            "loop_accepted": sum(
                row.get("accepted_loop_count", 0) for row in loop_cycles
            ),
            "resource_samples_total": len(resources),
            "resource_samples_in_runtime_window": len(runtime_resources),
            "resource_samples_in_steady_state_window": len(steady_resources),
            "resource_samples_outside_runtime_window": len(resources) - len(runtime_resources),
        },
        "runtime_timing_ms": runtime_stats,
        "steady_state_runtime_timing_ms": steady_runtime_stats,
        "tracking_timing_ms": tracking_stats,
        "visibility_shadow_timing_ms": visibility_shadow_stats,
        "loop_cycle_timing_ms": loop_stats,
        "loop_work_cycle_timing_ms": work_loop_stats,
        "callback_budget_overrun_ms": stats(budget_overrun, len(runtime)),
        "steady_state_callback_budget_overrun_ms": stats(
            steady_budget_overrun, len(steady_runtime)
        ),
        "callback_schedule_lag_ms": stats(schedule_lag, len(runtime)),
        "steady_state_callback_schedule_lag_ms": stats(
            steady_schedule_lag, len(steady_runtime)
        ),
        "process_resources": resource_summary(runtime_resources),
        "steady_state_process_resources": resource_summary(steady_resources),
        "process_cpu_cores": cpu_cores,
        "process_host_capacity_percent": host_capacity,
        "steady_state_process_cpu_cores": steady_cpu_cores,
        "steady_state_process_host_capacity_percent": steady_host_capacity,
        "resource_runtime_window": {
            "start_processing_time": runtime_start,
            "end_processing_time": runtime_end,
            "selection": "sample_interval_overlaps_callback_window",
        },
        "resource_steady_state_window": {
            "start_processing_time": steady_start,
            "end_processing_time": steady_end,
            "selection": "sample_interval_overlaps_steady_callback_window",
        },
        "accepted_keyframe_timing_ms": {
            field: stats((row.get(field) for row in accepted), len(accepted))
            for field in (
                "keyframe_commit_ms",
                "graph_update_ms",
                "descriptor_update_ms",
                "post_commit_refresh_ms",
            )
        },
        "dominant_p95_stage": {
            "callback_direct": dominant_stage(
                steady_runtime_stats,
                (
                    "callback_lock_wait_ms",
                    "ros_conversion_ms",
                    "core_frame_ms",
                    "authority_publish_ms",
                    "odometry_path_publish_ms",
                    "cloud_publish_ms",
                ),
            ),
            "tracking": dominant_stage(
                tracking_stats,
                (
                    "nearest_keyframe_ms",
                    "loaded_map_cache_ms",
                    "submap_build_ms",
                    "target_prepare_ms",
                    "source_prepare_ms",
                    "registration_ms",
                    "retry_registration_ms",
                    "visibility_ms",
                ),
            ),
            "loaded_map_tracking": dominant_stage(
                tracking_stats,
                (
                    "nearest_keyframe_ms",
                    "loaded_map_cache_ms",
                    "submap_build_ms",
                    "target_prepare_ms",
                    "source_prepare_ms",
                    "registration_ms",
                    "retry_registration_ms",
                    "visibility_ms",
                ),
            )
            if mode == "map_extension"
            else None,
            "mapping_loop_work": dominant_stage(
                work_loop_stats, ("lock_wait_ms", "core_ms", "publish_ms")
            )
            if mode == "mapping"
            else None,
        },
    }
    return report, EXIT_CODES[status]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--runtime-jsonl", type=Path)
    parser.add_argument("--tracking-jsonl", type=Path)
    parser.add_argument("--resource-jsonl", type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--sensor-period-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.input_dir is not None:
        args.runtime_jsonl = args.runtime_jsonl or (
            args.input_dir / "runtime_performance_debug.jsonl"
        )
        args.tracking_jsonl = args.tracking_jsonl or (
            args.input_dir / "relocalization_debug.jsonl"
        )
        args.resource_jsonl = args.resource_jsonl or (
            args.input_dir / "process_resource.jsonl"
        )
    if args.runtime_jsonl is None:
        parser.error("provide --input-dir or --runtime-jsonl")
    if args.tracking_jsonl is None:
        args.tracking_jsonl = args.runtime_jsonl.with_name(
            "relocalization_debug.jsonl"
        )
    if args.expected_frames is not None and args.expected_frames <= 0:
        parser.error("--expected-frames must be positive")
    if not finite(args.sensor_period_ms) or args.sensor_period_ms <= 0.0:
        parser.error("--sensor-period-ms must be positive and finite")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report, exit_code = analyze(
        args.runtime_jsonl,
        args.tracking_jsonl,
        args.resource_jsonl,
        expected_frames=args.expected_frames,
        sensor_period_ms=args.sensor_period_ms,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
