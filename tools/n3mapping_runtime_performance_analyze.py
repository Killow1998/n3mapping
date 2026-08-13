#!/usr/bin/env python3
"""Summarize PERF-ME-01 map-extension timing evidence without judging quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


RUNTIME_SCHEMA = "n3mapping_runtime_performance_v1"
RESOURCE_SCHEMA = "n3mapping_process_resource_v1"
REPORT_SCHEMA = "n3mapping_runtime_performance_summary_v1"
PROFILE_READY = "PROFILE_READY"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {PROFILE_READY: 0, INSUFFICIENT: 2, INVALID: 3}

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


def validate_runtime(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    previous_index = 0
    for line, record in enumerate(records, 1):
        missing = sorted(RUNTIME_REQUIRED - record.keys())
        if missing:
            errors.append(f"runtime record {line}: missing {','.join(missing)}")
            continue
        if record["schema"] != RUNTIME_SCHEMA:
            errors.append(f"runtime record {line}: unexpected schema")
        if record["record_type"] != "map_extension_frame":
            errors.append(f"runtime record {line}: unexpected record_type")
        index = record["frame_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index != previous_index + 1:
            errors.append(f"runtime record {line}: frame_index is not contiguous")
        else:
            previous_index = index
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


def validate_tracking(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    previous_index = 0
    for line, record in enumerate(records, 1):
        missing = sorted(TRACKING_REQUIRED - record.keys())
        if missing:
            errors.append(f"tracking record {line}: missing {','.join(missing)}")
            continue
        if record["record_type"] != "tracking" or record["strict_loaded_map"] is not True:
            errors.append(f"tracking record {line}: not strict loaded-map tracking")
        index = record["query_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index <= previous_index:
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
        if record["result_success"]:
            for field in (
                "tracking_total_ms",
                "nearest_keyframe_ms",
                "loaded_map_cache_ms",
                "submap_build_ms",
                "target_prepare_ms",
                "source_prepare_ms",
                "registration_ms",
                "visibility_ms",
            ):
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
        if not isinstance(index, int) or isinstance(index, bool) or index != previous_index + 1:
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
        and float(row["processing_time"]) - float(row["interval_s"])
        < runtime_end
    ]
    return selected, runtime_start, runtime_end


def analyze(
    runtime_path: Path,
    tracking_path: Path,
    resource_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    runtime, runtime_read_errors = read_jsonl(runtime_path)
    all_reloc, tracking_read_errors = read_jsonl(tracking_path)
    if resource_path is not None and resource_path.is_file():
        resources, resource_read_errors = read_jsonl(resource_path)
    else:
        resources, resource_read_errors = [], []
    strict_tracking = [
        record
        for record in all_reloc
        if record.get("record_type") == "tracking"
        and record.get("strict_loaded_map") is True
    ]
    errors = runtime_read_errors + tracking_read_errors + resource_read_errors
    if runtime:
        errors.extend(validate_runtime(runtime))
    if strict_tracking:
        errors.extend(validate_tracking(strict_tracking))
    if resources:
        errors.extend(validate_resources(resources))

    runtime_resources: list[dict[str, Any]] = []
    runtime_start: float | None = None
    runtime_end: float | None = None
    if runtime and resources and not errors:
        runtime_resources, runtime_start, runtime_end = select_runtime_resources(
            runtime, resources
        )

    loaded_runtime = [row for row in runtime if finite(row.get("loaded_map_tracking_ms"))]
    if runtime and strict_tracking and len(loaded_runtime) != len(strict_tracking):
        errors.append(
            "loaded-map tracking count mismatch: "
            f"runtime={len(loaded_runtime)} tracking={len(strict_tracking)}"
        )

    if errors:
        status = INVALID
    elif not runtime or not strict_tracking or not runtime_resources:
        status = INSUFFICIENT
    else:
        status = PROFILE_READY

    runtime_stats = {
        field: stats((row.get(field) for row in runtime), len(runtime))
        for field in RUNTIME_TIMINGS
    }
    tracking_stats = {
        field: stats((row.get(field) for row in strict_tracking), len(strict_tracking))
        for field in TRACKING_TIMINGS
    }
    budget_overrun = [
        float(row["callback_total_ms"]) - float(row["sensor_delta_ms"])
        for row in runtime
        if finite(row.get("callback_total_ms"))
        and finite(row.get("sensor_delta_ms"))
        and float(row["sensor_delta_ms"]) > 0.0
    ]
    schedule_lag = [
        float(row["callback_interarrival_ms"]) - float(row["sensor_delta_ms"])
        for row in runtime
        if finite(row.get("callback_interarrival_ms"))
        and finite(row.get("sensor_delta_ms"))
        and float(row["sensor_delta_ms"]) > 0.0
    ]
    accepted = [row for row in runtime if row.get("accepted_keyframe") is True]
    cpu_cores = [
        float(row["process_cpu_percent"]) / 100.0
        for row in runtime_resources
        if finite(row.get("process_cpu_percent"))
    ]
    host_capacity = [
        float(row["process_cpu_percent"]) / float(row["host_logical_cpus"])
        for row in runtime_resources
        if finite(row.get("process_cpu_percent"))
        and isinstance(row.get("host_logical_cpus"), int)
        and row["host_logical_cpus"] > 0
    ]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "scope": "measurement_only_no_optimization_or_quality_verdict",
        "runtime_jsonl": str(runtime_path),
        "tracking_jsonl": str(tracking_path),
        "resource_jsonl": str(resource_path) if resource_path is not None else None,
        "errors": errors,
        "counts": {
            "runtime_frames": len(runtime),
            "loaded_map_tracking_frames": len(loaded_runtime),
            "strict_tracking_records": len(strict_tracking),
            "strict_tracking_success": sum(
                row.get("result_success") is True for row in strict_tracking
            ),
            "strict_tracking_failure": sum(
                row.get("result_success") is False for row in strict_tracking
            ),
            "tracking_retry": sum(row.get("retry_used") is True for row in strict_tracking),
            "loaded_map_target_cache_observed": sum(
                row.get("loaded_map_target_cache_hit") is True
                or row.get("loaded_map_target_cache_miss") is True
                for row in strict_tracking
            ),
            "loaded_map_target_cache_hit": sum(
                row.get("loaded_map_target_cache_hit") is True
                for row in strict_tracking
            ),
            "loaded_map_target_cache_miss": sum(
                row.get("loaded_map_target_cache_miss") is True
                for row in strict_tracking
            ),
            "core_failure": sum(row.get("core_success") is False for row in runtime),
            "callback_skipped": sum(row.get("callback_skipped") is True for row in runtime),
            "accepted_keyframes": len(accepted),
            "callback_over_sensor_budget": sum(value > 0.0 for value in budget_overrun),
            "callback_interarrival_slower_than_sensor": sum(
                value > 0.0 for value in schedule_lag
            ),
            "resource_samples_total": len(resources),
            "resource_samples_in_runtime_window": len(runtime_resources),
            "resource_samples_outside_runtime_window": (
                len(resources) - len(runtime_resources)
            ),
        },
        "runtime_timing_ms": runtime_stats,
        "tracking_timing_ms": tracking_stats,
        "callback_budget_overrun_ms": stats(budget_overrun, len(runtime)),
        "callback_schedule_lag_ms": stats(schedule_lag, len(runtime)),
        "process_resources": {
            field: stats(
                (row.get(field) for row in runtime_resources),
                len(runtime_resources),
            )
            for field in (
                "process_cpu_percent",
                "rss_mib",
                "vm_hwm_mib",
                "thread_count",
            )
        },
        "process_cpu_cores": stats(cpu_cores, len(runtime_resources)),
        "process_host_capacity_percent": stats(
            host_capacity, len(runtime_resources)
        ),
        "resource_runtime_window": {
            "start_processing_time": runtime_start,
            "end_processing_time": runtime_end,
            "selection": "sample_interval_overlaps_callback_window",
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
                runtime_stats,
                (
                    "callback_lock_wait_ms",
                    "ros_conversion_ms",
                    "core_frame_ms",
                    "authority_publish_ms",
                    "odometry_path_publish_ms",
                    "cloud_publish_ms",
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
            ),
        },
    }
    return report, EXIT_CODES[status]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--runtime-jsonl", type=Path)
    parser.add_argument("--tracking-jsonl", type=Path)
    parser.add_argument("--resource-jsonl", type=Path)
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
    if args.runtime_jsonl is None or args.tracking_jsonl is None:
        parser.error("provide --input-dir or both JSONL paths")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report, exit_code = analyze(
        args.runtime_jsonl, args.tracking_jsonl, args.resource_jsonl
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
