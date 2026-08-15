#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "n3mapping_runtime_performance_analyze.py"
SPEC = importlib.util.spec_from_file_location("runtime_performance_analyze", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
SAMPLER_PATH = ROOT / "tools" / "n3mapping_process_resource_sample.py"
SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "process_resource_sample", SAMPLER_PATH
)
if SAMPLER_SPEC is None or SAMPLER_SPEC.loader is None:
    raise RuntimeError(f"cannot load {SAMPLER_PATH}")
SAMPLER = importlib.util.module_from_spec(SAMPLER_SPEC)
SAMPLER_SPEC.loader.exec_module(SAMPLER)


def runtime_record(
    index: int,
    sensor_timestamp: float,
    *,
    loaded_tracking_ms: float | None,
    callback_total_ms: float,
    accepted_keyframe: bool = False,
) -> dict[str, object]:
    first = index == 1
    return {
        "schema": "n3mapping_runtime_performance_v1",
        "record_type": "map_extension_frame",
        "processing_time": 1000.0 + index,
        "frame_index": index,
        "sensor_timestamp": sensor_timestamp,
        "sensor_delta_ms": None if first else 100.0,
        "callback_interarrival_ms": None if first else 105.0,
        "input_points": 4096,
        "core_success": not first,
        "accepted_keyframe": accepted_keyframe,
        "keyframe_id": 217 if accepted_keyframe else -1,
        "matched_keyframe_id": 53 if not first else -1,
        "relocalization_state": "SEARCHING" if first else "FULL_6DOF_LOCKED",
        "pose_source": "NONE" if first else "GEOMETRICALLY_CORRECTED",
        "relocalization_decision": (
            "initial_relocalization_rejected"
            if first
            else "loaded_map_tracking_geometric"
        ),
        "callback_skipped": first,
        "published_global_pose": not first,
        "published_body_cloud": not first,
        "published_world_cloud": not first,
        "callback_lock_wait_ms": 1.0,
        "ros_conversion_ms": 2.0,
        "core_frame_ms": 90.0 if not first else 20.0,
        "initial_relocalization_ms": 18.0 if first else None,
        "loaded_map_tracking_ms": loaded_tracking_ms,
        "keyframe_gate_ms": 0.1 if not first else None,
        "keyframe_commit_ms": 12.0 if accepted_keyframe else None,
        "graph_update_ms": 5.0 if accepted_keyframe else None,
        "descriptor_update_ms": 6.0 if accepted_keyframe else None,
        "post_commit_refresh_ms": 0.5 if accepted_keyframe else None,
        "authority_publish_ms": 0.2,
        "odometry_path_publish_ms": 0.3 if not first else None,
        "callback_locked_ms": callback_total_ms - 5.0,
        "cloud_publish_ms": 4.0 if not first else None,
        "callback_total_ms": callback_total_ms,
    }


def tracking_record(
    index: int,
    *,
    cache_hit: bool = False,
    strict: bool = True,
    localization_cache_hit: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": "tracking",
        "processing_time": 1000.0 + index,
        "query_index": index,
        "strict_loaded_map": strict,
        "predicted_pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        },
        "nearest_kf_id": 53,
        "submap_size": 200000,
        "tracking_total_ms": 85.0,
        "nearest_keyframe_ms": 1.0,
        "loaded_map_cache_ms": 2.0 if strict else None,
        "submap_build_ms": 3.0,
        "target_prepare_ms": 20.0,
        "loaded_map_target_cache_hit": cache_hit if strict else False,
        "loaded_map_target_cache_miss": (not cache_hit) if strict else False,
        "localization_target_cache_enabled": not strict,
        "localization_target_cache_hit": localization_cache_hit if not strict else False,
        "localization_target_cache_miss": (not localization_cache_hit) if not strict else False,
        "localization_target_cache_entry_bytes": 1024 if not strict else 0,
        "localization_target_cache_total_bytes": 4096 if not strict else 0,
        "localization_target_cache_entries": 4 if not strict else 0,
        "source_prepare_ms": 5.0,
        "registration_ms": 50.0,
        "retry_registration_ms": None,
        "visibility_ms": 4.0 if strict else None,
        "icp_converged": True,
        "fitness_score": 0.01,
        "inlier_ratio": 0.9,
        "retry_used": False,
        "consecutive_track_failures": 0,
        "result_success": True,
        "reject_reason": "",
    }
    return record


def runtime_record_v2(
    index: int,
    sensor_timestamp: float,
    *,
    mode: str,
    callback_total_ms: float,
    relocalization_locked: bool = False,
    tracking_attempted: bool = False,
    loaded_tracking_ms: float | None = None,
) -> dict[str, object]:
    record = runtime_record(
        index,
        sensor_timestamp,
        loaded_tracking_ms=loaded_tracking_ms,
        callback_total_ms=callback_total_ms,
    )
    record.update(
        {
            "schema": "n3mapping_runtime_performance_v2",
            "record_type": "runtime_frame",
            "mode": mode,
            "relocalization_locked": relocalization_locked,
            "tracking_attempted": tracking_attempted,
        }
    )
    if relocalization_locked:
        record["relocalization_state"] = "FULL_6DOF_LOCKED"
        record["core_success"] = True
    return record


def loop_record(
    index: int, *, queued: int, total_ms: float = 10.0
) -> dict[str, object]:
    return {
        "schema": "n3mapping_runtime_performance_v2",
        "record_type": "loop_cycle",
        "mode": "mapping",
        "processing_time": 1000.25 + index,
        "cycle_index": index,
        "queued_keyframe_count": queued,
        "detected_candidate_count": 2 if queued else 0,
        "place_candidate_count": 1 if queued else 0,
        "accepted_loop_count": 1 if queued else 0,
        "edge_count": 1 if queued else 0,
        "optimized": queued > 0,
        "lock_wait_ms": 1.0,
        "core_ms": total_ms - 2.0,
        "publish_ms": 1.0,
        "total_ms": total_ms,
    }


def resource_record(index: int) -> dict[str, object]:
    return {
        "schema": "n3mapping_process_resource_v1",
        "sample_index": index,
        "processing_time": 1000.5 + index,
        "monotonic_time": 2000.0 + index,
        "pid": 123,
        "process_start_ticks": 456,
        "host_logical_cpus": 8,
        "interval_s": 1.0,
        "process_cpu_percent": 300.0 + index,
        "rss_mib": 420.0 + index,
        "vm_hwm_mib": 430.0 + index,
        "thread_count": 12,
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


class RuntimePerformanceAnalyzeTest(unittest.TestCase):
    def test_v2_mapping_requires_and_summarizes_loop_work(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [
                    runtime_record_v2(
                        1, 10.0, mode="mapping", callback_total_ms=80.0
                    ),
                    loop_record(1, queued=0),
                    runtime_record_v2(
                        2, 10.1, mode="mapping", callback_total_ms=120.0
                    ),
                    loop_record(2, queued=1, total_ms=45.0),
                ],
            )
            write_jsonl(resource_path, [resource_record(1), resource_record(2)])

            report, exit_code = TOOL.analyze(
                runtime_path,
                tracking_path,
                resource_path,
                expected_frames=2,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "PROFILE_READY")
            self.assertEqual(report["mode"], "mapping")
            self.assertEqual(report["counts"]["steady_state_runtime_frames"], 2)
            self.assertEqual(report["counts"]["loop_work_cycles"], 1)
            self.assertEqual(report["counts"]["loop_detected_candidates"], 2)
            self.assertEqual(report["counts"]["processed_input_rate"], 1.0)
            self.assertEqual(report["counts"]["expected_frame_count_source"], "explicit")
            self.assertEqual(report["loop_work_cycle_timing_ms"]["total_ms"]["p95"], 45.0)

    def test_v2_localization_uses_only_post_lock_tracking_window(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [
                    runtime_record_v2(
                        1,
                        10.0,
                        mode="localization",
                        callback_total_ms=400.0,
                        relocalization_locked=True,
                    ),
                    runtime_record_v2(
                        2,
                        10.1,
                        mode="localization",
                        callback_total_ms=120.0,
                        tracking_attempted=True,
                    ),
                    runtime_record_v2(
                        3,
                        10.2,
                        mode="localization",
                        callback_total_ms=130.0,
                        tracking_attempted=True,
                    ),
                ],
            )
            write_jsonl(
                tracking_path,
                [
                    tracking_record(1, strict=False),
                    tracking_record(
                        2, strict=False, localization_cache_hit=True
                    ),
                ],
            )
            write_jsonl(resource_path, [resource_record(1), resource_record(2)])

            report, exit_code = TOOL.analyze(
                runtime_path, tracking_path, resource_path
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["mode"], "localization")
            self.assertEqual(report["steady_state"]["lock_frame_index"], 1)
            self.assertEqual(report["counts"]["runtime_frames"], 3)
            self.assertEqual(report["counts"]["steady_state_runtime_frames"], 2)
            self.assertEqual(report["counts"]["ordinary_tracking_records"], 2)
            self.assertEqual(
                report["counts"]["localization_target_cache_observed"], 2
            )
            self.assertEqual(
                report["counts"]["localization_target_cache_hit"], 1
            )
            self.assertEqual(
                report["counts"]["localization_target_cache_miss"], 1
            )
            self.assertEqual(
                report["counts"]["localization_target_cache_peak_bytes"],
                4096,
            )
            self.assertEqual(
                report["counts"]["steady_state_max_consecutive_over_sensor_budget"],
                2,
            )
            self.assertEqual(
                report["steady_state_runtime_timing_ms"]["callback_total_ms"]["p50"],
                125.0,
            )
            self.assertEqual(report["counts"]["expected_frame_count"], 3)
            self.assertEqual(
                report["counts"]["expected_frame_count_source"],
                "observed_sensor_timestamp_span",
            )

    def test_profile_ready_summarizes_stages_and_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [
                    runtime_record(1, 10.0, loaded_tracking_ms=None, callback_total_ms=25.0),
                    runtime_record(
                        2,
                        10.1,
                        loaded_tracking_ms=85.0,
                        callback_total_ms=120.0,
                        accepted_keyframe=True,
                    ),
                    runtime_record(3, 10.2, loaded_tracking_ms=85.0, callback_total_ms=80.0),
                ],
            )
            write_jsonl(
                tracking_path,
                [tracking_record(1), tracking_record(2, cache_hit=True)],
            )
            write_jsonl(resource_path, [resource_record(1), resource_record(2)])

            report, exit_code = TOOL.analyze(
                runtime_path, tracking_path, resource_path
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "PROFILE_READY")
            self.assertEqual(report["counts"]["runtime_frames"], 3)
            self.assertEqual(report["counts"]["strict_tracking_success"], 2)
            self.assertEqual(report["counts"]["strict_tracking_failure"], 0)
            self.assertEqual(
                report["counts"]["loaded_map_target_cache_observed"], 2
            )
            self.assertEqual(report["counts"]["loaded_map_target_cache_hit"], 1)
            self.assertEqual(report["counts"]["loaded_map_target_cache_miss"], 1)
            self.assertEqual(report["counts"]["accepted_keyframes"], 1)
            self.assertEqual(report["counts"]["callback_over_sensor_budget"], 1)
            self.assertEqual(report["counts"]["resource_samples_total"], 2)
            self.assertEqual(
                report["counts"]["resource_samples_in_runtime_window"], 2
            )
            self.assertEqual(
                report["process_resources"]["process_cpu_percent"]["p50"],
                301.5,
            )
            self.assertEqual(
                report["dominant_p95_stage"]["loaded_map_tracking"],
                "registration_ms",
            )
            self.assertEqual(
                report["accepted_keyframe_timing_ms"]["graph_update_ms"]["p95"],
                5.0,
            )

    def test_resource_statistics_exclude_samples_outside_callback_window(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [
                    runtime_record(
                        1, 10.0, loaded_tracking_ms=85.0, callback_total_ms=100.0
                    )
                ],
            )
            write_jsonl(tracking_path, [tracking_record(1)])
            before = resource_record(1)
            before["processing_time"] = 900.0
            during = resource_record(2)
            during["processing_time"] = 1001.05
            write_jsonl(resource_path, [before, during])

            report, exit_code = TOOL.analyze(
                runtime_path, tracking_path, resource_path
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["counts"]["resource_samples_total"], 2)
            self.assertEqual(
                report["counts"]["resource_samples_in_runtime_window"], 1
            )
            self.assertEqual(
                report["counts"]["resource_samples_outside_runtime_window"], 1
            )
            self.assertEqual(
                report["process_resources"]["process_cpu_percent"]["p50"],
                during["process_cpu_percent"],
            )

    def test_resource_samples_outside_callback_window_are_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [
                    runtime_record(
                        1, 10.0, loaded_tracking_ms=85.0, callback_total_ms=100.0
                    )
                ],
            )
            write_jsonl(tracking_path, [tracking_record(1)])
            outside = resource_record(1)
            outside["processing_time"] = 900.0
            write_jsonl(resource_path, [outside])

            report, exit_code = TOOL.analyze(
                runtime_path, tracking_path, resource_path
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(
                report["counts"]["resource_samples_in_runtime_window"], 0
            )

    def test_count_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [runtime_record(1, 10.0, loaded_tracking_ms=20.0, callback_total_ms=30.0)],
            )
            write_jsonl(tracking_path, [tracking_record(1), tracking_record(2)])
            write_jsonl(resource_path, [resource_record(1)])

            report, exit_code = TOOL.analyze(
                runtime_path, tracking_path, resource_path
            )

            self.assertEqual(exit_code, 3)
            self.assertEqual(report["status"], "INVALID_EVIDENCE")
            self.assertTrue(any("count mismatch" in error for error in report["errors"]))

    def test_no_strict_tracking_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            write_jsonl(
                runtime_path,
                [runtime_record(1, 10.0, loaded_tracking_ms=None, callback_total_ms=20.0)],
            )
            write_jsonl(
                tracking_path,
                [{"record_type": "relocalize", "query_index": 1}],
            )

            report, exit_code = TOOL.analyze(runtime_path, tracking_path)

            self.assertEqual(exit_code, 2)
            self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")

    def test_legacy_tracking_without_cache_outcome_remains_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            resource_path = directory / "process_resource.jsonl"
            write_jsonl(
                runtime_path,
                [
                    runtime_record(
                        1,
                        10.0,
                        loaded_tracking_ms=85.0,
                        callback_total_ms=100.0,
                    )
                ],
            )
            legacy_tracking = tracking_record(1)
            legacy_tracking.pop("loaded_map_target_cache_hit")
            legacy_tracking.pop("loaded_map_target_cache_miss")
            write_jsonl(tracking_path, [legacy_tracking])
            write_jsonl(resource_path, [resource_record(1)])

            report, exit_code = TOOL.analyze(
                runtime_path, tracking_path, resource_path
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "PROFILE_READY")
            self.assertEqual(
                report["counts"]["loaded_map_target_cache_observed"], 0
            )

    def test_cli_writes_report_and_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            runtime_path = directory / "runtime_performance_debug.jsonl"
            tracking_path = directory / "relocalization_debug.jsonl"
            runtime_path.write_text(
                '{"schema":"n3mapping_runtime_performance_v1","schema":"duplicate"}\n',
                encoding="utf-8",
            )
            write_jsonl(tracking_path, [tracking_record(1)])
            output = directory / "summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--input-dir",
                    str(directory),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 3)
            self.assertTrue(output.is_file())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "INVALID_EVIDENCE")
            self.assertTrue(any("duplicate key" in error for error in report["errors"]))

    def test_process_sampler_parses_comm_and_samples_live_process(self) -> None:
        parsed = SAMPLER.parse_proc_stat(
            "123 (name with spaces) R 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
        )
        self.assertEqual(parsed["pid"], 123)
        self.assertEqual(parsed["utime_ticks"], 11)
        self.assertEqual(parsed["stime_ticks"], 12)
        self.assertEqual(parsed["process_start_ticks"], 19)
        self.assertEqual(parsed["rss_pages"], 21)

        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir) / "process_resource.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SAMPLER_PATH),
                    "--pid",
                    str(os.getpid()),
                    "--output",
                    str(output),
                    "--interval",
                    "0.01",
                    "--duration",
                    "0.04",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(records), 2)
            self.assertEqual(records[0]["schema"], "n3mapping_process_resource_v1")
            self.assertGreaterEqual(records[0]["process_cpu_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
