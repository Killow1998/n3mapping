#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "n3mapping_runtime_performance_gate.py"
SPEC = importlib.util.spec_from_file_location("runtime_performance_gate", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

COMMIT = "a" * 40
PROFILE = "b" * 64


def stats(value: float, count: int = 1) -> dict[str, float | int]:
    return {
        "count": count,
        "max": value,
        "mean": value,
        "missing": 0,
        "p50": value,
        "p95": value,
    }


def contract(bag: str) -> dict[str, object]:
    return {
        "schema": TOOL.CONTRACT_SCHEMA,
        "fa01": {
            "modes": list(TOOL.MODES),
            "inputs": {
                "derived_lio_bag": bag,
                "sensor_period_ms": 100,
                "replay_rate": 1,
                "headless": True,
            },
            "steady_state_thresholds": {
                "minimum_runtime_frames": 8,
                "minimum_processed_input_rate": 0.99,
                "callback_total_p95_ms_max": 100,
                "callback_over_sensor_budget_rate_max": 0.05,
                "max_consecutive_over_sensor_budget": 5,
                "loop_cycle_p95_ms_max": 100,
                "loop_cycle_max_ms_max": 500,
                "queue_overflow_count_max": 0,
                "fatal_count_max": 0,
                "oom_count_max": 0,
                "unexpected_tracking_loss_count_max": 0,
                "nonfinite_pose_count_max": 0,
            },
        },
    }


MAPPING_STRUCTURE = {
    "version": "test",
    "keyframes": 2,
    "edges": 2,
    "odometry": 1,
    "loop": 1,
    "session_anchor": 0,
    "unknown_edges": 0,
    "dense_trajectory": 10,
    "metadata_match": True,
    "duplicate_keyframes": 0,
    "dangling_edges": 0,
    "session_anchors": [],
}
EXTENSION_STRUCTURE = {
    "version": "test",
    "keyframes": 3,
    "edges": 3,
    "odometry": 2,
    "loop": 0,
    "session_anchor": 1,
    "unknown_edges": 0,
    "dense_trajectory": 10,
    "metadata_match": True,
    "duplicate_keyframes": 0,
    "dangling_edges": 0,
    "session_anchors": [[0, 2]],
}


def quality(reference_sha: str) -> dict[str, object]:
    return {
        "schema": TOOL.QUALITY_SCHEMA,
        "reference_map_sha256": reference_sha,
        "mapping": {
            "accepted_keyframes": 2,
            "accepted_loops": 1,
            "output_map": copy.deepcopy(MAPPING_STRUCTURE),
        },
        "localization": {
            "lock_frame_index": 2,
            "lost_frames_max": 0,
            "recently_lost_frames_max": 1,
            "tracking_failure_max": 0,
        },
        "map_extension": {
            "lock_frame_index": 2,
            "accepted_keyframes": 1,
            "lost_frames_max": 0,
            "recently_lost_frames_max": 0,
            "strict_tracking_failure_max": 0,
            "output_map": copy.deepcopy(EXTENSION_STRUCTURE),
            "loaded_pose_delta_max": {
                "translation_m": 0.1,
                "rotation_rad": 0.01,
            },
        },
    }


def summary(mode: str) -> dict[str, object]:
    loaded = mode != "mapping"
    steady_frames = 8 if loaded else 10
    counts = {
        "accepted_keyframes": 1 if mode == "map_extension" else (2 if mode == "mapping" else 0),
        "expected_frame_count": 10,
        "runtime_frames": 10,
        "steady_state_runtime_frames": steady_frames,
        "processed_input_rate": 1.0,
        "steady_state_callback_over_sensor_budget": 0,
        "steady_state_max_consecutive_over_sensor_budget": 0,
        "loop_accepted": 1 if mode == "mapping" else 0,
        "tracking_records": 8 if mode == "localization" else 0,
        "tracking_success": 8 if mode == "localization" else 0,
        "tracking_failure": 0,
        "strict_tracking_records": 8 if mode == "map_extension" else 0,
        "strict_tracking_success": 8 if mode == "map_extension" else 0,
        "strict_tracking_failure": 0,
    }
    return {
        "schema": TOOL.SUMMARY_SCHEMA,
        "status": "PROFILE_READY",
        "mode": mode,
        "counts": counts,
        "steady_state": {
            "lock_frame_index": 2 if loaded else None,
            "selection": TOOL.STEADY_SELECTION[mode],
            "sensor_period_ms": 100,
        },
        "steady_state_runtime_timing_ms": {"callback_total_ms": stats(10.0, steady_frames)},
        "loop_work_cycle_timing_ms": {"total_ms": stats(10.0)},
        "steady_state_process_cpu_cores": stats(1.0),
        "steady_state_process_host_capacity_percent": stats(5.0),
        "steady_state_process_resources": {
            "rss_mib": stats(100.0),
            "vm_hwm_mib": stats(110.0),
            "thread_count": stats(4.0),
        },
    }


def manifest(mode: str, bag: str, reference_sha: str) -> dict[str, object]:
    identity = {
        "schema": "n3mapping_product_build_identity_v2",
        "commit": COMMIT,
        "product_profile_sha256": PROFILE,
        "build_type": "Release",
        "research_tools": "OFF",
        "verified": True,
    }
    fingerprint = {"path": "/tmp/node", "sha256": "c" * 64, "size_bytes": 1, "mtime_ns": 1}
    return {
        "schema": TOOL.RUN_SCHEMA,
        "status": "COMPLETE",
        "mode": mode,
        "performance_status": "PROFILE_READY",
        "expected_commit": COMMIT,
        "node_build_identity": identity,
        "node_executable": fingerprint,
        "bag": {"path": bag, "metadata": {"sha256": "d" * 64}, "payloads": []},
        "config": {"path": "/tmp/config", "sha256": "e" * 64},
        "map": None if mode == "mapping" else {"path": "/tmp/map", "sha256": reference_sha},
        "rate": 1,
        "start_offset_s": 100,
        "target_runtime_frames": 10,
        "observed_runtime_frames": 10,
        "commands": {"node": ["n3mapping_node"]},
        "process_return_codes": {"node": 0, "bag": 0, "sampler": 0, "analyzer": 0},
        "runtime_events": {
            "node": {
                "queue_overflow": 0,
                "fatal": 0,
                "oom": 0,
                "nonfinite": 0,
                "tracking_failed": 1 if mode == "localization" else 0,
                "message_filter_drop": 0,
            },
            "bag": {
                "queue_overflow": 0,
                "fatal": 0,
                "oom": 0,
                "nonfinite": 0,
                "tracking_failed": 0,
                "message_filter_drop": 0,
            },
        },
    }


def write_run(root: Path, mode: str, bag: str, reference_sha: str) -> Path:
    directory = root / mode
    directory.mkdir()
    (directory / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest(mode, bag, reference_sha)), encoding="utf-8"
    )
    (directory / "performance_summary.json").write_text(
        json.dumps(summary(mode)), encoding="utf-8"
    )
    states = ["SEARCHING", "PROVISIONAL"]
    if mode == "localization":
        states += ["FULL_6DOF_LOCKED"] * 7 + ["RECENTLY_LOST"]
    elif mode == "map_extension":
        states += ["FULL_6DOF_LOCKED"] * 8
    else:
        states += ["SEARCHING"] * 8
    (directory / "runtime_performance_debug.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema": "n3mapping_runtime_performance_v2",
                    "record_type": "runtime_frame",
                    "mode": mode,
                    "relocalization_state": state,
                }
            )
            + "\n"
            for state in states
        ),
        encoding="utf-8",
    )
    (directory / "n3map.pbstream").write_bytes(b"placeholder")
    return directory


def passing_map_quality(reference_sha: str) -> dict[str, object]:
    return {
        "reference_map_sha256": reference_sha,
        "mapping": {"output_map_sha256": "f" * 64, "structure": copy.deepcopy(MAPPING_STRUCTURE)},
        "map_extension": {
            "output_map_sha256": "1" * 64,
            "structure": copy.deepcopy(EXTENSION_STRUCTURE),
            "loaded_pose_delta": {
                "compared_keyframes": 2,
                "missing_keyframe_ids": [],
                "translation_m": {"p50": 0.01, "p95": 0.02, "max": 0.03},
                "rotation_rad": {"p50": 0.001, "p95": 0.002, "max": 0.003},
            },
        },
    }


class RuntimePerformanceGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference_map = self.root / "reference.pbstream"
        self.reference_map.write_bytes(b"reference")
        self.reference_sha = hashlib.sha256(b"reference").hexdigest()
        self.bag = str(self.root / "bag")
        self.run_dirs = {
            mode: write_run(self.root, mode, self.bag, self.reference_sha)
            for mode in TOOL.MODES
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evaluate(self, map_quality: dict[str, object] | None = None) -> dict[str, object]:
        observed = map_quality or passing_map_quality(self.reference_sha)
        return TOOL.evaluate(
            contract(self.bag),
            quality(self.reference_sha),
            self.run_dirs,
            self.reference_map,
            ROOT / "proto" / "n3map.proto",
            map_inspector=lambda *unused: copy.deepcopy(observed),
        )

    def test_complete_consistent_triad_passes_with_controlled_fallback(self) -> None:
        report = self.evaluate()
        self.assertEqual(report["classification"], TOOL.PASS)
        self.assertEqual(report["issue_counts"]["total"], 0)
        localization = report["modes"]["localization"]
        self.assertEqual(localization["controlled_recently_lost_frames"], 1)
        self.assertEqual(localization["unexpected_tracking_loss_frames"], 0)

    def test_over_budget_rate_is_not_rounded_down(self) -> None:
        path = self.run_dirs["map_extension"] / "performance_summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["counts"]["steady_state_callback_over_sensor_budget"] = 1
        path.write_text(json.dumps(value), encoding="utf-8")
        report = self.evaluate()
        self.assertEqual(report["classification"], TOOL.FAIL_PERFORMANCE)
        self.assertTrue(any(issue["code"] == "callback_over_sensor_budget_rate" for issue in report["issues"]))

    def test_build_identity_mismatch_invalidates_triad(self) -> None:
        path = self.run_dirs["localization"] / "run_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["node_build_identity"]["commit"] = "9" * 40
        value["expected_commit"] = "9" * 40
        path.write_text(json.dumps(value), encoding="utf-8")
        report = self.evaluate()
        self.assertEqual(report["classification"], TOOL.INVALID)
        self.assertTrue(any(issue["code"] == "commit_mismatch" for issue in report["issues"]))

    def test_lost_state_fails_performance_and_quality(self) -> None:
        path = self.run_dirs["localization"] / "runtime_performance_debug.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[-1])
        event["relocalization_state"] = "LOST"
        lines[-1] = json.dumps(event)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = self.evaluate()
        self.assertEqual(report["classification"], TOOL.FAIL_MULTIPLE)
        self.assertTrue(any(issue["code"] == "unexpected_tracking_loss" for issue in report["issues"]))
        self.assertTrue(any(issue["code"] == "tracking_lost" for issue in report["issues"]))

    def test_map_structure_regression_fails_quality(self) -> None:
        observed = passing_map_quality(self.reference_sha)
        observed["map_extension"]["structure"]["dangling_edges"] = 1
        report = self.evaluate(observed)
        self.assertEqual(report["classification"], TOOL.FAIL_QUALITY)
        self.assertTrue(any(issue["code"] == "map_structure_mismatch" for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
