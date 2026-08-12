#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "n3mapping_submap_graph_trial_qualify.py"
SPEC = importlib.util.spec_from_file_location("submap_graph_trial_qualify", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

COMMIT = "a" * 40
PROFILE = "b" * 64


def pose(x: float = 0.0) -> dict[str, list[float]]:
    return {
        "translation": [x, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def solved_record() -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "n3mapping_submap_graph_trial_v1",
        "record_type": "checkpoint",
        "runtime_source": "core",
        "context": "save_map",
        "mode": "mapping",
        "product_commit": COMMIT,
        "product_profile_sha256": PROFILE,
        "product_build_type": "Release",
        "product_research_tools": "OFF",
        "product_verified": True,
        "no_writeback": True,
        "snapshot_valid": True,
        "snapshot_failure_reason": "",
        "snapshot_node_count": 2,
        "snapshot_owned_keyframe_count": 2,
        "snapshot_source_edge_count": 1,
        "snapshot_intra_edge_count": 0,
        "snapshot_cross_edge_count": 1,
        "snapshot_unassigned_keyframe_count": 0,
        "snapshot_unassigned_edge_count": 0,
        "snapshot_floor_count": 0,
        "snapshot_assigned_floor_count": 0,
        "snapshot_unassigned_floor_count": 0,
        "snapshot_mean_cross_translation_residual_m": 0.2,
        "snapshot_max_cross_translation_residual_m": 0.2,
        "snapshot_mean_cross_rotation_residual_rad": 0.0,
        "snapshot_max_cross_rotation_residual_rad": 0.0,
        "valid": True,
        "attempted": True,
        "solved": True,
        "failure_reason": "",
        "node_count": 2,
        "gauge_anchor_count": 1,
        "active_edge_factor_count": 1,
        "intra_submap_constant_edge_count": 0,
        "full_6d_factor_count": 1,
        "xy_yaw_lifted_factor_count": 0,
        "robust_factor_count": 0,
        "session_odometry_factor_count": 0,
        "explicit_information_factor_count": 1,
        "fallback_noise_factor_count": 0,
        "floor_factor_count": 0,
        "initial_nonlinear_error": 4.0,
        "final_nonlinear_error": 0.0,
        "nonlinear_error_reduction": 4.0,
        "max_translation_delta_m": 0.2,
        "max_rotation_delta_rad": 0.0,
        "initial_keyframe_reference_count": 2,
        "initial_keyframe_reference_mean_translation_error_m": 0.1,
        "initial_keyframe_reference_p95_translation_error_m": 0.19,
        "initial_keyframe_reference_max_translation_error_m": 0.2,
        "initial_keyframe_reference_mean_rotation_error_rad": 0.0,
        "initial_keyframe_reference_p95_rotation_error_rad": 0.0,
        "initial_keyframe_reference_max_rotation_error_rad": 0.0,
        "optimized_keyframe_reference_count": 2,
        "optimized_keyframe_reference_mean_translation_error_m": 0.0,
        "optimized_keyframe_reference_p95_translation_error_m": 0.0,
        "optimized_keyframe_reference_max_translation_error_m": 0.0,
        "optimized_keyframe_reference_mean_rotation_error_rad": 0.0,
        "optimized_keyframe_reference_p95_rotation_error_rad": 0.0,
        "optimized_keyframe_reference_max_rotation_error_rad": 0.0,
        "nodes": [
            {
                "submap_id": 0,
                "gauge_anchor": True,
                "initial_pose": pose(),
                "optimized_pose": pose(),
                "translation_delta_m": 0.0,
                "rotation_delta_rad": 0.0,
            },
            {
                "submap_id": 1,
                "gauge_anchor": False,
                "initial_pose": pose(1.2),
                "optimized_pose": pose(1.0),
                "translation_delta_m": 0.2,
                "rotation_delta_rad": 0.0,
            },
        ],
        "keyframes": [
            {
                "keyframe_id": 10,
                "submap_id": 0,
                "reference_pose": pose(),
                "initial_shadow_pose": pose(),
                "optimized_shadow_pose": pose(),
                "initial_translation_error_m": 0.0,
                "initial_rotation_error_rad": 0.0,
                "optimized_translation_error_m": 0.0,
                "optimized_rotation_error_rad": 0.0,
            },
            {
                "keyframe_id": 20,
                "submap_id": 1,
                "reference_pose": pose(1.0),
                "initial_shadow_pose": pose(1.2),
                "optimized_shadow_pose": pose(1.0),
                "initial_translation_error_m": 0.2,
                "initial_rotation_error_rad": 0.0,
                "optimized_translation_error_m": 0.0,
                "optimized_rotation_error_rad": 0.0,
            },
        ],
    }
    return record


def failed_record() -> dict[str, object]:
    record = solved_record()
    record.update(
        {
            "context": "loop_commit",
            "snapshot_source_edge_count": 0,
            "snapshot_cross_edge_count": 0,
            "snapshot_mean_cross_translation_residual_m": 0.0,
            "snapshot_max_cross_translation_residual_m": 0.0,
            "valid": False,
            "attempted": False,
            "solved": False,
            "failure_reason": "topology_not_ready",
            "node_count": 0,
            "gauge_anchor_count": 0,
            "active_edge_factor_count": 0,
            "full_6d_factor_count": 0,
            "explicit_information_factor_count": 0,
            "initial_nonlinear_error": None,
            "final_nonlinear_error": None,
            "nonlinear_error_reduction": None,
            "max_translation_delta_m": 0.0,
            "initial_keyframe_reference_count": 0,
            "initial_keyframe_reference_mean_translation_error_m": 0.0,
            "initial_keyframe_reference_p95_translation_error_m": 0.0,
            "initial_keyframe_reference_max_translation_error_m": 0.0,
            "optimized_keyframe_reference_count": 0,
            "nodes": [],
            "keyframes": [],
        }
    )
    return record


def single_node_record() -> dict[str, object]:
    record = solved_record()
    record.update(
        {
            "snapshot_node_count": 1,
            "snapshot_owned_keyframe_count": 1,
            "snapshot_source_edge_count": 0,
            "snapshot_cross_edge_count": 0,
            "snapshot_mean_cross_translation_residual_m": 0.0,
            "snapshot_max_cross_translation_residual_m": 0.0,
            "node_count": 1,
            "active_edge_factor_count": 0,
            "full_6d_factor_count": 0,
            "explicit_information_factor_count": 0,
            "initial_nonlinear_error": 0.0,
            "nonlinear_error_reduction": 0.0,
            "max_translation_delta_m": 0.0,
            "initial_keyframe_reference_count": 1,
            "initial_keyframe_reference_mean_translation_error_m": 0.0,
            "initial_keyframe_reference_p95_translation_error_m": 0.0,
            "initial_keyframe_reference_max_translation_error_m": 0.0,
            "optimized_keyframe_reference_count": 1,
            "nodes": [copy.deepcopy(record["nodes"][0])],
            "keyframes": [copy.deepcopy(record["keyframes"][0])],
        }
    )
    return record


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


class SubmapGraphTrialQualificationTest(unittest.TestCase):
    def test_complete_latest_final_record_is_qualified_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trial.jsonl"
            write_jsonl(source, [failed_record(), solved_record()])
            report = TOOL.qualify(source, COMMIT, PROFILE)

        self.assertEqual(report["classification"], TOOL.QUALIFIED)
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["failed_trial_record_count"], 1)
        self.assertEqual(report["selected_record"]["line"], 2)
        self.assertIn(
            "reference_is_current_keyframe_graph_not_external_ground_truth",
            report["limitations"],
        )

    def test_single_node_final_record_is_valid_but_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trial.jsonl"
            write_jsonl(source, [single_node_record()])
            report = TOOL.qualify(source, COMMIT)

        self.assertEqual(report["classification"], TOOL.INSUFFICIENT)
        self.assertEqual(report["error_count"], 0)
        self.assertIsNone(report["selected_record"])

    def test_later_nonfinal_checkpoint_prevents_stale_final_qualification(self) -> None:
        later = solved_record()
        later["context"] = "loop_commit"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trial.jsonl"
            write_jsonl(source, [solved_record(), later])
            report = TOOL.qualify(source, COMMIT)

        self.assertEqual(report["classification"], TOOL.INSUFFICIENT)
        self.assertEqual(report["complete_final_multi_submap_record_count"], 1)

    def test_map_extension_accepts_shared_session_runtime_handoff(self) -> None:
        cross_session = solved_record()
        cross_session.update(
            {
                "mode": "map_extension",
                "runtime_source": "mapping_resuming",
                "context": "cross_session_loop",
            }
        )
        final_save = solved_record()
        final_save.update({"mode": "map_extension", "context": "save_map"})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trial.jsonl"
            write_jsonl(source, [cross_session, final_save])
            report = TOOL.qualify(source, COMMIT)

        self.assertEqual(report["classification"], TOOL.QUALIFIED)
        self.assertEqual(report["contexts"]["cross_session_loop"], 1)
        self.assertEqual(report["selected_record"]["line"], 2)

    def test_lineage_and_recomputed_metrics_fail_closed(self) -> None:
        wrong_commit = solved_record()
        wrong_commit["product_commit"] = "c" * 40
        wrong_metric = solved_record()
        wrong_metric["optimized_keyframe_reference_max_translation_error_m"] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trial.jsonl"
            write_jsonl(source, [wrong_commit, wrong_metric])
            report = TOOL.qualify(source, COMMIT)

        self.assertEqual(report["classification"], TOOL.INVALID)
        codes = {error["code"] for error in report["errors"]}
        self.assertIn("commit_mismatch", codes)
        self.assertIn("inconsistent_metric", codes)

    def test_duplicate_key_and_missing_terminal_newline_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                '{"schema":"one","schema":"two"}\n', encoding="utf-8"
            )
            duplicate_report = TOOL.qualify(duplicate, COMMIT)
            truncated = root / "truncated.jsonl"
            truncated.write_text(json.dumps(solved_record()), encoding="utf-8")
            truncated_report = TOOL.qualify(truncated, COMMIT)

        self.assertEqual(duplicate_report["classification"], TOOL.INVALID)
        self.assertEqual(truncated_report["classification"], TOOL.INVALID)
        self.assertEqual(truncated_report["errors"][0]["code"], "input_error")

    def test_cli_writes_report_and_uses_distinct_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("qualified", solved_record(), 0, TOOL.QUALIFIED),
                ("insufficient", single_node_record(), 2, TOOL.INSUFFICIENT),
                ("invalid", {**solved_record(), "product_verified": False}, 3, TOOL.INVALID),
            )
            for name, record, expected_code, expected_classification in cases:
                source = root / f"{name}.jsonl"
                output = root / f"{name}.report.json"
                write_jsonl(source, [record])
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(TOOL_PATH),
                        "--input",
                        str(source),
                        "--expected-commit",
                        COMMIT,
                        "--output",
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, expected_code, completed.stderr)
                persisted = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(persisted["classification"], expected_classification)
                self.assertEqual(json.loads(completed.stdout), persisted)


if __name__ == "__main__":
    unittest.main()
