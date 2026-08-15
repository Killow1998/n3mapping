#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import n3mapping_fa02_gate as fa02  # noqa: E402
import n3mapping_fa02b_gate as gate  # noqa: E402
import n3mapping_fa02b_run as runner  # noqa: E402


PROFILE = {
    "id": "moderate",
    "translation_scale_error_fraction": 0.01,
    "yaw_bias_deg_per_meter": 0.1,
    "translation_rw_std_m_per_sqrt_meter": 0.01,
    "rotation_rw_std_deg_per_sqrt_meter": 0.05,
}
ORACLE = {
    "place_translation_threshold_m": 5.0,
    "correct_measurement_translation_error_m_max": 1.0,
    "correct_measurement_rotation_error_deg_max": 10.0,
    "catastrophic_place_distance_m_min": 10.0,
    "catastrophic_measurement_translation_error_m_min": 2.0,
    "catastrophic_measurement_rotation_error_deg_min": 30.0,
}
THRESHOLDS = {
    "optimizer_error_count_max": 0,
    "minimum_injected_odom_ate_translation_rmse_m": 0.1,
    "catastrophic_false_loop_count_max": 0,
    "minimum_correct_accepted_loops_per_positive_case": 1,
    "ate_translation_improvement_ratio_min": 0.5,
    "rpe_translation_regression_ratio_max": 1.1,
    "rpe_rotation_regression_ratio_max": 1.1,
    "accepted_loop_count_max_per_control_case": 0,
}


class FakeMapHelpers:
    @staticmethod
    def parse_map(path: Path, module: object) -> Path:
        if not path.is_file():
            raise ValueError("map missing")
        return path

    @staticmethod
    def map_structure(path: Path, module: object) -> dict[str, object]:
        return {
            "keyframes": 25,
            "edges": 25 if path.parent.name == "loop_on" else 24,
            "odometry": 24,
            "loop": 1 if path.parent.name == "loop_on" else 0,
            "session_anchor": 0,
            "unknown_edges": 0,
            "dense_trajectory": 2,
            "metadata_match": True,
            "duplicate_keyframes": 0,
            "dangling_edges": 0,
        }


def write_keyframes(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["keyframe_id", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for keyframe_id in range(25):
            x = 0.5 if keyframe_id == 20 else 0.0 if keyframe_id == 0 else 100.0 + keyframe_id
            writer.writerow([keyframe_id, keyframe_id, x, 0, 0, 0, 0, 0, 1])


def write_pair(root: Path) -> tuple[dict[str, str], Path]:
    evaluator = root / "evaluator"
    evaluator.write_bytes(b"evaluator-fixture")
    evaluator.chmod(0o755)
    fingerprint = {
        "path": str(evaluator.resolve()),
        "size_bytes": evaluator.stat().st_size,
        "sha256": fa02.sha256_file(evaluator),
    }
    provenance = {
        "source_commit": "a" * 40,
        "input_manifest_sha256": "b" * 64,
        "episode_frames_sha256": "c" * 64,
    }
    config = gate.drift_config(PROFILE, 17)
    gt = "0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n"
    odom = "0 0 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n"
    for condition in ("loop_off", "loop_on"):
        directory = root / condition
        directory.mkdir()
        command = [str(evaluator), "--enable_correlated_odom_drift"]
        if condition == "loop_off":
            command.append("--disable_loop_closure")
        (directory / "run_manifest.json").write_text(json.dumps({
            "schema": gate.RUN_SCHEMA,
            "status": "COMPLETE",
            "case_id": "case",
            "condition": condition,
            "episode_id": "episode",
            "dataset": "kitti360",
            "sequence": "sequence",
            **provenance,
            "process_return_code": 0,
            "missing_artifacts": [],
            "drift_config": config,
            "evaluator": fingerprint,
            "command": command,
        }), encoding="utf-8")
        (directory / "metrics.json").write_text(json.dumps({
            "frames_processed": 2,
            "accepted_keyframes": 25,
            "accepted_loop_count": 0 if condition == "loop_off" else 1,
            "loop_closure_enabled": condition == "loop_on",
            "odom_source": "correlated_gt_derived",
            "backend_input_contract": "correlated_gt_derived_odom_plus_lidar",
            "correlated_odom_drift": config,
        }), encoding="utf-8")
        (directory / "trajectory_gt.txt").write_text(gt, encoding="utf-8")
        (directory / "trajectory_odom.txt").write_text(odom, encoding="utf-8")
        (directory / "trajectory_optimized.txt").write_text(
            gt if condition == "loop_on" else odom, encoding="utf-8"
        )
        write_keyframes(directory / "keyframes_gt.csv")
        if condition == "loop_on":
            (directory / "accepted_loops.csv").write_text(
                "query_id,match_id\n20,0\n", encoding="utf-8"
            )
            (directory / "loop_debug.jsonl").write_text(json.dumps({
                "record_type": "candidate",
                "query_id": 20,
                "match_id": 0,
                "gate_result": "accepted",
                "measurement_x": 0.5,
                "measurement_y": 0,
                "measurement_z": 0,
                "measurement_roll": 0,
                "measurement_pitch": 0,
                "measurement_yaw": 0,
            }) + "\n", encoding="utf-8")
        else:
            (directory / "accepted_loops.csv").write_text(
                "query_id,match_id\n", encoding="utf-8"
            )
            (directory / "loop_debug.jsonl").write_text("", encoding="utf-8")
        (directory / "n3map.pbstream").write_bytes(b"fixture")
        (directory / "stderr.log").write_text("", encoding="utf-8")
    return provenance, evaluator


class FA02BToolsTest(unittest.TestCase):
    def test_drift_arguments_are_explicit_and_complete(self) -> None:
        config = gate.drift_config(PROFILE, 73)
        arguments = runner.drift_arguments(config)
        self.assertIn("--enable_correlated_odom_drift", arguments)
        self.assertEqual(arguments[arguments.index("--odom_drift_seed") + 1], "73")
        self.assertIn("--odom_translation_scale_error", arguments)
        self.assertIn("--odom_yaw_bias_deg_per_meter", arguments)
        self.assertIn("--odom_translation_rw_std_m_per_sqrt_meter", arguments)
        self.assertIn("--odom_rotation_rw_std_deg_per_sqrt_meter", arguments)

    def test_pair_gate_accepts_same_input_and_correcting_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance, _ = write_pair(root)
            issues = fa02.Issues()
            report = gate.evaluate_pair(
                {"id": "case", "episode_id": "episode", "profile_id": "moderate", "seed": 17,
                 "expected_role": "positive_correction"},
                {"id": "episode", "dataset": "kitti360", "sequence": "sequence",
                 "max_frames": 2, "attitude_authoritative": True},
                PROFILE,
                root,
                ORACLE,
                THRESHOLDS,
                provenance,
                FakeMapHelpers,
                object(),
                issues,
            )
            self.assertTrue(report["pass"])
            self.assertEqual(report["loops"]["correct_accepted_loop_count"], 1)
            self.assertAlmostEqual(report["ate_translation_improvement_ratio"], 1.0)
            self.assertFalse(issues.items)

    def test_pair_gate_rejects_different_odom_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance, _ = write_pair(root)
            (root / "loop_on" / "trajectory_odom.txt").write_text(
                "0 0 0 0 0 0 0 1\n1 3 0 0 0 0 0 1\n", encoding="utf-8"
            )
            issues = fa02.Issues()
            gate.evaluate_pair(
                {"id": "case", "episode_id": "episode", "profile_id": "moderate", "seed": 17,
                 "expected_role": "positive_correction"},
                {"id": "episode", "dataset": "kitti360", "sequence": "sequence",
                 "max_frames": 2, "attitude_authoritative": True},
                PROFILE,
                root,
                ORACLE,
                THRESHOLDS,
                provenance,
                FakeMapHelpers,
                object(),
                issues,
            )
            self.assertTrue(any(item["code"] == "paired_input" for item in issues.items))


if __name__ == "__main__":
    unittest.main()
