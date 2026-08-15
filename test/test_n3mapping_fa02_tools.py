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

import n3mapping_fa02_freeze as freeze_tool  # noqa: E402
import n3mapping_fa02_gate as gate_tool  # noqa: E402
import n3mapping_fa02_run as run_tool  # noqa: E402


ORACLE = {
    "minimum_keyframe_id_gap": 20,
    "place_translation_threshold_m": 5.0,
    "correct_measurement_translation_error_m_max": 1.0,
    "correct_measurement_rotation_error_deg_max": 10.0,
    "catastrophic_place_distance_m_min": 10.0,
    "catastrophic_measurement_translation_error_m_min": 2.0,
    "catastrophic_measurement_rotation_error_deg_min": 30.0,
    "revisit_segment_max_query_id_gap": 5,
}
THRESHOLDS = {
    "minimum_correct_accepted_loops_per_positive_episode": 1,
    "optimizer_error_count_max": 0,
    "nonfinite_pose_count_max": 0,
    "ate_translation_rmse_m_max": 0.5,
    "ate_translation_p95_m_max": 1.0,
    "ate_rotation_rmse_deg_max": 5.0,
    "rpe_translation_rmse_m_max": 0.2,
    "rpe_rotation_rmse_deg_max": 2.0,
}


class FakeMapHelpers:
    @staticmethod
    def parse_map(path: Path, module: object) -> object:
        if not path.is_file():
            raise ValueError("map missing")
        return object()

    @staticmethod
    def map_structure(map_proto: object, module: object) -> dict[str, object]:
        return {
            "keyframes": 25,
            "edges": 25,
            "odometry": 24,
            "loop": 1,
            "session_anchor": 0,
            "unknown_edges": 0,
            "dense_trajectory": 2,
            "metadata_match": True,
            "duplicate_keyframes": 0,
            "dangling_edges": 0,
            "session_anchors": [],
        }


def write_episode(directory: Path, measurement_x: float = 0.5) -> None:
    directory.mkdir()
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": gate_tool.RUN_SCHEMA,
                "status": "COMPLETE",
                "process_return_code": 0,
            }
        ),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "frames_processed": 2,
                "accepted_keyframes": 25,
                "backend_input_contract": "gt_pose_plus_lidar",
                "trajectory_optimized_semantics": "final_dense_after_all_loop_updates",
            }
        ),
        encoding="utf-8",
    )
    with (directory / "keyframes_gt.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["keyframe_id", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for keyframe_id in range(25):
            x = 0.5 if keyframe_id == 20 else 0.0 if keyframe_id == 0 else 100.0 + 10.0 * keyframe_id
            writer.writerow([keyframe_id, keyframe_id, x, 0, 0, 0, 0, 0, 1])
    (directory / "accepted_loops.csv").write_text(
        "query_id,match_id\n20,0\n", encoding="utf-8"
    )
    (directory / "loop_debug.jsonl").write_text(
        json.dumps(
            {
                "record_type": "candidate",
                "query_id": 20,
                "match_id": 0,
                "gate_result": "accepted",
                "measurement_x": measurement_x,
                "measurement_y": 0,
                "measurement_z": 0,
                "measurement_roll": 0,
                "measurement_pitch": 0,
                "measurement_yaw": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory = "0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n"
    (directory / "trajectory_gt.txt").write_text(trajectory, encoding="utf-8")
    (directory / "trajectory_optimized.txt").write_text(trajectory, encoding="utf-8")
    (directory / "n3map.pbstream").write_bytes(b"fixture")
    (directory / "stderr.log").write_text("", encoding="utf-8")


class FA02ToolsTest(unittest.TestCase):
    def test_kitti_selection_matches_stride_then_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = "2013_05_28_drive_0005_sync"
            lidar = root / "data_3d_raw" / sequence / "velodyne_points" / "data"
            poses = root / "data_poses" / sequence
            calibration = root / "calibration"
            lidar.mkdir(parents=True)
            poses.mkdir(parents=True)
            calibration.mkdir()
            for frame_id in range(8):
                (lidar / f"{frame_id:010d}.bin").write_bytes(bytes([frame_id]))
            (poses / "poses.txt").write_text(
                "".join(f"{frame_id} 1 0 0 0 0 1 0 0 0 0 1 0\n" for frame_id in range(8)),
                encoding="utf-8",
            )
            (calibration / "calib_cam_to_velo.txt").write_text("fixture", encoding="utf-8")
            (calibration / "calib_cam_to_pose.txt").write_text("fixture", encoding="utf-8")
            selected, evidence = freeze_tool.select_kitti360(
                root,
                {"sequence": sequence, "stride": 2, "start_index": 1, "max_frames": 3, "id": "test"},
            )
            self.assertEqual([path.stem for path in selected], ["0000000002", "0000000004", "0000000006"])
            self.assertEqual(evidence["alignment_common_count"], 8)

    def test_m2dgr_selection_matches_cpp_nearest_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "M2DGR"
            repo = Path(temporary) / "repo"
            sequence_dir = root / "gate_02"
            lidar = sequence_dir / "velodyne_points"
            calibration = repo / "config" / "eval"
            lidar.mkdir(parents=True)
            calibration.mkdir(parents=True)
            for index in range(8):
                (lidar / f"{1000 + index * 0.1:.9f}.bin").write_bytes(bytes([index]))
            (sequence_dir / "gate_02.txt").write_text(
                "".join(
                    f"{1000 + index * 0.1:.9f} {index} 0 0 0 0 0 1\n"
                    for index in range(8)
                ),
                encoding="utf-8",
            )
            (calibration / "m2dgr.txt").write_text("fixture", encoding="utf-8")
            selected, evidence = freeze_tool.select_m2dgr(
                root,
                repo,
                {
                    "sequence": "gate_02",
                    "gt_file": "gate_02.txt",
                    "max_time_diff_s": 0.01,
                    "stride": 2,
                    "start_index": 1,
                    "max_frames": 3,
                    "calibration_file": "config/eval/m2dgr.txt",
                    "gt_sensor_frame": "xsens",
                    "id": "test",
                },
            )
            self.assertEqual(
                [path.stem for path in selected],
                ["1000.200000000", "1000.400000000", "1000.600000000"],
            )
            self.assertEqual(evidence["alignment_common_count"], 8)

    def test_episode_gate_accepts_correct_measurement_and_final_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "episode"
            write_episode(directory)
            issues = gate_tool.Issues()
            report = gate_tool.evaluate_episode(
                {
                    "id": "positive",
                    "dataset": "kitti360",
                    "expected_role": "positive_revisit",
                    "attitude_authoritative": True,
                    "max_frames": 2,
                },
                directory,
                ORACLE,
                THRESHOLDS,
                FakeMapHelpers,
                object(),
                issues,
            )
            self.assertEqual(report["correct_accepted_loop_count"], 1)
            self.assertEqual(report["catastrophic_false_loop_count"], 0)
            self.assertEqual(report["revisit_segment_hit_count"], 1)
            self.assertFalse(issues.items)

    def test_episode_gate_labels_bad_measurement_catastrophic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "episode"
            write_episode(directory, measurement_x=3.0)
            issues = gate_tool.Issues()
            report = gate_tool.evaluate_episode(
                {
                    "id": "positive",
                    "dataset": "kitti360",
                    "expected_role": "positive_revisit",
                    "attitude_authoritative": True,
                    "max_frames": 2,
                },
                directory,
                ORACLE,
                THRESHOLDS,
                FakeMapHelpers,
                object(),
                issues,
            )
            self.assertEqual(report["correct_accepted_loop_count"], 0)
            self.assertEqual(report["catastrophic_false_loop_count"], 1)
            self.assertTrue(any(issue["code"] == "positive_loop_miss" for issue in issues.items))

    def test_episode_gate_rejects_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "episode"
            write_episode(directory)
            issues = gate_tool.Issues()
            gate_tool.evaluate_episode(
                {
                    "id": "positive",
                    "dataset": "kitti360",
                    "sequence": "sequence",
                    "expected_role": "positive_revisit",
                    "attitude_authoritative": True,
                    "max_frames": 2,
                },
                directory,
                ORACLE,
                THRESHOLDS,
                FakeMapHelpers,
                object(),
                issues,
                {
                    "source_commit": "a" * 40,
                    "input_manifest_sha256": "b" * 64,
                    "episode_frames_sha256": "c" * 64,
                },
            )
            self.assertTrue(any(issue["code"] == "run_provenance" for issue in issues.items))
            self.assertTrue(any(issue["code"] == "evaluator_fingerprint" for issue in issues.items))

    def test_runner_m2dgr_command_is_explicit_about_gt_and_calibration(self) -> None:
        episode = {
            "id": "m2dgr",
            "dataset": "m2dgr",
            "sequence": "gate_02",
            "input_voxel_size_m": 0.5,
            "gt_sensor_frame": "xsens",
            "calibration_file": "config/eval/calibration.txt",
            "max_time_diff_s": 0.05,
            "normalize_gt_origin": True,
        }
        _, command = run_tool.episode_command(
            episode,
            {"inputs": {"gt": {"path": "/data/gate_02.txt"}}},
            {"fa02": {"dataset_roots": {"m2dgr": "/data"}}},
            Path("/tmp/frames.csv"),
            Path("/tmp/output"),
            Path("/repo"),
            Path("/bin/kitti"),
            Path("/bin/m2dgr"),
        )
        self.assertIn("--gt_sensor_frame", command)
        self.assertIn("--calibration_file", command)
        self.assertIn("--normalize_gt_origin", command)


if __name__ == "__main__":
    unittest.main()
