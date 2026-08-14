#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "n3mapping_submap_graph_trial_compare.py"
SPEC = importlib.util.spec_from_file_location("submap_graph_trial_compare", TOOL_PATH)
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


def trial_record() -> dict[str, object]:
    return {
        "schema": TOOL.TRIAL_SCHEMA,
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
        "solved": True,
        "node_count": 2,
        "snapshot_owned_keyframe_count": 2,
        "snapshot_cross_edge_count": 1,
        "nodes": [
            {"submap_id": 0, "optimized_pose": pose()},
            {"submap_id": 1, "optimized_pose": pose(1.2)},
        ],
        "keyframes": [
            {
                "keyframe_id": 10,
                "submap_id": 0,
                "reference_pose": pose(),
                "initial_shadow_pose": pose(),
                "optimized_shadow_pose": pose(),
            },
            {
                "keyframe_id": 20,
                "submap_id": 1,
                "reference_pose": pose(1.0),
                "initial_shadow_pose": pose(1.1),
                "optimized_shadow_pose": pose(1.2),
            },
        ],
    }


def write_trial_and_qualification(
    directory: Path,
    classification: str = TOOL.QUALIFIED,
) -> tuple[Path, Path]:
    trial = directory / "trial.jsonl"
    payload = (json.dumps(trial_record(), separators=(",", ":")) + "\n").encode()
    trial.write_bytes(payload)
    qualification = directory / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema": TOOL.QUALIFICATION_SCHEMA,
                "classification": classification,
                "input": {
                    "path": str(trial),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "lineage": {
                    "expected_commit": COMMIT,
                    "expected_profile_sha256": PROFILE,
                },
                "selected_record": {
                    "line": 1,
                    "node_count": 2,
                    "owned_keyframe_count": 2,
                    "cross_edge_count": 1,
                    "complete_final_multi_submap": True,
                },
                "record_count": 1,
                "error_count": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return trial, qualification


class SubmapGraphTrialCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proto_temporary = tempfile.TemporaryDirectory()
        cls.proto_dir = Path(cls.proto_temporary.name)
        subprocess.run(
            [
                "protoc",
                f"-I{ROOT / 'proto'}",
                f"--python_out={cls.proto_dir}",
                str(ROOT / "proto" / "n3map.proto"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        module_path = cls.proto_dir / "n3map_pb2.py"
        spec = importlib.util.spec_from_file_location("sg09_test_n3map_pb2", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {module_path}")
        cls.proto = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.proto)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proto_temporary.cleanup()

    def write_map(
        self,
        path: Path,
        *,
        second_reference_x: float = 1.0,
        second_dense_offset_s: float = 0.0,
        invalid_cloud_shape: bool = False,
    ) -> None:
        result = self.proto.N3Map()
        result.metadata.version = "2.5.0"
        result.metadata.num_keyframes = 2
        for keyframe_id, stamp, x in ((10, 1.0, 0.0), (20, 2.0, second_reference_x)):
            keyframe = result.keyframes.add()
            keyframe.id = keyframe_id
            keyframe.timestamp = stamp
            for target in (keyframe.pose_odom, keyframe.pose_optimized):
                target.tx = x
                target.qw = 1.0
            keyframe.cloud.num_points = 2
            keyframe.cloud.points.extend(
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0]
            )
        for submap_id, keyframe_id in ((0, 10), (1, 20)):
            submap = result.submaps.add()
            submap.id = submap_id
            submap.keyframe_ids.append(keyframe_id)
        if invalid_cloud_shape:
            result.keyframes[1].cloud.num_points = 3
        for seq, stamp, x in (
            (0, 1.0, 0.0),
            (1, 2.0 + second_dense_offset_s, 1.0),
        ):
            dense = result.dense_optimized_trajectory.add()
            dense.seq = seq
            dense.timestamp = stamp
            dense.pose_world_lidar.tx = x
            dense.pose_world_lidar.qw = 1.0
        path.write_bytes(result.SerializeToString())

    def run_compare(
        self,
        directory: Path,
        *,
        second_reference_x: float = 1.0,
        second_dense_offset_s: float = 0.0,
        invalid_cloud_shape: bool = False,
        max_dense_time_error_ms: float = 1.0,
    ) -> dict[str, object]:
        trial, qualification = write_trial_and_qualification(directory)
        map_path = directory / "map.pbstream"
        self.write_map(
            map_path,
            second_reference_x=second_reference_x,
            second_dense_offset_s=second_dense_offset_s,
            invalid_cloud_shape=invalid_cloud_shape,
        )
        return TOOL.compare(
            qualification,
            trial,
            map_path,
            self.proto_dir,
            directory / "comparison",
            max_dense_time_error_ms,
            max_points_per_keyframe=2,
            preview_points=4,
            vector_magnification=10.0,
        )

    def test_complete_comparison_is_review_ready_without_writeback_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.run_compare(root)
            output = root / "comparison"

            self.assertEqual(report["classification"], TOOL.READY)
            self.assertFalse(report["writeback_authorized"])
            self.assertTrue(report["pairing"]["keyframe_id_sets_match_exactly"])
            self.assertEqual(
                report["trajectory_proxy"]["optimized_translation_proxy_change_counts"],
                {
                    "lower_than_reference": 0,
                    "higher_than_reference": 1,
                    "equal_to_reference": 1,
                },
            )
            displacement = report["map_geometry_deformation"]["same_point_displacement_m"]
            self.assertEqual(displacement["count"], 4)
            self.assertAlmostEqual(displacement["max"], 0.2)
            self.assertTrue((output / "comparison_summary.json").is_file())
            self.assertTrue((output / "trajectory_comparison.csv").is_file())
            self.assertTrue((output / "map_geometry_displacement.csv").is_file())
            self.assertTrue((output / "reference_map_samples.pcd").is_file())
            self.assertTrue((output / "optimized_shadow_map_samples.pcd").is_file())
            self.assertIn("SG-09 shadow review", (output / "top_view_overlay.svg").read_text())

    def test_qualification_hash_binds_trial_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial, qualification = write_trial_and_qualification(root)
            trial.write_text(trial.read_text() + "\n", encoding="utf-8")
            map_path = root / "map.pbstream"
            self.write_map(map_path)
            with self.assertRaisesRegex(TOOL.ComparisonError, "does not bind"):
                TOOL.compare(
                    qualification,
                    trial,
                    map_path,
                    self.proto_dir,
                    root / "comparison",
                    1.0,
                )

    def test_nonqualified_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial, qualification = write_trial_and_qualification(
                root, TOOL.INSUFFICIENT
            )
            map_path = root / "map.pbstream"
            self.write_map(map_path)
            with self.assertRaisesRegex(TOOL.ComparisonError, "not QUALIFIED"):
                TOOL.compare(
                    qualification,
                    trial,
                    map_path,
                    self.proto_dir,
                    root / "comparison",
                    1.0,
                )

    def test_map_reference_pose_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TOOL.ComparisonError, "map pose does not match"):
                self.run_compare(Path(temporary), second_reference_x=1.01)

    def test_dense_time_gap_is_insufficient_not_quality_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_compare(
                Path(temporary),
                second_dense_offset_s=0.01,
                max_dense_time_error_ms=1.0,
            )
            self.assertEqual(report["classification"], TOOL.INSUFFICIENT)
            self.assertFalse(report["trajectory_proxy"]["complete"])

    def test_invalid_cloud_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TOOL.ComparisonError, "cloud shape"):
                self.run_compare(Path(temporary), invalid_cloud_shape=True)

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial, qualification = write_trial_and_qualification(root)
            map_path = root / "map.pbstream"
            self.write_map(map_path)
            output = root / "comparison"
            output.mkdir()
            (output / "stale").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(TOOL.ComparisonError, "not empty"):
                TOOL.compare(
                    qualification,
                    trial,
                    map_path,
                    self.proto_dir,
                    output,
                    1.0,
                )


if __name__ == "__main__":
    unittest.main()
