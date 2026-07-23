#!/usr/bin/env python3
"""Focused helper-level tests for relocalization data-contract tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]


def load_tool(name: str, relative_path: str):
    specification = importlib.util.spec_from_file_location(
        name, REPOSITORY / relative_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


EXTRACT = load_tool(
    "extract_relocalization_manifest",
    "tools/extract_ros1_relocalization_manifest.py",
)
JOIN = load_tool(
    "same_frame_reference_join",
    "tools/n3mapping_same_frame_reference_join.py",
)
INVENTORY = load_tool(
    "ros1_raw_bag_inventory",
    "tools/n3mapping_ros1_raw_bag_inventory.py",
)


def stamped_message(stamp_ns: int):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            )
        )
    )


class ExtractManifestHelpersTest(unittest.TestCase):
    def test_stamp_ns_preserves_integer_nanoseconds(self) -> None:
        expected = 1_784_778_343_699_753_523
        self.assertEqual(EXTRACT.stamp_ns(stamped_message(expected)), expected)

    def test_anyreader_path_supports_ros1_and_ros2_without_stamp_conversion(
        self,
    ) -> None:
        cloud_connection = SimpleNamespace(
            topic="/cloud", msgtype="sensor_msgs/msg/PointCloud2"
        )
        odom_connection = SimpleNamespace(
            topic="/odom", msgtype="nav_msgs/msg/Odometry"
        )
        cloud_stamp = 1_784_778_343_699_753_523
        odom_stamp = cloud_stamp
        messages = {
            b"cloud": stamped_message(cloud_stamp),
            b"odom": stamped_message(odom_stamp),
        }

        class FakeAnyReader:
            def __init__(self, paths, *, default_typestore):
                del default_typestore
                self.is2 = paths[0].suffix != ".bag"
                self.connections = [cloud_connection, odom_connection]

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def messages(self, connections):
                self.assert_connections = connections
                yield cloud_connection, 1, b"cloud"
                yield odom_connection, 2, b"odom"

            def deserialize(self, rawdata, unused_msgtype):
                return messages[rawdata]

        with mock.patch.object(EXTRACT, "AnyReader", FakeAnyReader):
            for path, expected_format in (
                (Path("query.bag"), "ros1"),
                (Path("query_ros2"), "ros2"),
            ):
                odometry, clouds, bag_format = EXTRACT.read_bag_messages(
                    path, "/cloud", "/odom"
                )
                self.assertEqual(bag_format, expected_format)
                self.assertEqual(list(odometry), [odom_stamp])
                self.assertEqual(clouds[0][0], cloud_stamp)

    def test_pcd_set_hash_binds_order_path_size_and_content_hash(self) -> None:
        entries = [
            {"path": "frame_000000.pcd", "bytes": 10, "sha256": "a" * 64},
            {"path": "frame_000001.pcd", "bytes": 20, "sha256": "b" * 64},
        ]
        original = EXTRACT.pcd_set_sha256(entries)
        self.assertEqual(original, EXTRACT.pcd_set_sha256(list(entries)))
        self.assertNotEqual(original, EXTRACT.pcd_set_sha256(list(reversed(entries))))
        changed = [dict(entry) for entry in entries]
        changed[1]["sha256"] = "c" * 64
        self.assertNotEqual(original, EXTRACT.pcd_set_sha256(changed))


class SameFrameReferenceJoinHelpersTest(unittest.TestCase):
    def test_raw_acquisition_stamp_uses_exact_header_nanoseconds(self) -> None:
        expected = 1_784_778_343_719_489_384
        self.assertEqual(
            JOIN.message_stamp_ns(stamped_message(expected)), expected
        )

    def test_unique_match_is_fixed_tolerance_and_fails_on_ambiguity(self) -> None:
        self.assertEqual(
            JOIN.unique_match_indices(
                [1_000, 3_000],
                [1_512, 3_001],
                tolerance_ns=JOIN.JOIN_TOLERANCE_NS,
                label="test",
            ),
            [0, 1],
        )
        with self.assertRaisesRegex(
            JOIN.SameFrameReferenceError, "2 candidates"
        ):
            JOIN.unique_match_indices(
                [1_000],
                [600, 1_400],
                tolerance_ns=JOIN.JOIN_TOLERANCE_NS,
                label="test",
            )
        with self.assertRaisesRegex(
            JOIN.SameFrameReferenceError, "0 candidates"
        ):
            JOIN.unique_match_indices(
                [1_000],
                [1_513],
                tolerance_ns=JOIN.JOIN_TOLERANCE_NS,
                label="test",
            )

    def test_reference_rows_are_exact_dense_subset(self) -> None:
        dense = [
            JOIN.DensePose(1_000, 10, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            JOIN.DensePose(2_001, 11, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
        ]
        lio = [
            {"stamp_ns": 1_000, "episode_id": "q", "frame_index": "0"},
            {"stamp_ns": 2_000, "episode_id": "q", "frame_index": "1"},
        ]
        rows = JOIN.build_reference_rows(lio, dense)
        self.assertEqual([row["stamp_ns"] for row in rows], [1_000, 2_000])
        self.assertEqual(rows[0]["manifest_minus_dense_ns"], 0)
        self.assertEqual(rows[1]["tx"], 4.0)

    def test_dense_evidence_binds_csv_and_pbstream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root / "dense.csv"
            trajectory.write_text("header\nrow\n", encoding="utf-8")
            pbstream = root / "map.pbstream"
            pbstream.write_bytes(b"map")
            evidence = root / "dense.evidence.json"
            evidence.write_text(
                json.dumps(
                    {
                        "schema": "n3mapping_dense_trajectory_evidence_v1",
                        "pbstream": str(pbstream),
                        "pbstream_sha256": JOIN.sha256_file(pbstream),
                        "trajectory_csv": str(trajectory),
                        "trajectory_csv_sha256": JOIN.sha256_file(trajectory),
                        "row_count": 1,
                        "source": "native",
                        "degraded": False,
                        "pose_convention": "T_world_body",
                    }
                ),
                encoding="utf-8",
            )
            validated = JOIN.validate_dense_evidence(
                evidence, trajectory, expected_rows=1
            )
            self.assertEqual(validated["source"], "native")
            pbstream.write_bytes(b"tampered map")
            with self.assertRaisesRegex(
                JOIN.SameFrameReferenceError, "pbstream SHA-256"
            ):
                JOIN.validate_dense_evidence(
                    evidence, trajectory, expected_rows=1
                )
            pbstream.write_bytes(b"map")
            trajectory.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                JOIN.SameFrameReferenceError, "SHA-256"
            ):
                JOIN.validate_dense_evidence(
                    evidence, trajectory, expected_rows=1
                )

    def test_inventory_contract_rejects_partial_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query = root / "query.bag"
            query.write_bytes(b"bag")
            inventory_path = root / "inventory.json"
            inventory_path.write_text(
                json.dumps(
                    {
                        "schema": "n3mapping_ros1_raw_bag_inventory_v1",
                        "file_hashes_included": True,
                        "full_bag": {"sha256": "a" * 64},
                        "query_bags": [
                            {
                                "file": query.name,
                                "sha256": "b" * 64,
                                "payload_overlap": {
                                    "reference_class": (
                                        "partial_same_frame_reference"
                                    ),
                                    "matched_payload_ratio": 0.5,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                JOIN.SameFrameReferenceError, "same_frame_reference"
            ):
                JOIN.validate_inventory(inventory_path, query)


class RawBagInventoryHelpersTest(unittest.TestCase):
    def test_repeated_query_payload_cannot_reuse_one_full_occurrence(
        self,
    ) -> None:
        connection = SimpleNamespace(topic="/lidar")

        class FakeReader:
            def __init__(self, unused_path):
                self.path = Path("query.bag")
                self.connections = [connection]

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def messages(self, connections):
                self.received_connections = connections
                yield connection, 100, b"same"
                yield connection, 200, b"same"

        fingerprint = INVENTORY.hashlib.sha256(b"same").hexdigest()
        with mock.patch.object(INVENTORY, "Reader", FakeReader):
            result = INVENTORY.query_payload_overlap(
                Path("query.bag"),
                "/lidar",
                {fingerprint: [50]},
            )
        self.assertEqual(result["matched_payload_count"], 2)
        self.assertEqual(result["unique_payload_count"], 1)
        self.assertEqual(
            result["reference_class"], "partial_same_frame_reference"
        )


if __name__ == "__main__":
    unittest.main()
