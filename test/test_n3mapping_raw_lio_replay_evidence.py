#!/usr/bin/env python3
"""Focused tests for the Product V1 raw-to-LIO replay evidence contract."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
TOOLS_DIRECTORY = REPOSITORY / "tools"
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))
TOOL_PATH = REPOSITORY / "tools/n3mapping_raw_lio_replay_evidence.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "n3mapping_raw_lio_replay_evidence", TOOL_PATH
)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise RuntimeError(f"cannot load {TOOL_PATH}")
EVIDENCE = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = EVIDENCE
SPECIFICATION.loader.exec_module(EVIDENCE)


def topic_summary(
    topic: str,
    message_type: str,
    count: int = 2,
    first: int = 1_000,
    last: int = 2_000,
    stamp_hash: str = "stamp",
    canonical_hash: str = "canonical",
    serialized_hash: str = "serialized",
):
    return {
        "topic": topic,
        "message_type": message_type,
        "message_count": count,
        "first_header_stamp_ns": first,
        "last_header_stamp_ns": last,
        "header_stamp_sequence_sha256": stamp_hash,
        "canonicalization": EVIDENCE.CANONICALIZATION,
        "canonical_payload_sequence_sha256": canonical_hash,
        "serialized_payload_sequence_sha256": serialized_hash,
    }


def replay_spec():
    return {
        "source_lidar_topic": "/source/lidar",
        "source_imu_topic": "/source/imu",
        "converted_lidar_topic": "/converted/lidar",
        "converted_imu_topic": "/converted/imu",
        "output_lidar_topic": "/livox/lidar",
        "output_imu_topic": "/livox/imu",
        "cloud_topic": "/cloud_registered_body",
        "odom_topic": "/Odometry",
    }


def complete_replay_spec():
    return {
        "source_ros1_bag": "/data/query.bag",
        "converted_ros2_bag": "/data/query_ros2",
        "lio_output_ros2_bag": "/data/query_lio",
        "extractor_output": "/data/query_manifest",
        "fast_lio_executable": "/opt/fast_lio",
        "fast_lio_config": "/etc/fast_lio.yaml",
        "calibration": "/etc/calibration.yaml",
        "replay_tool": "/opt/n3mapping_deterministic_ros2_replay.py",
        "replay_command": "ros2 bag play query_ros2",
        **replay_spec(),
    }


def valid_chain():
    spec = replay_spec()
    source = {
        "topics": {
            spec["source_lidar_topic"]: topic_summary(
                spec["source_lidar_topic"],
                "livox_ros_driver2/CustomMsg",
                canonical_hash="lidar",
                serialized_hash="ros1-lidar",
            ),
            spec["source_imu_topic"]: topic_summary(
                spec["source_imu_topic"],
                "sensor_msgs/Imu",
                count=20,
                first=900,
                last=2_100,
                stamp_hash="imu-stamps",
                canonical_hash="imu",
                serialized_hash="ros1-imu",
            ),
        }
    }
    converted = {
        "topics": {
            spec["converted_lidar_topic"]: topic_summary(
                spec["converted_lidar_topic"],
                "livox_ros_driver2/CustomMsg",
                canonical_hash="lidar",
                serialized_hash="ros2-lidar",
            ),
            spec["converted_imu_topic"]: topic_summary(
                spec["converted_imu_topic"],
                "sensor_msgs/Imu",
                count=20,
                first=900,
                last=2_100,
                stamp_hash="imu-stamps",
                canonical_hash="imu",
                serialized_hash="ros2-imu",
            ),
        }
    }
    output = {
        "topics": {
            spec["output_lidar_topic"]: copy.deepcopy(
                converted["topics"][spec["converted_lidar_topic"]]
            ),
            spec["output_imu_topic"]: copy.deepcopy(
                converted["topics"][spec["converted_imu_topic"]]
            ),
            spec["cloud_topic"]: topic_summary(
                spec["cloud_topic"],
                "sensor_msgs/PointCloud2",
                canonical_hash="cloud",
                serialized_hash="cloud-serialized",
            ),
            spec["odom_topic"]: topic_summary(
                spec["odom_topic"],
                "nav_msgs/Odometry",
                canonical_hash="odom",
                serialized_hash="odom-serialized",
            ),
        }
    }
    output["topics"][spec["output_lidar_topic"]]["topic"] = spec[
        "output_lidar_topic"
    ]
    output["topics"][spec["output_imu_topic"]]["topic"] = spec[
        "output_imu_topic"
    ]
    return source, converted, output, spec


@dataclass
class Stamp:
    sec: int
    nanosec: int


@dataclass
class Header:
    stamp: Stamp
    frame_id: str


@dataclass
class Message:
    header: Header
    values: object


@dataclass
class Ros1Header:
    seq: int
    stamp: Stamp
    frame_id: str
    __msgtype__: str = "std_msgs/msg/Header"


@dataclass
class Ros2Header:
    stamp: Stamp
    frame_id: str
    __msgtype__: str = "std_msgs/msg/Header"


class ReplayChainContractTest(unittest.TestCase):
    def test_exact_chain_passes(self) -> None:
        source, converted, output, spec = valid_chain()
        checks = EVIDENCE.validate_replay_chain(
            source, converted, output, spec
        )
        self.assertTrue(checks)
        self.assertEqual(set(checks.values()), {"PASS"})

    def test_source_conversion_payload_change_fails(self) -> None:
        source, converted, output, spec = valid_chain()
        converted["topics"][spec["converted_lidar_topic"]][
            "canonical_payload_sequence_sha256"
        ] = "changed"
        with self.assertRaisesRegex(
            EVIDENCE.ReplayEvidenceError,
            "ROS1-to-ROS2 LiDAR preservation",
        ):
            EVIDENCE.validate_replay_chain(source, converted, output, spec)

    def test_output_raw_serialization_change_fails(self) -> None:
        source, converted, output, spec = valid_chain()
        output["topics"][spec["output_imu_topic"]][
            "serialized_payload_sequence_sha256"
        ] = "changed"
        with self.assertRaisesRegex(
            EVIDENCE.ReplayEvidenceError,
            "converted-to-output IMU preservation",
        ):
            EVIDENCE.validate_replay_chain(source, converted, output, spec)

    def test_non_exact_cloud_odom_pairing_fails(self) -> None:
        source, converted, output, spec = valid_chain()
        output["topics"][spec["odom_topic"]][
            "header_stamp_sequence_sha256"
        ] = "different"
        with self.assertRaisesRegex(
            EVIDENCE.ReplayEvidenceError,
            "cloud/odometry exact-stamp pairing",
        ):
            EVIDENCE.validate_replay_chain(source, converted, output, spec)


class SourceSliceLineageContractTest(unittest.TestCase):
    def test_original_unsliced_source_remains_supported(self) -> None:
        spec = complete_replay_spec()
        self.assertEqual(EVIDENCE._validate_spec(spec), spec)
        self.assertIsNone(EVIDENCE.verify_source_slice_lineage(spec))

    def test_verified_slice_must_name_exact_replay_source_and_topics(self) -> None:
        spec = {
            **complete_replay_spec(),
            "source_slice_lineage": "/data/query.slice.lineage.json",
        }
        verified = {
            "slice_ros1_bag": spec["source_ros1_bag"],
            "topics": sorted(
                [spec["source_lidar_topic"], spec["source_imu_topic"]]
            ),
            "verdict": "PASS",
        }
        with mock.patch(
            "n3mapping_ros1_raw_bag_slice.verify_lineage",
            return_value=verified,
        ):
            self.assertEqual(
                EVIDENCE.verify_source_slice_lineage(spec), verified
            )

        wrong = {**verified, "slice_ros1_bag": "/data/other.bag"}
        with mock.patch(
            "n3mapping_ros1_raw_bag_slice.verify_lineage",
            return_value=wrong,
        ):
            with self.assertRaisesRegex(
                EVIDENCE.ReplayEvidenceError,
                "different replay source bag",
            ):
                EVIDENCE.verify_source_slice_lineage(spec)

        wrong_topics = {**verified, "topics": ["/wrong", "/topics"]}
        with mock.patch(
            "n3mapping_ros1_raw_bag_slice.verify_lineage",
            return_value=wrong_topics,
        ):
            with self.assertRaisesRegex(
                EVIDENCE.ReplayEvidenceError,
                "topics do not match",
            ):
                EVIDENCE.verify_source_slice_lineage(spec)

    def test_slice_lineage_is_embedded_in_replay_evidence(self) -> None:
        spec = {
            **complete_replay_spec(),
            "source_slice_lineage": "/data/query.slice.lineage.json",
        }
        source, converted, output, _ = valid_chain()
        for label, value in (
            ("source", source),
            ("converted", converted),
            ("output", output),
        ):
            value["identity"] = {
                "path": "/data/{}".format(label),
                "sha256": label,
            }
        lineage = {
            "lineage": spec["source_slice_lineage"],
            "lineage_sha256": "lineage-sha",
            "slice_ros1_bag": spec["source_ros1_bag"],
            "topics": sorted(
                [spec["source_lidar_topic"], spec["source_imu_topic"]]
            ),
            "verdict": "PASS",
        }
        extractor = {
            "identity": {"path": spec["extractor_output"], "sha256": "x"}
        }
        identities = []

        def fake_identity(path, label, executable=False):
            del label, executable
            identity = {"path": str(path), "sha256": str(path)}
            identities.append(identity)
            return identity

        with mock.patch.object(
            EVIDENCE,
            "verify_source_slice_lineage",
            return_value=lineage,
        ), mock.patch.object(
            EVIDENCE,
            "collect_bag_evidence",
            side_effect=[source, converted, output],
        ), mock.patch.object(
            EVIDENCE,
            "collect_ros2_typestore",
            return_value=object(),
        ), mock.patch.object(
            EVIDENCE,
            "validate_replay_chain",
            return_value={"raw_chain": "PASS"},
        ), mock.patch.object(
            EVIDENCE,
            "collect_extractor_evidence",
            return_value=extractor,
        ), mock.patch.object(
            EVIDENCE, "file_identity", side_effect=fake_identity
        ):
            document = EVIDENCE.collect_evidence(
                spec, created_at="2026-07-23T12:00:00+08:00"
            )
        self.assertEqual(
            document["checks"]["source_slice_lineage"], "PASS"
        )
        self.assertEqual(
            document["toolchain"]["source_slice_lineage"], lineage
        )
        self.assertEqual(
            document["artifacts"]["source_slice_lineage"]["path"],
            spec["source_slice_lineage"],
        )
        self.assertEqual(
            document["toolchain"]["replay_tool"]["path"],
            spec["replay_tool"],
        )
        self.assertTrue(identities)


class BagReaderContractTest(unittest.TestCase):
    def test_missing_required_raw_topic_fails_closed(self) -> None:
        lidar_connection = SimpleNamespace(
            topic="/livox/lidar",
            msgtype="livox_ros_driver2/msg/CustomMsg",
        )

        class FakeReader:
            def __init__(self, paths, *, default_typestore):
                del paths, default_typestore
                self.is2 = False
                self.connections = [lidar_connection]

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

        with tempfile.TemporaryDirectory() as temporary:
            bag = Path(temporary) / "query.bag"
            bag.write_bytes(b"bag")
            with mock.patch.object(EVIDENCE, "AnyReader", FakeReader):
                with self.assertRaisesRegex(
                    EVIDENCE.ReplayEvidenceError, "topics mismatch"
                ):
                    EVIDENCE.collect_bag_evidence(
                        bag,
                        "query",
                        "ros1",
                        ["/livox/lidar", "/livox/imu"],
                    )

    def test_duplicate_header_stamp_fails_closed(self) -> None:
        connection = SimpleNamespace(
            topic="/livox/imu", msgtype="sensor_msgs/msg/Imu"
        )
        repeated = Message(Header(Stamp(123, 456), "imu"), [1.0])

        class FakeReader:
            def messages(self, connections):
                self.connections = connections
                yield connection, 1, b"first"
                yield connection, 2, b"second"

            def deserialize(self, rawdata, message_type):
                del rawdata, message_type
                return repeated

        with self.assertRaisesRegex(
            EVIDENCE.ReplayEvidenceError,
            "header stamps are not strictly increasing",
        ):
            EVIDENCE.summarize_connection(FakeReader(), connection)


class CanonicalPayloadTest(unittest.TestCase):
    def test_canonical_hash_binds_nested_values_and_exact_stamp(self) -> None:
        first = Message(
            Header(Stamp(123, 456), "livox_frame"),
            [1, 2.5, b"payload"],
        )
        same = copy.deepcopy(first)
        changed = copy.deepcopy(first)
        changed.header.stamp.nanosec += 1
        self.assertEqual(
            EVIDENCE.canonical_message_sha256(first),
            EVIDENCE.canonical_message_sha256(same),
        )
        self.assertNotEqual(
            EVIDENCE.canonical_message_sha256(first),
            EVIDENCE.canonical_message_sha256(changed),
        )

    def test_message_stamp_preserves_integer_nanoseconds(self) -> None:
        message = Message(
            Header(Stamp(1_784_778_343, 719_489_384), "frame"),
            [],
        )
        self.assertEqual(
            EVIDENCE.message_stamp_ns(message),
            1_784_778_343_719_489_384,
        )

    def test_ros1_header_seq_is_the_only_cross_format_drop(self) -> None:
        ros1 = Ros1Header(42, Stamp(123, 456), "frame")
        ros2 = Ros2Header(Stamp(123, 456), "frame")
        self.assertEqual(
            EVIDENCE.canonical_message_sha256(ros1),
            EVIDENCE.canonical_message_sha256(ros2),
        )
        changed = Ros2Header(Stamp(123, 457), "frame")
        self.assertNotEqual(
            EVIDENCE.canonical_message_sha256(ros1),
            EVIDENCE.canonical_message_sha256(changed),
        )


class StrictArtifactIdentityTest(unittest.TestCase):
    def test_directory_identity_binds_paths_sizes_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"one")
            first = EVIDENCE.directory_identity(root, "tree")
            (root / "a").write_bytes(b"two")
            second = EVIDENCE.directory_identity(root, "tree")
            self.assertNotEqual(first["sha256"], second["sha256"])
            (root / "b").write_bytes(b"")
            third = EVIDENCE.directory_identity(root, "tree")
            self.assertNotEqual(second["sha256"], third["sha256"])
            (root / "empty").mkdir()
            fourth = EVIDENCE.directory_identity(root, "tree")
            self.assertNotEqual(third["sha256"], fourth["sha256"])

    def test_directory_identity_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"data")
            os.symlink(target, root / "link")
            with self.assertRaisesRegex(
                EVIDENCE.ReplayEvidenceError, "symlinks are forbidden"
            ):
                EVIDENCE.directory_identity(root, "tree")

    def test_strict_json_rejects_duplicate_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"schema": 1, "schema": 2}\\n', encoding="utf-8")
            with self.assertRaisesRegex(
                EVIDENCE.ReplayEvidenceError, "duplicate JSON key"
            ):
                EVIDENCE.load_strict_json(path, "test JSON")


class ExtractorContractTest(unittest.TestCase):
    def test_extractor_must_cover_every_exact_output_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extractor = root / "extractor"
            extractor.mkdir()
            stamps = [1_000, 2_000]
            pcd_entries = []
            for index in range(2):
                name = f"frame_{index:06d}.pcd"
                path = extractor / name
                path.write_bytes(f"pcd-{index}".encode("ascii"))
                pcd_entries.append(
                    {
                        "path": name,
                        "bytes": path.stat().st_size,
                        "sha256": EVIDENCE.sha256_file(path),
                    }
                )
            frames = extractor / "frames.csv"
            with frames.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(EVIDENCE.EXTRACTOR_COLUMNS)
                for index, stamp_ns in enumerate(stamps):
                    writer.writerow(
                        [
                            "episode",
                            index,
                            stamp_ns,
                            pcd_entries[index]["path"],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                        ]
                    )
            output_bag = root / "lio_output_ros2"
            stamp_hash = EVIDENCE._sequence_sha256(stamps)
            metadata = {
                "schema": "n3mapping_ros2_relocalization_manifest_v1",
                "bag_format": "ros2",
                "bag": str(output_bag),
                "cloud_topic": "/cloud_registered_body",
                "odom_topic": "/Odometry",
                "selected_start_offset_s": 0.0,
                "selected_duration_s": 0.0,
                "frame_count": 2,
                "pairing": "exact_header_stamp",
                "frames_csv_sha256": EVIDENCE.sha256_file(frames),
                "pcd_set_sha256": EVIDENCE.pcd_set_sha256(pcd_entries),
                "pcd_files": pcd_entries,
            }
            metadata_path = extractor / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_summary = {
                "message_count": 2,
                "first_header_stamp_ns": stamps[0],
                "last_header_stamp_ns": stamps[-1],
                "header_stamp_sequence_sha256": stamp_hash,
            }
            result = EVIDENCE.collect_extractor_evidence(
                extractor,
                output_bag,
                "/cloud_registered_body",
                "/Odometry",
                output_summary,
                output_summary,
            )
            self.assertEqual(result["frame_count"], 2)
            self.assertEqual(
                result["pcd_set_sha256"],
                EVIDENCE.pcd_set_sha256(pcd_entries),
            )

    def test_extractor_rejects_partial_output(self) -> None:
        source, converted, output, spec = valid_chain()
        del source, converted
        cloud = output["topics"][spec["cloud_topic"]]
        odom = output["topics"][spec["odom_topic"]]
        expected = {
            "message_count": 1,
            "first_header_stamp_ns": 1_000,
            "last_header_stamp_ns": 1_000,
            "header_stamp_sequence_sha256": "one",
        }
        with self.assertRaisesRegex(
            EVIDENCE.ReplayEvidenceError, "stamp coverage"
        ):
            EVIDENCE._require_equal_fields(
                cloud,
                expected,
                list(expected),
                "extractor-to-cloud stamp coverage",
            )
        self.assertEqual(odom["message_count"], 2)


if __name__ == "__main__":
    unittest.main()
