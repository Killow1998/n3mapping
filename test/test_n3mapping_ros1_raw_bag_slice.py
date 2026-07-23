#!/usr/bin/env python3
"""Focused tests for deterministic ROS1 raw bag slicing and lineage."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


REPOSITORY = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY / "tools/n3mapping_ros1_raw_bag_slice.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "n3mapping_ros1_raw_bag_slice", TOOL_PATH
)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise RuntimeError("cannot load {}".format(TOOL_PATH))
SLICER = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = SLICER
SPECIFICATION.loader.exec_module(SLICER)


def make_imu(typestore, stamp_ns, sequence):
    types = typestore.types
    time_type = types["builtin_interfaces/msg/Time"]
    header_type = types["std_msgs/msg/Header"]
    quaternion_type = types["geometry_msgs/msg/Quaternion"]
    vector_type = types["geometry_msgs/msg/Vector3"]
    imu_type = types["sensor_msgs/msg/Imu"]
    stamp = time_type(
        sec=stamp_ns // 1_000_000_000,
        nanosec=stamp_ns % 1_000_000_000,
    )
    return imu_type(
        header=header_type(
            seq=sequence,
            stamp=stamp,
            frame_id="sensor",
        ),
        orientation=quaternion_type(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.zeros(9, dtype=np.float64),
        angular_velocity=vector_type(x=0.0, y=0.0, z=0.0),
        angular_velocity_covariance=np.zeros(9, dtype=np.float64),
        linear_acceleration=vector_type(x=0.0, y=0.0, z=9.8),
        linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
    )


def write_source(path):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    records = [
        ("/lidar", 90, 1),
        ("/imu", 95, 2),
        ("/lidar", 100, 3),
        ("/imu", 110, 4),
        ("/other", 150, 5),
        ("/imu", 180, 6),
        ("/lidar", 199, 7),
        ("/imu", 200, 8),
        ("/lidar", 200, 9),
    ]
    with Writer(path) as writer:
        connections = {
            topic: writer.add_connection(
                topic, "sensor_msgs/msg/Imu", typestore=typestore
            )
            for topic in ("/lidar", "/imu", "/other")
        }
        for topic, storage_stamp, sequence in records:
            message = make_imu(
                typestore, storage_stamp - 10, sequence
            )
            writer.write(
                connections[topic],
                storage_stamp,
                typestore.serialize_ros1(
                    message, "sensor_msgs/msg/Imu"
                ),
            )


class RawBagSliceTest(unittest.TestCase):
    def create_fixture(self, root):
        source = root / "source.bag"
        sliced = root / "slice.bag"
        lineage = root / "slice.lineage.json"
        write_source(source)
        result = SLICER.create_slice_and_lineage(
            source,
            sliced,
            lineage,
            100,
            200,
            ["/lidar", "/imu"],
        )
        return source, sliced, lineage, result

    def test_half_open_slice_is_raw_byte_exact_and_verifiable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sliced, lineage, result = self.create_fixture(root)
            self.assertTrue(source.is_file())
            self.assertTrue(sliced.is_file())
            self.assertEqual(result["verdict"], "PASS")
            self.assertEqual(result["message_count"], 4)
            frozen = SLICER.load_strict_json(lineage, "lineage")
            selection = frozen["selection"]
            self.assertEqual(selection["message_count"], 4)
            self.assertEqual(selection["first_storage_stamp_ns"], 100)
            self.assertEqual(selection["last_storage_stamp_ns"], 199)
            self.assertEqual(
                selection["topics"]["/lidar"]["message_count"], 2
            )
            self.assertEqual(
                selection["topics"]["/imu"]["message_count"], 2
            )
            self.assertEqual(
                set(selection["topics"]), {"/imu", "/lidar"}
            )
            self.assertEqual(
                SLICER.verify_lineage(lineage), result
            )

    def test_source_or_slice_byte_change_fails_verification(self):
        for target_name in ("source", "slice"):
            with self.subTest(target=target_name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    source, sliced, lineage, _ = self.create_fixture(root)
                    target = source if target_name == "source" else sliced
                    with target.open("ab") as stream:
                        stream.write(b"tamper")
                    with self.assertRaisesRegex(
                        SLICER.SliceLineageError,
                        "does not match freshly rebuilt evidence",
                    ):
                        SLICER.verify_lineage(lineage)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema":1,"schema":2}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SLICER.SliceLineageError, "duplicate JSON key"
            ):
                SLICER.load_strict_json(duplicate, "lineage")
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                SLICER.SliceLineageError, "non-finite JSON"
            ):
                SLICER.load_strict_json(nonfinite, "lineage")

    def test_symlink_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bag"
            write_source(source)
            link = root / "source-link.bag"
            os.symlink(source, link)
            with self.assertRaisesRegex(
                SLICER.SliceLineageError, "symlinks are forbidden"
            ):
                SLICER.copy_raw_slice(
                    link,
                    root / "slice.bag",
                    100,
                    200,
                    ["/lidar", "/imu"],
                )

    def test_duplicate_topic_and_existing_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bag"
            write_source(source)
            with self.assertRaisesRegex(
                SLICER.SliceLineageError, "duplicates"
            ):
                SLICER.copy_raw_slice(
                    source,
                    root / "slice.bag",
                    100,
                    200,
                    ["/imu", "/imu"],
                )
            existing = root / "existing.bag"
            existing.write_bytes(b"owner")
            with self.assertRaisesRegex(
                SLICER.SliceLineageError, "already exists"
            ):
                SLICER.copy_raw_slice(
                    source,
                    existing,
                    100,
                    200,
                    ["/lidar", "/imu"],
                )
            self.assertEqual(existing.read_bytes(), b"owner")


if __name__ == "__main__":
    unittest.main()
