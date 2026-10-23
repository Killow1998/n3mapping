"""Check sampling-time invariance, including row padding and byte order."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as Record
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "normalize", Path(__file__).parents[1] / "tools/n3mapping_normalize_point_time.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cloud(endian):
    data = np.arange(104, dtype=np.uint8)
    times = np.ndarray((2, 3), dtype=endian + "f4", buffer=data,
                       offset=8, strides=(52, 16))
    times[:] = [[-.101, -.08, -.04], [-.02, 0, .00064]]
    return Record(
        header=Record(stamp=Record(sec=12, nanosec=50_000_000)),
        width=3, height=2, is_bigendian=endian == ">", point_step=16,
        row_step=52, data=data,
        fields=[Record(name="time", offset=8, datatype=7, count=1)],
    )


class SamplingTimeContract(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("rosbags"), "rosbags not installed")
    def test_cli_bag_roundtrip_preserves_records_and_other_streams(self):
        from rosbags.highlevel import AnyReader
        from rosbags.rosbag2 import Writer
        from rosbags.typesys import Stores, get_typestore

        store = get_typestore(Stores.ROS2_HUMBLE)
        types = store.types
        sample = cloud("<")
        message = types["sensor_msgs/msg/PointCloud2"](
            header=types["std_msgs/msg/Header"](
                types["builtin_interfaces/msg/Time"](12, 50_000_000), "lidar"),
            height=sample.height, width=sample.width,
            fields=[types["sensor_msgs/msg/PointField"]("time", 8, 7, 1)],
            is_bigendian=False, point_step=16, row_step=52,
            data=sample.data, is_dense=True,
        )
        other = bytes(store.serialize_cdr(
            types["std_msgs/msg/UInt64"](1234), "std_msgs/msg/UInt64"))
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "input", Path(directory) / "output"
            with Writer(source, version=8) as writer:
                lidar = writer.add_connection("/velodyne_points",
                    "sensor_msgs/msg/PointCloud2", typestore=store)
                observer = writer.add_connection("/other",
                    "std_msgs/msg/UInt64", typestore=store)
                writer.write(lidar, 99, store.serialize_cdr(message, lidar.msgtype))
                writer.write(observer, 100, other)
            subprocess.run([sys.executable, str(Path(module.__file__)),
                "--bag", str(source), "--output", str(target)],
                check=True, capture_output=True, text=True)
            with AnyReader([target]) as reader:
                records = list(reader.messages())
                self.assertEqual([t for _, t, _ in records], [99, 100])
                self.assertEqual(bytes(records[1][2]), other)
                normalized = reader.deserialize(records[0][2], records[0][0].msgtype)
                self.assertEqual(normalized.header.frame_id, "lidar")
                self.assertEqual(normalized.header.stamp.sec, 11)
                self.assertEqual(normalized.row_step, 52)

    def test_absolute_times_and_other_bytes_are_preserved(self):
        for endian in ["<", ">"]:
            with self.subTest(endian=endian):
                original = cloud(endian)
                before = bytes(original.data)
                result, audit = module.normalize_seconds(original)
                read = lambda m: np.ndarray((2, 3), dtype=endian + "f4",
                    buffer=m.data, offset=8, strides=(52, 16)).astype(float)
                original_times = read(original) + .05
                normalized_times = (read(result) + result.header.stamp.sec - 12
                                    + result.header.stamp.nanosec * 1e-9)
                np.testing.assert_allclose(original_times, normalized_times,
                                           atol=1e-8, rtol=0)
                self.assertGreaterEqual(read(result).min(), 0)
                self.assertEqual(bytes(original.data), before)
                mask = np.ones(len(before), dtype=bool)
                for row in range(2):
                    for col in range(3):
                        start = row * 52 + col * 16 + 8
                        mask[start:start + 4] = False
                np.testing.assert_array_equal(result.data[mask], original.data[mask])
                self.assertLessEqual(audit["absolute_point_time_max_error_s"], 1e-8)

    def test_invalid_timing_is_rejected(self):
        message = cloud("<")
        message.data[8:12] = np.frombuffer(np.float32(np.nan).tobytes(), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.normalize_seconds(message)
        message = cloud("<")
        message.fields[0].datatype = 8
        with self.assertRaisesRegex(ValueError, "FLOAT32"):
            module.normalize_seconds(message)


if __name__ == "__main__":
    unittest.main()
