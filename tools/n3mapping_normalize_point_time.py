#!/usr/bin/env python3
"""Normalize FLOAT32 second offsets for FAST_LIO without moving sample times.

Some ROS bags (including M2DGR) timestamp a cloud near its end, with negative
per-point offsets. FAST_LIO expects the header at the scan start and positive
offsets. This offline adapter changes that representation only. It preserves
bag record times, all other point fields, and all other message streams.
Configure the consuming frontend's timestamp_unit as seconds (0).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def normalize_seconds(message):
    """Return a new message and numerical audit; leave the input untouched."""
    fields = [field for field in message.fields if field.name == "time"]
    if len(fields) != 1 or fields[0].datatype != 7 or fields[0].count != 1:
        raise ValueError("expected exactly one scalar FLOAT32 'time' in seconds")
    if not message.width or not message.height:
        raise ValueError("cannot infer scan start from an empty cloud")
    field = fields[0]
    if field.offset + 4 > message.point_step:
        raise ValueError("time field extends beyond point_step")
    result = copy.copy(message)
    result.header = copy.deepcopy(message.header)
    result.data = np.frombuffer(bytes(message.data), dtype=np.uint8).copy()
    dtype = np.dtype({
        "names": ["time"],
        "formats": [(">" if message.is_bigendian else "<") + "f4"],
        "offsets": [field.offset],
        "itemsize": message.point_step,
    })
    offsets = np.ndarray(
        (message.height, message.width), dtype=dtype, buffer=result.data,
        strides=(message.row_step, message.point_step),
    )["time"]
    before = offsets.astype(np.float64)
    if not np.isfinite(before).all():
        raise ValueError("non-finite point time")
    shift_ns = int(np.floor(float(before.min()) * 1e9))
    offsets[:] = before - shift_ns * 1e-9
    error = float(np.max(np.abs(offsets.astype(np.float64) + shift_ns * 1e-9 - before)))
    stamp_ns = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
    result.header.stamp.sec, result.header.stamp.nanosec = divmod(stamp_ns + shift_ns, 10**9)
    return result, {
        "header_shift_ns": shift_ns,
        "absolute_point_time_max_error_s": error,
        "offset_min_s": float(offsets.min()),
        "offset_max_s": float(offsets.max()),
    }


def main():
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="ROS2 input bag")
    parser.add_argument("--output", type=Path, required=True, help="New ROS2 bag directory")
    parser.add_argument("--lidar-topic", default="/velodyne_points")
    args = parser.parse_args()
    store = get_typestore(Stores.ROS2_HUMBLE)
    count = 0
    max_error = 0.0
    hashes = {}
    with AnyReader([args.bag], default_typestore=store) as reader:
        if not reader.is2:
            parser.error("first convert ROS1 with rosbags-convert; this adapter expects ROS2")
        lidar = [c for c in reader.connections if c.topic == args.lidar_topic]
        if len(lidar) != 1 or lidar[0].msgtype != "sensor_msgs/msg/PointCloud2":
            parser.error("LiDAR topic must have one PointCloud2 connection")
        with Writer(args.output, version=8) as writer:
            destinations = {
                c.id: writer.add_connection(
                    c.topic, c.msgtype, typestore=reader.typestore,
                    offered_qos_profiles=c.ext.offered_qos_profiles,
                )
                for c in reader.connections
            }
            for connection, record_ns, data in reader.messages():
                if connection.id == lidar[0].id:
                    message, audit = normalize_seconds(reader.deserialize(data, connection.msgtype))
                    data = store.serialize_cdr(message, connection.msgtype)
                    count += 1
                    max_error = max(max_error, audit["absolute_point_time_max_error_s"])
                hashes.setdefault(connection.topic, hashlib.sha256()).update(data)
                writer.write(destinations[connection.id], record_ns, data)
    report = {
        "input": str(args.bag.resolve()), "frames": count,
        "absolute_point_time_max_error_s": max_error,
        "cdr_stream_sha256": {topic: digest.hexdigest() for topic, digest in hashes.items()},
    }
    (args.output / "point_time_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
