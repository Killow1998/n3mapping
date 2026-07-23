#!/usr/bin/env python3
"""Extract synchronized ROS1/ROS2 PointCloud2/Odometry frames without ROS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("rosbags is required: python3 -m pip install rosbags") from exc


POINT_FIELD_FLOAT32 = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcd_set_sha256(entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract exact-stamp ROS1/ROS2 PointCloud2/Odometry pairs into binary PCD files "
            "and a ROS-free n3mapping relocalization manifest."
        )
    )
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--cloud-topic", default="/cloud_registered_body")
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to extract; zero means through the end of the bag.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def stamp_ns(message: object) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ensure_fresh_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"output exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(f"output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def pointcloud_xyzi(message: object) -> np.ndarray:
    wanted = {field.name: field for field in message.fields}
    missing = sorted({"x", "y", "z", "intensity"} - set(wanted))
    if missing:
        raise RuntimeError(f"PointCloud2 fields missing: {', '.join(missing)}")
    for name in ("x", "y", "z", "intensity"):
        field = wanted[name]
        if field.datatype != POINT_FIELD_FLOAT32 or field.count != 1:
            raise RuntimeError(
                f"PointCloud2 field {name} must be scalar FLOAT32, got "
                f"datatype={field.datatype} count={field.count}"
            )

    byte_order = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "intensity"],
            "formats": [byte_order + "f4"] * 4,
            "offsets": [wanted[name].offset for name in ("x", "y", "z", "intensity")],
            "itemsize": int(message.point_step),
        }
    )
    structured = np.ndarray(
        shape=(int(message.height), int(message.width)),
        dtype=dtype,
        buffer=memoryview(message.data),
        strides=(int(message.row_step), int(message.point_step)),
    )
    cloud = np.empty((structured.size, 4), dtype="<f4")
    flat = structured.reshape(-1)
    for column, name in enumerate(("x", "y", "z", "intensity")):
        cloud[:, column] = flat[name]
    return cloud


def write_binary_pcd(path: Path, cloud: np.ndarray) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {cloud.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {cloud.shape[0]}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(cloud.tobytes(order="C"))


def quaternion_is_valid(values: tuple[float, float, float, float]) -> bool:
    norm = math.sqrt(sum(value * value for value in values))
    return math.isfinite(norm) and norm > 1e-9


def read_bag_messages(
    bag: Path, cloud_topic: str, odom_topic: str
) -> tuple[dict[int, object], list[tuple[int, object]], str]:
    """Read normalized messages while retaining their exact header nanoseconds."""
    odometry: dict[int, object] = {}
    cloud_records: list[tuple[int, object]] = []
    with AnyReader(
        [bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
    ) as reader:
        bag_format = "ros2" if reader.is2 else "ros1"
        selected_connections = [
            connection
            for connection in reader.connections
            if connection.topic in {cloud_topic, odom_topic}
        ]
        found_topics = {connection.topic for connection in selected_connections}
        required_topics = {cloud_topic, odom_topic}
        if found_topics != required_topics:
            missing = sorted(required_topics - found_topics)
            raise RuntimeError(f"bag is missing topics: {', '.join(missing)}")
        for connection, _, rawdata in reader.messages(
            connections=selected_connections
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            message_stamp = stamp_ns(message)
            if connection.topic == odom_topic:
                odometry[message_stamp] = message
            else:
                cloud_records.append((message_stamp, message))
    return odometry, cloud_records, bag_format


def main() -> int:
    args = parse_args()
    if args.start_offset < 0.0 or args.duration < 0.0 or args.max_frames < 0:
        raise RuntimeError("start-offset, duration, and max-frames must be non-negative")
    if not args.bag.exists():
        raise RuntimeError(f"bag does not exist: {args.bag}")
    ensure_fresh_output(args.output)

    odometry, cloud_records, bag_format = read_bag_messages(
        args.bag, args.cloud_topic, args.odom_topic
    )

    if not cloud_records:
        raise RuntimeError("bag contains no cloud messages")
    cloud_records.sort(key=lambda item: item[0])
    source_start_ns = cloud_records[0][0]
    selected_start_ns = source_start_ns + int(round(args.start_offset * 1e9))
    selected_end_ns = (
        selected_start_ns + int(round(args.duration * 1e9)) if args.duration > 0.0 else None
    )

    manifest_path = args.output / "frames.csv"
    frame_count = 0
    total_points = 0
    pcd_files: list[dict[str, object]] = []
    cloud_frame_id = None
    odom_frame_id = None
    child_frame_id = None
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "episode_id",
                "frame_index",
                "stamp_ns",
                "pcd_path",
                "tx",
                "ty",
                "tz",
                "qx",
                "qy",
                "qz",
                "qw",
            ]
        )
        for message_stamp, cloud_message in cloud_records:
            if message_stamp < selected_start_ns:
                continue
            if selected_end_ns is not None and message_stamp >= selected_end_ns:
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
            odom_message = odometry.get(message_stamp)
            if odom_message is None:
                raise RuntimeError(f"no exact-stamp odometry for cloud stamp {message_stamp}")

            position = odom_message.pose.pose.position
            orientation = odom_message.pose.pose.orientation
            quaternion = (
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            if not quaternion_is_valid(quaternion):
                raise RuntimeError(f"invalid odometry quaternion at stamp {message_stamp}")
            cloud = pointcloud_xyzi(cloud_message)
            pcd_name = f"frame_{frame_count:06d}.pcd"
            pcd_path = args.output / pcd_name
            write_binary_pcd(pcd_path, cloud)
            pcd_files.append(
                {
                    "path": pcd_name,
                    "bytes": pcd_path.stat().st_size,
                    "sha256": sha256_file(pcd_path),
                }
            )
            writer.writerow(
                [
                    args.episode,
                    frame_count,
                    message_stamp,
                    pcd_name,
                    format(float(position.x), ".17g"),
                    format(float(position.y), ".17g"),
                    format(float(position.z), ".17g"),
                    *(format(value, ".17g") for value in quaternion),
                ]
            )
            frame_count += 1
            total_points += int(cloud.shape[0])
            cloud_frame_id = cloud_message.header.frame_id
            odom_frame_id = odom_message.header.frame_id
            child_frame_id = odom_message.child_frame_id

    if frame_count == 0:
        raise RuntimeError("selected interval contains no synchronized frames")

    metadata = {
        "schema": (
            "n3mapping_ros2_relocalization_manifest_v1"
            if bag_format == "ros2"
            else "n3mapping_ros1_relocalization_manifest_v1"
        ),
        "bag_format": bag_format,
        "bag": str(args.bag.resolve()),
        "episode_id": args.episode,
        "cloud_topic": args.cloud_topic,
        "odom_topic": args.odom_topic,
        "cloud_frame_id": cloud_frame_id,
        "odom_frame_id": odom_frame_id,
        "child_frame_id": child_frame_id,
        "source_start_stamp_ns": source_start_ns,
        "selected_start_offset_s": args.start_offset,
        "selected_duration_s": args.duration,
        "frame_count": frame_count,
        "total_points": total_points,
        "pairing": "exact_header_stamp",
        "point_encoding": "binary_pcd_xyzi_float32",
        "frames_csv_sha256": sha256_file(manifest_path),
        "pcd_set_sha256": pcd_set_sha256(pcd_files),
        "pcd_files": pcd_files,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - CLI must report data-contract failures
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
