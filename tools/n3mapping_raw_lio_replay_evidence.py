#!/usr/bin/env python3
"""Create or verify fail-closed Product V1 raw-to-LIO replay evidence.

The evidence binds one complete acquisition chain:

* a ROS1 query bag containing only raw LiDAR and IMU;
* its ROS2 conversion containing the same two message streams;
* a ROS2 FAST_LIO output bag that preserves both raw streams and adds exact-
  stamp ``/Odometry`` and ``/cloud_registered_body`` streams;
* the ROS-free ``frames.csv``/``metadata.json``/PCD extractor output.

``create`` writes one immutable JSON sidecar. ``verify`` re-reads and re-hashes
every artifact and accepts only a byte-for-byte equivalent evidence document.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "rosbags is required: python3 -m pip install "
        "'rosbags>=0.9.23,<0.12'"
    ) from exc


SCHEMA = "n3mapping_raw_lio_replay_evidence_v1"
SCHEMA_VERSION = 1
SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
SHA256_HEX_LENGTH = 64
EXTRACTOR_COLUMNS = [
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
CANONICALIZATION = "ros_common_fields_v1_drop_ros1_header_seq"
REQUIRED_SPEC_KEYS = {
    "source_ros1_bag",
    "converted_ros2_bag",
    "lio_output_ros2_bag",
    "extractor_output",
    "fast_lio_executable",
    "fast_lio_config",
    "calibration",
    "replay_tool",
    "replay_command",
    "source_lidar_topic",
    "source_imu_topic",
    "converted_lidar_topic",
    "converted_imu_topic",
    "output_lidar_topic",
    "output_imu_topic",
    "cloud_topic",
    "odom_topic",
}
OPTIONAL_SPEC_KEYS = {"source_slice_lineage"}
TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "created_at",
    "spec",
    "artifacts",
    "bags",
    "toolchain",
    "extractor",
    "checks",
    "verdict",
}


class ReplayEvidenceError(RuntimeError):
    """A fail-closed replay-evidence contract violation."""


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_chain(path: Path, allow_missing_leaf: bool = False) -> Path:
    absolute = _absolute_without_resolving(path)
    parts = absolute.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if allow_missing_leaf:
                return absolute
            raise ReplayEvidenceError(f"path does not exist: {absolute}")
        if stat.S_ISLNK(mode):
            raise ReplayEvidenceError(f"symlinks are forbidden: {current}")
    return absolute


def require_regular_file(
    path: Path, label: str, executable: bool = False
) -> Path:
    absolute = _require_no_symlink_chain(path)
    mode = absolute.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ReplayEvidenceError(f"{label} must be a regular file: {absolute}")
    if executable and not os.access(os.fspath(absolute), os.X_OK):
        raise ReplayEvidenceError(f"{label} is not executable: {absolute}")
    return absolute


def require_directory(path: Path, label: str) -> Path:
    absolute = _require_no_symlink_chain(path)
    if not stat.S_ISDIR(absolute.lstat().st_mode):
        raise ReplayEvidenceError(f"{label} must be a directory: {absolute}")
    return absolute


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(
    path: Path, label: str, executable: bool = False
) -> Dict[str, Any]:
    regular = require_regular_file(path, label, executable=executable)
    return {
        "path": str(regular),
        "kind": "file",
        "bytes": regular.stat().st_size,
        "sha256": sha256_file(regular),
    }


def _framed_update(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def directory_identity(path: Path, label: str) -> Dict[str, Any]:
    directory = require_directory(path, label)
    directories: List[str] = []
    entries: List[Tuple[str, int, str]] = []
    for root, directory_names, file_names in os.walk(os.fspath(directory)):
        root_path = Path(root)
        for name in sorted(directory_names):
            child = root_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ReplayEvidenceError(f"symlinks are forbidden: {child}")
            if not stat.S_ISDIR(mode):
                raise ReplayEvidenceError(
                    f"{label} contains a non-directory entry: {child}"
                )
            directories.append(child.relative_to(directory).as_posix())
        for name in sorted(file_names):
            child = root_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ReplayEvidenceError(f"symlinks are forbidden: {child}")
            if not stat.S_ISREG(mode):
                raise ReplayEvidenceError(
                    f"{label} contains a non-regular file: {child}"
                )
            relative = child.relative_to(directory).as_posix()
            entries.append((relative, child.stat().st_size, sha256_file(child)))
    entries.sort()
    if not entries:
        raise ReplayEvidenceError(f"{label} is empty: {directory}")
    digest = hashlib.sha256()
    total_bytes = 0
    for relative in sorted(directories):
        digest.update(b"D")
        _framed_update(digest, relative.encode("utf-8"))
    for relative, byte_count, file_sha256 in entries:
        digest.update(b"F")
        _framed_update(digest, relative.encode("utf-8"))
        _framed_update(digest, str(byte_count).encode("ascii"))
        _framed_update(digest, file_sha256.encode("ascii"))
        total_bytes += byte_count
    return {
        "path": str(directory),
        "kind": "directory_tree",
        "bytes": total_bytes,
        "file_count": len(entries),
        "directory_count": len(directories),
        "sha256": digest.hexdigest(),
    }


def _canonical_update(digest: Any, value: Any) -> None:
    if isinstance(value, np.generic):
        _canonical_update(digest, value.item())
        return
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        digest.update(b"I")
        _framed_update(digest, str(value).encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"F")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, str):
        digest.update(b"S")
        _framed_update(digest, value.encode("utf-8"))
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"Y")
        _framed_update(digest, bytes(value))
        return
    if isinstance(value, np.ndarray):
        digest.update(b"A")
        _framed_update(digest, repr(tuple(value.shape)).encode("ascii"))
        if value.dtype.hasobject:
            _framed_update(digest, b"object")
            for item in value.flat:
                _canonical_update(digest, item)
        else:
            canonical = np.ascontiguousarray(value)
            dtype = canonical.dtype
            if dtype.byteorder == ">" or (
                dtype.byteorder == "=" and sys.byteorder == "big"
            ):
                canonical = canonical.byteswap().newbyteorder("<")
                dtype = canonical.dtype
            _framed_update(digest, dtype.str.encode("ascii"))
            _framed_update(digest, canonical.tobytes(order="C"))
        return
    if is_dataclass(value):
        digest.update(b"D")
        message_type = getattr(value, "__msgtype__", "")
        for field in fields(value):
            # ROS1 std_msgs/Header.seq has no ROS2 representation. Conversion
            # necessarily drops it, so the cross-format semantic digest binds
            # every common field while each bag's serialized digest still
            # binds the original bytes independently.
            if (
                field.name == "seq"
                and normalize_message_type(str(message_type))
                == "std_msgs/Header"
            ):
                continue
            _framed_update(digest, field.name.encode("utf-8"))
            _canonical_update(digest, getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        for key in sorted(value, key=lambda item: str(item)):
            _canonical_update(digest, key)
            _canonical_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L")
        _framed_update(digest, str(len(value)).encode("ascii"))
        for item in value:
            _canonical_update(digest, item)
        return
    slots = getattr(type(value), "__slots__", ())
    if slots:
        digest.update(b"O")
        for name in slots:
            if isinstance(name, str) and hasattr(value, name):
                _framed_update(digest, name.encode("utf-8"))
                _canonical_update(digest, getattr(value, name))
        return
    raise ReplayEvidenceError(
        f"cannot canonicalize message field type {type(value).__name__}"
    )


def canonical_message_sha256(message: object) -> str:
    digest = hashlib.sha256()
    _canonical_update(digest, message)
    return digest.hexdigest()


def normalize_message_type(message_type: str) -> str:
    return message_type.replace("/msg/", "/")


def message_stamp_ns(message: object) -> int:
    try:
        stamp = message.header.stamp
    except AttributeError as error:
        raise ReplayEvidenceError("message has no header.stamp") from error
    sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
    nanosec = getattr(
        stamp,
        "nanosec",
        getattr(stamp, "nsec", getattr(stamp, "nsecs", None)),
    )
    if sec is None or nanosec is None:
        raise ReplayEvidenceError("message header stamp has no sec/nanosec")
    second = int(sec)
    nanosecond = int(nanosec)
    if second < 0 or not 0 <= nanosecond < 1_000_000_000:
        raise ReplayEvidenceError(
            f"invalid header stamp sec={second} nanosec={nanosecond}"
        )
    return second * 1_000_000_000 + nanosecond


def _sequence_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, bytes):
            payload = value
        else:
            payload = str(value).encode("ascii")
        _framed_update(digest, payload)
    return digest.hexdigest()


def summarize_connection(reader: Any, connection: Any) -> Dict[str, Any]:
    header_stamps: List[int] = []
    canonical_sequence = hashlib.sha256()
    serialized_sequence = hashlib.sha256()
    for _, _, rawdata in reader.messages(connections=[connection]):
        message = reader.deserialize(rawdata, connection.msgtype)
        header_stamps.append(message_stamp_ns(message))
        _framed_update(
            canonical_sequence,
            canonical_message_sha256(message).encode("ascii"),
        )
        _framed_update(serialized_sequence, bytes(rawdata))
    if not header_stamps:
        raise ReplayEvidenceError(f"topic {connection.topic} has no messages")
    for index in range(1, len(header_stamps)):
        if header_stamps[index] <= header_stamps[index - 1]:
            raise ReplayEvidenceError(
                f"topic {connection.topic} header stamps are not strictly "
                f"increasing at index {index}"
            )
    return {
        "topic": connection.topic,
        "message_type": normalize_message_type(connection.msgtype),
        "message_count": len(header_stamps),
        "first_header_stamp_ns": header_stamps[0],
        "last_header_stamp_ns": header_stamps[-1],
        "header_stamp_sequence_sha256": _sequence_sha256(header_stamps),
        "canonicalization": CANONICALIZATION,
        "canonical_payload_sequence_sha256": canonical_sequence.hexdigest(),
        "serialized_payload_sequence_sha256": serialized_sequence.hexdigest(),
    }


def collect_bag_evidence(
    path: Path,
    label: str,
    expected_format: str,
    exact_topics: Sequence[str],
    default_typestore: Optional[Any] = None,
) -> Dict[str, Any]:
    if expected_format == "ros1":
        bag_path = require_regular_file(path, label)
        identity_before = file_identity(bag_path, label)
    elif expected_format == "ros2":
        bag_path = require_directory(path, label)
        identity_before = directory_identity(bag_path, label)
    else:  # pragma: no cover - internal programming guard
        raise ReplayEvidenceError(f"unsupported bag format {expected_format}")
    if len(set(exact_topics)) != len(exact_topics):
        raise ReplayEvidenceError(f"{label} contract contains duplicate topics")

    typestore = (
        default_typestore
        if default_typestore is not None
        else get_typestore(Stores.ROS2_HUMBLE)
    )
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        observed_format = "ros2" if reader.is2 else "ros1"
        if observed_format != expected_format:
            raise ReplayEvidenceError(
                f"{label} must be {expected_format}, got {observed_format}"
            )
        connections_by_topic: Dict[str, List[Any]] = {}
        for connection in reader.connections:
            connections_by_topic.setdefault(connection.topic, []).append(connection)
        observed_topics = set(connections_by_topic)
        expected_topics = set(exact_topics)
        if observed_topics != expected_topics:
            raise ReplayEvidenceError(
                f"{label} topics mismatch: expected {sorted(expected_topics)}, "
                f"got {sorted(observed_topics)}"
            )
        topic_summaries: Dict[str, Any] = {}
        for topic in exact_topics:
            connections = connections_by_topic[topic]
            if len(connections) != 1:
                raise ReplayEvidenceError(
                    f"{label} expected one connection for {topic}, "
                    f"found {len(connections)}"
                )
            topic_summaries[topic] = summarize_connection(
                reader, connections[0]
            )

    identity_after = (
        file_identity(bag_path, label)
        if expected_format == "ros1"
        else directory_identity(bag_path, label)
    )
    if identity_after != identity_before:
        raise ReplayEvidenceError(f"{label} changed while it was being read")
    return {
        "format": expected_format,
        "identity": identity_after,
        "topics": topic_summaries,
    }


def collect_ros2_typestore(path: Path, label: str) -> Any:
    """Load custom message definitions embedded in a converted ROS2 bag."""
    bag_path = require_directory(path, label)
    base = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader(
        [bag_path], default_typestore=base
    ) as reader:
        if not reader.is2:
            raise ReplayEvidenceError(f"{label} must be ROS2")
        custom_types = {
            name: definition
            for name, definition in reader.typestore.fielddefs.items()
            if name not in base.fielddefs
        }
    base.register(custom_types)
    return base


def _require_equal_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields_to_compare: Sequence[str],
    label: str,
) -> None:
    for field in fields_to_compare:
        if left.get(field) != right.get(field):
            raise ReplayEvidenceError(
                f"{label} mismatch for {field}: "
                f"{left.get(field)!r} != {right.get(field)!r}"
            )


def validate_replay_chain(
    source: Mapping[str, Any],
    converted: Mapping[str, Any],
    output: Mapping[str, Any],
    spec: Mapping[str, str],
) -> Dict[str, Any]:
    source_lidar = source["topics"][spec["source_lidar_topic"]]
    source_imu = source["topics"][spec["source_imu_topic"]]
    converted_lidar = converted["topics"][spec["converted_lidar_topic"]]
    converted_imu = converted["topics"][spec["converted_imu_topic"]]
    output_lidar = output["topics"][spec["output_lidar_topic"]]
    output_imu = output["topics"][spec["output_imu_topic"]]
    output_cloud = output["topics"][spec["cloud_topic"]]
    output_odom = output["topics"][spec["odom_topic"]]

    raw_semantic_fields = [
        "message_type",
        "message_count",
        "first_header_stamp_ns",
        "last_header_stamp_ns",
        "header_stamp_sequence_sha256",
        "canonicalization",
        "canonical_payload_sequence_sha256",
    ]
    raw_ros2_fields = raw_semantic_fields + [
        "serialized_payload_sequence_sha256"
    ]
    _require_equal_fields(
        source_lidar,
        converted_lidar,
        raw_semantic_fields,
        "ROS1-to-ROS2 LiDAR preservation",
    )
    _require_equal_fields(
        source_imu,
        converted_imu,
        raw_semantic_fields,
        "ROS1-to-ROS2 IMU preservation",
    )
    _require_equal_fields(
        converted_lidar,
        output_lidar,
        raw_ros2_fields,
        "converted-to-output LiDAR preservation",
    )
    _require_equal_fields(
        converted_imu,
        output_imu,
        raw_ros2_fields,
        "converted-to-output IMU preservation",
    )
    if output_cloud["message_type"] != "sensor_msgs/PointCloud2":
        raise ReplayEvidenceError(
            f"{spec['cloud_topic']} must carry sensor_msgs/PointCloud2"
        )
    if output_odom["message_type"] != "nav_msgs/Odometry":
        raise ReplayEvidenceError(
            f"{spec['odom_topic']} must carry nav_msgs/Odometry"
        )
    output_pair_fields = [
        "message_count",
        "first_header_stamp_ns",
        "last_header_stamp_ns",
        "header_stamp_sequence_sha256",
    ]
    _require_equal_fields(
        output_cloud,
        output_odom,
        output_pair_fields,
        "FAST_LIO cloud/odometry exact-stamp pairing",
    )
    return {
        "source_to_converted_raw_semantics": "PASS",
        "converted_to_output_raw_serialization": "PASS",
        "output_cloud_odom_exact_stamp_pairing": "PASS",
    }


def _strict_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path, label: str) -> Dict[str, Any]:
    regular = require_regular_file(path, label)
    try:
        value = json.loads(
            regular.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReplayEvidenceError(
                    f"{label} contains non-finite JSON value {token}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayEvidenceError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayEvidenceError(f"{label} root must be a JSON object")
    return value


def pcd_set_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_finite_csv_float(row: Mapping[str, str], field: str, label: str) -> None:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ReplayEvidenceError(f"{label} has invalid {field}") from error
    if not math.isfinite(value):
        raise ReplayEvidenceError(f"{label} has non-finite {field}")


def collect_extractor_evidence(
    extractor_output: Path,
    lio_output_bag: Path,
    cloud_topic: str,
    odom_topic: str,
    cloud_summary: Mapping[str, Any],
    odom_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    directory = require_directory(extractor_output, "extractor output")
    identity_before = directory_identity(directory, "extractor output")
    frames_path = require_regular_file(directory / "frames.csv", "frames.csv")
    metadata_path = require_regular_file(
        directory / "metadata.json", "extractor metadata"
    )
    metadata = load_strict_json(metadata_path, "extractor metadata")

    rows: List[Dict[str, str]] = []
    try:
        with frames_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != EXTRACTOR_COLUMNS:
                raise ReplayEvidenceError(
                    "frames.csv header does not match extractor schema"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ReplayEvidenceError(f"cannot parse frames.csv: {error}") from error
    if not rows:
        raise ReplayEvidenceError("frames.csv contains no rows")

    stamps: List[int] = []
    pcd_entries: List[Dict[str, Any]] = []
    pcd_paths: set = set()
    episode_id: Optional[str] = None
    for row_index, row in enumerate(rows):
        label = f"frames.csv row {row_index}"
        try:
            if int(row["frame_index"]) != row_index:
                raise ReplayEvidenceError(
                    f"{label} frame_index is not contiguous"
                )
            stamp_ns = int(row["stamp_ns"])
        except (KeyError, ValueError) as error:
            raise ReplayEvidenceError(
                f"{label} has invalid frame_index or stamp_ns"
            ) from error
        if stamp_ns <= 0 or (stamps and stamp_ns <= stamps[-1]):
            raise ReplayEvidenceError(
                f"{label} stamp_ns must be positive and strictly increasing"
            )
        stamps.append(stamp_ns)
        current_episode = row.get("episode_id", "")
        if not current_episode:
            raise ReplayEvidenceError(f"{label} has empty episode_id")
        if episode_id is None:
            episode_id = current_episode
        elif current_episode != episode_id:
            raise ReplayEvidenceError("frames.csv contains multiple episode_id values")
        for field in ("tx", "ty", "tz", "qx", "qy", "qz", "qw"):
            _require_finite_csv_float(row, field, label)

        raw_pcd_path = row.get("pcd_path", "")
        pure_path = PurePosixPath(raw_pcd_path)
        if (
            not raw_pcd_path
            or pure_path.is_absolute()
            or pure_path.as_posix() != raw_pcd_path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ReplayEvidenceError(f"{label} has invalid pcd_path")
        if raw_pcd_path in pcd_paths:
            raise ReplayEvidenceError(f"frames.csv reuses PCD {raw_pcd_path}")
        pcd_paths.add(raw_pcd_path)
        identity = file_identity(
            directory / Path(*pure_path.parts), f"PCD {raw_pcd_path}"
        )
        pcd_entries.append(
            {
                "path": raw_pcd_path,
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
            }
        )

    expected_files = {"frames.csv", "metadata.json"} | pcd_paths
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ReplayEvidenceError(
            "extractor output files mismatch: expected "
            f"{sorted(expected_files)}, got {sorted(actual_files)}"
        )

    if metadata.get("schema") != "n3mapping_ros2_relocalization_manifest_v1":
        raise ReplayEvidenceError("extractor metadata schema is not ROS2 V1")
    expected_metadata = {
        "bag_format": "ros2",
        "bag": str(_absolute_without_resolving(lio_output_bag)),
        "cloud_topic": cloud_topic,
        "odom_topic": odom_topic,
        "frame_count": len(rows),
        "pairing": "exact_header_stamp",
        "frames_csv_sha256": sha256_file(frames_path),
        "pcd_set_sha256": pcd_set_sha256(pcd_entries),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ReplayEvidenceError(
                f"extractor metadata {key} mismatch: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    if metadata.get("selected_start_offset_s") != 0.0:
        raise ReplayEvidenceError("extractor must use selected_start_offset_s=0")
    if metadata.get("selected_duration_s") != 0.0:
        raise ReplayEvidenceError("extractor must use selected_duration_s=0")
    if metadata.get("pcd_files") != pcd_entries:
        raise ReplayEvidenceError("extractor metadata pcd_files mismatch")
    expected_stamp_summary = {
        "message_count": len(stamps),
        "first_header_stamp_ns": stamps[0],
        "last_header_stamp_ns": stamps[-1],
        "header_stamp_sequence_sha256": _sequence_sha256(stamps),
    }
    _require_equal_fields(
        cloud_summary,
        expected_stamp_summary,
        list(expected_stamp_summary),
        "extractor-to-cloud stamp coverage",
    )
    _require_equal_fields(
        odom_summary,
        expected_stamp_summary,
        list(expected_stamp_summary),
        "extractor-to-odometry stamp coverage",
    )

    identity_after = directory_identity(directory, "extractor output")
    if identity_after != identity_before:
        raise ReplayEvidenceError(
            "extractor output changed while it was being read"
        )
    return {
        "identity": identity_after,
        "episode_id": episode_id,
        "frame_count": len(rows),
        "first_stamp_ns": stamps[0],
        "last_stamp_ns": stamps[-1],
        "stamp_sequence_sha256": _sequence_sha256(stamps),
        "frames_csv": file_identity(frames_path, "frames.csv"),
        "metadata_json": file_identity(metadata_path, "extractor metadata"),
        "pcd_set_sha256": pcd_set_sha256(pcd_entries),
        "pcd_files": pcd_entries,
    }


def command_identity(command: str) -> Dict[str, Any]:
    if not isinstance(command, str) or not command.strip():
        raise ReplayEvidenceError("replay_command must be a non-empty string")
    if "\0" in command:
        raise ReplayEvidenceError("replay_command contains a NUL byte")
    encoded = command.encode("utf-8")
    return {
        "value": command,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_spec(spec: Mapping[str, Any]) -> Dict[str, str]:
    observed_keys = set(spec) if isinstance(spec, dict) else set()
    allowed_keys = REQUIRED_SPEC_KEYS | OPTIONAL_SPEC_KEYS
    if (
        not isinstance(spec, dict)
        or not REQUIRED_SPEC_KEYS.issubset(observed_keys)
        or not observed_keys.issubset(allowed_keys)
    ):
        raise ReplayEvidenceError(
            "spec keys mismatch: expected all required keys "
            f"{sorted(REQUIRED_SPEC_KEYS)} and optional keys "
            f"{sorted(OPTIONAL_SPEC_KEYS)}, "
            f"got {sorted(spec) if isinstance(spec, dict) else type(spec).__name__}"
        )
    validated: Dict[str, str] = {}
    for key in observed_keys:
        value = spec[key]
        if not isinstance(value, str) or not value:
            raise ReplayEvidenceError(f"spec.{key} must be a non-empty string")
        validated[key] = value
    topic_keys = [
        "source_lidar_topic",
        "source_imu_topic",
        "converted_lidar_topic",
        "converted_imu_topic",
        "output_lidar_topic",
        "output_imu_topic",
        "cloud_topic",
        "odom_topic",
    ]
    for key in topic_keys:
        if not validated[key].startswith("/"):
            raise ReplayEvidenceError(f"spec.{key} must be an absolute ROS topic")
    for label, keys in (
        (
            "source",
            ["source_lidar_topic", "source_imu_topic"],
        ),
        (
            "converted",
            ["converted_lidar_topic", "converted_imu_topic"],
        ),
        (
            "output",
            [
                "output_lidar_topic",
                "output_imu_topic",
                "cloud_topic",
                "odom_topic",
            ],
        ),
    ):
        if len({validated[key] for key in keys}) != len(keys):
            raise ReplayEvidenceError(f"{label} topic names must be distinct")
    return validated


def verify_source_slice_lineage(
    spec: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    lineage_text = spec.get("source_slice_lineage")
    if lineage_text is None:
        return None
    try:
        from n3mapping_ros1_raw_bag_slice import (
            SliceLineageError,
            verify_lineage,
        )
    except ImportError as error:
        raise ReplayEvidenceError(
            "ROS1 source-slice lineage verifier is unavailable"
        ) from error
    try:
        verified = verify_lineage(Path(lineage_text))
    except SliceLineageError as error:
        raise ReplayEvidenceError(
            f"ROS1 source-slice lineage verification failed: {error}"
        ) from error
    source_path = _absolute_without_resolving(Path(spec["source_ros1_bag"]))
    lineage_slice = _absolute_without_resolving(
        Path(verified["slice_ros1_bag"])
    )
    if lineage_slice != source_path:
        raise ReplayEvidenceError(
            "ROS1 source-slice lineage names a different replay source bag"
        )
    expected_topics = sorted(
        [spec["source_lidar_topic"], spec["source_imu_topic"]]
    )
    if verified.get("topics") != expected_topics:
        raise ReplayEvidenceError(
            "ROS1 source-slice lineage topics do not match replay source topics"
        )
    return verified


def _created_at_now() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def _validate_created_at(value: str) -> str:
    if not isinstance(value, str):
        raise ReplayEvidenceError("created_at must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ReplayEvidenceError("created_at is not ISO-8601") from error
    if parsed.utcoffset() != timedelta(hours=8):
        raise ReplayEvidenceError("created_at must use Asia/Shanghai UTC+8")
    return value


def collect_evidence(
    raw_spec: Mapping[str, Any], created_at: Optional[str] = None
) -> Dict[str, Any]:
    spec = _validate_spec(raw_spec)
    created = (
        _validate_created_at(created_at)
        if created_at is not None
        else _created_at_now()
    )
    source_slice_lineage = verify_source_slice_lineage(spec)

    source = collect_bag_evidence(
        Path(spec["source_ros1_bag"]),
        "source ROS1 bag",
        "ros1",
        [spec["source_lidar_topic"], spec["source_imu_topic"]],
    )
    converted = collect_bag_evidence(
        Path(spec["converted_ros2_bag"]),
        "converted ROS2 bag",
        "ros2",
        [spec["converted_lidar_topic"], spec["converted_imu_topic"]],
    )
    converted_typestore = collect_ros2_typestore(
        Path(spec["converted_ros2_bag"]),
        "converted ROS2 bag typestore",
    )
    output = collect_bag_evidence(
        Path(spec["lio_output_ros2_bag"]),
        "FAST_LIO output ROS2 bag",
        "ros2",
        [
            spec["output_lidar_topic"],
            spec["output_imu_topic"],
            spec["cloud_topic"],
            spec["odom_topic"],
        ],
        default_typestore=converted_typestore,
    )
    checks = validate_replay_chain(source, converted, output, spec)
    extractor = collect_extractor_evidence(
        Path(spec["extractor_output"]),
        Path(spec["lio_output_ros2_bag"]),
        spec["cloud_topic"],
        spec["odom_topic"],
        output["topics"][spec["cloud_topic"]],
        output["topics"][spec["odom_topic"]],
    )
    checks["extractor_covers_all_output_pairs"] = "PASS"

    executable = file_identity(
        Path(spec["fast_lio_executable"]),
        "FAST_LIO executable",
        executable=True,
    )
    config = file_identity(Path(spec["fast_lio_config"]), "FAST_LIO config")
    calibration = file_identity(Path(spec["calibration"]), "calibration")
    replay_tool = file_identity(
        Path(spec["replay_tool"]), "deterministic replay tool", executable=True
    )
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": created,
        "spec": spec,
        "artifacts": {
            "source_ros1_bag": source["identity"],
            "converted_ros2_bag": converted["identity"],
            "lio_output_ros2_bag": output["identity"],
            "extractor_output": extractor["identity"],
        },
        "bags": {
            "source_ros1": source,
            "converted_ros2": converted,
            "lio_output_ros2": output,
        },
        "toolchain": {
            "fast_lio_executable": executable,
            "fast_lio_config": config,
            "calibration": calibration,
            "replay_tool": replay_tool,
            "replay_command": command_identity(spec["replay_command"]),
        },
        "extractor": extractor,
        "checks": checks,
        "verdict": "PASS",
    }
    if source_slice_lineage is not None:
        document["artifacts"]["source_slice_lineage"] = file_identity(
            Path(spec["source_slice_lineage"]),
            "ROS1 source-slice lineage",
        )
        document["toolchain"][
            "source_slice_lineage"
        ] = source_slice_lineage
        document["checks"]["source_slice_lineage"] = "PASS"
    return document


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    absolute = _require_no_symlink_chain(path, allow_missing_leaf=True)
    if absolute.exists():
        raise ReplayEvidenceError(
            f"refusing to overwrite existing evidence: {absolute}"
        )
    parent = absolute.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    parent = require_directory(parent, "evidence parent directory")
    temporary = parent / f".{absolute.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise ReplayEvidenceError(f"temporary path already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(os.fspath(temporary), os.fspath(absolute))
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_evidence(path: Path) -> Dict[str, Any]:
    frozen = load_strict_json(path, "raw-to-LIO replay evidence")
    if set(frozen) != TOP_LEVEL_KEYS:
        raise ReplayEvidenceError(
            f"evidence keys mismatch: expected {sorted(TOP_LEVEL_KEYS)}, "
            f"got {sorted(frozen)}"
        )
    if frozen.get("schema") != SCHEMA:
        raise ReplayEvidenceError("evidence schema mismatch")
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise ReplayEvidenceError("evidence schema_version mismatch")
    if frozen.get("verdict") != "PASS":
        raise ReplayEvidenceError("frozen evidence verdict is not PASS")
    rebuilt = collect_evidence(
        frozen.get("spec", {}), created_at=frozen.get("created_at")
    )
    if rebuilt != frozen:
        raise ReplayEvidenceError(
            "frozen evidence does not match freshly reconstructed evidence"
        )
    return {
        "evidence": str(require_regular_file(path, "raw-to-LIO replay evidence")),
        "evidence_sha256": sha256_file(path),
        "verdict": "PASS",
    }


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify Product V1 raw-to-LIO replay evidence."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-ros1-bag", type=Path, required=True)
    create.add_argument("--source-slice-lineage", type=Path)
    create.add_argument("--converted-ros2-bag", type=Path, required=True)
    create.add_argument("--lio-output-ros2-bag", type=Path, required=True)
    create.add_argument("--extractor-output", type=Path, required=True)
    create.add_argument("--fast-lio-executable", type=Path, required=True)
    create.add_argument("--fast-lio-config", type=Path, required=True)
    create.add_argument("--calibration", type=Path, required=True)
    create.add_argument("--replay-tool", type=Path, required=True)
    create.add_argument("--replay-command", required=True)
    create.add_argument(
        "--source-lidar-topic", default="/go2w/livox/lidar"
    )
    create.add_argument("--source-imu-topic", default="/go2w/livox/imu")
    create.add_argument(
        "--converted-lidar-topic", default="/go2w/livox/lidar"
    )
    create.add_argument("--converted-imu-topic", default="/go2w/livox/imu")
    create.add_argument("--output-lidar-topic", default="/livox/lidar")
    create.add_argument("--output-imu-topic", default="/livox/imu")
    create.add_argument("--cloud-topic", default="/cloud_registered_body")
    create.add_argument("--odom-topic", default="/Odometry")
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence", type=Path, required=True)
    return parser


def _create_spec(args: argparse.Namespace) -> Dict[str, str]:
    path_keys = [
        "source_ros1_bag",
        "converted_ros2_bag",
        "lio_output_ros2_bag",
        "extractor_output",
        "fast_lio_executable",
        "fast_lio_config",
        "calibration",
        "replay_tool",
    ]
    spec: Dict[str, str] = {
        key: str(_absolute_without_resolving(getattr(args, key)))
        for key in path_keys
    }
    for key in REQUIRED_SPEC_KEYS - set(path_keys):
        spec[key] = str(getattr(args, key))
    if args.source_slice_lineage is not None:
        spec["source_slice_lineage"] = str(
            _absolute_without_resolving(args.source_slice_lineage)
        )
    return spec


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _create_parser().parse_args(argv)
    if args.action == "create":
        output = _absolute_without_resolving(args.output)
        forbidden_directories = [
            _absolute_without_resolving(args.converted_ros2_bag),
            _absolute_without_resolving(args.lio_output_ros2_bag),
            _absolute_without_resolving(args.extractor_output),
        ]
        for directory in forbidden_directories:
            try:
                output.relative_to(directory)
            except ValueError:
                continue
            raise ReplayEvidenceError(
                f"evidence output must not be inside an input directory: {directory}"
            )
        evidence = collect_evidence(_create_spec(args))
        write_new_json(output, evidence)
        print(
            json.dumps(
                {
                    "evidence": str(output),
                    "evidence_sha256": sha256_file(output),
                    "verdict": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0
    result = verify_evidence(args.evidence)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayEvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
