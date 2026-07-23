#!/usr/bin/env python3
"""Inventory ROS1 raw bags and detect exact query/full LiDAR payload overlap."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

from rosbags.rosbag1 import Reader


SCHEMA_VERSION = 1
SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")


class InventoryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_time(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, SHANGHAI).isoformat(
        timespec="microseconds"
    )


def connection_summary(reader: Reader) -> list[dict[str, Any]]:
    return [
        {
            "topic": connection.topic,
            "message_type": connection.msgtype,
            "message_count": connection.msgcount,
        }
        for connection in sorted(
            reader.connections, key=lambda value: (value.topic, value.msgtype)
        )
    ]


def lidar_connections(reader: Reader, topic: str) -> list[Any]:
    connections = [
        connection
        for connection in reader.connections
        if connection.topic == topic
    ]
    if len(connections) != 1:
        raise InventoryError(
            f"{reader.path.name}: expected exactly one {topic} connection, "
            f"found {len(connections)}"
        )
    return connections


def bag_metadata(path: Path, hash_file: bool) -> dict[str, Any]:
    with Reader(path) as reader:
        metadata = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "start_record_time_ns": reader.start_time,
            "end_record_time_ns": reader.end_time,
            "start_record_time": iso_time(reader.start_time),
            "end_record_time": iso_time(reader.end_time),
            "duration_s": (reader.end_time - reader.start_time) / 1e9,
            "message_count": reader.message_count,
            "connections": connection_summary(reader),
        }
    if hash_file:
        metadata["sha256"] = sha256_file(path)
    return metadata


def full_payload_index(path: Path, lidar_topic: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    with Reader(path) as reader:
        connections = lidar_connections(reader, lidar_topic)
        for _, timestamp, rawdata in reader.messages(connections=connections):
            index[hashlib.sha256(rawdata).hexdigest()].append(timestamp)
    return dict(index)


def query_payload_overlap(
    path: Path, lidar_topic: str, full_index: dict[str, list[int]]
) -> dict[str, Any]:
    count = 0
    matched = 0
    unique = 0
    deltas_ns: list[int] = []
    unmatched_examples: list[str] = []
    used_unique_fingerprints: set[str] = set()
    with Reader(path) as reader:
        connections = lidar_connections(reader, lidar_topic)
        for _, query_timestamp, rawdata in reader.messages(
            connections=connections
        ):
            count += 1
            fingerprint = hashlib.sha256(rawdata).hexdigest()
            full_timestamps = full_index.get(fingerprint, [])
            if not full_timestamps:
                if len(unmatched_examples) < 3:
                    unmatched_examples.append(fingerprint)
                continue
            matched += 1
            if (
                len(full_timestamps) == 1
                and fingerprint not in used_unique_fingerprints
            ):
                used_unique_fingerprints.add(fingerprint)
                unique += 1
                deltas_ns.append(query_timestamp - full_timestamps[0])

    ratio = matched / count if count else 0.0
    unique_ratio = unique / count if count else 0.0
    result: dict[str, Any] = {
        "lidar_message_count": count,
        "matched_payload_count": matched,
        "matched_payload_ratio": ratio,
        "unique_payload_count": unique,
        "unique_payload_ratio": unique_ratio,
        "unmatched_fingerprint_examples": unmatched_examples,
    }
    if deltas_ns:
        ordered = sorted(deltas_ns)
        result["record_time_delta_ns"] = {
            "min": ordered[0],
            "median": int(statistics.median(ordered)),
            "max": ordered[-1],
        }
    if count > 0 and matched == count and unique == count:
        result["reference_class"] = "same_frame_reference"
    elif count > 0 and matched > 0:
        result["reference_class"] = "partial_same_frame_reference"
    else:
        result["reference_class"] = "no_payload_identity"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hash ROS1 bags and compare serialized LiDAR payloads between a "
            "full recording and query recordings."
        )
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--full-bag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lidar-topic", default="/go2w/livox/lidar"
    )
    parser.add_argument("--hash-files", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = args.directory.resolve()
    full_path = directory / args.full_bag
    if not directory.is_dir():
        print(f"ERROR: directory does not exist: {directory}", file=sys.stderr)
        return 1
    if not full_path.is_file():
        print(f"ERROR: full bag does not exist: {full_path}", file=sys.stderr)
        return 1
    output = args.output.resolve()
    if output.exists():
        print(
            f"ERROR: refusing to overwrite frozen inventory: {output}",
            file=sys.stderr,
        )
        return 1
    bag_paths = sorted(directory.glob("*.bag"))
    query_paths = [path for path in bag_paths if path != full_path]
    if not query_paths:
        print("ERROR: no query bags found", file=sys.stderr)
        return 1

    try:
        full_metadata = bag_metadata(full_path, args.hash_files)
        query_metadata = [
            bag_metadata(path, args.hash_files) for path in query_paths
        ]
        full_index = full_payload_index(full_path, args.lidar_topic)
        overlap_by_file = {
            path.name: query_payload_overlap(
                path, args.lidar_topic, full_index
            )
            for path in query_paths
        }
    except (InventoryError, OSError, ReaderError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    full_start = full_metadata["start_record_time_ns"]
    full_end = full_metadata["end_record_time_ns"]
    for metadata in query_metadata:
        metadata["record_time_inside_full"] = (
            full_start <= metadata["start_record_time_ns"]
            and metadata["end_record_time_ns"] <= full_end
        )
        metadata["payload_overlap"] = overlap_by_file[metadata["file"]]

    class_counts: dict[str, int] = defaultdict(int)
    for metadata in query_metadata:
        class_counts[
            metadata["payload_overlap"]["reference_class"]
        ] += 1
    inventory = {
        "schema": "n3mapping_ros1_raw_bag_inventory_v1",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "input_directory": str(directory),
        "lidar_topic": args.lidar_topic,
        "file_hashes_included": args.hash_files,
        "full_bag": full_metadata,
        "query_bags": query_metadata,
        "summary": {
            "bag_count": len(bag_paths),
            "query_bag_count": len(query_paths),
            "all_query_record_times_inside_full": all(
                metadata["record_time_inside_full"]
                for metadata in query_metadata
            ),
            "reference_class_counts": dict(sorted(class_counts.items())),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(output),
                "query_bag_count": len(query_paths),
                "summary": inventory["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


try:
    from rosbags.rosbag1 import ReaderError
except ImportError:  # pragma: no cover - compatibility with older rosbags
    ReaderError = Exception


if __name__ == "__main__":
    raise SystemExit(main())
