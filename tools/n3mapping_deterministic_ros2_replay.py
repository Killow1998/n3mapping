#!/usr/bin/env python3
"""Replay selected ROS2 bag topics after matching and optional consumer flow control.

Humble's rosbag2 player may publish the first storage record before a recorder
has matched the newly created publisher.  Product-data preparation cannot
silently lose that record.  This small transport tool creates all publishers,
waits for an explicit subscription count, and only then replays the original
serialized messages in storage-time order. An optional LiDAR acknowledgement
channel bounds the LIO consumer backlog without changing sensor messages.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Dict, MutableMapping, Optional, Sequence, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


class ReplayError(RuntimeError):
    """A deterministic replay contract violation."""


def parse_topic_depth(value: str) -> Tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected TOPIC=DEPTH")
    topic, depth_text = value.rsplit("=", 1)
    if not topic.startswith("/"):
        raise argparse.ArgumentTypeError("topic must be absolute")
    try:
        depth = int(depth_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("depth must be an integer") from error
    if depth <= 0:
        raise argparse.ArgumentTypeError("depth must be positive")
    return topic, depth


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Replay exact ROS2 bag messages after waiting for DDS matches."
        )
    )
    command.add_argument("--bag", required=True, type=Path)
    command.add_argument(
        "--topic-depth",
        required=True,
        action="append",
        type=parse_topic_depth,
        metavar="TOPIC=DEPTH",
    )
    command.add_argument("--rate", type=float, default=1.0)
    command.add_argument("--min-subscriptions", type=int, default=1)
    command.add_argument("--match-timeout-s", type=float, default=30.0)
    command.add_argument("--ack-timeout-s", type=float, default=10.0)
    command.add_argument(
        "--flow-control-lidar-topic",
        help=(
            "LiDAR topic whose next record is held until the previous "
            "synchronized frame is acknowledged"
        ),
    )
    command.add_argument(
        "--lidar-ack-topic",
        help="std_msgs/msg/UInt64 sequence published by the LIO consumer",
    )
    command.add_argument("--lidar-ack-timeout-s", type=float, default=60.0)
    return command


def require_arguments(args: argparse.Namespace) -> Dict[str, int]:
    if not math.isfinite(args.rate) or args.rate <= 0.0:
        raise ReplayError("--rate must be finite and positive")
    if args.min_subscriptions <= 0:
        raise ReplayError("--min-subscriptions must be positive")
    if (
        not math.isfinite(args.match_timeout_s)
        or args.match_timeout_s <= 0.0
    ):
        raise ReplayError("--match-timeout-s must be finite and positive")
    if (
        not math.isfinite(args.ack_timeout_s)
        or args.ack_timeout_s <= 0.0
    ):
        raise ReplayError("--ack-timeout-s must be finite and positive")
    flow_control_requested = (
        args.flow_control_lidar_topic is not None
        or args.lidar_ack_topic is not None
    )
    if flow_control_requested and (
        args.flow_control_lidar_topic is None
        or args.lidar_ack_topic is None
    ):
        raise ReplayError(
            "--flow-control-lidar-topic and --lidar-ack-topic must be "
            "provided together"
        )
    if args.flow_control_lidar_topic is not None:
        if not args.flow_control_lidar_topic.startswith("/"):
            raise ReplayError("--flow-control-lidar-topic must be absolute")
        if not args.lidar_ack_topic.startswith("/"):
            raise ReplayError("--lidar-ack-topic must be absolute")
        if (
            not math.isfinite(args.lidar_ack_timeout_s)
            or args.lidar_ack_timeout_s <= 0.0
        ):
            raise ReplayError(
                "--lidar-ack-timeout-s must be finite and positive"
            )
    depths: Dict[str, int] = {}
    for topic, depth in args.topic_depth:
        if topic in depths:
            raise ReplayError(f"duplicate topic: {topic}")
        depths[topic] = depth
    if not depths:
        raise ReplayError("at least one topic is required")
    if (
        args.flow_control_lidar_topic is not None
        and args.flow_control_lidar_topic not in depths
    ):
        raise ReplayError(
            "--flow-control-lidar-topic must also be a --topic-depth topic"
        )
    bag = args.bag.resolve()
    if args.bag.is_symlink() or not bag.is_dir():
        raise ReplayError(f"bag must be a non-symlink directory: {bag}")
    args.bag = bag
    return depths


def open_reader(
    bag: Path, topics: Sequence[str]
) -> Tuple[rosbag2_py.SequentialReader, Dict[str, str]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    available: Dict[str, str] = {}
    for metadata in reader.get_all_topics_and_types():
        if metadata.name in available:
            raise ReplayError(f"duplicate bag topic metadata: {metadata.name}")
        available[metadata.name] = metadata.type
    missing = sorted(set(topics) - set(available))
    if missing:
        raise ReplayError(f"requested topics missing from bag: {missing}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader, {topic: available[topic] for topic in topics}


def wait_for_subscriptions(
    node: Node,
    publishers: Dict[str, object],
    minimum: int,
    timeout_s: float,
) -> Dict[str, int]:
    deadline = time.monotonic() + timeout_s
    while rclpy.ok():
        counts = {
            topic: publisher.get_subscription_count()
            for topic, publisher in publishers.items()
        }
        if all(count >= minimum for count in counts.values()):
            return counts
        if time.monotonic() >= deadline:
            raise ReplayError(
                "subscription match timeout: "
                + json.dumps(counts, sort_keys=True)
            )
        rclpy.spin_once(node, timeout_sec=0.05)
    raise ReplayError("ROS shutdown before subscriber matching")


def wait_until(node: Node, target_ns: int) -> None:
    while rclpy.ok():
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        rclpy.spin_once(
            node,
            timeout_sec=min(0.01, remaining_ns / 1_000_000_000.0),
        )
    raise ReplayError("ROS shutdown during replay")


def wait_for_ack_publisher(
    node: Node, topic: str, timeout_s: float
) -> int:
    deadline = time.monotonic() + timeout_s
    while rclpy.ok():
        count = node.count_publishers(topic)
        if count == 1:
            return 1
        if count > 1:
            raise ReplayError(
                "LiDAR acknowledgement requires exactly one publisher: "
                f"topic={topic}, publishers={count}"
            )
        if time.monotonic() >= deadline:
            raise ReplayError(
                f"LiDAR acknowledgement publisher timeout: {topic}"
            )
        rclpy.spin_once(node, timeout_sec=0.05)
    raise ReplayError("ROS shutdown before LiDAR acknowledgement matching")


def wait_for_lidar_ack(
    node: Node,
    state: MutableMapping[str, object],
    expected_sequence: int,
    timeout_s: float,
) -> int:
    start_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_s
    while rclpy.ok():
        error = state.get("error")
        if error is not None:
            raise ReplayError(str(error))
        if int(state["sequence"]) >= expected_sequence:
            return time.monotonic_ns() - start_ns
        if time.monotonic() >= deadline:
            raise ReplayError(
                "LiDAR acknowledgement timeout: "
                f"expected={expected_sequence}, "
                f"received={state['sequence']}"
            )
        rclpy.spin_once(node, timeout_sec=0.01)
    raise ReplayError("ROS shutdown while waiting for LiDAR acknowledgement")


def update_lidar_ack_state(
    state: MutableMapping[str, object], received_sequence: int
) -> None:
    expected = int(state["sequence"]) + 1
    if received_sequence != expected:
        state["error"] = (
            "non-contiguous LiDAR acknowledgement: "
            f"expected={expected}, received={received_sequence}"
        )
        return
    state["sequence"] = received_sequence


def replay(
    reader: rosbag2_py.SequentialReader,
    node: Node,
    publishers: Dict[str, object],
    message_types: Dict[str, object],
    rate: float,
    flow_control_lidar_topic: Optional[str] = None,
    lidar_ack_state: Optional[MutableMapping[str, object]] = None,
    lidar_ack_timeout_s: float = 60.0,
) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    first_storage_ns: Optional[int] = None
    last_storage_ns: Optional[int] = None
    start_monotonic_ns: Optional[int] = None
    flow_control_ack_wait_ns = 0
    flow_control_schedule_shift_ns = 0
    published_lidar_frames = 0

    while reader.has_next():
        topic, serialized, storage_ns = reader.read_next()
        if topic not in publishers:
            raise ReplayError(f"reader returned an unrequested topic: {topic}")
        storage_ns = int(storage_ns)
        if first_storage_ns is None:
            first_storage_ns = storage_ns
            start_monotonic_ns = time.monotonic_ns()
        if storage_ns < (last_storage_ns or storage_ns):
            raise ReplayError("bag storage timestamps are not monotonic")
        assert start_monotonic_ns is not None
        target_ns = (
            start_monotonic_ns
            + int((storage_ns - first_storage_ns) / rate)
            + flow_control_schedule_shift_ns
        )
        if (
            flow_control_lidar_topic is not None
            and topic == flow_control_lidar_topic
            and published_lidar_frames > 0
        ):
            assert lidar_ack_state is not None
            flow_control_ack_wait_ns += wait_for_lidar_ack(
                node,
                lidar_ack_state,
                published_lidar_frames,
                lidar_ack_timeout_s,
            )
            ack_complete_ns = time.monotonic_ns()
            if ack_complete_ns > target_ns:
                overrun_ns = ack_complete_ns - target_ns
                flow_control_schedule_shift_ns += overrun_ns
                target_ns += overrun_ns
        wait_until(node, target_ns)
        message = deserialize_message(
            serialized, message_types[topic]
        )
        publishers[topic].publish(message)
        counts[topic] += 1
        if topic == flow_control_lidar_topic:
            published_lidar_frames += 1
        last_storage_ns = storage_ns

    if first_storage_ns is None or last_storage_ns is None:
        raise ReplayError("selected bag topics contain no messages")
    if flow_control_lidar_topic is not None and published_lidar_frames > 0:
        assert lidar_ack_state is not None
        flow_control_ack_wait_ns += wait_for_lidar_ack(
            node,
            lidar_ack_state,
            published_lidar_frames,
            lidar_ack_timeout_s,
        )
    return {
        "counts": dict(sorted(counts.items())),
        "first_storage_ns": first_storage_ns,
        "last_storage_ns": last_storage_ns,
        "flow_control": {
            "enabled": flow_control_lidar_topic is not None,
            "lidar_topic": flow_control_lidar_topic,
            "acknowledged_lidar_frames": (
                int(lidar_ack_state["sequence"])
                if lidar_ack_state is not None
                else 0
            ),
            "ack_wait_ns": flow_control_ack_wait_ns,
            "schedule_shift_ns": flow_control_schedule_shift_ns,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    node: Optional[Node] = None
    lidar_ack_subscription = None
    try:
        depths = require_arguments(args)
        reader, type_names = open_reader(args.bag, list(depths))
        message_types = {
            topic: get_message(type_name)
            for topic, type_name in type_names.items()
        }

        rclpy.init(args=None)
        node = Node("n3mapping_deterministic_ros2_replay")
        publishers = {
            topic: node.create_publisher(
                message_types[topic],
                topic,
                QoSProfile(
                    depth=depth,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
            for topic, depth in depths.items()
        }
        lidar_ack_state: Optional[MutableMapping[str, object]] = None
        if args.lidar_ack_topic is not None:
            lidar_ack_state = {"sequence": 0, "error": None}
            ack_type = get_message("std_msgs/msg/UInt64")

            def on_lidar_ack(message: object) -> None:
                assert lidar_ack_state is not None
                update_lidar_ack_state(lidar_ack_state, int(message.data))

            lidar_ack_subscription = node.create_subscription(
                ack_type,
                args.lidar_ack_topic,
                on_lidar_ack,
                QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
        matches = wait_for_subscriptions(
            node,
            publishers,
            args.min_subscriptions,
            args.match_timeout_s,
        )
        lidar_ack_publishers = 0
        if args.lidar_ack_topic is not None:
            lidar_ack_publishers = wait_for_ack_publisher(
                node, args.lidar_ack_topic, args.match_timeout_s
            )
        result = replay(
            reader,
            node,
            publishers,
            message_types,
            args.rate,
            args.flow_control_lidar_topic,
            lidar_ack_state,
            args.lidar_ack_timeout_s,
        )
        acknowledgements = {
            topic: publisher.wait_for_all_acked(
                Duration(seconds=args.ack_timeout_s)
            )
            for topic, publisher in publishers.items()
        }
        if not all(acknowledgements.values()):
            raise ReplayError(
                "publisher acknowledgement timeout: "
                + json.dumps(acknowledgements, sort_keys=True)
            )
        print(
            json.dumps(
                {
                    "schema": "n3mapping_deterministic_ros2_replay_v2",
                    "bag": str(args.bag),
                    "rate": args.rate,
                    "matched_subscriptions": matches,
                    "lidar_ack_publishers": lidar_ack_publishers,
                    "acknowledged": acknowledgements,
                    **result,
                    "status": "PASS",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (ReplayError, RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1
    finally:
        if node is not None:
            if lidar_ack_subscription is not None:
                node.destroy_subscription(lidar_ack_subscription)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
