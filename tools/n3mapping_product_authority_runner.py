#!/usr/bin/env python3
"""Actively observe Product V1 authority outputs from the exact Humble node."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


OBSERVATION_SCHEMA = "n3mapping_authority_observation_v2"
SCENARIOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "localization_authority_sequence",
        "LOCALIZATION",
        (
            "searching",
            "provisional",
            "first_full_lock",
            "steady_full_lock",
            "recently_lost",
            "lost",
            "recovered_full_lock",
        ),
    ),
    (
        "invalid_pose_suppression",
        "LOCALIZATION",
        ("invalid_full_pose",),
    ),
    (
        "map_extension_legacy_compatibility",
        "MAP_EXTENSION",
        ("map_extension_lock",),
    ),
)
STATE_NAMES = {
    0: "SEARCHING",
    1: "PROVISIONAL",
    2: "FULL_6DOF_LOCKED",
    3: "RECENTLY_LOST",
    4: "DEGRADED_TRACKING",
    5: "LOST",
}
POSE_SOURCE_NAMES = {
    0: "NONE",
    1: "ODOM_PREDICTED",
    2: "GEOMETRICALLY_CORRECTED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def stamp_ns(header: Any) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def observed_pose(header: Any, pose: Any) -> dict[str, Any]:
    return {
        "stamp_ns": stamp_ns(header),
        "frame_id": header.frame_id,
        "position": [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        "orientation_xyzw": [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
    }


def backend_pose_matrix(event_index: int, invalid: bool) -> list[float]:
    yaw = 0.1 * event_index
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return [
        2.0 if invalid else cosine,
        -sine,
        0.0,
        10.0 * event_index,
        sine,
        cosine,
        0.0,
        float(event_index),
        0.0,
        0.0,
        1.0,
        0.1 * event_index,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def backend_contract(label: str, event_index: int) -> dict[str, Any]:
    state = "SEARCHING"
    pose_source = "NONE"
    lock_event = False
    if label == "provisional":
        state = "PROVISIONAL"
    elif label in {
        "first_full_lock",
        "steady_full_lock",
        "recovered_full_lock",
        "invalid_full_pose",
        "map_extension_lock",
    }:
        state = "FULL_6DOF_LOCKED"
        pose_source = "GEOMETRICALLY_CORRECTED"
        lock_event = label != "steady_full_lock"
    elif label == "recently_lost":
        state = "RECENTLY_LOST"
        pose_source = "ODOM_PREDICTED"
    elif label == "lost":
        state = "LOST"
    return {
        "state": state,
        "pose_source": pose_source,
        "pose_matrix": backend_pose_matrix(
            event_index, label == "invalid_full_pose"
        ),
        "lock_event": lock_event,
    }


def topic_contract() -> dict[str, dict[str, Any]]:
    def topic(
        name: str, message_type: str, depth: int, durability: str
    ) -> dict[str, Any]:
        return {
            "name": name,
            "type": message_type,
            "qos": {
                "reliability": "reliable",
                "durability": durability,
                "depth": depth,
            },
        }

    return {
        "relocalization_status": topic(
            "/n3mapping/relocalization_status",
            "n3mapping/msg/RelocalizationStatus",
            1,
            "transient_local",
        ),
        "authoritative_pose": topic(
            "/n3mapping/relocalization_pose",
            "geometry_msgs/msg/PoseStamped",
            1,
            "volatile",
        ),
        "legacy_lock": topic(
            "/n3mapping/relocalization_lock",
            "std_msgs/msg/UInt32",
            10,
            "volatile",
        ),
        "global_odometry": topic(
            "/n3mapping/odometry",
            "nav_msgs/msg/Odometry",
            10,
            "volatile",
        ),
        "global_world_cloud": topic(
            "/n3mapping/cloud_world",
            "sensor_msgs/msg/PointCloud2",
            10,
            "volatile",
        ),
    }


def probe_identity(node: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(node), "--build-identity-json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "candidate node identity probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("candidate node returned invalid build identity") from error
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != "n3mapping_product_build_identity_v2"
        or identity.get("build_type") != "Release"
        or identity.get("research_tools") != "OFF"
    ):
        raise RuntimeError("candidate node is not a Product V1 Release/OFF build")
    return identity


def qos_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(value).split(".")[-1].lower()


def inspect_publishers(
    collector: Any, expected: dict[str, dict[str, Any]]
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        ready = True
        for descriptor in expected.values():
            endpoints = collector.get_publishers_info_by_topic(
                descriptor["name"]
            )
            matches = [
                endpoint
                for endpoint in endpoints
                if endpoint.topic_type == descriptor["type"]
            ]
            if len(matches) != 1:
                ready = False
                break
            qos = matches[0].qos_profile
            observed = {
                "reliability": qos_name(qos.reliability),
                "durability": qos_name(qos.durability),
            }
            expected_graph_qos = {
                "reliability": descriptor["qos"]["reliability"],
                "durability": descriptor["qos"]["durability"],
            }
            if observed != expected_graph_qos:
                raise RuntimeError(
                    f"publisher QoS mismatch for {descriptor['name']}: "
                    f"{observed}"
                )
            if int(qos.depth) not in {0, descriptor["qos"]["depth"]}:
                raise RuntimeError(
                    f"publisher depth mismatch for {descriptor['name']}: "
                    f"{int(qos.depth)}"
                )
        if ready:
            return
        time.sleep(0.05)
    raise RuntimeError("candidate authority publishers were not discovered")


def make_event_observation(
    label: str,
    event_index: int,
    messages: dict[int, dict[str, list[Any] | int]],
) -> dict[str, Any]:
    stamp = (99 + event_index) * 1_000_000_000
    observed = messages.get(
        stamp,
        {
            "statuses": [],
            "authoritative_poses": [],
            "global_poses": [],
            "world_cloud_count": 0,
        },
    )
    legacy: list[int] = []
    if label in {"first_full_lock", "map_extension_lock"}:
        legacy = [1]
    elif label == "recovered_full_lock":
        legacy = [2]
    return {
        "label": label,
        "backend": backend_contract(label, event_index),
        "observed": {
            "statuses": list(observed["statuses"]),
            "authoritative_poses": list(observed["authoritative_poses"]),
            "legacy_lock_epochs": legacy,
            "global_poses": list(observed["global_poses"]),
            "world_cloud_count": int(observed["world_cloud_count"]),
        },
    }


def run_scenario(
    rclpy: Any,
    node_path: Path,
    scenario_id: str,
    run_mode: str,
    labels: tuple[str, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    from geometry_msgs.msg import PoseStamped
    from n3mapping.msg import RelocalizationStatus
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import UInt32

    collector = rclpy.create_node(
        "n3mapping_product_authority_observer_" + str(os.getpid())
    )
    messages: dict[int, dict[str, list[Any] | int]] = {}
    legacy_epochs: list[int] = []

    def bucket(stamp: int) -> dict[str, list[Any] | int]:
        return messages.setdefault(
            stamp,
            {
                "statuses": [],
                "authoritative_poses": [],
                "global_poses": [],
                "world_cloud_count": 0,
            },
        )

    def status_callback(message: Any) -> None:
        stamp = stamp_ns(message.header)
        payload = {
            "state": STATE_NAMES.get(int(message.state), "INVALID"),
            "pose_source": POSE_SOURCE_NAMES.get(
                int(message.pose_source), "INVALID"
            ),
            "lock_epoch": int(message.lock_epoch),
        }
        bucket(stamp)["statuses"].append(payload)
        records.append(
            {
                "kind": "ros_message",
                "scenario": scenario_id,
                "topic": "/n3mapping/relocalization_status",
                "stamp_ns": stamp,
                "payload": payload,
            }
        )

    def pose_callback(message: Any) -> None:
        stamp = stamp_ns(message.header)
        payload = observed_pose(message.header, message.pose)
        bucket(stamp)["authoritative_poses"].append(payload)
        records.append(
            {
                "kind": "ros_message",
                "scenario": scenario_id,
                "topic": "/n3mapping/relocalization_pose",
                "stamp_ns": stamp,
                "payload": payload,
            }
        )

    def lock_callback(message: Any) -> None:
        epoch = int(message.data)
        legacy_epochs.append(epoch)
        records.append(
            {
                "kind": "ros_message",
                "scenario": scenario_id,
                "topic": "/n3mapping/relocalization_lock",
                "payload": {"lock_epoch": epoch},
            }
        )

    def odom_callback(message: Any) -> None:
        stamp = stamp_ns(message.header)
        payload = observed_pose(message.header, message.pose.pose)
        bucket(stamp)["global_poses"].append(payload)
        records.append(
            {
                "kind": "ros_message",
                "scenario": scenario_id,
                "topic": "/n3mapping/odometry",
                "stamp_ns": stamp,
                "payload": payload,
            }
        )

    def cloud_callback(message: Any) -> None:
        stamp = stamp_ns(message.header)
        current = bucket(stamp)
        current["world_cloud_count"] = int(current["world_cloud_count"]) + 1
        records.append(
            {
                "kind": "ros_message",
                "scenario": scenario_id,
                "topic": "/n3mapping/cloud_world",
                "stamp_ns": stamp,
                "payload": {
                    "frame_id": message.header.frame_id,
                    "width": int(message.width),
                    "height": int(message.height),
                },
            }
        )

    subscriptions = [
        collector.create_subscription(
            RelocalizationStatus,
            "/n3mapping/relocalization_status",
            status_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        ),
        collector.create_subscription(
            PoseStamped,
            "/n3mapping/relocalization_pose",
            pose_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        ),
        collector.create_subscription(
            UInt32,
            "/n3mapping/relocalization_lock",
            lock_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        ),
        collector.create_subscription(
            Odometry,
            "/n3mapping/odometry",
            odom_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        ),
        collector.create_subscription(
            PointCloud2,
            "/n3mapping/cloud_world",
            cloud_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        ),
    ]
    process = subprocess.Popen(
        [
            str(node_path),
            "--product-authority-probe",
            scenario_id,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        inspect_publishers(collector, topic_contract())
        deadline = time.monotonic() + 15.0
        while process.poll() is None and time.monotonic() < deadline:
            rclpy.spin_once(collector, timeout_sec=0.05)
        if process.poll() is None:
            process.terminate()
            raise RuntimeError(f"authority probe timed out: {scenario_id}")
        for _ in range(10):
            rclpy.spin_once(collector, timeout_sec=0.02)
        output = process.communicate(timeout=2)[0]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        collector.destroy_node()

    records.append(
        {
            "kind": "process_output",
            "scenario": scenario_id,
            "returncode": int(process.returncode),
            "output": output,
        }
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"authority probe failed for {scenario_id}: {output.strip()}"
        )
    done_marker = f"N3MAPPING_AUTHORITY_PROBE_DONE {scenario_id}"
    if done_marker not in output:
        raise RuntimeError(f"authority probe completion marker missing: {scenario_id}")
    qos_lines = [
        line.removeprefix("N3MAPPING_AUTHORITY_PROBE_QOS ")
        for line in output.splitlines()
        if line.startswith("N3MAPPING_AUTHORITY_PROBE_QOS ")
    ]
    if len(qos_lines) != 1:
        raise RuntimeError(
            f"candidate actual-QoS record missing for {scenario_id}"
        )
    try:
        actual_depths = json.loads(qos_lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"candidate actual-QoS record is invalid for {scenario_id}"
        ) from error
    expected_depths = {
        role: descriptor["qos"]["depth"]
        for role, descriptor in topic_contract().items()
    }
    if actual_depths != expected_depths:
        raise RuntimeError(
            f"candidate actual-QoS depth mismatch for {scenario_id}: "
            f"{actual_depths}"
        )

    expected_legacy = (
        [1, 2]
        if scenario_id == "localization_authority_sequence"
        else ([] if scenario_id == "invalid_pose_suppression" else [1])
    )
    if legacy_epochs != expected_legacy:
        raise RuntimeError(
            f"legacy lock sequence mismatch for {scenario_id}: {legacy_epochs}"
        )
    return {
        "id": scenario_id,
        "run_mode": run_mode,
        "events": [
            make_event_observation(label, index + 1, messages)
            for index, label in enumerate(labels)
        ],
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe authority publications from an exact Humble node."
    )
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ros-domain-id", type=int)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    node = args.node.resolve(strict=True)
    harness = Path(__file__).resolve(strict=True)
    if not node.is_file() or not os.access(node, os.X_OK):
        raise RuntimeError(f"candidate node is not executable: {node}")
    for output in (args.observation, args.log):
        if output.exists():
            raise RuntimeError(f"runner output already exists: {output}")
        if not output.parent.is_dir():
            raise RuntimeError(f"runner output parent does not exist: {output.parent}")

    domain_id = args.ros_domain_id
    if domain_id is None:
        domain_id = 100 + random.SystemRandom().randrange(100)
    if not 0 <= domain_id <= 232:
        raise RuntimeError("ROS_DOMAIN_ID must be in [0, 232]")
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

    identity = probe_identity(node)
    try:
        import rclpy
    except ImportError as error:
        raise RuntimeError(
            "rclpy is unavailable; source the exact Humble candidate setup"
        ) from error

    records: list[dict[str, Any]] = [
        {
            "kind": "runner_start",
            "distro": "humble",
            "ros_domain_id": domain_id,
            "runtime_node": str(node),
            "runtime_node_sha256": sha256_file(node),
        }
    ]
    rclpy.init(args=[])
    try:
        scenarios = [
            run_scenario(
                rclpy,
                node,
                scenario_id,
                run_mode,
                labels,
                records,
            )
            for scenario_id, run_mode, labels in SCENARIOS
        ]
    finally:
        rclpy.shutdown()

    with args.log.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())

    observation = {
        "schema": OBSERVATION_SCHEMA,
        "schema_version": 2,
        "distro": "humble",
        "candidate_commit": identity["commit"],
        "product_profile_sha256": identity["product_profile_sha256"],
        "files": {
            "runtime_node": {
                "path": str(node),
                "sha256": sha256_file(node),
            },
            "harness": {
                "path": str(harness),
                "sha256": sha256_file(harness),
            },
            "log": {
                "path": str(args.log.resolve()),
                "sha256": sha256_file(args.log),
            },
        },
        "topics": topic_contract(),
        "scenarios": scenarios,
    }
    write_json(args.observation, observation)
    print(
        json.dumps(
            {
                "distro": "humble",
                "observation": str(args.observation.resolve()),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
