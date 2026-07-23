#!/usr/bin/env python3
"""Bind query LIO frames to a same-run native dense map trajectory.

The frozen inventory proves every query Livox payload occurs exactly once in
the full mapping bag. FAST_LIO may skip raw scans, and its output stamp depends
on preprocessing and rolling scan-time state, so this tool does not infer a
raw-frame index from a timestamp window. Instead it:

* verifies the query and full raw bag hashes plus exact payload-overlap class;
* takes acquisition start from the query's first raw LiDAR header stamp;
* requires every query LIO stamp to match one native dense pose within 512 ns.

No time offset is estimated or fitted.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


JOIN_TOLERANCE_NS = 512


class SameFrameReferenceError(RuntimeError):
    """A fail-closed violation of the same-frame reference contract."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RawBagSummary:
    lidar_count: int
    first_header_stamp_ns: int
    last_header_stamp_ns: int


@dataclass(frozen=True)
class DensePose:
    stamp_ns: int
    seq: int | None
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_stamp_ns(message: object) -> int:
    stamp = message.header.stamp
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise SameFrameReferenceError(
            f"invalid ROS header stamp sec={sec} nanosec={nanosec}"
        )
    return sec * 1_000_000_000 + nanosec


def read_raw_bag_summary(query_bag: Path, lidar_topic: str) -> RawBagSummary:
    if not query_bag.is_file() or query_bag.suffix != ".bag":
        raise SameFrameReferenceError(
            f"query raw input must be a ROS1 .bag file: {query_bag}"
        )
    header_stamps: list[int] = []
    with AnyReader(
        [query_bag], default_typestore=get_typestore(Stores.ROS1_NOETIC)
    ) as reader:
        if reader.is2:
            raise SameFrameReferenceError("query raw input must be ROS1")
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == lidar_topic
        ]
        if len(connections) != 1:
            raise SameFrameReferenceError(
                f"expected exactly one {lidar_topic} connection, "
                f"found {len(connections)}"
            )
        connection = connections[0]
        if not connection.msgtype.endswith("/CustomMsg"):
            raise SameFrameReferenceError(
                f"{lidar_topic} must carry CustomMsg, got {connection.msgtype}"
            )
        for _, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            header_stamps.append(message_stamp_ns(message))
    if not header_stamps:
        raise SameFrameReferenceError("query bag contains no raw lidar frames")
    require_strictly_increasing(
        header_stamps,
        "raw lidar header stamps",
    )
    return RawBagSummary(
        lidar_count=len(header_stamps),
        first_header_stamp_ns=header_stamps[0],
        last_header_stamp_ns=header_stamps[-1],
    )


def require_strictly_increasing(values: list[int], label: str) -> None:
    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            raise SameFrameReferenceError(
                f"{label} contain duplicate stamp {values[index]}"
            )
        if values[index] < values[index - 1]:
            raise SameFrameReferenceError(f"{label} are not strictly increasing")


def parse_csv_int(row: dict[str, str], field: str, label: str) -> int:
    value = row.get(field)
    if value is None or value.strip() == "":
        raise SameFrameReferenceError(f"{label} has missing {field}")
    try:
        return int(value)
    except ValueError as error:
        raise SameFrameReferenceError(
            f"{label} has invalid integer {field}={value!r}"
        ) from error


def read_lio_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SameFrameReferenceError(f"LIO frames.csv does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "stamp_ns" not in reader.fieldnames:
            raise SameFrameReferenceError("LIO frames.csv is missing stamp_ns")
        for row_index, row in enumerate(reader):
            stamp_ns = parse_csv_int(row, "stamp_ns", f"LIO row {row_index}")
            rows.append(
                {
                    "stamp_ns": stamp_ns,
                    "episode_id": row.get("episode_id", ""),
                    "frame_index": row.get("frame_index", str(row_index)),
                }
            )
    if not rows:
        raise SameFrameReferenceError("LIO frames.csv contains no rows")
    require_strictly_increasing(
        [int(row["stamp_ns"]) for row in rows], "LIO manifest stamps"
    )
    return rows


def parse_csv_float(row: dict[str, str], field: str, label: str) -> float:
    value = row.get(field)
    if value is None or value.strip() == "":
        raise SameFrameReferenceError(f"{label} has missing {field}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise SameFrameReferenceError(
            f"{label} has invalid float {field}={value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise SameFrameReferenceError(f"{label} has non-finite {field}")
    return parsed


def read_dense_trajectory(path: Path) -> list[DensePose]:
    if not path.is_file():
        raise SameFrameReferenceError(
            f"dense trajectory CSV does not exist: {path}"
        )
    required = {"stamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    poses: list[DensePose] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SameFrameReferenceError(
                "dense trajectory CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        for row_index, row in enumerate(reader):
            label = f"dense trajectory row {row_index}"
            quaternion = tuple(
                parse_csv_float(row, field, label)
                for field in ("qx", "qy", "qz", "qw")
            )
            quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
            if quaternion_norm <= 1e-12:
                raise SameFrameReferenceError(
                    f"{label} has an invalid zero quaternion"
                )
            seq_value = row.get("seq")
            try:
                seq = (
                    int(seq_value)
                    if seq_value is not None and seq_value.strip() != ""
                    else None
                )
            except ValueError as error:
                raise SameFrameReferenceError(
                    f"{label} has invalid seq={seq_value!r}"
                ) from error
            poses.append(
                DensePose(
                    stamp_ns=parse_csv_int(row, "stamp_ns", label),
                    seq=seq,
                    tx=parse_csv_float(row, "tx", label),
                    ty=parse_csv_float(row, "ty", label),
                    tz=parse_csv_float(row, "tz", label),
                    qx=quaternion[0],
                    qy=quaternion[1],
                    qz=quaternion[2],
                    qw=quaternion[3],
                )
            )
    if not poses:
        raise SameFrameReferenceError("dense trajectory CSV contains no rows")
    require_strictly_increasing(
        [pose.stamp_ns for pose in poses], "dense trajectory stamps"
    )
    return poses


def validate_dense_evidence(
    evidence_path: Path,
    trajectory_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SameFrameReferenceError(
            f"failed to read dense trajectory evidence: {error}"
        ) from error
    if evidence.get("schema") != "n3mapping_dense_trajectory_evidence_v1":
        raise SameFrameReferenceError(
            "unsupported dense trajectory evidence schema"
        )
    if evidence.get("source") != "native" or evidence.get("degraded") is not False:
        raise SameFrameReferenceError(
            "dense trajectory evidence must be native and non-degraded"
        )
    if evidence.get("pose_convention") != "T_world_body":
        raise SameFrameReferenceError(
            "dense trajectory evidence pose convention must be T_world_body"
        )
    if evidence.get("row_count") != expected_rows:
        raise SameFrameReferenceError(
            "dense trajectory evidence row count mismatch"
        )
    recorded_trajectory = Path(
        str(evidence.get("trajectory_csv", ""))
    ).resolve()
    if recorded_trajectory != trajectory_path.resolve():
        raise SameFrameReferenceError(
            "dense trajectory evidence points to a different CSV"
        )
    expected_csv_sha = str(evidence.get("trajectory_csv_sha256", ""))
    if not SHA256_RE.fullmatch(expected_csv_sha):
        raise SameFrameReferenceError(
            "dense trajectory evidence is missing CSV SHA-256"
        )
    if sha256_file(trajectory_path) != expected_csv_sha:
        raise SameFrameReferenceError(
            "dense trajectory CSV SHA-256 disagrees with evidence"
        )
    pbstream_path = Path(str(evidence.get("pbstream", ""))).resolve()
    expected_pbstream_sha = str(evidence.get("pbstream_sha256", ""))
    if not pbstream_path.is_file() or not SHA256_RE.fullmatch(
        expected_pbstream_sha
    ):
        raise SameFrameReferenceError(
            "dense trajectory evidence is missing pbstream provenance"
        )
    if sha256_file(pbstream_path) != expected_pbstream_sha:
        raise SameFrameReferenceError(
            "dense trajectory pbstream SHA-256 disagrees with evidence"
        )
    return evidence


def validate_inventory(
    inventory_path: Path, query_bag: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SameFrameReferenceError(
            f"failed to read frozen inventory JSON: {error}"
        ) from error
    if inventory.get("schema") != "n3mapping_ros1_raw_bag_inventory_v1":
        raise SameFrameReferenceError("unsupported frozen inventory schema")
    if inventory.get("file_hashes_included") is not True:
        raise SameFrameReferenceError(
            "formal reference inventory must include bag SHA-256 values"
        )
    full_bag = inventory.get("full_bag")
    if not isinstance(full_bag, dict) or not SHA256_RE.fullmatch(
        str(full_bag.get("sha256", ""))
    ):
        raise SameFrameReferenceError(
            "frozen inventory is missing the full bag SHA-256"
        )
    matches = [
        entry
        for entry in inventory.get("query_bags", [])
        if entry.get("file") == query_bag.name
    ]
    if len(matches) != 1:
        raise SameFrameReferenceError(
            f"inventory must contain query filename exactly once: {query_bag.name}"
        )
    query_entry = matches[0]
    if not SHA256_RE.fullmatch(str(query_entry.get("sha256", ""))):
        raise SameFrameReferenceError(
            f"{query_bag.name} inventory entry is missing SHA-256"
        )
    overlap = query_entry.get("payload_overlap", {})
    if overlap.get("reference_class") != "same_frame_reference":
        raise SameFrameReferenceError(
            f"{query_bag.name} is not classified as same_frame_reference"
        )
    if overlap.get("matched_payload_ratio") != 1.0:
        raise SameFrameReferenceError(
            f"{query_bag.name} payload match ratio is not 1.0"
        )
    count = overlap.get("lidar_message_count")
    if (
        not isinstance(count, int)
        or count <= 0
        or overlap.get("matched_payload_count") != count
        or overlap.get("unique_payload_count") != count
        or overlap.get("unique_payload_ratio") != 1.0
    ):
        raise SameFrameReferenceError(
            f"{query_bag.name} inventory payload counts are not uniquely complete"
        )
    return inventory, query_entry


def unique_match_indices(
    targets: Iterable[int],
    references: list[int],
    *,
    tolerance_ns: int,
    label: str,
) -> list[int]:
    if tolerance_ns < 0:
        raise SameFrameReferenceError("join tolerance must be non-negative")
    matched_indices: list[int] = []
    used: set[int] = set()
    for target in targets:
        begin = bisect_left(references, target - tolerance_ns)
        end = bisect_right(references, target + tolerance_ns)
        candidates = list(range(begin, end))
        if len(candidates) != 1:
            raise SameFrameReferenceError(
                f"{label} stamp {target} has {len(candidates)} candidates "
                f"within +/-{tolerance_ns} ns"
            )
        candidate = candidates[0]
        if candidate in used:
            raise SameFrameReferenceError(
                f"{label} reference stamp {references[candidate]} was reused"
            )
        used.add(candidate)
        matched_indices.append(candidate)
    return matched_indices


def build_reference_rows(
    lio_rows: list[dict[str, Any]],
    dense_poses: list[DensePose],
) -> list[dict[str, Any]]:
    manifest_stamps = [int(row["stamp_ns"]) for row in lio_rows]
    dense_indices = unique_match_indices(
        manifest_stamps,
        [pose.stamp_ns for pose in dense_poses],
        tolerance_ns=JOIN_TOLERANCE_NS,
        label="dense trajectory join",
    )

    output: list[dict[str, Any]] = []
    for lio_row, dense_index in zip(lio_rows, dense_indices):
        manifest_stamp = int(lio_row["stamp_ns"])
        dense = dense_poses[dense_index]
        output.append(
            {
                "stamp_ns": manifest_stamp,
                "episode_id": lio_row.get("episode_id", ""),
                "frame_index": lio_row.get("frame_index", ""),
                "dense_stamp_ns": dense.stamp_ns,
                "manifest_minus_dense_ns": manifest_stamp - dense.stamp_ns,
                "dense_seq": "" if dense.seq is None else dense.seq,
                "tx": dense.tx,
                "ty": dense.ty,
                "tz": dense.tz,
                "qx": dense.qx,
                "qy": dense.qy,
                "qz": dense.qz,
                "qw": dense.qw,
            }
        )
    return output


def residual_summary(values: list[int]) -> dict[str, int]:
    return {"min": min(values), "max": max(values)}


REFERENCE_FIELDS = [
    "stamp_ns",
    "episode_id",
    "frame_index",
    "dense_stamp_ns",
    "manifest_minus_dense_ns",
    "dense_seq",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
]


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise SameFrameReferenceError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", dir=str(output_dir.parent)
        )
    )
    try:
        with (temporary / "reference_trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=REFERENCE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        evidence["output"]["reference_trajectory_sha256"] = sha256_file(
            temporary / "reference_trajectory.csv"
        )
        (temporary / "reference_evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed same-frame reference trajectory from a query "
            "ROS1 CustomMsg bag and a native dense pbstream trajectory export."
        )
    )
    parser.add_argument("--query-bag", required=True, type=Path)
    parser.add_argument("--lio-frames", required=True, type=Path)
    parser.add_argument("--dense-trajectory", required=True, type=Path)
    parser.add_argument("--dense-evidence", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory, query_entry = validate_inventory(args.inventory, args.query_bag)
    query_sha256 = sha256_file(args.query_bag)
    inventory_sha256 = query_entry["sha256"]
    if inventory_sha256 != query_sha256:
        raise SameFrameReferenceError(
            "query bag SHA-256 does not match frozen inventory"
        )
    input_directory = Path(str(inventory.get("input_directory", ""))).resolve()
    full_entry = inventory["full_bag"]
    full_bag = input_directory / str(full_entry.get("file", ""))
    if not full_bag.is_file():
        raise SameFrameReferenceError(
            f"frozen full bag does not exist: {full_bag}"
        )
    full_bag_sha256 = sha256_file(full_bag)
    if full_bag_sha256 != full_entry["sha256"]:
        raise SameFrameReferenceError(
            "full bag SHA-256 does not match frozen inventory"
        )

    lidar_topic = inventory.get("lidar_topic")
    if not isinstance(lidar_topic, str) or not lidar_topic:
        raise SameFrameReferenceError("frozen inventory is missing lidar_topic")
    raw_summary = read_raw_bag_summary(args.query_bag, lidar_topic)
    inventory_count = query_entry["payload_overlap"]["lidar_message_count"]
    if raw_summary.lidar_count != inventory_count:
        raise SameFrameReferenceError(
            f"raw bag/inventory lidar count mismatch: "
            f"{raw_summary.lidar_count} != {inventory_count}"
        )
    lio_rows = read_lio_manifest(args.lio_frames)
    if len(lio_rows) > raw_summary.lidar_count:
        raise SameFrameReferenceError(
            f"LIO output cannot exceed raw frame count: "
            f"{len(lio_rows)} > {raw_summary.lidar_count}"
        )
    dense_poses = read_dense_trajectory(args.dense_trajectory)
    dense_evidence = validate_dense_evidence(
        args.dense_evidence, args.dense_trajectory, len(dense_poses)
    )
    rows = build_reference_rows(lio_rows, dense_poses)

    dense_residuals = [int(row["manifest_minus_dense_ns"]) for row in rows]
    evidence = {
        "schema": "n3mapping_same_frame_reference_evidence_v1",
        "reference_class": "same_run_dense_optimized_reference",
        "inventory_payload_overlap": {
            "matched_payload_ratio": query_entry["payload_overlap"][
                "matched_payload_ratio"
            ],
            "unique_payload_ratio": query_entry["payload_overlap"][
                "unique_payload_ratio"
            ],
            "lidar_message_count": query_entry["payload_overlap"][
                "lidar_message_count"
            ],
        },
        "query_bag": str(args.query_bag.resolve()),
        "query_bag_sha256": query_sha256,
        "inventory_query_bag_sha256": inventory_sha256,
        "full_bag": str(full_bag),
        "full_bag_sha256": full_bag_sha256,
        "lio_frames": str(args.lio_frames.resolve()),
        "lio_frames_sha256": sha256_file(args.lio_frames),
        "dense_trajectory": str(args.dense_trajectory.resolve()),
        "dense_trajectory_sha256": sha256_file(args.dense_trajectory),
        "dense_evidence": str(args.dense_evidence.resolve()),
        "dense_evidence_sha256": sha256_file(args.dense_evidence),
        "pbstream": dense_evidence["pbstream"],
        "pbstream_sha256": dense_evidence["pbstream_sha256"],
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": sha256_file(args.inventory),
        "lidar_topic": lidar_topic,
        "raw_lidar_count": raw_summary.lidar_count,
        "raw_first_header_stamp_ns": raw_summary.first_header_stamp_ns,
        "raw_last_header_stamp_ns": raw_summary.last_header_stamp_ns,
        "lio_manifest_count": len(lio_rows),
        "skipped_raw_lidar_count": raw_summary.lidar_count - len(lio_rows),
        "lio_to_raw_coverage_ratio": (
            len(lio_rows) / raw_summary.lidar_count
        ),
        "reference_pose_count": len(rows),
        "dense_trajectory_count": len(dense_poses),
        "timestamp_alignment": {
            "raw_to_lio_frame_index_inferred": False,
            "acquisition_start_definition": (
                "first query CustomMsg header.stamp_ns"
            ),
            "join_tolerance_ns": JOIN_TOLERANCE_NS,
            "fitted_time_offset": False,
            "manifest_minus_dense_ns": residual_summary(dense_residuals),
        },
        "pose_convention": "T_world_body",
        "pose_source": "native pbstream dense trajectory",
        "output": {
            "reference_trajectory": "reference_trajectory.csv",
            "reference_evidence": "reference_evidence.json",
        },
    }
    write_outputs(args.output.resolve(), rows, evidence)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "reference_pose_count": len(rows),
                "join_tolerance_ns": JOIN_TOLERANCE_NS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - CLI must fail closed on bad evidence
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
