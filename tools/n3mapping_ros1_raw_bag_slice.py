#!/usr/bin/env python3
"""Create and verify deterministic, raw-byte-preserving ROS1 bag slices.

The slice interval is the half-open storage-time range ``[start_ns, stop_ns)``.
Only the explicitly selected topics are copied.  The lineage sidecar binds the
complete source bag, the resulting slice, connection metadata, storage/header
timestamps, and every serialized message byte.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "rosbags is required: python3 -m pip install "
        "'rosbags>=0.9.23,<0.12'"
    ) from exc


SCHEMA = "n3mapping_ros1_raw_bag_slice_lineage_v1"
SCHEMA_VERSION = 1
SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "created_at",
    "spec",
    "artifacts",
    "selection",
    "checks",
    "verdict",
}
SPEC_KEYS = {
    "source_ros1_bag",
    "slice_ros1_bag",
    "start_ns",
    "stop_ns",
    "topics",
}
SUMMARY_KEYS = {
    "interval_semantics",
    "message_count",
    "first_storage_stamp_ns",
    "last_storage_stamp_ns",
    "message_sequence_sha256",
    "topics",
}
TOPIC_SUMMARY_KEYS = {
    "topic",
    "message_type",
    "ros1_md5sum",
    "message_definition_sha256",
    "callerid",
    "latching",
    "message_count",
    "first_storage_stamp_ns",
    "last_storage_stamp_ns",
    "first_header_stamp_ns",
    "last_header_stamp_ns",
    "message_sequence_sha256",
}


class SliceLineageError(RuntimeError):
    """A fail-closed slice or lineage contract violation."""


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_chain(
    path: Path, allow_missing_leaf: bool = False
) -> Path:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if allow_missing_leaf:
                return absolute
            raise SliceLineageError(f"path does not exist: {absolute}")
        if stat.S_ISLNK(mode):
            raise SliceLineageError(f"symlinks are forbidden: {current}")
    return absolute


def require_regular_file(path: Path, label: str) -> Path:
    absolute = _require_no_symlink_chain(path)
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise SliceLineageError(
            f"{label} must be a regular file: {absolute}"
        )
    return absolute


def _prepare_new_file(path: Path, label: str) -> Path:
    absolute = _require_no_symlink_chain(path, allow_missing_leaf=True)
    if absolute.exists():
        raise SliceLineageError(f"{label} already exists: {absolute}")
    parent = _require_no_symlink_chain(
        absolute.parent, allow_missing_leaf=True
    )
    parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_chain(parent)
    return absolute


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, label: str) -> Dict[str, Any]:
    regular = require_regular_file(path, label)
    return {
        "path": str(regular),
        "bytes": regular.stat().st_size,
        "sha256": sha256_file(regular),
    }


def _framed_update(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SliceLineageError(f"{label} must be an integer")
    return value


def _strict_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SliceLineageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path, label: str) -> Dict[str, Any]:
    regular = require_regular_file(path, label)
    try:
        value = json.loads(
            regular.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SliceLineageError(
                    f"{label} contains non-finite JSON value {token}"
                )
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SliceLineageError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise SliceLineageError(f"{label} root must be an object")
    return value


def _exact_keys(value: Any, expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    if not isinstance(value, dict) or set(value) != expected_set:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise SliceLineageError(
            f"{label} keys mismatch: expected {sorted(expected_set)}, "
            f"got {observed}"
        )


def _created_at_now() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def _validate_created_at(value: Any) -> str:
    if not isinstance(value, str):
        raise SliceLineageError("created_at must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SliceLineageError("created_at is not ISO-8601") from error
    if parsed.utcoffset() != timedelta(hours=8):
        raise SliceLineageError("created_at must use Asia/Shanghai UTC+8")
    return value


def _normalized_topics(raw_topics: Sequence[str]) -> List[str]:
    if not raw_topics:
        raise SliceLineageError("at least one topic is required")
    topics: List[str] = []
    for topic in raw_topics:
        if not isinstance(topic, str) or not topic.startswith("/") or topic == "/":
            raise SliceLineageError(
                "each selected topic must be a non-root absolute ROS topic"
            )
        if "\0" in topic:
            raise SliceLineageError("selected topic contains a NUL byte")
        topics.append(topic)
    if len(set(topics)) != len(topics):
        raise SliceLineageError("selected topics contain duplicates")
    return sorted(topics)


def _header_stamp_ns(message: Any, label: str) -> int:
    try:
        stamp = message.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as error:
        raise SliceLineageError(
            f"{label} does not contain a valid header stamp"
        ) from error
    if value < 0:
        raise SliceLineageError(f"{label} has a negative header stamp")
    return value


def _message_definition_text(connection: Any) -> str:
    raw_definition = connection.msgdef
    message_definition = getattr(raw_definition, "data", raw_definition)
    if not isinstance(message_definition, str) or not message_definition:
        raise SliceLineageError(
            f"{connection.topic} has no embedded ROS1 message definition"
        )
    return message_definition


def _connection_fields(connection: Any) -> Dict[str, Any]:
    message_definition = _message_definition_text(connection)
    extension = connection.ext
    callerid = getattr(extension, "callerid", None)
    latching = getattr(extension, "latching", None)
    if callerid is not None and not isinstance(callerid, str):
        raise SliceLineageError(f"{connection.topic} callerid is invalid")
    if latching is not None:
        latching = _integer(latching, f"{connection.topic} latching")
    return {
        "topic": connection.topic,
        "message_type": connection.msgtype,
        "ros1_md5sum": connection.digest,
        "message_definition_sha256": hashlib.sha256(
            message_definition.encode("utf-8")
        ).hexdigest(),
        "callerid": callerid,
        "latching": latching,
    }


def _selected_connections(
    reader: Any, topics: Sequence[str], exact_topics: bool
) -> Dict[str, Any]:
    if reader.is2:
        raise SliceLineageError("input must be a ROS1 bag")
    by_topic: Dict[str, List[Any]] = {}
    for connection in reader.connections:
        by_topic.setdefault(connection.topic, []).append(connection)
    observed = set(by_topic)
    expected = set(topics)
    if exact_topics and observed != expected:
        raise SliceLineageError(
            f"slice topics mismatch: expected {sorted(expected)}, "
            f"got {sorted(observed)}"
        )
    missing = sorted(expected - observed)
    if missing:
        raise SliceLineageError(
            "source bag is missing selected topics: " + ", ".join(missing)
        )
    result: Dict[str, Any] = {}
    for topic in topics:
        connections = by_topic[topic]
        if len(connections) != 1:
            raise SliceLineageError(
                f"expected exactly one connection for {topic}, "
                f"found {len(connections)}"
            )
        result[topic] = connections[0]
    return result


def collect_stream_summary(
    path: Path,
    topics: Sequence[str],
    start_ns: Optional[int],
    stop_ns: Optional[int],
    exact_topics: bool,
) -> Dict[str, Any]:
    bag = require_regular_file(path, "ROS1 bag")
    selected_topics = _normalized_topics(topics)
    if (start_ns is None) != (stop_ns is None):
        raise SliceLineageError(
            "start_ns and stop_ns must both be set or both be omitted"
        )
    if start_ns is not None:
        start_ns = _integer(start_ns, "start_ns")
        stop_ns = _integer(stop_ns, "stop_ns")
        if start_ns < 0 or stop_ns <= start_ns:
            raise SliceLineageError(
                "slice interval must satisfy 0 <= start_ns < stop_ns"
            )

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = _selected_connections(
            reader, selected_topics, exact_topics
        )
        states: Dict[str, Dict[str, Any]] = {}
        for topic in selected_topics:
            states[topic] = {
                "metadata": _connection_fields(connections[topic]),
                "count": 0,
                "first_storage": None,
                "last_storage": None,
                "first_header": None,
                "last_header": None,
                "digest": hashlib.sha256(),
            }
        selected = list(connections.values())
        for connection, storage_stamp, rawdata in reader.messages(
            connections=selected, start=start_ns, stop=stop_ns
        ):
            topic = connection.topic
            state = states[topic]
            storage_value = _integer(
                storage_stamp, f"{topic} storage timestamp"
            )
            if (
                start_ns is not None
                and not (start_ns <= storage_value < stop_ns)
            ):
                raise SliceLineageError(
                    f"{topic} message escaped the half-open interval"
                )
            message = reader.deserialize(rawdata, connection.msgtype)
            header_value = _header_stamp_ns(message, topic)
            if (
                state["last_storage"] is not None
                and storage_value <= state["last_storage"]
            ):
                raise SliceLineageError(
                    f"{topic} storage timestamps are not strictly increasing"
                )
            if (
                state["last_header"] is not None
                and header_value <= state["last_header"]
            ):
                raise SliceLineageError(
                    f"{topic} header timestamps are not strictly increasing"
                )
            if state["count"] == 0:
                state["first_storage"] = storage_value
                state["first_header"] = header_value
            state["last_storage"] = storage_value
            state["last_header"] = header_value
            state["count"] += 1
            record = hashlib.sha256()
            _framed_update(record, topic.encode("utf-8"))
            _framed_update(record, struct.pack(">q", storage_value))
            _framed_update(record, struct.pack(">q", header_value))
            _framed_update(record, bytes(rawdata))
            _framed_update(state["digest"], record.digest())

    topic_summaries: Dict[str, Any] = {}
    aggregate = hashlib.sha256()
    total_count = 0
    first_storage: Optional[int] = None
    last_storage: Optional[int] = None
    for topic in selected_topics:
        state = states[topic]
        if state["count"] == 0:
            raise SliceLineageError(
                f"selected interval contains no messages for {topic}"
            )
        summary = {
            **state["metadata"],
            "message_count": state["count"],
            "first_storage_stamp_ns": state["first_storage"],
            "last_storage_stamp_ns": state["last_storage"],
            "first_header_stamp_ns": state["first_header"],
            "last_header_stamp_ns": state["last_header"],
            "message_sequence_sha256": state["digest"].hexdigest(),
        }
        topic_summaries[topic] = summary
        total_count += state["count"]
        first_storage = (
            state["first_storage"]
            if first_storage is None
            else min(first_storage, state["first_storage"])
        )
        last_storage = (
            state["last_storage"]
            if last_storage is None
            else max(last_storage, state["last_storage"])
        )
        _framed_update(aggregate, topic.encode("utf-8"))
        _framed_update(
            aggregate, summary["message_sequence_sha256"].encode("ascii")
        )
    return {
        "interval_semantics": "half_open_storage_time_[start_ns,stop_ns)",
        "message_count": total_count,
        "first_storage_stamp_ns": first_storage,
        "last_storage_stamp_ns": last_storage,
        "message_sequence_sha256": aggregate.hexdigest(),
        "topics": topic_summaries,
    }


def copy_raw_slice(
    source: Path,
    output: Path,
    start_ns: int,
    stop_ns: int,
    topics: Sequence[str],
) -> None:
    source_path = require_regular_file(source, "source ROS1 bag")
    output_path = _prepare_new_file(output, "slice ROS1 bag")
    selected_topics = _normalized_topics(topics)
    start_value = _integer(start_ns, "start_ns")
    stop_value = _integer(stop_ns, "stop_ns")
    if start_value < 0 or stop_value <= start_value:
        raise SliceLineageError(
            "slice interval must satisfy 0 <= start_ns < stop_ns"
        )
    temporary = output_path.with_name(
        ".{}.{}.tmp".format(output_path.name, os.getpid())
    )
    if temporary.exists():
        raise SliceLineageError(
            f"temporary slice path already exists: {temporary}"
        )
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    try:
        with AnyReader(
            [source_path], default_typestore=typestore
        ) as reader, Writer(temporary) as writer:
            connections = _selected_connections(
                reader, selected_topics, exact_topics=False
            )
            output_connections: Dict[int, Any] = {}
            for topic in selected_topics:
                connection = connections[topic]
                extension = connection.ext
                output_connections[connection.id] = writer.add_connection(
                    connection.topic,
                    connection.msgtype,
                    msgdef=_message_definition_text(connection),
                    md5sum=connection.digest,
                    callerid=getattr(extension, "callerid", None),
                    latching=getattr(extension, "latching", None),
                )
            copied = 0
            for connection, storage_stamp, rawdata in reader.messages(
                connections=list(connections.values()),
                start=start_value,
                stop=stop_value,
            ):
                writer.write(
                    output_connections[connection.id],
                    storage_stamp,
                    rawdata,
                )
                copied += 1
            if copied == 0:
                raise SliceLineageError(
                    "selected interval contains no messages"
                )
        os.replace(os.fspath(temporary), os.fspath(output_path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_spec(raw: Mapping[str, Any]) -> Dict[str, Any]:
    _exact_keys(raw, SPEC_KEYS, "spec")
    source_text = raw["source_ros1_bag"]
    slice_text = raw["slice_ros1_bag"]
    if not isinstance(source_text, str) or not source_text:
        raise SliceLineageError("spec.source_ros1_bag must be a path")
    if not isinstance(slice_text, str) or not slice_text:
        raise SliceLineageError("spec.slice_ros1_bag must be a path")
    source = _absolute_without_resolving(Path(source_text))
    sliced = _absolute_without_resolving(Path(slice_text))
    if str(source) != source_text or str(sliced) != slice_text:
        raise SliceLineageError("lineage paths must be normalized and absolute")
    if source == sliced:
        raise SliceLineageError("source and slice ROS1 bags must differ")
    start_ns = _integer(raw["start_ns"], "spec.start_ns")
    stop_ns = _integer(raw["stop_ns"], "spec.stop_ns")
    if start_ns < 0 or stop_ns <= start_ns:
        raise SliceLineageError(
            "spec interval must satisfy 0 <= start_ns < stop_ns"
        )
    if not isinstance(raw["topics"], list):
        raise SliceLineageError("spec.topics must be an array")
    topics = _normalized_topics(raw["topics"])
    if topics != raw["topics"]:
        raise SliceLineageError(
            "spec.topics must be sorted and contain no duplicates"
        )
    return {
        "source_ros1_bag": str(source),
        "slice_ros1_bag": str(sliced),
        "start_ns": start_ns,
        "stop_ns": stop_ns,
        "topics": topics,
    }


def collect_lineage(
    raw_spec: Mapping[str, Any], created_at: Optional[str] = None
) -> Dict[str, Any]:
    spec = _validated_spec(raw_spec)
    created = (
        _validate_created_at(created_at)
        if created_at is not None
        else _created_at_now()
    )
    source_path = Path(spec["source_ros1_bag"])
    slice_path = Path(spec["slice_ros1_bag"])
    source_before = file_identity(source_path, "source ROS1 bag")
    slice_before = file_identity(slice_path, "slice ROS1 bag")
    source_selection = collect_stream_summary(
        source_path,
        spec["topics"],
        spec["start_ns"],
        spec["stop_ns"],
        exact_topics=False,
    )
    slice_selection = collect_stream_summary(
        slice_path,
        spec["topics"],
        None,
        None,
        exact_topics=True,
    )
    source_after = file_identity(source_path, "source ROS1 bag")
    slice_after = file_identity(slice_path, "slice ROS1 bag")
    if source_before != source_after or slice_before != slice_after:
        raise SliceLineageError(
            "source or slice ROS1 bag changed while lineage was collected"
        )
    if source_selection != slice_selection:
        raise SliceLineageError(
            "slice does not exactly preserve the selected source messages"
        )
    first_storage = source_selection["first_storage_stamp_ns"]
    last_storage = source_selection["last_storage_stamp_ns"]
    if (
        first_storage is None
        or last_storage is None
        or first_storage < spec["start_ns"]
        or last_storage >= spec["stop_ns"]
    ):
        raise SliceLineageError(
            "slice messages violate the half-open storage-time interval"
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": created,
        "spec": spec,
        "artifacts": {
            "source_ros1_bag": source_after,
            "slice_ros1_bag": slice_after,
        },
        "selection": source_selection,
        "checks": {
            "source_to_slice_raw_serialized_bytes": "PASS",
            "slice_exact_topics": "PASS",
            "half_open_storage_time_interval": "PASS",
        },
        "verdict": "PASS",
    }


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    absolute = _prepare_new_file(path, "lineage output")
    temporary = absolute.with_name(
        ".{}.{}.tmp".format(absolute.name, os.getpid())
    )
    if temporary.exists():
        raise SliceLineageError(
            f"temporary lineage path already exists: {temporary}"
        )
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(os.fspath(temporary), os.fspath(absolute))
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_lineage(path: Path) -> Dict[str, Any]:
    lineage_path = require_regular_file(path, "slice lineage")
    frozen = load_strict_json(lineage_path, "slice lineage")
    _exact_keys(frozen, TOP_LEVEL_KEYS, "lineage")
    if frozen.get("schema") != SCHEMA:
        raise SliceLineageError("slice lineage schema mismatch")
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise SliceLineageError("slice lineage schema_version mismatch")
    if frozen.get("verdict") != "PASS":
        raise SliceLineageError("slice lineage verdict is not PASS")
    _validate_created_at(frozen.get("created_at"))
    _validated_spec(frozen.get("spec", {}))
    _exact_keys(frozen.get("selection"), SUMMARY_KEYS, "selection")
    topics = frozen["selection"].get("topics")
    if not isinstance(topics, dict):
        raise SliceLineageError("selection.topics must be an object")
    for topic, summary in topics.items():
        _exact_keys(
            summary, TOPIC_SUMMARY_KEYS, "selection.topics.{}".format(topic)
        )
    rebuilt = collect_lineage(
        frozen["spec"], created_at=frozen["created_at"]
    )
    if rebuilt != frozen:
        raise SliceLineageError(
            "frozen slice lineage does not match freshly rebuilt evidence"
        )
    return {
        "lineage": str(lineage_path),
        "lineage_sha256": sha256_file(lineage_path),
        "source_ros1_bag": rebuilt["spec"]["source_ros1_bag"],
        "slice_ros1_bag": rebuilt["spec"]["slice_ros1_bag"],
        "start_ns": rebuilt["spec"]["start_ns"],
        "stop_ns": rebuilt["spec"]["stop_ns"],
        "topics": rebuilt["spec"]["topics"],
        "message_count": rebuilt["selection"]["message_count"],
        "verdict": "PASS",
    }


def create_slice_and_lineage(
    source: Path,
    output_bag: Path,
    lineage: Path,
    start_ns: int,
    stop_ns: int,
    topics: Sequence[str],
) -> Dict[str, Any]:
    source_path = require_regular_file(source, "source ROS1 bag")
    output_path = _absolute_without_resolving(output_bag)
    lineage_path = _absolute_without_resolving(lineage)
    if output_path == lineage_path:
        raise SliceLineageError(
            "slice ROS1 bag and lineage output must differ"
        )
    _prepare_new_file(lineage_path, "lineage output")
    copy_raw_slice(
        source_path, output_path, start_ns, stop_ns, topics
    )
    lineage_written = False
    try:
        spec = {
            "source_ros1_bag": str(source_path),
            "slice_ros1_bag": str(output_path),
            "start_ns": start_ns,
            "stop_ns": stop_ns,
            "topics": _normalized_topics(topics),
        }
        evidence = collect_lineage(spec)
        write_new_json(lineage_path, evidence)
        lineage_written = True
        return verify_lineage(lineage_path)
    except Exception:
        if output_path.exists():
            output_path.unlink()
        if lineage_written and lineage_path.exists():
            lineage_path.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify a deterministic raw-byte-preserving ROS1 bag "
            "slice and its strict lineage evidence."
        )
    )
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output-bag", type=Path, required=True)
    create.add_argument("--lineage", type=Path, required=True)
    create.add_argument("--start-ns", type=int, required=True)
    create.add_argument("--stop-ns", type=int, required=True)
    create.add_argument("--topic", action="append", required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("--lineage", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "create":
        result = create_slice_and_lineage(
            args.source,
            args.output_bag,
            args.lineage,
            args.start_ns,
            args.stop_ns,
            args.topic,
        )
    else:
        result = verify_lineage(args.lineage)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SliceLineageError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
