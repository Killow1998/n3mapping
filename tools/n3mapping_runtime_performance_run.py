#!/usr/bin/env python3
"""Run one headless FA-01 runtime profile with ordinary local processes."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, TextIO
from zoneinfo import ZoneInfo


SCHEMA = "n3mapping_runtime_performance_run_v1"
TIMEZONE = "America/Los_Angeles"
MODES = ("mapping", "localization", "map_extension")


class RunError(RuntimeError):
    pass


def write_json(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(resolved)
    return result


def bag_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise RunError(f"bag is not a directory: {resolved}")
    payloads = sorted(
        child for child in resolved.iterdir() if child.is_file() and child.suffix == ".db3"
    )
    metadata = resolved / "metadata.yaml"
    if not metadata.is_file() or not payloads:
        raise RunError(f"bag lacks metadata.yaml or db3 payload: {resolved}")
    return {
        "path": str(resolved),
        "metadata": file_identity(metadata, include_sha256=True),
        "payloads": [file_identity(payload, include_sha256=False) for payload in payloads],
    }


def count_record_type(path: Path, record_type: str) -> int:
    if not path.is_file():
        return 0
    needle = f'"record_type":"{record_type}"'
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(needle in line for line in stream)


def scan_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {
        "fatal": text.lower().count("fatal"),
        "oom": sum(
            text.lower().count(pattern)
            for pattern in ("out of memory", "std::bad_alloc", "oom-kill", "[oom]")
        ),
        "tracking_failed": text.count("Tracking failed"),
        "message_filter_drop": text.count("Message Filter dropping"),
        "queue_overflow": text.lower().count("queue overflow"),
        "nonfinite": text.lower().count("non-finite") + text.lower().count("nonfinite"),
    }


def signal_group(process: subprocess.Popen[Any] | None, sig: signal.Signals) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_process(
    process: subprocess.Popen[Any] | None,
    *,
    interrupt_timeout_s: float = 15.0,
) -> int | None:
    if process is None:
        return None
    if process.poll() is None:
        signal_group(process, signal.SIGINT)
        try:
            return process.wait(timeout=interrupt_timeout_s)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGTERM)
        try:
            return process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
    return process.wait(timeout=5.0)


def wait_for_text(
    path: Path, text: str, process: subprocess.Popen[Any], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunError(f"node exited before readiness: rc={process.returncode}")
        if path.is_file() and text in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.2)
    raise RunError(f"node readiness timeout after {timeout_s:.1f}s")


def build_environment(output: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ROS_LOG_DIR": str(output / "ros_log"),
            "GLOG_v": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def run_checked(
    command: list[str],
    *,
    timeout_s: float = 30.0,
    environment: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=environment,
    )
    if completed.returncode != 0:
        raise RunError(
            f"command failed rc={completed.returncode}: {command!r}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def locate_node(environment: dict[str, str]) -> Path:
    prefix = Path(
        run_checked(
            ["ros2", "pkg", "prefix", "n3mapping"], environment=environment
        ).strip()
    )
    node = prefix / "lib" / "n3mapping" / "n3mapping_node"
    if not node.is_file() or not os.access(node, os.X_OK):
        raise RunError(f"installed node missing or not executable: {node}")
    return node.resolve()


def ensure_node_absent(environment: dict[str, str]) -> None:
    completed = subprocess.run(
        ["ros2", "node", "list"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    if completed.returncode == 0 and any(
        name.strip() == "/n3mapping_node" for name in completed.stdout.splitlines()
    ):
        raise RunError("/n3mapping_node already exists; refusing competing profile")


def parse_identity(
    node: Path, expected_commit: str, environment: dict[str, str]
) -> dict[str, Any]:
    try:
        identity = json.loads(
            run_checked(
                [str(node), "--build-identity-json"], environment=environment
            )
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RunError(f"invalid node build identity: {error}") from error
    if not isinstance(identity, dict):
        raise RunError("node build identity is not an object")
    if identity.get("commit") != expected_commit or identity.get("verified") is not True:
        raise RunError(
            "node build identity mismatch: "
            f"expected={expected_commit} actual={identity.get('commit')} "
            f"verified={identity.get('verified')}"
        )
    return identity


def make_node_command(args: argparse.Namespace, node: Path) -> list[str]:
    command = [
        "stdbuf",
        "-oL",
        "-eL",
        str(node),
        "--ros-args",
        "--params-file",
        str(args.config.resolve()),
        "-p",
        f"mode:={args.mode}",
        "-p",
        f"map_save_path:={args.output_dir.resolve()}",
        "-p",
        "floor_attitude_enable:=false",
        "-p",
        "save_global_map_on_shutdown:=false",
        "-p",
        "reloc_debug_enable:=true",
        "-p",
        f"reloc_debug_path:={args.output_dir.resolve() / 'relocalization_debug.jsonl'}",
    ]
    if args.mode != "mapping":
        command.extend(["-p", f"map_path:={args.map.resolve()}"])
    return command


def make_bag_command(args: argparse.Namespace) -> list[str]:
    return [
        "ros2",
        "bag",
        "play",
        str(args.bag.resolve()),
        "--start-offset",
        str(args.start_offset_s),
        "--rate",
        str(args.rate),
        "--read-ahead-queue-size",
        "100",
        "--delay",
        "3",
        "--disable-keyboard-controls",
        "--topics",
        "/cloud_registered_body",
        "/Odometry",
    ]


def run_profile(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise RunError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    (output / "ros_log").mkdir()
    environment = build_environment(output)
    manifest_path = output / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "STARTING",
        "recorded_timezone": TIMEZONE,
        "started_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "mode": args.mode,
        "target_runtime_frames": args.target_frames,
        "start_offset_s": args.start_offset_s,
        "rate": args.rate,
        "expected_commit": args.expected_commit,
    }
    write_json(manifest_path, manifest)

    node_process: subprocess.Popen[Any] | None = None
    sampler_process: subprocess.Popen[Any] | None = None
    bag_process: subprocess.Popen[Any] | None = None
    handles: list[TextIO] = []
    return_codes: dict[str, int | None] = {}
    try:
        if not args.config.is_file():
            raise RunError(f"config missing: {args.config}")
        if args.mode != "mapping" and not args.map.is_file():
            raise RunError(f"map missing: {args.map}")
        ensure_node_absent(environment)
        node = locate_node(environment)
        identity = parse_identity(node, args.expected_commit, environment)
        sampler = Path(__file__).resolve().with_name(
            "n3mapping_process_resource_sample.py"
        )
        analyzer = Path(__file__).resolve().with_name(
            "n3mapping_runtime_performance_analyze.py"
        )
        for tool in (sampler, analyzer):
            if not tool.is_file():
                raise RunError(f"required sibling tool missing: {tool}")

        node_command = make_node_command(args, node)
        bag_command = make_bag_command(args)
        sampler_command: list[str] = []
        manifest.update(
            {
                "node_build_identity": identity,
                "node_executable": file_identity(node, include_sha256=True),
                "config": file_identity(args.config, include_sha256=True),
                "map": (
                    file_identity(args.map, include_sha256=True)
                    if args.mode != "mapping"
                    else None
                ),
                "bag": bag_identity(args.bag),
                "commands": {"node": node_command, "bag": bag_command},
                "status": "LAUNCHING",
            }
        )
        write_json(manifest_path, manifest)

        node_log = (output / "n3mapping.log").open("w", encoding="utf-8")
        handles.append(node_log)
        node_process = subprocess.Popen(
            node_command,
            stdout=node_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        wait_for_text(
            output / "n3mapping.log",
            "N3Mapping node initialized",
            node_process,
            args.node_ready_timeout_s,
        )

        sampler_command = [
            sys.executable,
            "-B",
            str(sampler),
            "--pid",
            str(node_process.pid),
            "--output",
            str(output / "process_resource.jsonl"),
            "--interval",
            str(args.resource_interval_s),
        ]
        sampler_log = (output / "resource_sampler.log").open(
            "w", encoding="utf-8"
        )
        handles.append(sampler_log)
        sampler_process = subprocess.Popen(
            sampler_command,
            stdout=sampler_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        bag_log = (output / "bag.log").open("w", encoding="utf-8")
        handles.append(bag_log)
        bag_process = subprocess.Popen(
            bag_command,
            stdout=bag_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        manifest["commands"]["sampler"] = sampler_command
        manifest["pids"] = {
            "node": node_process.pid,
            "sampler": sampler_process.pid,
            "bag": bag_process.pid,
        }
        manifest["status"] = "RUNNING"
        write_json(manifest_path, manifest)
        print(f"RUNNING mode={args.mode} output={output}", flush=True)

        runtime_path = output / "runtime_performance_debug.jsonl"
        deadline = time.monotonic() + args.run_timeout_s
        last_reported = 0
        while time.monotonic() < deadline:
            if node_process.poll() is not None:
                raise RunError(f"node exited during replay: rc={node_process.returncode}")
            if bag_process.poll() is not None:
                raise RunError(
                    f"bag exited before target frames: rc={bag_process.returncode}"
                )
            frame_count = count_record_type(runtime_path, "runtime_frame")
            if frame_count >= args.target_frames:
                break
            if frame_count >= last_reported + 100:
                last_reported = (frame_count // 100) * 100
                print(f"PROGRESS mode={args.mode} frames={frame_count}", flush=True)
            time.sleep(0.5)
        else:
            raise RunError(
                f"runtime frame timeout: observed={count_record_type(runtime_path, 'runtime_frame')}"
            )

        return_codes["bag"] = stop_process(bag_process)
        time.sleep(args.drain_s)
        final_frames = count_record_type(runtime_path, "runtime_frame")
        if final_frames < args.target_frames:
            raise RunError(
                f"runtime frame target not reached: {final_frames} < {args.target_frames}"
            )

        save_output = "not_applicable"
        if args.mode in ("mapping", "map_extension"):
            save_output = run_checked(
                [
                    "ros2",
                    "service",
                    "call",
                    "/n3mapping/save_map",
                    "std_srvs/srv/Trigger",
                    "{}",
                ],
                timeout_s=120.0,
                environment=environment,
            )
            (output / "save_map.log").write_text(save_output, encoding="utf-8")
            if "success=True" not in save_output:
                raise RunError("save_map did not return success=True")

        return_codes["node"] = stop_process(node_process, interrupt_timeout_s=30.0)
        return_codes["sampler"] = stop_process(sampler_process)
        for handle in handles:
            handle.flush()

        analyzer_command = [
            sys.executable,
            "-B",
            str(analyzer),
            "--input-dir",
            str(output),
            "--sensor-period-ms",
            str(args.sensor_period_ms),
            "--output",
            str(output / "performance_summary.json"),
        ]
        analyzer_stdout = output / "performance_summary.stdout.json"
        analyzer_stderr = output / "performance_analyzer.stderr.log"
        with analyzer_stdout.open("w", encoding="utf-8") as stdout, analyzer_stderr.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                analyzer_command,
                check=False,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=60.0,
                env=environment,
            )
        return_codes["analyzer"] = completed.returncode
        summary = json.loads(
            (output / "performance_summary.json").read_text(encoding="utf-8")
        )
        events = {
            "node": scan_log(output / "n3mapping.log"),
            "bag": scan_log(output / "bag.log"),
        }
        write_json(output / "runtime_events.json", events)
        manifest.update(
            {
                "status": (
                    "COMPLETE"
                    if completed.returncode == 0
                    and summary.get("status") == "PROFILE_READY"
                    else "EVIDENCE_NOT_READY"
                ),
                "completed_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "observed_runtime_frames": final_frames,
                "process_return_codes": return_codes,
                "commands": {
                    **manifest["commands"],
                    "analyzer": analyzer_command,
                },
                "performance_status": summary.get("status"),
                "runtime_events": events,
                "save_map_response": save_output.strip(),
            }
        )
        write_json(manifest_path, manifest)
        if manifest["status"] == "COMPLETE":
            (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
            print(f"COMPLETE mode={args.mode} frames={final_frames}", flush=True)
            return 0
        print(
            f"EVIDENCE_NOT_READY mode={args.mode} analyzer_rc={completed.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "completed_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "error": str(error),
            }
        )
        write_json(manifest_path, manifest)
        print(f"FAILED mode={args.mode}: {error}", file=sys.stderr, flush=True)
        return 3
    finally:
        if "bag" not in return_codes:
            return_codes["bag"] = stop_process(bag_process)
        if "node" not in return_codes:
            return_codes["node"] = stop_process(
                node_process, interrupt_timeout_s=30.0
            )
        if "sampler" not in return_codes:
            return_codes["sampler"] = stop_process(sampler_process)
        for handle in handles:
            handle.close()
        manifest["process_return_codes"] = return_codes
        try:
            write_json(manifest_path, manifest)
        except OSError:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--target-frames", type=int, default=620)
    parser.add_argument("--start-offset-s", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--sensor-period-ms", type=float, default=100.0)
    parser.add_argument("--resource-interval-s", type=float, default=1.0)
    parser.add_argument("--node-ready-timeout-s", type=float, default=60.0)
    parser.add_argument("--run-timeout-s", type=float, default=600.0)
    parser.add_argument("--drain-s", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.mode != "mapping" and args.map is None:
        parser.error("--map is required for localization and map_extension")
    if len(args.expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.expected_commit
    ):
        parser.error("--expected-commit must be a lowercase 40-character SHA-1")
    if args.target_frames < 500:
        parser.error("--target-frames must be at least 500")
    for field in (
        "start_offset_s",
        "rate",
        "sensor_period_ms",
        "resource_interval_s",
        "node_ready_timeout_s",
        "run_timeout_s",
        "drain_s",
    ):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be positive and finite")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run_profile(args)
    except RunError as error:
        print(f"FAILED before output initialization: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
