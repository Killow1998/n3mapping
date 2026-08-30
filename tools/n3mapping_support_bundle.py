#!/usr/bin/env python3
"""Prepare a manual ROS 2 test and export bounded, local-only support logs."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import zipfile


MIB = 1024 * 1024
ROOT_FILES = (
    "run.json", "build_identity.json", "config.yaml", "overrides.yaml",
    "parameters.yaml", "issue.md", "n3mapping.log", "frontend.log",
    "rviz.log", "bag.log", "save_map.log", "optimization.log",
    "relocalization_debug.jsonl", "loop_debug.jsonl",
    "runtime_performance_debug.jsonl", "process_resource.jsonl",
)
ISSUE_TEMPLATE = """# 实机问题反馈

- 模式、传感器和前端版本：
- 预期行为：
- 实际行为：
- 问题发生时间（注明时区，最好同时提供日志中的时间戳）：
- 最小复现步骤，是否每次发生：
- 地图来源、是否在地图覆盖范围内（不必上传地图）：
- CPU/内存限制，以及是否同时运行其他重负载：
- 截图或视频（单独附件）：

上传前请检查本目录的配置、日志和路径，移除凭据及不适合公开的位置信息。
此工具不自动脱敏、不上传文件；不要将日志或地图提交进 Git。
"""


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def prepare(args: argparse.Namespace) -> Path:
    with args.config.open("rb") as source:
        config = source.read(MIB + 1)
    if not config or len(config) > MIB:
        raise ValueError("config must be non-empty and no larger than 1 MiB")
    # Do not resolve __file__: under symlink-install the node is beside the
    # installed script, not beside the script's source-tree target.
    node = (args.node_executable or
            Path(__file__).absolute().with_name("n3mapping_node")).resolve()
    result = subprocess.run([str(node), "--build-identity-json"],
                            capture_output=True, text=True, timeout=10, check=True)
    identity = json.loads(result.stdout)
    if not isinstance(identity, dict) or not isinstance(identity.get("commit"), str):
        raise ValueError("node did not return a build identity with a commit")
    output = args.run_dir.absolute()
    # mkdir, rather than exist_ok, prevents overwriting a previous run's logs.
    output.mkdir(parents=True)
    output = output.resolve()
    (output / "config.yaml").write_bytes(config)
    write_json(output / "build_identity.json", identity)
    write_json(output / "run.json", {
        "prepared_at": now(), "mode": args.mode,
        "node_executable": str(node), "ros_distro": os.environ.get("ROS_DISTRO"),
        "diagnostics_requested": args.diagnostics,
        "note": "Preparation only; no node was started. Config is a snapshot, "
                "not a live parameter dump. Use this recorded executable.",
    })
    overrides = {
        "mode": args.mode, "map_save_path": str(output),
        "reloc_debug_path": str(output / "relocalization_debug.jsonl"),
        "loop_debug_path": str(output / "loop_debug.jsonl"),
    }
    if args.diagnostics:
        overrides.update(reloc_debug_enable=True, loop_debug_enable=True)
    # JSON is valid YAML. No extra YAML dependency is required for this tool.
    write_json(output / "overrides.yaml",
               {"n3mapping_node": {"ros__parameters": overrides}})
    (output / "issue.md").write_text(ISSUE_TEMPLATE, encoding="utf-8")
    if identity.get("verified") is not True:
        print("WARNING: runtime build identity is UNVERIFIED; do not treat the "
              "checkout HEAD as proof of the installed binary's version.", file=sys.stderr)
    return output


def candidates(root: Path):
    for name in ROOT_FILES:
        path = root / name
        if path.exists() or path.is_symlink():
            yield path
    ros_logs = root / "ros_log"
    if ros_logs.is_symlink():
        yield ros_logs
    elif ros_logs.is_dir():
        for directory, dirs, files in os.walk(ros_logs, followlinks=False):
            dirs.sort()
            for name in list(dirs):
                child = Path(directory) / name
                if child.is_symlink():
                    yield child
                    dirs.remove(name)
            for name in sorted(files):
                if name.endswith(".log"):
                    yield Path(directory) / name


def pack(args: argparse.Namespace) -> Path:
    root = args.run_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("run-dir must be a directory")
    output = args.output.absolute()
    if root == output.resolve() or root in output.resolve().parents:
        raise ValueError("output ZIP must be outside run-dir")
    per_file = args.max_file_mib * MIB
    remaining = args.max_total_mib * MIB
    entries = []
    report = {
        "packed_at": now(), "files": entries,
        "missing": [name for name in ("run.json", "build_identity.json",
                                       "config.yaml", "n3mapping.log")
                    if not (root / name).is_file()],
        "privacy": "Review before sharing. No automatic redaction or upload. "
                   "Only allowlisted text files are selected; maps/bags are excluded.",
        "limits": "Payload byte limits, before ZIP compression. Large logs keep "
                  "their first and last bytes. JSONL excerpts can have partial lines.",
    }
    # Exclusive creation protects an existing archive, including symlink targets.
    with output.open("xb") as stream:
        try:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=1) as archive:
                for path in candidates(root):
                    relative = path.relative_to(root).as_posix()
                    entry = {"path": relative}
                    entries.append(entry)
                    if (path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
                            or root not in path.resolve().parents):
                        entry["skipped"] = "not a regular in-directory file"
                        continue
                    size = path.stat().st_size
                    entry["source_bytes"] = size
                    if remaining <= 0 or len(entries) > 256:
                        entry["skipped"] = "bundle limit reached"
                        continue
                    limit = min(per_file, remaining)
                    # Keep structured metadata intact; never silently clip config.
                    if size > limit and path.suffix not in (".log", ".jsonl"):
                        entry["skipped"] = "metadata exceeds remaining byte limit"
                        continue
                    with path.open("rb") as source:
                        if size > limit:
                            head = limit // 2
                            payload = source.read(head)
                            source.seek(size - (limit - head))
                            payload += source.read(limit - head)
                            entry["retained_ranges"] = [[0, head], [size - (limit - head), size]]
                        else:
                            payload = source.read(size)
                    # Never present clipped JSONL as a complete analyzer input.
                    archive_name = "logs/" + relative + (".excerpt" if size > limit else "")
                    entry["archive_path"] = archive_name
                    entry["included_bytes"] = len(payload)
                    entry["truncated"] = size > limit
                    entry["changed_during_pack"] = path.stat().st_size != size
                    remaining -= len(payload)
                    archive.writestr(archive_name, payload)
                report["incomplete"] = bool(report["missing"]) or any(
                    item.get("skipped") or item.get("truncated") or
                    item.get("changed_during_pack") for item in entries)
                archive.writestr("bundle_report.json", json.dumps(
                    report, indent=2, ensure_ascii=False) + "\n")
        except BaseException:
            output.unlink()  # Only the exclusively created, incomplete ZIP.
            raise
    if report["incomplete"]:
        print("WARNING: bundle is incomplete; inspect bundle_report.json. "
              "Original logs were not changed.", file=sys.stderr)
    return output


def positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="snapshot a manual ROS 2 test; never starts nodes")
    prep.add_argument("--run-dir", required=True, type=Path)
    prep.add_argument("--config", required=True, type=Path)
    prep.add_argument("--mode", required=True, choices=("mapping", "localization", "map_extension"))
    prep.add_argument("--node-executable", type=Path)
    prep.add_argument("--diagnostics", action="store_true",
                      help="enable existing JSONL diagnostics for a bounded reproduction")
    bundle = commands.add_parser("pack", help="export local text logs; does not upload")
    bundle.add_argument("--run-dir", required=True, type=Path)
    bundle.add_argument("--output", required=True, type=Path)
    bundle.add_argument("--max-file-mib", type=positive, default=8)
    bundle.add_argument("--max-total-mib", type=positive, default=64)
    args = parser.parse_args()
    try:
        path = prepare(args) if args.command == "prepare" else pack(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
