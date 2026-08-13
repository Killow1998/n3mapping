#!/usr/bin/env python3
"""Sample one Linux process's CPU/RSS for PERF-ME-01 evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


SCHEMA = "n3mapping_process_resource_v1"


def parse_proc_stat(text: str) -> dict[str, int]:
    close = text.rfind(")")
    open_ = text.find("(")
    if open_ < 0 or close <= open_:
        raise ValueError("malformed /proc stat comm field")
    pid_text = text[:open_].strip()
    fields = text[close + 1 :].split()
    if len(fields) <= 21:
        raise ValueError("truncated /proc stat")
    return {
        "pid": int(pid_text),
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
        "process_start_ticks": int(fields[19]),
        "rss_pages": int(fields[21]),
    }


def parse_status(text: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {"vm_hwm_mib": 0.0, "thread_count": 0}
    for raw in text.splitlines():
        if raw.startswith("VmHWM:"):
            result["vm_hwm_mib"] = float(raw.split()[1]) / 1024.0
        elif raw.startswith("Threads:"):
            result["thread_count"] = int(raw.split()[1])
    return result


def read_process(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    stat = parse_proc_stat((proc / "stat").read_text(encoding="utf-8"))
    status = parse_status((proc / "status").read_text(encoding="utf-8"))
    stat.update(status)
    return stat


def sample(pid: int, interval: float, output: Path, duration: float | None) -> int:
    ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    logical_cpus = os.cpu_count() or 1
    if ticks_per_second <= 0 or page_size <= 0:
        raise RuntimeError("invalid host clock/page configuration")

    try:
        previous = read_process(pid)
    except (FileNotFoundError, ProcessLookupError):
        print(f"process not found: {pid}", file=sys.stderr)
        return 2
    identity = (previous["pid"], previous["process_start_ticks"])
    previous_monotonic = time.monotonic()
    started = previous_monotonic
    sample_index = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as stream:
        try:
            while duration is None or time.monotonic() - started < duration:
                remaining = interval
                if duration is not None:
                    remaining = min(remaining, max(0.0, duration - (time.monotonic() - started)))
                if remaining <= 0.0:
                    break
                time.sleep(remaining)
                now_monotonic = time.monotonic()
                try:
                    current = read_process(pid)
                except (FileNotFoundError, ProcessLookupError):
                    break
                if (current["pid"], current["process_start_ticks"]) != identity:
                    print(f"process identity changed for pid {pid}", file=sys.stderr)
                    return 3
                elapsed = now_monotonic - previous_monotonic
                if elapsed <= 0.0:
                    continue
                tick_delta = (
                    current["utime_ticks"]
                    + current["stime_ticks"]
                    - previous["utime_ticks"]
                    - previous["stime_ticks"]
                )
                cpu_percent = max(
                    0.0,
                    100.0 * tick_delta / (ticks_per_second * elapsed),
                )
                sample_index += 1
                record = {
                    "schema": SCHEMA,
                    "sample_index": sample_index,
                    "processing_time": time.time(),
                    "monotonic_time": now_monotonic,
                    "pid": pid,
                    "process_start_ticks": identity[1],
                    "host_logical_cpus": logical_cpus,
                    "interval_s": elapsed,
                    # 100 means one fully occupied logical CPU; multi-threaded
                    # processes can legitimately exceed 100.
                    "process_cpu_percent": cpu_percent,
                    "rss_mib": current["rss_pages"] * page_size / (1024.0 * 1024.0),
                    "vm_hwm_mib": current["vm_hwm_mib"],
                    "thread_count": current["thread_count"],
                }
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                previous = current
                previous_monotonic = now_monotonic
        except KeyboardInterrupt:
            pass
    return 0 if sample_index > 0 else 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args(argv)
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if not math.isfinite(args.interval) or args.interval <= 0.0:
        parser.error("--interval must be finite and positive")
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0.0
    ):
        parser.error("--duration must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return sample(args.pid, args.interval, args.output, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
