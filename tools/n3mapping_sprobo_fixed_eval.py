#!/usr/bin/env python3
"""Replay every fixed-location acquisition serially with a frozen core/probe.

Run variants in the same CPU/thread configuration. Process CPU includes all
workers; first-lock wall time starts at the first input after map loading.
The independent mapping-session anchor is shared by all query truth scores.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from n3mapping_sprobo_score import score


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=Path, required=True,
                        help="Directory with matching libn3mapping_core.so and replay_probe; repeat for A/B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpus", default="0,2,4,6")
    parser.add_argument("--rate", type=float, default=1.)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.)
    parser.add_argument("--case", action="append",
                        help="Replay named positions only; omit for the entire frozen plan")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.dataset / "plan.json").read_text())
    if args.case:
        unknown = set(args.case) - {position["id"] for position in plan["positions"]}
        if unknown:
            parser.error("Unknown fixed positions: " + ", ".join(sorted(unknown)))
        plan["positions"] = [position for position in plan["positions"] if position["id"] in args.case]
    map_file = args.mapping / "map/n3map.pbstream"
    atlas = map_file.with_name(map_file.name + ".localization_atlas.pb")
    report = dict(dataset=str(args.dataset), mapping=str(args.mapping),
                  map_sha256=sha256(map_file), atlas_sha256=sha256(atlas),
                  plan_sha256=sha256(args.dataset / "plan.json"),
                  cpus=args.cpus, threads=args.threads, rate=args.rate,
                  max_frames=args.max_frames, variants={}, results=[])
    report["selected_positions"] = [position["id"] for position in plan["positions"]]
    for variant in args.variant:
        report["variants"][variant.name] = {name: sha256(variant / name)
            for name in ("libn3mapping_core.so", "replay_probe", "replay_probe.cpp")}
    for index, position in enumerate(plan["positions"]):
        query = args.dataset / position["id"]
        acquisition = json.loads((query / "summary.json").read_text())
        if not acquisition["passed"]:
            raise ValueError("Unqualified acquisition: " + str(query))
        # Alternate A/B order across positions to avoid a fixed ordering bias.
        for variant in (args.variant if index % 2 == 0 else list(reversed(args.variant))):
            tag = position["id"] + "_" + variant.name
            destination = args.output / (tag + ".csv")
            env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", N3M_BENCH_THREADS=str(args.threads))
            # Detailed diagnostics do additional searches and must not leak
            # from a caller's environment into a performance comparison.
            env.pop("N3M_BENCH_DEBUG", None)
            env["LD_LIBRARY_PATH"] = ":".join(map(str, [variant,
                args.workspace / "install/gtsam/lib", args.workspace / "install/small_gicp/lib"]))
            command = ["taskset", "-c", args.cpus, str(variant / "replay_probe"),
                       str(map_file), str(atlas), str(query / "manifest/frames.csv"),
                       str(destination), str(args.rate), str(args.max_frames)]
            print("Evaluating " + tag, flush=True)
            with (args.output / (tag + ".log")).open("x") as log:
                try:
                    process = subprocess.run(command, env=env, stdout=log, stderr=log,
                                             timeout=args.timeout)
                    status = process.returncode
                except subprocess.TimeoutExpired:
                    status = "timeout"
            rows = list(csv.DictReader(destination.open())) if destination.exists() else []
            lock = next((row for row in rows if row["lock"] == "1"), None)
            result = dict(id=position["id"], variant=variant.name, returncode=status,
                          frames=len(rows), outputs=sum(row["live"] == "1" for row in rows),
                          successful_frames=sum(row["success"] == "1" for row in rows),
                          lock_frames=[int(row["frame"]) for row in rows if row["lock"] == "1"],
                          manifest_sha256=sha256(query / "manifest/frames.csv"),
                          csv=str(destination), first_lock_frame=int(lock["frame"]) if lock else None,
                          first_lock_wall_s=float(lock["cumulative_wall_ms"]) / 1000 if lock else None,
                          first_lock_cpu_s=float(lock["cumulative_cpu_ms"]) / 1000 if lock else None)
            if rows:
                input_s = (int(rows[-1]["stamp_ns"]) - int(rows[0]["stamp_ns"])) / 1e9
                cpu_s = float(rows[-1]["cumulative_cpu_ms"]) / 1000
                result.update(cpu_s=cpu_s, wall_s=float(rows[-1]["cumulative_wall_ms"]) / 1000,
                              input_s=input_s, cpu_seconds_per_input_second=cpu_s / input_s if input_s else None,
                              alignments=sum(int(row["alignments"]) for row in rows),
                              peak_rss_mib=max(int(row["rss_kib"]) for row in rows) / 1024)
            if result["outputs"]:
                try:
                    result["truth"] = score(args.mapping, query, probe_csv=destination)
                except ValueError as error:
                    # A coasted pose needs the actual output transform; never
                    # fit or silently score the backend pose as that output.
                    result["truth_score_error"] = str(error)
            ready = re.search(r"MAP_READY wall_ms=([\d.e+-]+)", (args.output / (tag + ".log")).read_text())
            result["map_load_wall_s"] = float(ready[1]) / 1000 if ready else None
            report["results"].append(result)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key != "truth"}), flush=True)


if __name__ == "__main__":
    main()
