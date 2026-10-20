#!/usr/bin/env python3
"""Prepare map × deskew diagnostic fixtures, separate from production replay.

Requires independently verified native 200 Hz CSVs, frozen LIO manifests, and
the mapping session's fixed alignment. Does not fit any query output to truth.
Generated truth seeds and reference geometry must never qualify localization.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from extract_ros1_relocalization_manifest import sha256_file, write_binary_pcd
from sprobo_accuracy_geometry import (
    NativeTrajectory, RawScans, authored_surface_cloud,
    pose_matrix, read_binary_pcd, transform)


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def odometry(row):
    return pose_matrix(
        np.array([float(row[k]) for k in ("tx", "ty", "tz")]),
        Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]))


def write_cloud(path, xyz):
    write_binary_pcd(path, np.column_stack([xyz, np.full(len(xyz), 100)]).astype("<f4"))


def prepare_window(rows, end, manifest, native, raw):
    last_pose = odometry(rows[end])
    original, corrected, errors = [], [], []
    for row in rows[end-4:end+1]:
        stamp = int(row["stamp_ns"])
        points, times, intensity = raw.preprocessed(stamp)
        exact = native.deskew(points, times, stamp)
        cloud = read_binary_pcd(manifest.parent / row["pcd_path"])
        if len(cloud) != len(exact) or not np.array_equal(cloud[:, 3], intensity):
            raise ValueError("Raw scan/preprocessing does not match the frozen LIO cloud")
        errors.extend(np.linalg.norm(cloud[:, :3] - exact, axis=1))
        relative = np.linalg.inv(last_pose) @ odometry(row)
        original.append(transform(cloud[:, :3], relative))
        corrected.append(transform(exact, relative))
    return np.concatenate(original), np.concatenate(corrected), np.array(errors)


def select_cases(args):
    cases = []
    plan = json.loads((args.fixed_dataset / "plan.json").read_text())
    for position in plan["positions"]:
        run = args.fixed_dataset / position["id"]
        cases.append((position["id"], run, run / "manifest/frames.csv", [4]))
    route_manifest = args.route / "manifest/frames.csv"
    count = len(read_rows(route_manifest))
    cases.append(("independent_route", args.route, route_manifest, list(range(4, count, 50))))
    if args.mapping_keyframes:
        rows = read_rows(args.mapping_manifest)
        lookup = {int(row["stamp_ns"]): i for i, row in enumerate(rows)}
        indices = [lookup[int(round(float(k["timestamp"])*1e9))]
                   for k in read_rows(args.mapping_keyframes)[::20]]
        if min(indices) < 4:
            raise ValueError("Mapping selection needs five preceding input scans")
        cases.append(("coverage_v2", args.mapping, args.mapping_manifest, indices))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mapping", "fixed-dataset", "route", "native-root", "alignment-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mapping-manifest", type=Path)
    parser.add_argument("--mapping-keyframes", type=Path)
    parser.add_argument("--livox-msg-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "livox_ros_driver2/msg")
    args = parser.parse_args()
    if bool(args.mapping_manifest) != bool(args.mapping_keyframes):
        parser.error("Supply both mapping manifest and mapping keyframes, or neither")
    cases = select_cases(args)
    args.output.mkdir(parents=True, exist_ok=False)
    alignment = json.loads(args.alignment_json.read_text())["alignment"]
    world_rotation = Rotation.from_quat(alignment["quaternion_xyzw"])
    world_translation = np.array(alignment["translation"])
    scene = json.loads((args.mapping / "session.json").read_text())["scene"]
    reference = authored_surface_cloud(scene["objects"], world_rotation, world_translation)
    write_cloud(args.output / "reference.pcd", reference)
    jobs, results, inputs = [], [], {}
    for path in (args.alignment_json, args.mapping / "session.json",
                 args.fixed_dataset / "plan.json"):
        inputs[str(path.resolve())] = sha256_file(path)
    for name, run, manifest, indices in cases:
        native_file = args.native_root / name / "poses.csv"
        extrinsics_file = run / "extrinsics.json"
        native = NativeTrajectory(native_file, json.loads(extrinsics_file.read_text()))
        for path in (native_file, extrinsics_file, manifest):
            inputs[str(path.resolve())] = sha256_file(path)
        rows = read_rows(manifest)
        raw = RawScans(run / "bag", args.livox_msg_dir)
        try:
            for end in indices:
                label = f"{name}_{end:05}"
                target = args.output / label
                target.mkdir()
                stamp = int(rows[end]["stamp_ns"])
                position, rotation = native.at(stamp, "imu_frame")
                truth = pose_matrix(world_rotation.inv().apply(position[0] - world_translation),
                                    world_rotation.inv() * rotation[0])
                original, corrected, error = prepare_window(rows, end, manifest, native, raw)
                write_cloud(target / "fastlio.pcd", original)
                write_cloud(target / "native_deskew.pcd", corrected)
                jobs.append(dict(id=label, path=str(target.resolve()), truth=truth.ravel().tolist()))
                results.append(dict(id=label, run=name, frame=end, stamp_ns=stamp,
                                    points=len(original), point_deskew_delta_m=dict(
                                        rmse=float(np.sqrt(np.mean(error*error))),
                                        p95=float(np.percentile(error, 95)),
                                        maximum=float(error.max()))))
                print(label, flush=True)
        finally:
            raw.close()
    audit = dict(scope="Offline diagnosis only; truth-seeded ICP is not localization",
                 alignment=alignment, reference_points=len(reference), inputs=inputs,
                 preprocessing=dict(point_filter_num=2, blind_m=.2, scan_lines=4),
                 native_sample_period_s=.005,
                 aggregation="Same five scans and same LIO relative poses in both variants",
                 selection="All 14 first-lock windows; every 50th route input; "
                           "optional every 20th mapping keyframe",
                 results=results)
    (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
