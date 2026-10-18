#!/usr/bin/env python3
"""Score a query with simulation truth using one fixed mapping-session alignment.

The map origin is calibrated once from the mapping run's first selected output
(n3m by default).
Query estimates never participate in alignment, so wrong-location/heading errors
cannot be fitted away. Truth is base pose composed with the MID360 IMU extrinsic
because this workspace's FAST_LIO publishes its IMU/body pose and body cloud.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_run(directory):
    streams = {}
    with (directory / "poses.csv").open() as stream:
        for row in csv.DictReader(stream):
            streams.setdefault(row["topic"], []).append(
                [int(row["stamp_ns"])] + [float(row[k]) for k in
                                          ("tx", "ty", "tz", "qx", "qy", "qz", "qw")])
    streams = {name: np.array(sorted(rows, key=lambda row: row[0]))
               for name, rows in streams.items()}
    extrinsics = json.loads((directory / "extrinsics.json").read_text())
    imu = extrinsics["imu_frame"]
    if imu["parent"] != "base_link":
        raise ValueError("Expected a base-relative MID360 IMU transform")
    return streams, imu


def truth_at(streams, imu, stamps):
    truth = streams["/ground_truth/odom"]
    times = truth[:, 0]/1e9
    if not np.all(np.diff(times) > 0):
        raise ValueError("Truth timestamps must strictly increase")
    rotations = Slerp(times, Rotation.from_quat(truth[:, 4:]))(stamps/1e9)
    positions = np.column_stack([np.interp(stamps/1e9, times, truth[:, i]) for i in (1, 2, 3)])
    positions += rotations.apply(np.array(imu["position"]))
    rotations = rotations * Rotation.from_quat(imu["quaternion_xyzw"])
    return positions, rotations


def comparable(streams, topic="/n3mapping/odometry"):
    estimates = streams.get(topic)
    if estimates is None:
        raise ValueError("No scored poses on "+topic)
    truth = streams["/ground_truth/odom"]
    valid = estimates[(estimates[:, 0] >= truth[0, 0]) & (estimates[:, 0] <= truth[-1, 0])]
    if not len(valid):
        raise ValueError("No overlapping estimate/truth timestamps")
    return valid


def score(mapping_dir, query_dir, topic="/n3mapping/odometry", probe_csv=None):
    mapping, mapping_imu = read_run(mapping_dir)
    anchor = comparable(mapping, topic)[:1]
    anchor_p, anchor_r = truth_at(mapping, mapping_imu, anchor[:, 0])
    world_from_map_r = anchor_r[0] * Rotation.from_quat(anchor[0, 4:]).inv()
    world_from_map_p = anchor_p[0] - world_from_map_r.apply(anchor[0, 1:4])
    query, query_imu = read_run(query_dir)
    if probe_csv:
        rows = []
        with probe_csv.open() as stream:
            for row in csv.DictReader(stream):
                if row["live"] != "1":
                    continue
                if row["success"] != "1":
                    raise ValueError("Probe has coasted outputs; backend pose alone cannot score them")
                pose = np.array([float(row[f"pose{i}"]) for i in range(16)]).reshape(4, 4).T
                rows.append([int(row["stamp_ns"]), *pose[:3, 3],
                             *Rotation.from_matrix(pose[:3, :3]).as_quat()])
        if not rows:
            raise ValueError("Probe emitted no poses; absence of output is not an accuracy score")
        query[topic] = np.array(rows)
    estimates = comparable(query, topic)
    truth_p, truth_r = truth_at(query, query_imu, estimates[:, 0])
    prediction_p = world_from_map_r.apply(estimates[:, 1:4]) + world_from_map_p
    prediction_r = world_from_map_r * Rotation.from_quat(estimates[:, 4:])
    position_error = np.linalg.norm(prediction_p-truth_p, axis=1)
    rotation_error = (truth_r.inv()*prediction_r).magnitude()*180/np.pi

    def stats(values):
        return dict(rmse=float(np.sqrt(np.mean(values**2))), median=float(np.median(values)),
                    p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))

    return dict(mapping=str(mapping_dir.resolve()), query=str(query_dir.resolve()), topic=topic,
        probe_csv=str(probe_csv.resolve()) if probe_csv else None,
        alignment=dict(source="first mapping pose on "+topic+" only; no query fitting",
                       stamp_ns=int(anchor[0, 0]), translation=world_from_map_p.tolist(),
                       quaternion_xyzw=world_from_map_r.as_quat().tolist()),
        evaluated_poses=len(estimates), first_output_sim_s=float(estimates[0, 0]/1e9),
        position_error_m=stats(position_error), rotation_error_deg=stats(rotation_error),
        limitation="Simulation truth and ideal sensors; includes mapping/frontend errors; no hardware or CPU qualification")


def mapping_quality(directory):
    """Reject gross drift before using an acquired map for localization tests."""
    result = dict(passed=False, limits=dict(maximum_position_error_m=1., maximum_rotation_error_deg=5.),
                  scope="Gross mapping drift guard; not a localization acceptance threshold")
    try:
        for topic in ("/Odometry", "/n3mapping/odometry"):
            result[topic] = score(directory, directory, topic)
        result["passed"] = all(
            result[topic]["position_error_m"]["maximum"] <= result["limits"]["maximum_position_error_m"] and
            result[topic]["rotation_error_deg"]["maximum"] <= result["limits"]["maximum_rotation_error_deg"]
            for topic in ("/Odometry", "/n3mapping/odometry"))
    except (ValueError, KeyError) as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/n3mapping/odometry",
                        help="Pose stream to score and anchor; /Odometry audits the frontend")
    parser.add_argument("--probe-csv", type=Path,
                        help="Score successful replay_probe backend outputs against the query truth")
    args = parser.parse_args()
    result = score(args.mapping, args.query, args.topic, args.probe_csv)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
