#!/usr/bin/env python3
"""Score a query with simulation truth using one fixed mapping-session alignment.

The map origin is calibrated once from the mapping run's first n3m output.
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


def comparable(streams):
    estimates = streams.get("/n3mapping/odometry")
    if estimates is None:
        raise ValueError("No n3m output; localization did not produce a scored pose")
    truth = streams["/ground_truth/odom"]
    valid = estimates[(estimates[:, 0] >= truth[0, 0]) & (estimates[:, 0] <= truth[-1, 0])]
    if not len(valid):
        raise ValueError("No overlapping estimate/truth timestamps")
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping, mapping_imu = read_run(args.mapping)
    anchor = comparable(mapping)[:1]
    anchor_p, anchor_r = truth_at(mapping, mapping_imu, anchor[:, 0])
    world_from_map_r = anchor_r[0] * Rotation.from_quat(anchor[0, 4:]).inv()
    world_from_map_p = anchor_p[0] - world_from_map_r.apply(anchor[0, 1:4])
    query, query_imu = read_run(args.query)
    estimates = comparable(query)
    truth_p, truth_r = truth_at(query, query_imu, estimates[:, 0])
    prediction_p = world_from_map_r.apply(estimates[:, 1:4]) + world_from_map_p
    prediction_r = world_from_map_r * Rotation.from_quat(estimates[:, 4:])
    position_error = np.linalg.norm(prediction_p-truth_p, axis=1)
    rotation_error = (truth_r.inv()*prediction_r).magnitude()*180/np.pi

    def stats(values):
        return dict(rmse=float(np.sqrt(np.mean(values**2))), median=float(np.median(values)),
                    p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))

    result = dict(mapping=str(args.mapping.resolve()), query=str(args.query.resolve()),
        alignment=dict(source="first mapping n3m pose only; no query fitting",
                       stamp_ns=int(anchor[0, 0]), translation=world_from_map_p.tolist(),
                       quaternion_xyzw=world_from_map_r.as_quat().tolist()),
        evaluated_poses=len(estimates), first_output_sim_s=float(estimates[0, 0]/1e9),
        position_error_m=stats(position_error), rotation_error_deg=stats(rotation_error),
        limitation="Simulation truth, one short route, ideal sensors; includes mapping/frontend errors")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
