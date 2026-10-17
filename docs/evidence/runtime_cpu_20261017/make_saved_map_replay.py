"""Make a cost-only 10 Hz replay from retained scans and a map's dense path.

These scans were used to build the map. This is not an independent localization
accuracy dataset. Interpolated views retain the nearest saved scan's points;
they do not simulate visibility or sensor noise.
Requires n3map_pb2 generated from this repository's proto/n3map.proto.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import n3map_pb2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("map", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--offset", type=float, default=0)
parser.add_argument("--frames", type=int, default=1800)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
data = args.map.read_bytes()
map_ = n3map_pb2.N3Map()
map_.ParseFromString(data)
keyframes = sorted(map_.keyframes, key=lambda k: k.timestamp)
dense = sorted(map_.dense_optimized_trajectory, key=lambda p: p.timestamp)
assert keyframes and len(dense) > 1

def pose(p):
    return np.array([p.tx, p.ty, p.tz]), Rotation.from_quat([p.qx, p.qy, p.qz, p.qw])

times = np.array([p.timestamp for p in dense])
translations, rotations = zip(*(pose(p.pose_world_lidar) for p in dense))
translations = np.asarray(translations)
orientation = Slerp(times, Rotation.concatenate(rotations))
start = max(times[0], keyframes[0].timestamp) + args.offset
query_times = start + np.arange(args.frames) / 10
assert query_times[-1] <= times[-1]
query_positions = np.column_stack([np.interp(query_times, times, translations[:, i]) for i in range(3)])
query_rotations = orientation(query_times)
k_times = np.array([k.timestamp for k in keyframes])
k_poses = [pose(k.pose_optimized) for k in keyframes]
world_scans = [r.apply(np.array(k.cloud.points, dtype="<f4").reshape(-1, 4)[:, :3]) + t
               for k, (t, r) in zip(keyframes, k_poses)]
intensities = [np.array(k.cloud.points, dtype="<f4").reshape(-1, 4)[:, 3] for k in keyframes]
fake_rotation = Rotation.from_euler("z", 90, degrees=True)
fake_translation = np.array([20., -10., 1.])
digest = hashlib.sha256()
counts = []
with (args.output / "frames.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["episode_id", "frame_index", "stamp_ns", "pcd_path", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    for i, (timestamp, position, rotation) in enumerate(zip(query_times, query_positions, query_rotations)):
        nearest = int(np.argmin(abs(k_times - timestamp)))
        xyz = rotation.inv().apply(world_scans[nearest] - position)
        points = np.column_stack([xyz, intensities[nearest]]).astype("<f4")
        counts.append(len(points))
        filename = f"frame_{i:06d}.pcd"
        header = ("VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
                  f"COUNT 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n").encode()
        encoded = header + points.tobytes()
        (args.output / filename).write_bytes(encoded)
        digest.update(encoded)
        translation = fake_rotation.inv().apply(position - fake_translation)
        quaternion = (fake_rotation.inv() * rotation).as_quat()
        writer.writerow(["saved_map_cost_only", i, 1_000_000_000 + i * 100_000_000, filename,
                         *[format(x, ".17g") for x in translation],
                         *[format(x, ".17g") for x in quaternion]])
metadata = {"kind": "same_map_retained_scan_cost_probe", "independent_accuracy_evidence": False,
            "map_sha256": hashlib.sha256(data).hexdigest(), "frame_count": args.frames,
            "rate_hz": 10, "start_offset_s": args.offset, "pcd_sequence_sha256": digest.hexdigest(),
            "frames_csv_sha256": hashlib.sha256((args.output / "frames.csv").read_bytes()).hexdigest(),
            "points_min_median_max": np.percentile(counts, [0, 50, 100]).tolist()}
(args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata))
