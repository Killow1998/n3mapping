#!/usr/bin/env python3
"""Dumps keyframe poses and edge counts from an n3map pbstream.

Writes <out>_optimized.csv and <out>_odom.csv, the back end's poses and the
front end's, so the same consistency check can be run on each. Comparing the
two is what showed the back end to be a no-op on the 0723 map: it moved poses
by 0.38 mm at most while the front end had accumulated 2.49 m of height error.

Requires the generated protobuf module:

    protoc -I proto --python_out=<dir> proto/n3map.proto
    python3 tools/dump_map_poses.py <map.pbstream> <out_prefix> [<dir>]
"""
import csv
import sys
from pathlib import Path

pb_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/pb"
sys.path.insert(0, pb_dir)
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

src = Path(sys.argv[1])
out = sys.argv[2]

m = n3map_pb2.N3Map()
m.ParseFromString(src.read_bytes())

loop = sum(1 for e in m.edges if e.type == n3map_pb2.EdgeProto.LOOP)
print("keyframes=%d  edges=%d  (odometry=%d, loop=%d)  dense=%d" %
      (len(m.keyframes), len(m.edges), len(m.edges) - loop, loop,
       len(m.dense_optimized_trajectory)))

for name in ("optimized", "odom"):
    path = out + "_" + name + ".csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for k in m.keyframes:
            p = k.pose_optimized if name == "optimized" else k.pose_odom
            w.writerow([int(k.timestamp * 1e9),
                        p.tx, p.ty, p.tz, p.qx, p.qy, p.qz, p.qw])
    print("wrote", path)
