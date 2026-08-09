#!/usr/bin/env python3
"""Dumps keyframe poses and edge counts from an n3map pbstream.

Writes <out>_optimized.csv and <out>_odom.csv, the back end's poses and the
front end's, so the same consistency check can be run on each. Comparing the
two is what showed the back end to be a no-op on the 0723 map: it moved poses
by 0.38 mm at most while the front end had accumulated 2.49 m of height error.

With ``-`` as the output prefix, only the map/metadata summary is printed and
no CSV files are created.

Requires the generated protobuf module:

    protoc -I proto --python_out=<dir> proto/n3map.proto
    python3 tools/dump_map_poses.py <map.pbstream> <out_prefix|-> [<dir>]
"""
import csv
import sys
from pathlib import Path

if len(sys.argv) < 3:
    raise SystemExit(
        "usage: dump_map_poses.py <map.pbstream> <out_prefix|-> [<proto_dir>]"
    )

pb_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/pb"
sys.path.insert(0, pb_dir)
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

src = Path(sys.argv[1])
out = None if sys.argv[2] == "-" else sys.argv[2]

m = n3map_pb2.N3Map()
m.ParseFromString(src.read_bytes())


def measurement_text(edge):
    pose = edge.measurement
    return ("%d->%d t=(%.9g,%.9g,%.9g) q=(%.9g,%.9g,%.9g,%.9g)" %
            (edge.from_id, edge.to_id, pose.tx, pose.ty, pose.tz,
             pose.qx, pose.qy, pose.qz, pose.qw))


odometry = sum(1 for e in m.edges
               if e.type == n3map_pb2.EdgeProto.ODOMETRY)
loop = sum(1 for e in m.edges if e.type == n3map_pb2.EdgeProto.LOOP)
session_anchor = sum(1 for e in m.edges
                     if e.type == n3map_pb2.EdgeProto.SESSION_ANCHOR)
known_edge_count = odometry + loop + session_anchor
keyframe_ids = {keyframe.id for keyframe in m.keyframes}
duplicate_keyframes = len(m.keyframes) - len(keyframe_ids)
dangling_edges = sum(1 for edge in m.edges
                     if edge.from_id not in keyframe_ids or
                     edge.to_id not in keyframe_ids)
metadata_match = (
    m.metadata.num_keyframes == len(m.keyframes) and
    m.metadata.num_odometry_edges == odometry and
    m.metadata.num_loop_edges == loop and
    m.metadata.num_session_anchor_edges == session_anchor
)

print("version=%s keyframes=%d edges=%d odometry=%d loop=%d "
      "session_anchor=%d unknown=%d dense=%d" %
      (m.metadata.version, len(m.keyframes), len(m.edges), odometry, loop,
       session_anchor, len(m.edges) - known_edge_count,
       len(m.dense_optimized_trajectory)))
print("metadata keyframes=%d odometry=%d loop=%d session_anchor=%d "
      "match=%s duplicate_keyframes=%d dangling_edges=%d" %
      (m.metadata.num_keyframes, m.metadata.num_odometry_edges,
       m.metadata.num_loop_edges, m.metadata.num_session_anchor_edges,
       str(metadata_match).lower(), duplicate_keyframes, dangling_edges))
anchor_pairs = [(edge.from_id, edge.to_id) for edge in m.edges
                if edge.type == n3map_pb2.EdgeProto.SESSION_ANCHOR]
print("session_anchors=%s" %
      (",".join("%d->%d" % pair for pair in anchor_pairs)
       if anchor_pairs else "none"))
anchor_edges = [edge for edge in m.edges
                if edge.type == n3map_pb2.EdgeProto.SESSION_ANCHOR]
print("session_anchor_measurements=%s" %
      (";".join(measurement_text(edge) for edge in anchor_edges)
       if anchor_edges else "none"))
odometry_pairs = [(edge.from_id, edge.to_id) for edge in m.edges
                  if edge.type == n3map_pb2.EdgeProto.ODOMETRY]
print("odometry_tail=%s" %
      (",".join("%d->%d" % pair for pair in odometry_pairs[-3:])
       if odometry_pairs else "none"))
odometry_edges = [edge for edge in m.edges
                  if edge.type == n3map_pb2.EdgeProto.ODOMETRY]
print("odometry_latest_measurement=%s" %
      (measurement_text(odometry_edges[-1]) if odometry_edges else "none"))

if out is not None:
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
