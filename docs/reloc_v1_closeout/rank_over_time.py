#!/usr/bin/env python3
"""How often did the per-frame descriptor put the true place first?

The lock is decided on accumulated hypothesis evidence, not on this frame's
candidate list. If the candidate list was right most of the time then the
accumulator is discarding the one channel that knew the answer.
"""
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

DBG = "/home/user/ros_ws/n3mapping_v1_closeout/dbg_115034"
MAP = "/home/user/ros_ws/n3mapping_v1_closeout/atlas_rebuild/n3map.pbstream"
NEAR_M = 3.0     # a candidate this close to the truth counts as the right place

m = n3map_pb2.N3Map()
m.ParseFromString(Path(MAP).read_bytes())
kf_xy = {k.id: (k.pose_optimized.tx, k.pose_optimized.ty) for k in m.keyframes}

# The robot is moving, so the truth moves too; take it per frame from the dense
# trajectory via the query stamps in the frame status file.
import csv
import bisect
DENSE = ("/home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_product_v1/"
         "20260725/candidate_dd25f86/f7_0723_map/dense_trajectory.csv")
rows = []
with open(DENSE) as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["stamp_ns"]), float(r["tx"]), float(r["ty"])))
rows.sort()
stamps = [s for s, _, _ in rows]


def truth_at(ns):
    i = bisect.bisect_left(stamps, ns)
    best = None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(rows):
            d = abs(stamps[j] - ns)
            if best is None or d < best[0]:
                best = (d, rows[j][1], rows[j][2])
    return (best[1], best[2]) if best and best[0] < 10_000_000 else None


status = list(csv.DictReader(open(Path(DBG, "frame_status.csv"))))
recs = [json.loads(l) for l in open(Path(DBG, "relocalization_debug.jsonl"))]

rank_hist = collections.Counter()
n = 0
for i, r in enumerate(recs):
    cands = r.get("top_candidates") or []
    if not cands or i >= len(status):
        continue
    t = truth_at(int(status[i]["stamp_ns"]))
    if t is None:
        continue
    n += 1
    hit = None
    for rank, c in enumerate(cands):
        xy = kf_xy.get(c["match_id"])
        if xy and math.dist(xy, t) <= NEAR_M:
            hit = rank + 1
            break
    rank_hist["miss" if hit is None else hit] += 1

print("frames with a candidate list: %d" % n)
print("rank at which the true place (within %.1f m) first appears:" % NEAR_M)
for k in sorted(rank_hist, key=lambda x: (x == "miss", x)):
    print("   %-5s %4d   %5.1f%%" % (k, rank_hist[k], 100.0 * rank_hist[k] / n))
top1 = rank_hist[1]
print("\ntrue place ranked first: %d/%d = %.0f%%" % (top1, n, 100.0 * top1 / n))
inlist = n - rank_hist["miss"]
print("true place anywhere in the list: %d/%d = %.0f%%" % (inlist, n, 100.0 * inlist / n))
