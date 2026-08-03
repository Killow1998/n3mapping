#!/usr/bin/env python3
"""Is the ghost at the true place smaller on the better map?

The chain in the worklog predicts it should be: the wrong lock is blamed on a
half-metre vertical offset between two passes over the same spot, so a map whose
worst-case revisit |dz| is four times smaller ought to hold that spot once rather
than twice. Passes are identified by timestamp, which does not depend on either
map's frame or keyframe numbering.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

# The pair of visits that matters: the query's true position was mapped at
# t ~ ...630 and again at t ~ ...064, 434 s apart.
PASS_A = (1784778620.0, 1784778680.0)
PASS_B = (1784779050.0, 1784779080.0)
XY_TOL = 2.5


def load(path):
    m = n3map_pb2.N3Map()
    m.ParseFromString(Path(path).read_bytes())
    return sorted(m.keyframes, key=lambda k: k.timestamp)


def report(label, path):
    kfs = load(path)
    a = [k for k in kfs if PASS_A[0] <= k.timestamp <= PASS_A[1]]
    b = [k for k in kfs if PASS_B[0] <= k.timestamp <= PASS_B[1]]
    print("\n=== %s" % label)
    print("    keyframes %d   pass A %d   pass B %d" % (len(kfs), len(a), len(b)))
    if not a or not b:
        print("    one of the passes is missing; cannot compare")
        return
    pairs = []
    for ka in a:
        for kb in b:
            d = math.dist((ka.pose_optimized.tx, ka.pose_optimized.ty),
                          (kb.pose_optimized.tx, kb.pose_optimized.ty))
            if d <= XY_TOL:
                pairs.append((d, ka.id, kb.id,
                              kb.pose_optimized.tz - ka.pose_optimized.tz))
    if not pairs:
        print("    no pair within %.1f m -- the two passes do not overlap here"
              % XY_TOL)
        return
    pairs.sort()
    for d, ia, ib, dz in pairs[:6]:
        print("    kf %-4d <-> kf %-4d   xy %.3f m   dz %+.3f m" % (ia, ib, d, dz))
    dzs = [abs(p[3]) for p in pairs]
    dzs.sort()
    print("    pairs %d   |dz| median %.3f   max %.3f"
          % (len(dzs), dzs[len(dzs) // 2], dzs[-1]))


C = "/home/user/ros_ws/n3mapping_v1_closeout"
report("product map (used by today20, wrong lock)", C + "/atlas_rebuild/n3map.pbstream")
report("s16 static-start-guard map", C + "/s16_staticguard/map/n3map.pbstream")
