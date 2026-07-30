#!/usr/bin/env python3
"""Asks whether the loop closures in a map agree with each other.

Two true loops must be mutually consistent: walk from keyframe a to b along one
loop's measurement, on to d along odometry, back along the other loop, and home
along odometry, and you should arrive where you started. A false loop has no
reason to close any such cycle, so it disagrees with almost everything. This is
the pairwise-consistency test, and it needs no ground truth and no prior -- only
the loop measurements and the odometry between them.

The point of measuring it: with the robust kernel off the graph moved poses by
1.2 m median and the map got worse, which is what applying wrong loops looks
like. If a large consistent set exists inside these loops, the discriminator is
missing rather than the loops; if no such set exists, the loops themselves are
mostly wrong.

    python3 loop_pcm.py <map.pbstream> [<pb_dir>]
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/tmp/pb")
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

TRANS_TOL = 0.60   # metres of cycle error still called consistent
ROT_TOL = 0.05     # radians, about 3 degrees


def q2R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def iso(p):
    T = np.eye(4)
    T[:3, :3] = q2R(p.qx, p.qy, p.qz, p.qw)
    T[:3, 3] = [p.tx, p.ty, p.tz]
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


m = n3map_pb2.N3Map()
m.ParseFromString(Path(sys.argv[1]).read_bytes())

# Odometry poses give the chain the loops are supposed to be consistent with.
odom = {k.id: iso(k.pose_odom) for k in m.keyframes}
loops = [e for e in m.edges if e.type == n3map_pb2.EdgeProto.LOOP]
loops = [e for e in loops if e.from_id in odom and e.to_id in odom]
print("闭环边 %d 条" % len(loops))
if len(loops) < 2:
    raise SystemExit("需要至少两条闭环边")

# Each loop asserts T[from -> to]. Odometry asserts a different one; the loop's
# own disagreement with odometry is the correction it proposes.
corr = []
for e in loops:
    T_meas = iso(e.measurement)
    T_odom = inv(odom[e.from_id]) @ odom[e.to_id]
    d = inv(T_odom) @ T_meas
    corr.append(np.linalg.norm(d[:3, 3]))
print("各闭环提出的修正量 (m): " + ", ".join("%.2f" % c for c in corr))
print()

n = len(loops)
A = np.zeros((n, n), dtype=bool)
detail = []
for i, j in itertools.combinations(range(n), 2):
    a, b = loops[i], loops[j]
    # cycle: a.from -> a.to (loop a) -> b.to (odom) -> b.from (loop b inverse)
    #        -> a.from (odom)
    T = (iso(a.measurement)
         @ (inv(odom[a.to_id]) @ odom[b.to_id])
         @ inv(iso(b.measurement))
         @ (inv(odom[b.from_id]) @ odom[a.from_id]))
    et = float(np.linalg.norm(T[:3, 3]))
    c = max(-1.0, min(1.0, (np.trace(T[:3, :3]) - 1.0) / 2.0))
    er = float(np.arccos(c))
    ok = et <= TRANS_TOL and er <= ROT_TOL
    A[i, j] = A[j, i] = ok
    detail.append((et, er, ok))

ets = np.array([d[0] for d in detail])
print("成对回路闭合误差 (%d 对): 中位 %.3f m  p10 %.3f m  最小 %.3f m"
      % (len(detail), np.median(ets), np.percentile(ets, 10), ets.min()))
print("判为互相一致的对数: %d / %d (容差 %.2f m / %.0f deg)"
      % (A.sum() // 2, len(detail), TRANS_TOL, np.degrees(ROT_TOL)))
print()

# Greedy maximum clique is enough to see whether a large agreeing set exists.
best = []
order = sorted(range(n), key=lambda k: -A[k].sum())
for start in order:
    clique = [start]
    for cand in order:
        if cand in clique:
            continue
        if all(A[cand, c] for c in clique):
            clique.append(cand)
    if len(clique) > len(best):
        best = clique
print("最大互相一致集合: %d / %d 条" % (len(best), n))
for k in sorted(best):
    e = loops[k]
    print("   %3d -> %-3d  修正 %.2f m  互相一致度 %d/%d"
          % (e.from_id, e.to_id, corr[k], A[k].sum(), n - 1))
print()
outliers = [k for k in range(n) if k not in best]
if outliers:
    print("不在该集合内的 %d 条:" % len(outliers))
    for k in outliers:
        e = loops[k]
        print("   %3d -> %-3d  修正 %.2f m  互相一致度 %d/%d"
              % (e.from_id, e.to_id, corr[k], A[k].sum(), n - 1))
