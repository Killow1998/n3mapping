#!/usr/bin/env python3
"""Does the mapping trajectory come back to the same height at the same place?

An office floor is flat. If the same (x, y) is visited at two different times
and the recorded z differs by metres, the map is warped and the 7.41 m 'floor
z span' is drift rather than architecture. This needs no ground truth: it is an
internal consistency test of the map against itself.
"""
import csv, math, sys
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "dense_trajectory.csv"
XY_TOL = 1.0     # metres: same place
T_GAP = 30.0     # seconds: genuinely a revisit, not the adjacent sample

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["stamp_ns"]),
                     float(r["tx"]), float(r["ty"]), float(r["tz"])))
rows.sort()
T = np.array([r[0] for r in rows], dtype=np.int64)
P = np.array([[r[1], r[2], r[3]] for r in rows])
t0 = T[0]
ts = (T - t0) / 1e9

print("样本 %d   时长 %.1f s" % (len(P), ts[-1]))
print("z 范围 %.2f .. %.2f m   跨度 %.2f m" % (P[:, 2].min(), P[:, 2].max(),
                                              P[:, 2].max() - P[:, 2].min()))
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
print("轨迹总长 %.1f m" % seg.sum())
print()

# z as a function of time, sampled
print("z 随时间（每 60 s 采样）:")
for tt in range(0, int(ts[-1]) + 1, 60):
    i = int(np.searchsorted(ts, tt))
    i = min(i, len(P) - 1)
    print("  t=%6.0f s   xy=(%7.2f,%7.2f)   z=%7.3f" % (ts[i], P[i, 0], P[i, 1], P[i, 2]))
print()

# Revisits: same xy, far apart in time.
try:
    from scipy.spatial import cKDTree
    tree = cKDTree(P[:, :2])
    pairs = tree.query_pairs(XY_TOL, output_type="ndarray")
except ImportError:
    print("scipy 不可用，跳过 revisit 检测")
    sys.exit(0)

dt = np.abs(ts[pairs[:, 0]] - ts[pairs[:, 1]])
keep = dt > T_GAP
pairs, dt = pairs[keep], dt[keep]
if len(pairs) == 0:
    print("没有找到时间间隔 >%.0f s 的重访对" % T_GAP)
    sys.exit(0)

dz = np.abs(P[pairs[:, 0], 2] - P[pairs[:, 1], 2])
dxy = np.linalg.norm(P[pairs[:, 0], :2] - P[pairs[:, 1], :2], axis=1)
print("重访对 %d 个 (xy 距离 <%.1f m, 时间间隔 >%.0f s)" % (len(pairs), XY_TOL, T_GAP))
print("  |dz|  中位 %.3f m   p90 %.3f m   最大 %.3f m" %
      (np.median(dz), np.percentile(dz, 90), dz.max()))
print()

k = np.argsort(-dz)[:12]
print("最严重的 12 个重访:")
print("  %-9s %-9s %8s %8s %9s %9s" % ("t_a(s)", "t_b(s)", "dxy", "|dz|", "z_a", "z_b"))
for i in k:
    a, b = pairs[i]
    print("  %-9.1f %-9.1f %8.2f %8.3f %9.3f %9.3f" %
          (ts[a], ts[b], dxy[i], dz[i], P[a, 2], P[b, 2]))
print()
big = (dz > 0.5).sum()
print("|dz| > 0.5 m 的重访对: %d / %d  (%.1f%%)" % (big, len(dz), 100.0 * big / len(dz)))
print()
print("判读: 办公楼层是平的。重访 |dz| 若达到米级，说明建图存在 z 漂移，")
print("      §69 记录的「地板 z 跨度 7.41 m」就不是多标高结构，而是地图被拉歪。")
