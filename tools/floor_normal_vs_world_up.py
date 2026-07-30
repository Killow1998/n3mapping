#!/usr/bin/env python3
"""Tracks where the floor is, according to the LiDAR, in the LIO world frame.

A gravity estimate that is simply wrong by a fixed angle tilts the whole map
rigidly, and a rigid tilt cannot make the same place come back at a different
height -- revisit consistency is invariant under it. The measured 2.49 m says
the attitude error changed over the session. This asks the LiDAR the same
question independently of the IMU: fit the floor under each keyframe from that
keyframe's own cloud, rotate the normal into the world frame, and see whether
world 'up' wanders away from the floor as the session goes on.

This is not the section 69 measurement. That one fitted planes in 5 m cells of
the finished map and found 0.41 deg, but a cell built from a single pass looks
flat however far the frame has drifted; it measured local flatness, not
agreement between the world frame and the floor over time.

    python3 floor_vs_gravity.py <map.pbstream> [<pb_dir>]
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/tmp/pb")
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

FLOOR_BAND = 0.35    # metres above the lowest returns still counted as floor
MIN_PTS = 400
MAX_RADIUS = 8.0     # metres: keep to the patch the robot is standing on


def q2R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


m = n3map_pb2.N3Map()
m.ParseFromString(Path(sys.argv[1]).read_bytes())
kfs = sorted(m.keyframes, key=lambda k: k.timestamp)
t0 = kfs[0].timestamp

rows = []
for k in kfs:
    # PointCloudData stores x, y, z, intensity interleaved.
    if k.cloud.num_points < MIN_PTS:
        continue
    P = np.asarray(k.cloud.points, dtype=np.float64).reshape(-1, 4)[:, :3]
    if len(P) < MIN_PTS:
        continue

    R_wb = q2R(k.pose_odom.qx, k.pose_odom.qy, k.pose_odom.qz, k.pose_odom.qw)
    # Body-frame cloud rotated into world axes, so 'down' is whatever the LIO
    # world frame currently believes it to be.
    t_wb = np.array([k.pose_odom.tx, k.pose_odom.ty, k.pose_odom.tz])
    W = P @ R_wb.T
    r = np.hypot(W[:, 0], W[:, 1])
    W = W[r < MAX_RADIUS]
    if len(W) < MIN_PTS:
        continue
    zlo = np.percentile(W[:, 2], 2.0)
    F = W[(W[:, 2] > zlo - 0.10) & (W[:, 2] < zlo + FLOOR_BAND)]
    if len(F) < MIN_PTS:
        continue
    c = F.mean(axis=0)
    _, s, Vt = np.linalg.svd(F - c, full_matrices=False)
    nrm = Vt[2]
    if nrm[2] < 0:
        nrm = -nrm
    planarity = s[2] / max(s[1], 1e-9)
    if planarity > 0.20:
        continue
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, nrm[2]))))
    floor_world_z = float(np.percentile(F[:, 2], 50.0)) + t_wb[2]
    rows.append((k.timestamp - t0, k.id, tilt, nrm, len(F), floor_world_z,
                 float(t_wb[2])))

print("可用关键帧 %d / %d" % (len(rows), len(kfs)))
if not rows:
    raise SystemExit("no usable floor fits")
tilts = np.array([r[2] for r in rows])
print("地板法向偏离 world z: 中位 %.2f deg  p90 %.2f deg  最大 %.2f deg"
      % (np.median(tilts), np.percentile(tilts, 90), tilts.max()))
print()
fz = np.array([r[5] for r in rows])
bz = np.array([r[6] for r in rows])
print("地板在 world 系的高度: 中位 %.3f m  跨度 %.3f m  (最低 %.3f, 最高 %.3f)"
      % (np.median(fz), fz.max() - fz.min(), fz.min(), fz.max()))
print("机体在 world 系的高度: 跨度 %.3f m" % (bz.max() - bz.min()))
print("机体离地高度 (body_z - floor_z): 中位 %.3f m  跨度 %.3f m"
      % (np.median(bz - fz), (bz - fz).max() - (bz - fz).min()))
print()
print("%-9s %-6s %-9s %-10s %-10s %-10s"
      % ("t(s)", "kf", "偏离(deg)", "地板z", "机体z", "离地高"))
step = max(1, len(rows) // 22)
for t, kid, tilt, nrm, npts, floor_z, body_z in rows[::step]:
    print("%-9.1f %-6d %-9.2f %-10.3f %-10.3f %-10.3f"
          % (t, kid, tilt, floor_z, body_z, body_z - floor_z))
print()
# A wandering world frame shows up as the normal's horizontal part rotating,
# not merely as a bigger angle.
early = np.array([r[3] for r in rows[:max(3, len(rows) // 10)]]).mean(axis=0)
late = np.array([r[3] for r in rows[-max(3, len(rows) // 10):]]).mean(axis=0)
early /= np.linalg.norm(early)
late /= np.linalg.norm(late)
ang = math.degrees(math.acos(max(-1.0, min(1.0, float(early @ late)))))
print("前 10%% 与后 10%% 的平均法向夹角: %.2f deg" % ang)
print("  前段 [%6.3f %6.3f %6.3f]   后段 [%6.3f %6.3f %6.3f]"
      % (early[0], early[1], early[2], late[0], late[1], late[2]))
