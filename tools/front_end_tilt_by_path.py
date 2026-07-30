#!/usr/bin/env python3
"""Asks whether the front end's world frame is tilted at the start of a session.

The 0723 map's height error is concentrated in its first 50 m: revisit pairs
that begin there miss by up to 2.47 m while pairs past 150 m come in at 0.10 m.
Section 82 traced that to a startup transient in the LIO attitude, and section
92 showed no back-end change can undo it, because the early legs were integrated
under the wrong attitude and the error is baked into the odometry edges as
measurements. Whether that transient is a property of one recording or of the
front end is the difference between re-recording once and fixing FAST_LIO.

The floor answers it directly and without revisits, so it works on a multi-storey
session too. Fit the floor under each keyframe from that keyframe's own returns,
carry the normal into the world frame through the *raw* LIO pose, and see where
it points. Using pose_odom rather than pose_optimized matters: the attitude
factor drives exactly this residual to zero, so measured through the optimised
pose the question answers itself.

What is averaged is the normal as a vector, not its angle from vertical. An
angle is unsigned, so isotropic noise rectifies into it: with the 2.7 deg scatter
of section 88.4 and no tilt at all, the mean angle still reads about 2.4 deg, and
on 0723 the bins came out 2.1 to 3.3 with the largest in the middle of the
session -- an instrument that could not see what it was pointed at. Averaging the
horizontal components lets the noise cancel and leaves what is systematic. The
standard error is printed so a difference can be told from nothing, and so is the
bearing, because a tilt that stays put while its size changes is a different
story from one that swings around.

    python3 front_end_tilt_by_path.py <map.pbstream> [<pb_dir>]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/tmp/pb")
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

FLOOR_BAND = 0.35
FLOOR_BAND_BELOW = 0.10
MAX_RADIUS = 8.0
MIN_POINTS = 400
MAX_PLANARITY = 0.20
BIN_EDGES = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]


def rotation(q):
    return np.array([
        [1 - 2 * (q.qy * q.qy + q.qz * q.qz), 2 * (q.qx * q.qy - q.qz * q.qw),
         2 * (q.qx * q.qz + q.qy * q.qw)],
        [2 * (q.qx * q.qy + q.qz * q.qw), 1 - 2 * (q.qx * q.qx + q.qz * q.qz),
         2 * (q.qy * q.qz - q.qx * q.qw)],
        [2 * (q.qx * q.qz - q.qy * q.qw), 2 * (q.qy * q.qz + q.qx * q.qw),
         1 - 2 * (q.qx * q.qx + q.qy * q.qy)]])


def floor_normal_world(kf):
    """The floor normal under one keyframe, in the LIO world frame."""
    if kf.cloud.num_points < MIN_POINTS:
        return None
    pts = np.asarray(kf.cloud.points, dtype=np.float64).reshape(-1, 4)[:, :3]
    R = rotation(kf.pose_odom)
    # The floor is picked out in the sensor frame, where "below" is known
    # without trusting the pose that is under examination.
    down = R.T @ np.array([0.0, 0.0, -1.0])
    h = -(pts @ down)
    horiz2 = np.einsum("ij,ij->i", pts, pts) - h * h
    keep = horiz2 < MAX_RADIUS ** 2
    near, h = pts[keep], h[keep]
    if len(near) < MIN_POINTS:
        return None
    lo = np.percentile(h, 2.0)
    floor = near[(h > lo - FLOOR_BAND_BELOW) & (h < lo + FLOOR_BAND)]
    if len(floor) < MIN_POINTS:
        return None
    centred = floor - floor.mean(axis=0)
    evals, evecs = np.linalg.eigh(centred.T @ centred / len(floor))
    if np.sqrt(max(evals[0], 0)) / max(np.sqrt(max(evals[1], 0)), 1e-9) > MAX_PLANARITY:
        return None
    n = evecs[:, 0]
    if n @ (-down) < 0:
        n = -n
    world_n = R @ n
    return -world_n if world_n[2] < 0 else world_n


def summarise(rows):
    """Mean tilt of a set of normals, in degrees, with its standard error."""
    mean_n = rows.mean(axis=0)
    tilt = np.degrees(np.hypot(mean_n[0], mean_n[1]))
    bearing = np.degrees(np.arctan2(mean_n[1], mean_n[0]))
    var = rows[:, 0].var(ddof=1) + rows[:, 1].var(ddof=1)
    return tilt, bearing, np.degrees(np.sqrt(var / len(rows))), np.degrees(np.sqrt(var))


def main():
    m = n3map_pb2.N3Map()
    m.ParseFromString(Path(sys.argv[1]).read_bytes())
    kfs = sorted(m.keyframes, key=lambda k: k.timestamp)
    P = np.array([[k.pose_odom.tx, k.pose_odom.ty, k.pose_odom.tz] for k in kfs])
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])

    rows = []
    for i, k in enumerate(kfs):
        n = floor_normal_world(k)
        if n is not None:
            rows.append((path[i], n[0], n[1], n[2]))
    if not rows:
        print("no keyframe produced a usable floor fit")
        return
    A = np.array(rows)
    print("关键帧 %d，可拟合地板 %d，路程 %.1f m" % (len(kfs), len(A), path[-1]))
    print("前端世界系相对地板的系统性倾斜（矢量平均，用 pose_odom，与后端无关）")
    print("   路程段      帧数   倾角 deg   方位 deg   标准误   单帧散布")
    for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:] + [float("inf")]):
        s = A[(A[:, 0] >= lo) & (A[:, 0] < hi), 1:4]
        if len(s) < 3:
            continue
        tilt, bearing, se, spread = summarise(s)
        label = "%d-%s" % (lo, "%d" % hi if hi != float("inf") else "+")
        print("  %9s   %5d   %8.2f  %9.0f  %7.2f  %9.2f"
              % (label, len(s), tilt, bearing, se, spread))

    early = A[A[:, 0] < 50.0, 1:4]
    late = A[A[:, 0] >= 100.0, 1:4]
    if len(early) >= 3 and len(late) >= 3:
        d = early.mean(axis=0)[:2] - late.mean(axis=0)[:2]
        se = np.sqrt((early[:, 0].var(ddof=1) + early[:, 1].var(ddof=1)) / len(early)
                     + (late[:, 0].var(ddof=1) + late[:, 1].var(ddof=1)) / len(late))
        diff = np.hypot(d[0], d[1])
        print()
        print("前 50 m 与 100 m 之后：倾角差 %.2f deg（标准误 %.2f，%.1f sigma）"
              % (np.degrees(diff), np.degrees(se), diff / se if se > 0 else float("inf")))
        print("判定：%s" % ("存在开机暂态" if diff > 3 * se else "无法分辨（差值在噪声内）"))


main()
