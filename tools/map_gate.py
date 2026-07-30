#!/usr/bin/env python3
"""Scores a rebuilt map against the pre-registered criteria, with no ground truth.

An office floor is flat and the robot walked it once, so the map has to agree
with itself: the same place, revisited, must come back to the same height. That
is checkable without any external reference, which is what makes it usable as a
mapping gate. The folding check is here because height consistency alone can be
satisfied the wrong way -- a map crushed by a bad loop is consistent and wrong,
so the geometry must survive as well.

    python3 map_gate.py <candidate.pbstream> [<baseline.pbstream>] [<pb_dir>]
"""
import bisect
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PB_DIR = sys.argv[3] if len(sys.argv) > 3 else "/tmp/pb"
sys.path.insert(0, PB_DIR)
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

# Two poses at the same horizontal position on different floors are not a
# revisit, and pairing them reports the storey height as drift: b22, which is
# genuinely two levels about 11 m apart, came out at 14.28 m of revisit error
# purely from that. Pairs are therefore also required to see the floor at a
# similar height, which keeps the test to one level without needing to know how
# many levels there are.
SAME_LEVEL_TOL = 1.5   # metres of floor-height difference still one level
XY_TOL = 1.0      # metres: close enough to be the same place
T_GAP = 30.0      # seconds: far enough apart to be a revisit
# Distance bands the floor residual is summarised over. Coarse enough that each
# holds tens of keyframes, fine enough that a section sinking on its own shows up.
RESIDUAL_BANDS = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100),
                  (100, 150), (150, 200), (200, 300), (300, 10 ** 9)]

GATE = {
    "loop_edges_min": 20,
    "dz_p90_max": 0.10,
    "dz_max_max": 0.30,
    "geometry_drift_max": 0.05,   # fraction, vs baseline
    # How far apart two bands of the session may place the same floor. Read on
    # the optimised poses with the map's fixed tilt taken out, and excluding the
    # opening ten metres, which hold a front-end startup transient no pose graph
    # can undo. Unlike the revisit figure this cannot be improved by leaning
    # harder on the loop edges, because it is not what they constrain.
    #
    # Single-storey only, the same precondition the revisit test has: one plane
    # cannot describe a session that changes level, so on b22 -- two levels
    # 5.05 m apart -- this reads 3.838 m and calls the storey drift. For a
    # multi-storey recording the structural check is whether the floor-height
    # histogram keeps its modes and their separation.
    "residual_band_span_max": 0.50,
}


def load(path):
    m = n3map_pb2.N3Map()
    m.ParseFromString(Path(path).read_bytes())
    kfs = sorted(m.keyframes, key=lambda k: k.timestamp)
    P = np.array([[k.pose_optimized.tx, k.pose_optimized.ty, k.pose_optimized.tz]
                  for k in kfs])
    Podom = np.array([[k.pose_odom.tx, k.pose_odom.ty, k.pose_odom.tz]
                      for k in kfs])
    t = np.array([k.timestamp for k in kfs])
    loop = sum(1 for e in m.edges if e.type == n3map_pb2.EdgeProto.LOOP)
    # Floor height under each keyframe, fitted from that keyframe's own returns.
    # Used only to keep the revisit test on one level; deriving it from the scans
    # means the test needs no prior knowledge of how many levels exist.
    floor_z = np.full(len(kfs), np.nan)
    for i, k in enumerate(kfs):
        if k.cloud.num_points < 400:
            continue
        pts = np.asarray(k.cloud.points, dtype=np.float64).reshape(-1, 4)[:, :3]
        q = k.pose_optimized
        R = np.array([
            [1 - 2 * (q.qy * q.qy + q.qz * q.qz), 2 * (q.qx * q.qy - q.qz * q.qw),
             2 * (q.qx * q.qz + q.qy * q.qw)],
            [2 * (q.qx * q.qy + q.qz * q.qw), 1 - 2 * (q.qx * q.qx + q.qz * q.qz),
             2 * (q.qy * q.qz - q.qx * q.qw)],
            [2 * (q.qx * q.qz - q.qy * q.qw), 2 * (q.qy * q.qz + q.qx * q.qw),
             1 - 2 * (q.qx * q.qx + q.qy * q.qy)]])
        W = pts @ R.T
        r = np.hypot(W[:, 0], W[:, 1])
        W = W[r < 8.0]
        if len(W) < 400:
            continue
        zlo = np.percentile(W[:, 2], 2.0)
        band = W[(W[:, 2] > zlo - 0.10) & (W[:, 2] < zlo + 0.35)]
        if len(band) < 400:
            continue
        floor_z[i] = float(np.percentile(band[:, 2], 50.0)) + q.tz
    return {"P": P, "Podom": Podom, "t": t - t[0], "loop": loop,
            "edges": len(m.edges), "n": len(kfs), "floor_z": floor_z,
            "kfs": kfs}


def revisit(d):
    from scipy.spatial import cKDTree
    P, t = d["P"], d["t"]
    pairs = cKDTree(P[:, :2]).query_pairs(XY_TOL, output_type="ndarray")
    if len(pairs) == 0:
        return None
    gap = np.abs(t[pairs[:, 0]] - t[pairs[:, 1]])
    pairs = pairs[gap > T_GAP]
    if len(pairs) == 0:
        return None
    # Deliberately not filtered. Dropping pairs whose observed floors sit far
    # apart looks like it separates storeys from drift, and on 0723 -- a single
    # floor -- it took the worst revisit error from 2.33 m to 0.68 m by removing
    # exactly the pairs that prove the map has drifted. On a drifted single floor
    # the same place does report two floor heights, so the filter hides the
    # defect it is meant to measure. The count is reported instead, and the
    # caller has to know whether the recording is single-storey.
    fz = d.get("floor_z")
    far = 0
    if fz is not None:
        dz_floor = np.abs(fz[pairs[:, 0]] - fz[pairs[:, 1]])
        far = int((dz_floor >= SAME_LEVEL_TOL).sum())
    dz = np.abs(P[pairs[:, 0], 2] - P[pairs[:, 1], 2])
    return {"n": len(dz), "median": float(np.median(dz)),
            "p90": float(np.percentile(dz, 90)), "max": float(dz.max()),
            "over_half": int((dz > 0.5).sum()), "far_floor": far}


def geometry(d):
    P = d["P"]
    return {"path": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
            "x": float(P[:, 0].ptp()), "y": float(P[:, 1].ptp()),
            "z": float(P[:, 2].ptp())}


def residual_band_span(kfs, pose_attr="pose_optimized"):
    """Spread of the floor height between bands, with one fixed plane removed.

    A map tilted by a constant angle puts the same place at the same height
    every time, so the revisit test is blind to it and it does not belong in
    this number either. What is left after subtracting the best-fit plane is the
    part that changes as the session goes on, which is what makes a revisit
    disagree. Returns (span, tilt_deg, rows) or (None, None, []) when too few
    keyframes yield a floor.
    """
    import numpy as np

    rows = []
    path = 0.0
    previous = None
    for k in kfs:
        q = getattr(k, pose_attr)
        here = np.array([q.tx, q.ty, q.tz])
        if previous is not None:
            path += float(np.linalg.norm(here - previous))
        previous = here
        if k.cloud.num_points < 400:
            continue
        pts = np.asarray(k.cloud.points, dtype=np.float64).reshape(-1, 4)[:, :3]
        R = np.array([
            [1 - 2 * (q.qy * q.qy + q.qz * q.qz), 2 * (q.qx * q.qy - q.qz * q.qw),
             2 * (q.qx * q.qz + q.qy * q.qw)],
            [2 * (q.qx * q.qy + q.qz * q.qw), 1 - 2 * (q.qx * q.qx + q.qz * q.qz),
             2 * (q.qy * q.qz - q.qx * q.qw)],
            [2 * (q.qx * q.qz - q.qy * q.qw), 2 * (q.qy * q.qz + q.qx * q.qw),
             1 - 2 * (q.qx * q.qx + q.qy * q.qy)]])
        W = pts @ R.T
        W = W[np.hypot(W[:, 0], W[:, 1]) < 8.0]
        if len(W) < 400:
            continue
        lo = np.percentile(W[:, 2], 2.0)
        band = W[(W[:, 2] > lo - 0.10) & (W[:, 2] < lo + 0.35)]
        if len(band) < 400:
            continue
        rows.append((path, q.tx, q.ty, float(np.median(band[:, 2])) + q.tz))
    if len(rows) < 20:
        return None, None, []

    A = np.array(rows)
    M = np.column_stack([A[:, 1], A[:, 2], np.ones(len(A))])
    coef, *_ = np.linalg.lstsq(M, A[:, 3], rcond=None)
    residual = A[:, 3] - M @ coef
    tilt = math.degrees(math.atan(math.hypot(coef[0], coef[1])))

    out = []
    for lo, hi in RESIDUAL_BANDS:
        sel = residual[(A[:, 0] >= lo) & (A[:, 0] < hi)]
        if len(sel) >= 3:
            out.append((lo, hi, len(sel), float(np.median(sel)), float(sel.std())))
    # The opening band carries the startup transient, which is a property of the
    # recording rather than of the pose graph, and including it would report the
    # same failure whatever the back end did.
    settled = [r[3] for r in out if r[0] >= 10]
    if len(settled) < 2:
        return None, tilt, out
    return max(settled) - min(settled), tilt, out


cand = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else None

rc, rb = revisit(cand), (revisit(base) if base else None)
gc, gb = geometry(cand), (geometry(base) if base else None)

back_end = np.linalg.norm(cand["P"] - cand["Podom"], axis=1)

print("关键帧 %d   边 %d (闭环 %d)" % (cand["n"], cand["edges"], cand["loop"]))
print("前提：重访判据只对单层录制成立。多层建筑里同一 xy 的不同楼层会被当成重访。")
if rc and rc.get("far_floor"):
    print("注意：%d / %d 对配对的观测地板高度相差 >%.1f m。单层录制下这是漂移的证据；"
          % (rc["far_floor"], rc["n"], SAME_LEVEL_TOL))
    print("      多层录制下这些配对不是重访，本表的 |dz| 不可解读。")
print("后端对前端的位移修正: 中位 %.3e m  最大 %.4f m"
      % (float(np.median(back_end)), float(back_end.max())))
print()
hdr = "%-26s %12s" % ("", "候选")
if rb:
    hdr += " %12s" % "基线"
print(hdr)
print("-" * len(hdr))


def row(label, a, b, fmt="%.4f"):
    line = "%-26s %12s" % (label, fmt % a)
    if b is not None:
        line += " %12s" % (fmt % b)
    print(line)


row("闭环边", cand["loop"], base["loop"] if base else None, "%d")
row("重访对数", rc["n"], rb["n"] if rb else None, "%d")
row("|dz| 中位 m", rc["median"], rb["median"] if rb else None)
row("|dz| p90 m", rc["p90"], rb["p90"] if rb else None)
row("|dz| 最大 m", rc["max"], rb["max"] if rb else None)
row("|dz|>0.5m 的对数", rc["over_half"], rb["over_half"] if rb else None, "%d")
row("轨迹总长 m", gc["path"], gb["path"] if gb else None, "%.1f")
row("x 范围 m", gc["x"], gb["x"] if gb else None, "%.2f")
row("y 范围 m", gc["y"], gb["y"] if gb else None, "%.2f")
row("z 范围 m", gc["z"], gb["z"] if gb else None, "%.2f")

span_c, tilt_c, bands_c = residual_band_span(cand["kfs"])
span_b = residual_band_span(base["kfs"])[0] if base else None
if span_c is not None:
    row("地板残差段间跨度 m", span_c, span_b, "%.3f")
    if span_c > GATE["residual_band_span_max"]:
        print("      注意：该项与重访判据一样只对单层录制成立。"
              "多层录制下层高会被算成漂移。")

print()
if bands_c:
    print("地板残差按路程（已去掉 %.2f° 的固定倾斜，该倾斜对重访不可见）："
          % (tilt_c or 0.0))
    print("  %-12s %6s %10s %10s" % ("路程段", "帧数", "中位", "标准差"))
    for lo, hi, n, med, sd in bands_c:
        label = "%d-%s" % (lo, "%d" % hi if hi < 10 ** 9 else "+")
        note = "  ← 开机暂态，不计入跨度" if lo < 10 else ""
        print("  %-12s %6d %10.3f %10.3f%s" % (label, n, med, sd, note))
    print()
checks = [
    ("闭环边 >= %d" % GATE["loop_edges_min"], cand["loop"] >= GATE["loop_edges_min"]),
    ("|dz| p90 < %.2f m" % GATE["dz_p90_max"], rc["p90"] < GATE["dz_p90_max"]),
    ("|dz| max < %.2f m" % GATE["dz_max_max"], rc["max"] < GATE["dz_max_max"]),
]
if span_c is not None:
    # The revisit figure alone can be driven down by leaning harder on the loop
    # edges, because it scores very nearly the pairs those edges constrain:
    # loop_noise_position 0.5 -> 0.05 took it from 0.4605 to 0.3283 while the
    # floor under the last stretch of the session sank 0.33 m. This is the check
    # that notices.
    checks.append(("地板残差段间跨度 < %.2f m" % GATE["residual_band_span_max"],
                   span_c < GATE["residual_band_span_max"]))
if gb:
    worst = max(abs(gc[k] - gb[k]) / gb[k] for k in ("path", "x", "y"))
    checks.append(("几何未折叠 (变化 %.1f%% < %.0f%%)"
                   % (100 * worst, 100 * GATE["geometry_drift_max"]),
                   worst < GATE["geometry_drift_max"]))
for name, ok in checks:
    print("  %-42s %s" % (name, "PASS" if ok else "FAIL"))
print()
print("总判定: %s" % ("PASS" if all(ok for _, ok in checks) else "FAIL"))

