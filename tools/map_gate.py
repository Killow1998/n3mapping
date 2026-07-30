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

XY_TOL = 1.0      # metres: close enough to be the same place
T_GAP = 30.0      # seconds: far enough apart to be a revisit
GATE = {
    "loop_edges_min": 20,
    "dz_p90_max": 0.10,
    "dz_max_max": 0.30,
    "geometry_drift_max": 0.05,   # fraction, vs baseline
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
    return {"P": P, "Podom": Podom, "t": t - t[0], "loop": loop,
            "edges": len(m.edges), "n": len(kfs)}


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
    dz = np.abs(P[pairs[:, 0], 2] - P[pairs[:, 1], 2])
    return {"n": len(dz), "median": float(np.median(dz)),
            "p90": float(np.percentile(dz, 90)), "max": float(dz.max()),
            "over_half": int((dz > 0.5).sum())}


def geometry(d):
    P = d["P"]
    return {"path": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
            "x": float(P[:, 0].ptp()), "y": float(P[:, 1].ptp()),
            "z": float(P[:, 2].ptp())}


cand = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else None

rc, rb = revisit(cand), (revisit(base) if base else None)
gc, gb = geometry(cand), (geometry(base) if base else None)

back_end = np.linalg.norm(cand["P"] - cand["Podom"], axis=1)

print("关键帧 %d   边 %d (闭环 %d)" % (cand["n"], cand["edges"], cand["loop"]))
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

print()
checks = [
    ("闭环边 >= %d" % GATE["loop_edges_min"], cand["loop"] >= GATE["loop_edges_min"]),
    ("|dz| p90 < %.2f m" % GATE["dz_p90_max"], rc["p90"] < GATE["dz_p90_max"]),
    ("|dz| max < %.2f m" % GATE["dz_max_max"], rc["max"] < GATE["dz_max_max"]),
]
if gb:
    worst = max(abs(gc[k] - gb[k]) / gb[k] for k in ("path", "x", "y"))
    checks.append(("几何未折叠 (变化 %.1f%% < %.0f%%)"
                   % (100 * worst, 100 * GATE["geometry_drift_max"]),
                   worst < GATE["geometry_drift_max"]))
for name, ok in checks:
    print("  %-42s %s" % (name, "PASS" if ok else "FAIL"))
print()
print("总判定: %s" % ("PASS" if all(ok for _, ok in checks) else "FAIL"))
