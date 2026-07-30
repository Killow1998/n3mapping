#!/usr/bin/env python3
"""Searches all quality-passing loop candidates for a mutually consistent set.

The 19 loops the pipeline selected agree with each other on 1 of 171 pairs, and
the largest agreeing subset is the two that propose no correction at all. But
selection keeps one candidate per query keyframe and picks it by fitness and
referee energy, never asking whether it agrees with the others -- so a large
consistent set could exist in the full pool and be lost to that choice.

This reads every candidate's measured transform from the debug stream, uses the
odometry chain from the map for the legs between them, and looks for the largest
mutually consistent subset. What comes back decides where the defect is: a big
consistent set means the discriminator is missing, no set means the candidates
themselves are perceptual aliases and the front end is where to work.

    python3 pcm_all.py <loop_debug.jsonl> <map.pbstream> [<pb_dir>]
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else "/tmp/pb")
sys.setrecursionlimit(10000)
import n3map_pb2  # noqa: E402

# The cycle closes through two odometry legs, so the tolerance has to cover what
# those legs themselves are wrong by. Measured drift on this session is 2.49 m
# over about 33 keyframes, so roughly 0.075 m per keyframe, and a fixed 0.60 m
# tolerance with legs up to 12 keyframes long was rejecting genuinely consistent
# pairs on the strength of the odometry error inside the test.
TRANS_TOL_BASE = 0.30
TRANS_TOL_PER_KF = 0.075
ROT_TOL_BASE = 0.05
ROT_TOL_PER_KF = 0.004
MAX_LEG = 12


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def q2R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


m = n3map_pb2.N3Map()
m.ParseFromString(Path(sys.argv[2]).read_bytes())
odom = {}
for k in m.keyframes:
    p = k.pose_odom
    T = np.eye(4)
    T[:3, :3] = q2R(p.qx, p.qy, p.qz, p.qw)
    T[:3, 3] = [p.tx, p.ty, p.tz]
    odom[k.id] = T

cands = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if e.get("record_type") != "candidate":
        continue

    def num(k):
        v = e.get(k)
        return v if isinstance(v, (int, float)) and v == v else None

    f, ir = num("fitness_score"), num("inlier_ratio")
    if f is None or ir is None or f >= 0.2 or ir < 0.7:
        continue
    vals = [num("measured_match_query_" + k)
            for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    if any(v is None for v in vals) or e["query_id"] not in odom or e["match_id"] not in odom:
        continue
    x, y, z, ro, pi, ya = vals
    T = np.eye(4)
    T[:3, :3] = Rx(ro) @ Ry(pi) @ Rz(ya)
    T[:3, 3] = [x, y, z]
    cands.append({"q": e["query_id"], "m": e["match_id"], "T": T, "f": f})

print("质量合格且有完整变换的候选: %d  (覆盖 %d 个 query)"
      % (len(cands), len({c["q"] for c in cands})))
corr = []
for c in cands:
    T_odom = inv(odom[c["m"]]) @ odom[c["q"]]
    corr.append(float(np.linalg.norm((inv(T_odom) @ c["T"])[:3, 3])))
corr = np.array(corr)
print("提出的修正量: 中位 %.2f m  p10 %.2f m  最小 %.2f m"
      % (np.median(corr), np.percentile(corr, 10), corr.min()))
print()

n = len(cands)
A = np.zeros((n, n), dtype=bool)
legs_used = 0
errs = []
for i, j in itertools.combinations(range(n), 2):
    a, b = cands[i], cands[j]
    if a["q"] == b["q"] or a["m"] == b["m"]:
        continue                       # share an endpoint: cycle degenerates
    if abs(a["q"] - b["q"]) > MAX_LEG or abs(a["m"] - b["m"]) > MAX_LEG:
        continue                       # odometry leg too long to trust
    legs_used += 1
    T = (a["T"]
         @ (inv(odom[a["q"]]) @ odom[b["q"]])
         @ inv(b["T"])
         @ (inv(odom[b["m"]]) @ odom[a["m"]]))
    et = float(np.linalg.norm(T[:3, 3]))
    c = max(-1.0, min(1.0, (np.trace(T[:3, :3]) - 1.0) / 2.0))
    er = float(np.arccos(c))
    errs.append(et)
    legs = abs(a["q"] - b["q"]) + abs(a["m"] - b["m"])
    if (et <= TRANS_TOL_BASE + TRANS_TOL_PER_KF * legs
            and er <= ROT_TOL_BASE + ROT_TOL_PER_KF * legs):
        A[i, j] = A[j, i] = True

errs = np.array(errs) if errs else np.array([np.nan])
print("可比较的候选对 (端点不共享且里程计腿 <= %d 帧): %d" % (MAX_LEG, legs_used))
print("回路闭合误差: 中位 %.3f m  p10 %.3f m  最小 %.3f m"
      % (np.nanmedian(errs), np.nanpercentile(errs, 10), np.nanmin(errs)))
print("互相一致的对数: %d  (容差 %.2f m + %.3f m/帧)"
      % (A.sum() // 2, TRANS_TOL_BASE, TRANS_TOL_PER_KF))
print()

deg = A.sum(axis=1)
order = sorted(range(n), key=lambda k: -deg[k])
best = []
for start in order[:60]:
    clique = [start]
    for cand in order:
        if cand in clique:
            continue
        if all(A[cand, c] for c in clique):
            clique.append(cand)
    if len(clique) > len(best):
        best = clique
print("最大互相一致集合: %d 条  (覆盖 %d 个 query)"
      % (len(best), len({cands[k]["q"] for k in best})))
for k in sorted(best, key=lambda k: cands[k]["q"]):
    c = cands[k]
    print("   q=%-4d m=%-4d 修正 %6.2f m  fitness %.3f  一致度 %d"
          % (c["q"], c["m"], corr[k], c["f"], deg[k]))
print()
top = order[:10]
print("互相一致度最高的 10 个候选:")
for k in top:
    c = cands[k]
    print("   q=%-4d m=%-4d 修正 %6.2f m  一致度 %d" % (c["q"], c["m"], corr[k], deg[k]))
