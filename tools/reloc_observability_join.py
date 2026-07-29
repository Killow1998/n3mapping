#!/usr/bin/env python3
"""Joins the corrected observability probe against the measured attitude error.

Answers one question: once the rotation information is marginalised over
translation and attributed to the pose the hypothesis actually carries, does it
separate the cases that miss the 2 deg attitude spec from the ones that meet it?
"""
import bisect, csv, json, math, os, re, sys

RUN = "/home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_product_v1/20260725/candidate_dd25f86"
EV = RUN + "/baseline_72fa6f7/eval_fsveto"
PROBE = sys.argv[1]

dense = []
for r in csv.DictReader(open(RUN + "/f7_0723_map/dense_trajectory.csv")):
    dense.append((int(r["stamp_ns"]), r))
dense.sort()
stamps = [s for s, _ in dense]


def rpy(x, y, z, w):
    return (math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x))))),
            math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def ad(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def ref(t):
    i = bisect.bisect_left(stamps, t)
    best = None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(dense):
            dd = abs(stamps[j] - t)
            if best is None or dd < best[0]:
                best = (dd, dense[j][1])
    return best[1] if best and best[0] < 10 ** 6 else None


probe = {}
for line in open(PROBE, encoding="utf-8", errors="replace"):
    parts = line.split()
    if len(parts) < 2:
        continue
    q = parts[0]
    m1 = re.search(r"pose=(\w+)", line)
    m2 = re.search(r"rot_info_marginal_min=([0-9.e+-]+)", line)
    m3 = re.search(r"rot_info_min=([0-9.e+-]+)", line)
    m4 = re.search(r"prod_quality=([01])", line)
    if m1 and m2 and m3:
        probe[q] = (m1.group(1), float(m2.group(1)), float(m3.group(1)), m4.group(1))

rows = []
for q, (pose, marg, cond, pq) in probe.items():
    p = os.path.join(EV, q, "result.json")
    if not os.path.exists(p):
        continue
    res = json.load(open(p))
    if not res["algorithm_lock"]:
        continue
    g = ref(res["lock_stamp_ns"])
    fs = list(csv.DictReader(open(os.path.join(EV, q, "frame_status.csv"))))
    lr = next((f for f in fs if int(f["frame_index"]) == res["lock_frame_index"]), None)
    if g is None or lr is None:
        continue
    pp = {k: float(lr["map_" + k]) for k in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")}
    et = math.dist((pp["tx"], pp["ty"], pp["tz"]),
                   (float(g["tx"]), float(g["ty"]), float(g["tz"])))
    r1, p1, y1 = rpy(pp["qx"], pp["qy"], pp["qz"], pp["qw"])
    r2, p2, y2 = rpy(float(g["qx"]), float(g["qy"]), float(g["qz"]), float(g["qw"]))
    er, ep, ey = ad(r1, r2), ad(p1, p2), ad(y1, y2)
    att_fail = er > 2.0 or ep > 2.0 or ey > 3.0
    rows.append((marg, cond, pose, pq, q, et, er, ep, att_fail))

rows.sort()
print("%-22s %12s %12s %10s %4s %8s %7s %7s" %
      ("case", "marginal", "conditional", "pose", "pq", "t_err", "roll", "pitch"))
print("-" * 92)
for marg, cond, pose, pq, q, et, er, ep, fail in rows:
    print("%-22s %12.4g %12.4g %10s %4s %8.3f %7.2f %7.2f %s" %
          (q[-8:], marg, cond, pose, pq, et, er, ep, "  <-- 姿态超标" if fail else ""))

bad = [r[0] for r in rows if r[8]]
good = [r[0] for r in rows if not r[8]]
print()
if bad and good:
    print("姿态超标例 marginal: %s" % ", ".join("%.3g" % b for b in bad))
    print("达标例     marginal: min=%.3g  max=%.3g" % (min(good), max(good)))
    overlap = sum(1 for b in bad for g in good if g < b)
    print("有 %d 个 (超标例, 达标例) 组合中达标例的信息量更低 → %s" %
          (overlap, "不可分" if overlap else "可分"))
