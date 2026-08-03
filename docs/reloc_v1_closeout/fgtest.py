#!/usr/bin/env python3
"""Does foreground conflict track how many passes contributed to a place?

The claim to test: a spot driven through twice holds twice the accumulated
surface, and from one vantage much of the second pass's surface is occluded yet
still projects in front of the measured return. If so, foreground conflict should
rise with the number of temporally separate passes near the matched keyframe,
independently of how well the scan registered.
"""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

C = "/home/user/ros_ws/n3mapping_v1_closeout"
NEAR_M = 4.0        # what counts as "at the same place"
PASS_GAP_S = 120.0  # a gap this long means a separate visit


def load(p):
    m = n3map_pb2.N3Map()
    m.ParseFromString(Path(p).read_bytes())
    return {k.id: k for k in m.keyframes}, list(m.keyframes)


def passes_near(kf_by_id, all_kfs, kid):
    k = kf_by_id.get(kid)
    if k is None:
        return None, None
    x, y = k.pose_optimized.tx, k.pose_optimized.ty
    ts = sorted(kk.timestamp for kk in all_kfs
                if math.dist((kk.pose_optimized.tx, kk.pose_optimized.ty), (x, y)) <= NEAR_M)
    if not ts:
        return None, None
    n_pass = 1
    for a, b in zip(ts, ts[1:]):
        if b - a > PASS_GAP_S:
            n_pass += 1
    return n_pass, len(ts)


def collect(map_path, dbg_dir, label):
    kf_by_id, all_kfs = load(map_path)
    rows = []
    for line in open(Path(dbg_dir, "relocalization_debug.jsonl")):
        r = json.loads(line)
        for c in (r.get("per_basin_best") or []):
            kid = c.get("matched_kf_id")
            fg = c.get("visibility_foreground_conflict_ratio")
            fit = c.get("fitness_score")
            if kid is None or fg is None:
                continue
            n_pass, n_near = passes_near(kf_by_id, all_kfs, kid)
            if n_pass is None:
                continue
            rows.append((n_pass, n_near, fg, fit, kid))
    print("\n=== %s   (%d candidate observations)" % (label, len(rows)))
    by_pass = {}
    for n_pass, n_near, fg, fit, kid in rows:
        by_pass.setdefault(min(n_pass, 3), []).append((fg, fit))
    print("   passes   n     fg_conflict median   fitness median")
    for p in sorted(by_pass):
        vals = by_pass[p]
        print("   %-8s %-5d %.4f                %.4f"
              % ("%d%s" % (p, "+" if p == 3 else ""), len(vals),
                 statistics.median(v[0] for v in vals),
                 statistics.median(v[1] for v in vals)))
    return rows


a = collect(C + "/atlas_rebuild/n3map.pbstream", C + "/dbg_115034", "product map")
b = collect(C + "/s16_staticguard/map/n3map.pbstream", C + "/dbg_s16", "s16 map")

rows = a + b
if rows:
    single = [r[2] for r in rows if r[0] == 1]
    multi = [r[2] for r in rows if r[0] >= 2]
    print("\n=== both maps pooled")
    print("   single-pass places: n=%-4d fg_conflict median %.4f" % (len(single), statistics.median(single)) if single else "   no single-pass")
    print("   multi-pass places:  n=%-4d fg_conflict median %.4f" % (len(multi), statistics.median(multi)) if multi else "   no multi-pass")
    # Is it the passes or just the density?
    lo = [r[2] for r in rows if r[1] <= 10]
    hi = [r[2] for r in rows if r[1] > 10]
    if lo and hi:
        print("   few neighbours (<=10):  n=%-4d fg median %.4f" % (len(lo), statistics.median(lo)))
        print("   many neighbours (>10):  n=%-4d fg median %.4f" % (len(hi), statistics.median(hi)))
