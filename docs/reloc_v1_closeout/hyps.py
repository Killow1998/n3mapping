#!/usr/bin/env python3
"""Was the true place ever a hypothesis, or only ever a per-frame candidate?

The lock took the third-ranked candidate, so the choice is made over persistent
hypotheses rather than over this frame's descriptor ranking. If the basin around
keyframe 102 never became a hypothesis then no amount of per-frame evidence for
it could have mattered.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

DBG = "/home/user/ros_ws/n3mapping_v1_closeout/dbg_115034"
MAP = "/home/user/ros_ws/n3mapping_v1_closeout/atlas_rebuild/n3map.pbstream"

m = n3map_pb2.N3Map()
m.ParseFromString(Path(MAP).read_bytes())
by_id = {k.id: k for k in m.keyframes}
TRUE_XY = (-5.31, -5.64)          # reference at the lock stamp

recs = [json.loads(l) for l in open(Path(DBG, "relocalization_debug.jsonl"))]
print("records %d" % len(recs))

first = recs[84]
hs = first.get("hypotheses") or []
print("\nhypothesis record keys: %s" % (sorted(hs[0].keys()) if hs else "(none)"))

for idx in (20, 50, 80, 83, 84):
    r = recs[idx]
    hs = r.get("hypotheses") or []
    print("\n--- rec %d   lock=%s  reject=%r  margin=%s ratio=%s"
          % (idx, r.get("lock_accepted"), r.get("reject_reason"),
             r.get("margin"), r.get("ratio")))
    print("    hypotheses: %d" % len(hs))
    for h in hs[:8]:
        kid = h.get("seed_keyframe_id", h.get("match_id", h.get("keyframe_id")))
        k = by_id.get(kid)
        d = (math.dist((k.pose_optimized.tx, k.pose_optimized.ty), TRUE_XY)
             if k is not None else float("nan"))
        interesting = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                       for kk, vv in h.items()
                       if kk not in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")}
        print("      kf %-5s %6.2f m from truth   %s" % (kid, d, interesting))
