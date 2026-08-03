#!/usr/bin/env python3
"""The criterion failed. Establish exactly what happened before saying why.

The prediction was that removing the half-metre ghost at the true place would
stop 11-50-34 locking 25 m away. It locked at the same wrong place. Three things
to pin down: that kf 168 on the s16 map is the same physical spot as kf 176 on
the product map, whether the true place was still ranked first per frame, and
what the visibility evidence looks like now.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

C = "/home/user/ros_ws/n3mapping_v1_closeout"


def load(p):
    m = n3map_pb2.N3Map()
    m.ParseFromString(Path(p).read_bytes())
    return {k.id: k for k in m.keyframes}

prod = load(C + "/atlas_rebuild/n3map.pbstream")
s16 = load(C + "/s16_staticguard/map/n3map.pbstream")

print("timestamps of the locked keyframe on each map:")
print("   product kf 176   t=%.1f" % prod[176].timestamp)
print("   s16     kf 168   t=%.1f" % s16[168].timestamp)
print("   difference %.1f s  -> %s"
      % (abs(prod[176].timestamp - s16[168].timestamp),
         "same physical spot" if abs(prod[176].timestamp - s16[168].timestamp) < 5
         else "DIFFERENT spot"))

# The true place, by timestamp rather than by frame.
PASS = [(1784778620.0, 1784778680.0), (1784779050.0, 1784779080.0)]
true_ids = [k.id for k in s16.values()
            if any(a <= k.timestamp <= b for a, b in PASS)]
print("\ns16 keyframes at the true place (by timestamp): %s" % sorted(true_ids)[:12])

recs = [json.loads(l) for l in open(Path(C, "dbg_s16", "relocalization_debug.jsonl"))]
res = json.loads(Path(C, "dbg_s16", "result.json").read_text())
lock = res["lock_frame_index"]
print("\nlock frame %d" % lock)

r = recs[lock]
print("  margin %.4f  ratio %.4f  basin_sep %.3f"
      % (r.get("margin", float("nan")), r.get("ratio", float("nan")),
         r.get("basin_separation", float("nan"))))
print("  candidates:")
for rank, c in enumerate(r.get("top_candidates") or []):
    kid = c["match_id"]
    tag = "  <- TRUE PLACE" if kid in true_ids else ""
    print("    %2d. kf %-4d fused %.4f  rhpd %.3f%s"
          % (rank + 1, kid, c["fused_score"], c["rhpd_distance"], tag))

pb = r.get("per_basin_best") or []
print("  per-basin (%d):" % len(pb))
for c in pb:
    kid = c.get("matched_kf_id")
    tag = "  <- TRUE PLACE" if kid in true_ids else ""
    print("    kf %-4d fitness %.4f inlier %.3f  consist %.4f cover %.4f "
          "fg %.4f  log_odds %+.4f%s"
          % (kid, c.get("fitness_score", float("nan")),
             c.get("inlier_ratio", float("nan")),
             c.get("visibility_consistency_ratio", float("nan")),
             c.get("visibility_observed_coverage", float("nan")),
             c.get("visibility_foreground_conflict_ratio", float("nan")),
             c.get("visibility_evidence_log_odds", float("nan")), tag))

# How often was the true place ranked first, on this map?
hit = miss = 0
for rr in recs:
    cands = rr.get("top_candidates") or []
    if not cands:
        continue
    if cands[0]["match_id"] in true_ids:
        hit += 1
    else:
        miss += 1
print("\ntrue place ranked first: %d/%d frames" % (hit, hit + miss))
