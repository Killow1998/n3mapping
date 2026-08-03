#!/usr/bin/env python3
"""Where are the candidates the descriptor proposed, and where should the robot be?

The wrong lock at 11-50-34 took the third-ranked candidate. If the top-ranked one
sits near the true position then the ranking was right and the choice was wrong,
which is a different defect from the descriptor aliasing everything else here has
been about.
"""
import bisect
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

MAP = "/home/user/ros_ws/n3mapping_v1_closeout/atlas_rebuild/n3map.pbstream"
DBG = "/home/user/ros_ws/n3mapping_v1_closeout/dbg_115034"
DENSE = ("/home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_product_v1/"
         "20260725/candidate_dd25f86/f7_0723_map/dense_trajectory.csv")

m = n3map_pb2.N3Map()
m.ParseFromString(Path(MAP).read_bytes())
by_id = {k.id: k for k in m.keyframes}
print("keyframes %d   id range %d..%d" % (len(m.keyframes),
                                          min(by_id), max(by_id)))

result = json.loads(Path(DBG, "result.json").read_text())
lock_stamp = result["lock_stamp_ns"]

rows = []
with open(DENSE) as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["stamp_ns"]), r))
rows.sort()
stamps = [s for s, _ in rows]
i = bisect.bisect_left(stamps, lock_stamp)
best = min((abs(stamps[j] - lock_stamp), rows[j][1])
           for j in (i - 1, i, i + 1) if 0 <= j < len(rows))
ref = best[1]
rx, ry, rz = float(ref["tx"]), float(ref["ty"]), float(ref["tz"])
print("reference at lock stamp   (%.2f, %.2f, %.2f)   dt %.1f us"
      % (rx, ry, rz, best[0] / 1e3))

reported = result["final_map_body_pose"]
print("locked pose               (%.2f, %.2f, %.2f)"
      % (reported["tx"], reported["ty"], reported["tz"]))

# Which keyframe is actually nearest the truth?
near = sorted(((math.dist((k.pose_optimized.tx, k.pose_optimized.ty),
                          (rx, ry)), k.id) for k in m.keyframes))[:5]
print("\nkeyframes nearest the reference position:")
for d, kid in near:
    print("   kf %-4d  %.2f m away" % (kid, d))

recs = [json.loads(l) for l in open(Path(DBG, "relocalization_debug.jsonl"))]
cands = recs[84]["top_candidates"]
print("\ncandidates the descriptor proposed at the accepting frame:")
for rank, c in enumerate(cands):
    kid = c["match_id"]
    k = by_id.get(kid)
    if k is None:
        print("   %d. kf %-4d  (not in map)" % (rank + 1, kid))
        continue
    d = math.dist((k.pose_optimized.tx, k.pose_optimized.ty), (rx, ry))
    print("   %d. kf %-4d  fused %.4f  rhpd %.3f  sc %.4f   %6.2f m from truth%s"
          % (rank + 1, kid, c["fused_score"], c["rhpd_distance"],
             c["sc_distance"], d, "   <- LOCKED" if kid == 176 else ""))
