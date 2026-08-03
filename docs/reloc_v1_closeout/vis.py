#!/usr/bin/env python3
"""Is the true place punished for disagreeing, or for not being predicted?

evidence_log_odds is logit(consistent_bins / observed_bins), and observed_bins
counts every bin the query saw, including the ones the map made no prediction
for. If the true place has lower observed_coverage than the winner then the
penalty is for absence of map, not for contradiction by map.
"""
import json
import math
from pathlib import Path

DBG = Path("/home/user/ros_ws/n3mapping_v1_closeout/dbg_115034")
recs = [json.loads(l) for l in open(DBG / "relocalization_debug.jsonl")]

# Find the arrays that carry per-candidate visibility.
r = recs[84]
for key, val in r.items():
    if isinstance(val, list) and val and isinstance(val[0], dict):
        if any("visibility" in k for k in val[0]):
            print("array with visibility fields: %r  (%d entries)" % (key, len(val)))
            print("  keys: %s" % sorted(val[0]))

TRUE_KF = {101, 102, 103}
WRONG_KF = {175, 176, 179, 180, 181}


def show(idx):
    r = recs[idx]
    for key, val in r.items():
        if not (isinstance(val, list) and val and isinstance(val[0], dict)):
            continue
        if not any("visibility" in k for k in val[0]):
            continue
        print("\n--- rec %d   %s" % (idx, key))
        print("    %-6s %8s %8s %8s %9s" %
              ("kf", "consist", "coverage", "fg_confl", "log_odds"))
        for c in val:
            kid = c.get("match_id", c.get("seed_match_id"))
            tag = ""
            if kid in TRUE_KF:
                tag = "  <- TRUE PLACE"
            elif kid in WRONG_KF:
                tag = "  <- wrong place"
            def g(k):
                v = c.get(k)
                return float("nan") if v is None else v
            print("    %-6s %8.4f %8.4f %8.4f %9.4f%s"
                  % (kid, g("visibility_consistency_ratio"),
                     g("visibility_observed_coverage"),
                     g("visibility_foreground_conflict_ratio"),
                     g("visibility_evidence_log_odds"), tag))


for i in (20, 80, 84):
    show(i)
