#!/usr/bin/env python3
"""Is the true place a doubled part of the map?

Registration at the true place fits seven times worse than at a place 25 m away,
and half its bins have map surface in front of the observed return. That is what
a place mapped twice with an offset looks like. Keyframes 102/103 and 223/224 are
all within two metres of the true position, so the session passed there twice --
the question is how far apart the two passes ended up.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pb")
import n3map_pb2  # noqa: E402

MAP = "/home/user/ros_ws/n3mapping_v1_closeout/atlas_rebuild/n3map.pbstream"
m = n3map_pb2.N3Map()
m.ParseFromString(Path(MAP).read_bytes())
kf = {k.id: k for k in m.keyframes}

TRUE_XY = (-5.31, -5.64)


def xyz(k):
    p = k.pose_optimized
    return (p.tx, p.ty, p.tz)


print("keyframes within 3 m of the true position (xy):")
near = []
for k in m.keyframes:
    x, y, z = xyz(k)
    d = math.dist((x, y), TRUE_XY)
    if d <= 3.0:
        near.append((d, k.id, x, y, z, k.timestamp))
for d, kid, x, y, z, t in sorted(near):
    print("   kf %-4d  %.2f m   (%7.2f, %7.2f, %7.2f)   t=%.1f" % (kid, d, x, y, z, t))

if len(near) >= 2:
    ids = [n[1] for n in sorted(near)]
    print("\npairwise, first pass vs second pass:")
    early = [i for i in ids if i < 150]
    late = [i for i in ids if i >= 150]
    print("   first pass  %s" % early)
    print("   second pass %s" % late)
    for a in early:
        for b in late:
            ka, kb = kf[a], kf[b]
            xa, ya, za = xyz(ka)
            xb, yb, zb = xyz(kb)
            print("   kf %-4d <-> kf %-4d   xy %.3f m   dz %+.3f m   dt %.0f s"
                  % (a, b, math.dist((xa, ya), (xb, yb)), zb - za,
                     kb.timestamp - ka.timestamp))

# For contrast, the place that won.
print("\nsame question at the wrong place (kf 176):")
w = xyz(kf[176])
for k in m.keyframes:
    x, y, z = xyz(k)
    if k.id != 176 and math.dist((x, y), (w[0], w[1])) <= 3.0:
        print("   kf %-4d  %.2f m   dz %+.3f   dt %.0f s"
              % (k.id, math.dist((x, y), (w[0], w[1])), z - w[2],
                 k.timestamp - kf[176].timestamp))
