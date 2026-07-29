#!/usr/bin/env python3
"""Reads FAST_LIO's own gravity estimate over the mapping session.

The map disagrees with itself by 2.49 m in height at one spot, and the back end
corrected that by 0.38 mm, so the drift is the front end's. This asks the front
end directly: how far did its estimate of 'down' move while it was mapping? A
gravity direction that rotates by theta turns straight travel into a height
error of distance * sin(theta), which is the arithmetic that has to match the
5.71 m of z spread over 235.7 m of path.
"""
import math, sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

bag = Path(sys.argv[1])
rows = []
with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
    conns = [c for c in reader.connections if c.topic == "/lio_gravity"]
    if not conns:
        raise SystemExit("no /lio_gravity in bag")
    for conn, ts, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, conn.msgtype)
        st = m.header.stamp
        v = m.vector
        rows.append((int(st.sec) + int(st.nanosec) * 1e-9, v.x, v.y, v.z))

rows.sort()
t = np.array([r[0] for r in rows])
G = np.array([[r[1], r[2], r[3]] for r in rows])
t -= t[0]
n = np.linalg.norm(G, axis=1)
U = G / n[:, None]

print("samples %d   span %.1f s" % (len(G), t[-1]))
print("|g| min %.4f  max %.4f  (m/s^2)" % (n.min(), n.max()))
print()

ref = U[0]
ang = np.degrees(np.arccos(np.clip(U @ ref, -1, 1)))
print("重力方向相对首帧的偏转角:")
print("  最终 %.3f deg   最大 %.3f deg   中位 %.3f deg" %
      (ang[-1], ang.max(), np.median(ang)))
print()
print("  %-9s %-9s %-11s %-11s %-11s" % ("t(s)", "偏转(deg)", "gx", "gy", "gz"))
for tt in range(0, int(t[-1]) + 1, 60):
    i = int(np.searchsorted(t, tt))
    i = min(i, len(G) - 1)
    print("  %-9.0f %-9.3f %-11.5f %-11.5f %-11.5f" %
          (t[i], ang[i], G[i, 0], G[i, 1], G[i, 2]))
print()

# What height error does that rotation imply over the measured path?
PATH = 235.7
print("若重力方向偏 %.3f deg，沿 %.1f m 直线行进产生的 z 误差 = %.2f m" %
      (ang.max(), PATH, PATH * math.sin(math.radians(ang.max()))))
print("实测地图 z 跨度 5.71 m，同点重访最大 |dz| 2.49 m")
print()

# Is the estimate still moving at the end, or has it settled?
w = max(1, len(ang) // 20)
print("偏转角分段均值（20 段）:")
seg = [ang[i * w:(i + 1) * w].mean() for i in range(20)]
print("  " + "  ".join("%.2f" % s for s in seg))
