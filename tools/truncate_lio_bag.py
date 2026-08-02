#!/usr/bin/env python3
"""Cuts a LiDAR-odometry bag to start where another one does.

Comparing the still-start fix against the original means comparing two maps, and
the two runs currently cover different routes: the fixed one begins 34.7 s later
because it suppressed the stationary opening. Anything measured along the path --
the residual profile, the band span -- is not comparable across them, which is
what made the first verdict on the fix wrong.

Truncating the original to the same first stamp leaves the two runs covering the
same route, with whether the estimator ran through the stationary opening as the
only difference between them.

    python3 truncate_lio_bag.py <source_bag> <reference_bag> <out_bag>
"""
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

SRC, REF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
KEEP = ("/cloud_registered_body", "/Odometry")
ts = get_typestore(Stores.ROS2_HUMBLE)


def header_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


# Where the reference run starts, in sensor time.
with AnyReader([Path(REF)], default_typestore=ts) as reader:
    conns = [c for c in reader.connections if c.topic == "/Odometry"]
    cut_ns = None
    for conn, _, raw in reader.messages(connections=conns):
        cut_ns = header_ns(reader.deserialize(raw, conn.msgtype))
        break
if cut_ns is None:
    print("reference bag has no /Odometry")
    sys.exit(2)
print("cut at header stamp %d" % cut_ns)

kept = {t: 0 for t in KEEP}
dropped = {t: 0 for t in KEEP}
with AnyReader([Path(SRC)], default_typestore=ts) as reader, Writer(Path(OUT)) as writer:
    conns = [c for c in reader.connections if c.topic in KEEP]
    out_conns = {
        c.topic: writer.add_connection(c.topic, c.msgtype, typestore=ts)
        for c in conns
    }
    for conn, stamp, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        if header_ns(msg) < cut_ns:
            dropped[conn.topic] += 1
            continue
        writer.write(out_conns[conn.topic], stamp, raw)
        kept[conn.topic] += 1

for t in KEEP:
    print("  %-26s kept %6d  dropped %5d" % (t, kept[t], dropped[t]))
