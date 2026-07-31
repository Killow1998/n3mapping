#!/usr/bin/env python3
"""Asks whether the estimator's attitude wanders while the robot is standing still.

A stationary accelerometer measures gravity and nothing else, so the direction it
reports in the body frame *is* the attitude, to within whatever residual
acceleration is present. The odometry reports its own attitude. While the robot
is not moving the two must agree; where they part company, the estimator has
drifted, and on 0723 it parts company by roughly eight degrees over the opening
ten seconds while the robot travels three quarters of a metre.

Header stamps are used rather than arrival stamps: the 0723 LiDAR odometry was
recorded during a replay at rate 0.5, so its arrival timeline runs at twice real
duration while its header stamps are the sensor's own.

    python3 lio_static_attitude_drift.py <lio_bag> [imu_topic] [window_s]
"""
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

BAG = sys.argv[1]
IMU_TOPIC = sys.argv[2] if len(sys.argv) > 2 else None
WINDOW_S = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
# Long enough to average the sample noise down, short enough that a real
# posture change would not be smoothed away.
SMOOTH_S = 0.2


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def rpy(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return np.degrees([roll, pitch])


def main():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(BAG)], default_typestore=ts) as reader:
        topics = {c.topic for c in reader.connections}
        imu_topic = IMU_TOPIC
        if imu_topic is None:
            candidates = [t for t in topics if t.endswith("/imu")]
            if not candidates:
                print("bag 里没有 IMU 话题: %s" % sorted(topics))
                return 2
            imu_topic = candidates[0]

        conns = [c for c in reader.connections if c.topic in (imu_topic, "/Odometry")]
        imu, odom = [], []
        t0 = None
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = stamp_s(msg)
            if t0 is None:
                t0 = t
            if t - t0 > WINDOW_S + 2.0:
                break
            if conn.topic == imu_topic:
                a = msg.linear_acceleration
                imu.append((t - t0, a.x, a.y, a.z))
            else:
                q = msg.pose.pose.orientation
                p = msg.pose.pose.position
                odom.append((t - t0, *rpy(q.w, q.x, q.y, q.z), p.x, p.y, p.z))

    if len(imu) < 50 or len(odom) < 5:
        print("样本不足: imu %d, odom %d" % (len(imu), len(odom)))
        return 2

    A = np.array(imu)
    O = np.array(odom)
    A = A[A[:, 0] <= WINDOW_S]
    O = O[O[:, 0] <= WINDOW_S]

    # True attitude change: where measured gravity points, in the body frame.
    unit = A[:, 1:4] / np.linalg.norm(A[:, 1:4], axis=1, keepdims=True)
    ref = unit[A[:, 0] < 0.5].mean(axis=0)
    ref /= np.linalg.norm(ref)
    angle = np.degrees(np.arccos(np.clip(unit @ ref, -1.0, 1.0)))
    win = max(1, int(SMOOTH_S * len(A) / max(A[-1, 0], 1e-6)))
    true_deg = np.convolve(angle, np.ones(win) / win, mode="same")

    # Reported attitude change, as the swing of roll and pitch.
    roll_swing = O[:, 1].ptp()
    pitch_swing = O[:, 2].ptp()
    reported = max(roll_swing, pitch_swing)

    travelled = float(np.linalg.norm(np.diff(O[:, 3:6], axis=0), axis=1).sum())
    acc_norm = np.linalg.norm(A[:, 1:4], axis=1)
    # Residual acceleration masquerades as tilt; this bounds how much of the
    # measured change could be motion rather than attitude.
    bound = np.degrees(np.arctan(acc_norm.std() / max(np.median(acc_norm), 1e-9)))

    print("bag        %s" % BAG)
    print("IMU 话题    %s   窗口 %.1f s   IMU %d 帧 / odom %d 帧"
          % (imu_topic, WINDOW_S, len(A), len(O)))
    print("窗口内路程  %.3f m" % travelled)
    print()
    print("  真实姿态变化(加速度计)   %6.2f deg   上界(残余加速度) %.2f deg"
          % (true_deg.max(), bound))
    print("  里程计报告的姿态摆动     %6.2f deg   (roll %.2f / pitch %.2f)"
          % (reported, roll_swing, pitch_swing))
    print("  ------------------------------------------")
    print("  差值                     %6.2f deg" % (reported - true_deg.max()))
    return 0


sys.exit(main())
