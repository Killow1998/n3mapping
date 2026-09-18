#!/usr/bin/env python3
"""Verify unified N3 outputs on an isolated master without robot connections."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import xmlrpc.client


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for test observation")


def run(binary, directory):
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:{}".format(port)
    os.environ["ROS_IP"] = "127.0.0.1"
    os.environ.pop("ROS_HOSTNAME", None)
    os.environ["ROS_HOME"] = str(directory / "ros")
    (directory / "logs").mkdir()
    os.environ["GLOG_log_dir"] = str(directory / "logs")
    processes = []
    with (directory / "processes.log").open("w") as log:
        def start(command):
            child = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            processes.append(child)
            return child

        def stop(child):
            os.killpg(child.pid, signal.SIGINT)
            child.wait(timeout=10)
            assert child.returncode == 0

        try:
            start(["roscore", "-p", str(port)])
            api = xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"])

            def master_ready():
                try:
                    return api.lookupNode("/probe", "/rosout")[0] == 1
                except OSError:
                    return False

            wait_until(master_ready)
            import rospy
            from nav_msgs.msg import Odometry
            from tf2_msgs.msg import TFMessage
            from sensor_msgs import point_cloud2
            from sensor_msgs.msg import PointCloud2
            from std_msgs.msg import Header, String
            rospy.init_node("n3_output_smoke", disable_signals=True)
            received, body, world, transforms, statuses = [], [], [], [], []
            rospy.Subscriber("/probe/output", Odometry, received.append, queue_size=50)
            rospy.Subscriber("/probe/body", PointCloud2, body.append, queue_size=50)
            rospy.Subscriber("/probe/world", PointCloud2, world.append, queue_size=50)
            rospy.Subscriber("/tf", TFMessage, lambda m: transforms.extend(m.transforms), queue_size=50)
            rospy.Subscriber("/probe/status", String,
                             lambda m: statuses.append(json.loads(m.data)), queue_size=50)
            cloud_pub = rospy.Publisher("/probe/cloud", PointCloud2, queue_size=100)
            odom_pub = rospy.Publisher("/probe/odom", Odometry, queue_size=100)
            points = [(0.1*x, 0.1*y, -0.4) for x in range(-10, 11) for y in range(-10, 11)]

            def node_for(mode, extra=()):
                node = start([binary, "__name:=n3_output_probe", "_mode:=" + mode,
                    "_map_save_path:=" + str(directory / mode),
                    "_cloud_topic:=/probe/cloud", "_odom_topic:=/probe/odom",
                    "_output_odom_topic:=/probe/output", "_output_status_topic:=/probe/status",
                    "_output_cloud_body_topic:=/probe/body", "_output_cloud_world_topic:=/probe/world",
                    "_world_frame:=map", "_body_frame:=base_link",
                    "_input_linear_velocity_frame:=parent", *extra])
                wait_until(lambda: cloud_pub.get_num_connections() and odom_pub.get_num_connections())
                return node

            def publish(velocity=0.0, stamp=None, cloud_only=False, odom_only=False, x=0.0):
                stamp = rospy.Time.now() if stamp is None else stamp
                odom = Odometry()
                odom.header = Header(stamp=stamp, frame_id="camera_init")
                odom.child_frame_id = "base_link"
                odom.pose.pose.position.x = x
                odom.pose.pose.orientation.w = math.sqrt(0.5)
                odom.pose.pose.orientation.z = math.sqrt(0.5)
                odom.twist.twist.linear.x = velocity
                odom.twist.twist.angular.z = 0.3
                odom.twist.covariance[0] = 4.0
                odom.twist.covariance[7] = 1.0
                if not cloud_only:
                    odom_pub.publish(odom)
                if not odom_only:
                    cloud_pub.publish(point_cloud2.create_cloud_xyz32(
                        Header(stamp=stamp, frame_id="base_link"), points))
                return stamp

            node = node_for("mapping", ("_mapping_static_start_guard_enable:=true",))
            for _ in range(10):
                publish()
                time.sleep(0.1)
            wait_until(lambda: statuses and body)
            assert not received and not world
            assert statuses[-1]["state"] == "initializing"
            for _ in range(25):
                publish()
                time.sleep(0.1)
            wait_until(lambda: received and world and statuses[-1]["state"] == "tracking")
            stamp = publish(1.0)
            wait_until(lambda: received[-1].header.stamp == stamp)
            output = received[-1]
            assert output.header.frame_id == "map" and output.child_frame_id == "base_link"
            assert abs(output.twist.twist.linear.x) < 1e-9
            assert abs(output.twist.twist.linear.y + 1.0) < 1e-9
            assert abs(output.twist.twist.angular.z - 0.3) < 1e-9
            assert abs(output.twist.covariance[0] - 1.0) < 1e-9
            # Backend completion is reported by the next real input, never by
            # inventing a new estimate timestamp for a completed old scan.
            deadline = time.monotonic() + 10.0
            while statuses[-1]["correction_stamp"] + 1e-6 < stamp.to_sec() and time.monotonic() < deadline:
                publish(odom_only=True)
                time.sleep(0.02)
            assert statuses[-1]["correction_stamp"] + 1e-6 >= stamp.to_sec()
            # Exact paired fixture stamps let us check snapshot coherence while
            # the backend and realtime callback run on different threads.
            # Epoch-second doubles cannot represent nanosecond precision.
            incoherent = [(status["correction_stamp"], status["observation_stamp"], status["stamp"])
                          for status in statuses
                          if not (0.0 <= status["correction_stamp"] <= status["observation_stamp"] + 1e-6
                                  and status["observation_stamp"] <= status["stamp"] + 1e-6)]
            assert not incoherent, incoherent[:3]
            observation = statuses[-1]["observation_stamp"]
            correction = statuses[-1]["correction_stamp"]
            assert all(t.child_frame_id != "base_link" for t in transforms)
            published = {entry[0] for entry in api.getPublishedTopics("/probe", "")[2]}
            assert "/probe/output/local" not in published and "/probe/output/correction" not in published

            scan_stamp = publish(0.2, odom_only=True, x=0.02)
            time.sleep(0.03)
            newest = publish(0.2, odom_only=True, x=0.04)
            wait_until(lambda: received[-1].header.stamp == newest)
            assert abs(received[-1].pose.pose.position.x - 0.04) < 1e-6
            publish(stamp=scan_stamp, cloud_only=True)
            wait_until(lambda: world[-1].header.stamp == scan_stamp)
            first = next(point_cloud2.read_points(world[-1], field_names=("x", "y", "z")))
            assert abs(first[0] - 1.02) < 1e-5 and abs(first[1] + 1.0) < 1e-5

            time.sleep(0.65)
            fresh = publish(odom_only=True, x=0.04)
            wait_until(lambda: received[-1].header.stamp == fresh and statuses[-1]["stamp"] == fresh.to_sec())
            # The delayed scan can update the backend; capture its actual source stamp.
            assert any(abs(statuses[-1]["observation_stamp"] - expected) <= 1e-6
                       for expected in (observation, scan_stamp.to_sec()))
            assert any(abs(statuses[-1]["correction_stamp"] - expected) <= 1e-6
                       for expected in (correction, scan_stamp.to_sec()))
            assert statuses[-1]["state"] == "tracking"
            invalid = publish(float("nan"), odom_only=True)
            wait_until(lambda: statuses[-1]["stamp"] == invalid.to_sec())
            assert statuses[-1]["state"] == "error" and received[-1].header.stamp < invalid
            fresh = publish(1.0, odom_only=True)
            wait_until(lambda: received[-1].header.stamp == fresh)
            stop(node)
            saved = directory / "mapping/n3map.pbstream"
            assert saved.is_file() and saved.stat().st_size > 0
            wait_until(lambda: not odom_pub.get_num_connections())

            for mode, expected in (("localization", "localizing"), ("map_extension", "initializing")):
                received.clear()
                statuses.clear()
                body.clear()
                world.clear()
                node = node_for(mode, ("_map_path:=" + str(directory / "absent.pbstream"),))
                for _ in range(3):
                    stamp = publish()
                    time.sleep(0.1)
                wait_until(lambda: statuses and body and statuses[-1]["stamp"] == stamp.to_sec())
                assert statuses[-1]["state"] == expected and not received and not world
                stamp = publish(odom_only=True)
                wait_until(lambda: statuses[-1]["stamp"] == stamp.to_sec())
                assert statuses[-1]["correction_stamp"] == 0.0
                stop(node)
                wait_until(lambda: not odom_pub.get_num_connections())
            rospy.signal_shutdown("fixture complete")
            print("Noetic smoke passed: unified outputs, mapping/save, scan-time pairing, source times, "
                  "retained anchor, velocity/covariance, invalid input, localization/extension initialization")
        except Exception as exception:
            log.flush()
            detail = (directory / "processes.log").read_text(errors="replace")[-6000:]
            raise RuntimeError("{}\\n{}".format(exception, detail)) from exception
        finally:
            for child in reversed(processes):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("node", help="Noetic n3mapping_node executable")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="n3-output-smoke-") as temporary:
        run(str(Path(args.node).resolve()), Path(temporary))
