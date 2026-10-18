#!/usr/bin/env python3
"""Record SpRobo -> ROS 2 -> FAST_LIO -> n3m acquisitions.

Run on the authorized host after sourcing Humble and the n3m workspace. This
checks input completeness; coverage also checks route completion and gross map
drift against independent truth. Passing does not qualify localization or CPU.
All generated data go into a new output directory; existing sessions are unused.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import uuid

from n3mapping_sprobo_scene import session
from n3mapping_sprobo_route import RouteDriver


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def start_session(directory):
    """Use the simulator's existing owner request; never touch a desktop session."""
    def packet(kind, body):
        data = json.dumps(body).encode()
        return struct.pack("<4sHHQII", b"SRBT", 1, kind, 0, len(data), 0)+data

    connections = []
    hello = packet(1, dict(role="gui", token=uuid.uuid4().hex, subscriptions=[]))
    try:
        for name in ("control", "data"):
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(5)
            connection.connect(str(directory / (name+".sock")))
            connection.sendall(hello)
            connections.append(connection)
        connections[0].sendall(packet(2, dict(op="start")))
        # Leave the owner alive until the caller closes it after the run.
        return connections
    except BaseException:
        for connection in connections:
            connection.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=35.)
    parser.add_argument("--domain-id", type=int, default=181)
    parser.add_argument("--map", type=Path, help="Existing pbstream: run localization instead of mapping")
    parser.add_argument("--spawn", nargs=3, type=float, default=[-44, 0, 0])
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--variant", choices=["reference", "changed", "repetitive"], default="reference")
    parser.add_argument("--route", choices=["smoke", "coverage", "stationary"], default="smoke")
    parser.add_argument("--input-only", action="store_true", help="Record independent sensor/LIO queries without running n3m")
    args = parser.parse_args()
    if not 20 <= args.duration <= 2400 or not 1 <= args.domain_id <= 232:
        parser.error("duration must be 20..2400 s and domain-id 1..232")
    if args.input_only and (args.map or args.route == "coverage"):
        parser.error("input-only cannot be combined with map or coverage route")
    if args.map and args.route == "coverage":
        parser.error("coverage is a mapping acquisition route")
    args.sim_root, args.workspace, args.adapter = (
        path.resolve() for path in (args.sim_root, args.workspace, args.adapter))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(ROS_DOMAIN_ID=str(args.domain_id), ROS_LOCALHOST_ONLY="1",
                      OMP_WAIT_POLICY="PASSIVE", ROS_LOG_DIR=str(args.output / "ros_log"),
                      SIMPLE_RBT_LOG_DIR=str(args.output / "sim_log"))
    # The development node and core must come from the same build. A sourced
    # workspace may otherwise select an older installed core via LD_LIBRARY_PATH.
    os.environ["LD_LIBRARY_PATH"] = str(args.workspace / "build/n3mapping")+":"+os.environ.get("LD_LIBRARY_PATH", "")
    import rclpy
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, PointCloud2
    from livox_ros_driver2.msg import CustomMsg
    from rosgraph_msgs.msg import Clock
    from tf2_msgs.msg import TFMessage
    from std_srvs.srv import Trigger

    core = args.sim_root / "build/desktop/simple-rbt-sim"
    driver = RouteDriver() if args.route == "coverage" else None
    frontend = args.workspace / "install/fast_lio/lib/fast_lio/fastlio_mapping"
    backend = args.workspace / "build/n3mapping/n3mapping_node"
    config = session(args.variant, args.spawn, args.yaw, duration=args.duration)
    config["output"] = str(args.output / "simulation")
    config_path = args.output / "session.json"
    config_path.write_text(json.dumps(config, indent=2)+"\n")
    session_dir = args.output / "ipc"
    session_dir.mkdir()
    # The raw simulator stream uses g, matching this FAST_LIO configuration.
    # The wire unit divides SI by standard gravity (9.80665). MuJoCo defaults
    # to 9.81; FAST_LIO also normalizes the initial accelerometer mean. Retain
    # the existing MID360 extrinsics and record this override explicitly.
    commands = {
        "sim": [str(core), "--project-root", str(args.sim_root), "--session-config",
                str(config_path), "--session-dir", str(session_dir)],
        "adapter": [str(args.adapter), "--session-dir", str(session_dir)],
        "fast_lio": [str(frontend), "--ros-args", "--params-file",
                     str(args.workspace / "src/FAST_LIO/config/mid360.yaml"),
                     "-p", "use_sim_time:=true", "-p", "common.gravity_m_s2:=9.80665",
                     "-p", "pcd_save.pcd_save_en:=false", "-p", "runtime_pos_log_enable:=false"],
        "n3m": [str(backend), "--ros-args", "--params-file",
                str(args.workspace / "src/n3mapping/humble/config/n3mapping.yaml"),
                "-p", "use_sim_time:=true", "-p", "mode:=mapping",
                "-p", "map_save_path:="+str(args.output / "map"),
                "-p", "save_global_map_on_shutdown:=false"],
    }
    if args.map:
        commands["n3m"] += ["-p", "mode:=localization", "-p", "map_path:="+str(args.map.resolve())]
    if args.input_only:
        del commands["n3m"]
    identity = {name: dict(command=command, executable_sha256=file_hash(command[0]))
                for name, command in commands.items()}
    if not args.input_only:
        identity["n3m"]["core_sha256"] = file_hash(args.workspace / "build/n3mapping/libn3mapping_core.so")
    identity["inputs"] = {str(path): file_hash(path) for path in
                          (config_path, Path(__file__), Path(__file__).with_name("n3mapping_sprobo_scene.py"),
                           Path(__file__).with_name("n3mapping_sprobo_route.py"),
                           args.workspace / "src/FAST_LIO/config/mid360.yaml",
                           args.workspace / "src/n3mapping/humble/config/n3mapping.yaml")}
    if args.map:
        identity["inputs"][str(args.map.resolve())] = file_hash(args.map)
    (args.output / "commands.json").write_text(json.dumps(identity, indent=2)+"\n")
    rclpy.init()
    node = rclpy.create_node("n3m_sprobo_observer")
    counts, stamps = Counter(), {}
    poses, static_tf, lidar_sample = [], {}, {}
    sim_ns = 0
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.output / "bag"), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    sensor_qos = QoSProfile(depth=2000, reliability=ReliabilityPolicy.RELIABLE)
    clock_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
    static_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def receive(topic, cls):
        def callback(data):
            nonlocal sim_ns
            counts[topic] += 1
            writer.write(topic, data, sim_ns)
            if cls is CustomMsg and lidar_sample:
                return
            message = deserialize_message(data, cls)
            if cls is Clock:
                sim_ns = message.clock.sec*10**9+message.clock.nanosec
            elif cls is TFMessage:
                for transform in message.transforms:
                    t, q = transform.transform.translation, transform.transform.rotation
                    static_tf[transform.child_frame_id] = dict(parent=transform.header.frame_id,
                        position=[t.x, t.y, t.z], quaternion_xyzw=[q.x, q.y, q.z, q.w])
            else:
                stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
                stamps.setdefault(topic, []).append(stamp)
                if cls is Odometry:
                    p, q = message.pose.pose.position, message.pose.pose.orientation
                    poses.append([topic, stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                    if driver and topic == "/ground_truth/odom":
                        driver.observe(message, stamp)
                elif cls is CustomMsg:
                    lidar_sample.update(points=message.point_num, timebase=message.timebase,
                        stamp=stamp, max_offset_ns=max(p.offset_time for p in message.points),
                        offsets_ordered=all(a.offset_time <= b.offset_time
                                            for a, b in zip(message.points, message.points[1:])))
        return callback

    topics = [("/clock", Clock), ("/livox/lidar", CustomMsg), ("/livox/imu", Imu),
              ("/ground_truth/odom", Odometry), ("/tf_static", TFMessage),
              ("/Odometry", Odometry), ("/cloud_registered_body", PointCloud2),
              ("/n3mapping/odometry", Odometry)]
    subscriptions = []
    for topic, cls in topics:
        type_name = cls.__module__.split(".")[0]+"/msg/"+cls.__name__
        writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=type_name, serialization_format="cdr"))
        qos = static_qos if cls is TFMessage else clock_qos if cls is Clock else sensor_qos
        subscriptions.append(node.create_subscription(cls, topic, receive(topic, cls), qos, raw=True))
    publisher = node.create_publisher(Twist, "/scout_mini/cmd_vel", 10)
    processes, logs, owner = {}, [], []
    saved, error, truth_subscribers = False, None, []
    started = time.monotonic()
    try:
        for name, command in commands.items():
            log = (args.output / (name+".log")).open("w")
            logs.append(log)
            processes[name] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                cwd=args.output)
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if any(process.poll() is not None for process in processes.values()):
                raise RuntimeError("A child exited during startup; inspect its log")
            if (static_tf and node.count_subscribers("/scout_mini/cmd_vel") and
                    node.count_publishers("/cloud_registered_body") and
                    node.count_subscribers("/cloud_registered_body") >= (1 if args.input_only else 2)):
                break
        else:
            raise RuntimeError("ROS graph did not become ready")
        owner = start_session(session_dir)
        deadline = time.monotonic()+args.duration*4+30
        last_command, last_progress, stopping = 0., 0., False
        while time.monotonic() < deadline and processes["sim"].poll() is None:
            for name in (name for name in processes if name != "sim"):
                if processes[name].poll() is not None:
                    raise RuntimeError(name+" exited during the run")
            rclpy.spin_once(node, timeout_sec=.01)
            if time.monotonic()-last_command >= .05:
                elapsed = sim_ns/1e9
                command = Twist()
                if driver:
                    command.linear.x, command.angular.z = driver.command(elapsed)
                    if driver.completed_at is not None and elapsed > driver.completed_at+2 and not stopping:
                        processes["sim"].send_signal(signal.SIGINT)
                        stopping = True
                elif args.route == "smoke" and 7 <= elapsed < 22:
                    command.linear.x = .5
                elif args.route == "smoke" and 22 <= elapsed < 26:
                    command.angular.z = .3
                publisher.publish(command)
                last_command = time.monotonic()
                if driver and elapsed-last_progress >= 10:
                    print(json.dumps(dict(sim_s=elapsed, route=driver.index, total=len(driver.points),
                                          pose=driver.pose, counts=dict(counts))), flush=True)
                    last_progress = elapsed
        if processes["sim"].poll() is None:
            raise RuntimeError("Simulation exceeded its wall-time allowance")
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
        truth_subscribers = [info.node_name for info in
                             node.get_subscriptions_info_by_topic("/ground_truth/odom")]
        if not args.map and not args.input_only:
            save = node.create_client(Trigger, "/n3mapping/save_map")
            if save.wait_for_service(timeout_sec=5):
                future = save.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=30)
                saved = future.done() and future.result() is not None and future.result().success
    except Exception as exc:
        error = str(exc)
    finally:
        for connection in owner:
            connection.close()
        for process in reversed(list(processes.values())):
            stop(process)
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()
        del writer
    with (args.output / "poses.csv").open("w") as stream:
        output = csv.writer(stream)
        output.writerow(["topic", "stamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        output.writerows(poses)
    (args.output / "extrinsics.json").write_text(json.dumps(static_tf, indent=2)+"\n")
    pairs = set(stamps.get("/Odometry", [])) & set(stamps.get("/cloud_registered_body", []))
    native_path = args.output / "simulation/summary.json"
    native = json.loads(native_path.read_text()) if native_path.exists() else {}
    checks = dict(no_error=error is None,
                  simulation_exited_cleanly="sim" in processes and processes["sim"].returncode == 0,
                  raw_lidar=counts["/livox/lidar"] >= 100, raw_imu=counts["/livox/imu"] >= 2000,
                  complete_raw_lidar=counts["/livox/lidar"] == native.get("lidar_frames"),
                  complete_raw_imu=counts["/livox/imu"] == native.get("imu_samples"),
                  lio_pairs=len(pairs) >= 100,
                  complete_lio_pairs=len(pairs) == counts["/Odometry"] == counts["/cloud_registered_body"],
                  timestamps_increase=all(all(a < b for a, b in zip(values, values[1:]))
                                          for values in stamps.values()),
                  truth_isolated=truth_subscribers == ["n3m_sprobo_observer"],
                  rolling_scan=bool(lidar_sample.get("offsets_ordered")) and
                      90_000_000 < lidar_sample.get("max_offset_ns", 0) < 100_000_000)
    if not args.input_only:
        checks["backend_output"] = counts["/n3mapping/odometry"] >= 100
    if not args.map and not args.input_only:
        checks["saved_map"] = bool(saved)
    if driver:
        checks["route_completed"] = driver.completed_at is not None
        (args.output / "route.json").write_text(json.dumps(driver.report(), indent=2)+"\n")
        from n3mapping_sprobo_score import mapping_quality
        quality = mapping_quality(args.output)
        checks["mapping_geometry"] = quality["passed"]
        (args.output / "mapping_quality.json").write_text(json.dumps(quality, indent=2)+"\n")
    summary = dict(passed=all(checks.values()), checks=checks, error=error, counts=dict(counts),
                   lio_exact_stamp_pairs=len(pairs), first_lidar=lidar_sample,
                   truth_subscribers=truth_subscribers, wall_seconds=time.monotonic()-started,
                   simulation_seconds=sim_ns/1e9,
                   simulation_lidar_frames=native.get("lidar_frames"),
                   simulation_imu_samples=native.get("imu_samples"),
                   mode="input_only" if args.input_only else "localization" if args.map else "mapping",
                   limitation="Acquisition checks only; not localization accuracy or half-core qualification")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
