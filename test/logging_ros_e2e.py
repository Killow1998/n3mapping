#!/usr/bin/env python3
"""Real-node logging fault and live-output test; no robot or external master."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time


def run(args, root):
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ['ROS_HOME'] = str(root / 'ros')
    os.environ['ROS_LOG_DIR'] = str(root / 'ros_logs')
    os.environ['GLOG_log_dir'] = str(root / 'old_glog')
    os.environ['N3MAPPING_LOG_DIR'] = str(root / 'managed')
    os.environ['N3MAPPING_LOG_TOTAL_BYTES'] = '1048576'
    for name in ['managed', 'old_glog']:
        (root / name).mkdir()
    if not args.baseline:
        (root / 'managed/runtime.log').mkdir()  # Real EISDIR, not a mocked sink.
    children = []
    stop_spin = threading.Event()
    ros = None
    with (root / 'process.log').open('w') as output:
        def start(command):
            child = subprocess.Popen(command, stdout=output, stderr=output, start_new_session=True)
            children.append(child)
            return child

        try:
            if args.ros == 'noetic':
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:' + str(port)
                os.environ['ROS_IP'] = '127.0.0.1'
                start(['roscore', '-p', str(port)])
                import xmlrpc.client
                api = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
                for _ in range(150):
                    try:
                        if api.getPid('/logging_probe')[0] == 1:
                            break
                    except OSError:
                        pass
                    time.sleep(.02)
                import rospy as ros
                ros.init_node('logging_observer', disable_signals=True)
            else:
                import rclpy as ros
                ros.init()
                observer = ros.create_node('logging_observer')
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import PointCloud2, PointField
            from std_msgs.msg import String
            from std_srvs.srv import Trigger
            sent, received, health, poses, statuses = {}, {}, [], [], []
            lock = threading.Lock()

            def stamp_ns(stamp):
                return stamp.to_nsec() if args.ros == 'noetic' else stamp.sec * 10**9 + stamp.nanosec

            def receive(message):
                with lock:
                    received[stamp_ns(message.header.stamp)] = time.monotonic()
                    p = message.pose.pose.position
                    poses.append((p.x, p.y, p.z))

            def receive_health(message):
                health.append(json.loads(message.data))

            def receive_status(message):
                statuses.append(json.loads(message.data))

            if args.ros == 'noetic':
                ros.Subscriber('/logging_test/status', String, receive_status, queue_size=100)
                ros.Subscriber('/logging_test/output', Odometry, receive, queue_size=100)
                ros.Subscriber('/n3mapping/logging_status', String, receive_health, queue_size=10)
                odom_pub = ros.Publisher('/logging_test/odom', Odometry, queue_size=100)
                cloud_pub = ros.Publisher('/logging_test/cloud', PointCloud2, queue_size=100)
            else:
                observer.create_subscription(String, '/logging_test/status', receive_status, 100)
                observer.create_subscription(Odometry, '/logging_test/output', receive, 100)
                observer.create_subscription(String, '/n3mapping/logging_status', receive_health, 10)
                odom_pub = observer.create_publisher(Odometry, '/logging_test/odom', 100)
                cloud_pub = observer.create_publisher(PointCloud2, '/logging_test/cloud', 100)
                def spin():
                    while not stop_spin.is_set():
                        ros.spin_once(observer, timeout_sec=.005)
                spin_thread = threading.Thread(target=spin)
                spin_thread.start()
            settings = {
                'mode': 'mapping', 'mapping_static_start_guard_enable': 'false',
                'cloud_topic': '/logging_test/cloud', 'odom_topic': '/logging_test/odom',
                'output_status_topic': '/logging_test/status',
                'output_odom_topic': '/logging_test/output', 'world_frame': 'map', 'body_frame': 'body',
                'reloc_debug_enable': 'true', 'map_save_path': str(root / 'map'),
                'reloc_debug_path': str(root / 'managed/relocalization_debug.jsonl'),
                'loop_debug_enable': 'true', 'loop_debug_path': str(root / 'managed/loop_debug.jsonl'),
            }
            command = [args.node]
            if args.ros == 'noetic':
                command += ['__name:=n3_logging_probe'] + ['_' + k + ':=' + v for k, v in settings.items()]
            else:
                command += ['--ros-args', '-r', '__node:=n3_logging_probe']
                for k, v in settings.items():
                    command += ['-p', k + ':=' + v]
            node_started = time.monotonic()
            node = start(command)
            cloud = PointCloud2()
            cloud.height, cloud.width = 1, 441
            cloud.fields = [PointField(name=name, offset=4*i, datatype=7, count=1)
                            for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
            cloud.point_step, cloud.row_step = 16, 441 * 16
            cloud.is_dense = True
            cloud.data = b''.join(struct.pack('<ffff', x*.1, y*.1, -.5, 0.)
                                  for x in range(-10, 11) for y in range(-10, 11))

            def publish(index, measure=False, invalid=False):
                message = Odometry()
                ns = time.time_ns()
                if args.ros == 'noetic':
                    message.header.stamp = ros.Time(ns // 10**9, ns % 10**9)
                else:
                    message.header.stamp.sec, message.header.stamp.nanosec = divmod(ns, 10**9)
                message.header.frame_id, message.child_frame_id = 'odom', 'body'
                message.pose.pose.orientation.w = 1.0
                if invalid:
                    message.twist.twist.linear.x = float("nan")
                if measure:
                    with lock:
                        sent[ns] = time.monotonic()
                odom_pub.publish(message)
                if index % 20 == 0:
                    cloud.header.stamp = message.header.stamp
                    cloud.header.frame_id = 'body'
                    cloud_pub.publish(cloud)
                return ns

            # Input starts during discovery and initialization. Waiting for a
            # match or health first hides reliable-history startup stalls.
            # Establish a real trusted correction before measuring delivery.
            for i in range(600):
                if node.poll() is not None:
                    raise AssertionError('node exited during initialization')
                publish(i)
                if received and (args.baseline or health):
                    break
                time.sleep(.005)
            assert received, 'no authorized output with continuous startup input'
            first_output_s = min(received.values()) - node_started
            if not args.baseline:
                assert health and health[-1]['write_failures'] > 0, health

            ticks = os.sysconf('SC_CLK_TCK')
            def cpu():
                fields = Path('/proc/{}/stat'.format(node.pid)).read_text().split()
                return (int(fields[13]) + int(fields[14])) / ticks
            cpu_start = cpu()
            begin = time.monotonic()
            samples = int(args.seconds * 200)
            for i in range(samples):
                delay = begin + i / 200. - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                publish(i, True)
            elapsed = time.monotonic() - begin
            cpu_used = cpu() - cpu_start
            time.sleep(.1)
            with lock:
                latencies = sorted((received[k] - v) * 1000. for k, v in sent.items() if k in received)
                ratio = len(latencies) / len(sent)
            assert ratio >= .95, ('delivery ratio', ratio)
            assert latencies[int(.99 * (len(latencies)-1))] < 50., ('p99 ms', latencies[-10:])
            assert max(abs(v) for p in poses for v in p) < 1e-6, 'pose changed under logging load'
            if not args.baseline:
                (root / 'managed/runtime.log').rmdir()
                invalid_stamp = publish(1, invalid=True)
                deadline = time.monotonic() + 2
                while not any(item.get('projection', {}).get('odom_stamp_nsec') == invalid_stamp
                              for item in statuses) and time.monotonic() < deadline:
                    time.sleep(.005)
                invalid_status = next(item for item in statuses
                                      if item.get('projection', {}).get('odom_stamp_nsec') == invalid_stamp)
                assert invalid_status['reason'] == 'invalid_twist_or_body_frame'
                assert invalid_stamp not in received
                # Valid odometry alone remains blocked until a new paired scan.
                blocked_stamp = publish(1)
                time.sleep(.05)
                assert blocked_stamp not in received
                publish(0)
                for _ in range(200):
                    recovered_stamp = publish(1)
                    time.sleep(.01)
                    if recovered_stamp in received:
                        break
                assert recovered_stamp in received, 'paired recovery failed'
                if args.ros == 'noetic':
                    ros.wait_for_service('/n3mapping/save_map', timeout=5)
                    assert ros.ServiceProxy('/n3mapping/save_map', Trigger)().success
                else:
                    client = observer.create_client(Trigger, '/n3mapping/save_map')
                    assert client.wait_for_service(timeout_sec=5)
                    future = client.call_async(Trigger.Request())
                    deadline = time.monotonic() + 5
                    while not future.done() and time.monotonic() < deadline:
                        time.sleep(.01)
                    assert future.done() and future.result().success
                time.sleep(.3)
            os.killpg(node.pid, signal.SIGINT)
            node.wait(timeout=15)
            assert node.returncode == 0, node.returncode
            first_status = None
            if not args.baseline:
                first_status = json.loads((root/'managed/logger_health.json').read_text())
                assert first_status['write_failures'] > 0
                # Restart the real executable with the same instance catalog.
                with lock:
                    received.clear()
                health.clear()
                restart = start(command)
                for i in range(600):
                    restarted_stamp = publish(i)
                    assert restart.poll() is None
                    time.sleep(.005)
                    if restarted_stamp in received and health:
                        break
                assert health, 'restarted live worker did not publish health'
                assert restarted_stamp in received, 'no fresh output after restart'
                os.killpg(restart.pid, signal.SIGINT)
                restart.wait(timeout=15)
                assert restart.returncode == 0
            result = {'schema': 'n3mapping_logging_e2e_v1', 'ros': args.ros,
                      'baseline': args.baseline, 'startup_input': 'continuous_before_ready',
                      'first_output_s': first_output_s, 'sent': len(sent), 'received': len(latencies),
                      'delivery_ratio': ratio, 'p50_ms': latencies[len(latencies)//2],
                      'p99_ms': latencies[int(.99*(len(latencies)-1))], 'max_ms': latencies[-1],
                      'node_cpu_s': cpu_used, 'wall_s': elapsed, 'average_cores': cpu_used/elapsed}
            if not args.baseline:
                native_logs = list((root/'old_glog').rglob('*')) + [
                    p for p in (root/'ros_logs').rglob('*')
                    if p.name.startswith(('n3_logging_probe', 'n3mapping_node'))]
                unmanaged_node_bytes = sum(p.stat().st_size for p in native_logs if p.is_file())
                assert unmanaged_node_bytes == 0, native_logs
                catalog = json.loads((root / 'managed/catalog.json').read_text())
                managed = list((root / 'managed').glob('*'))
                for entry in catalog['paths']:
                    path = Path(entry)
                    managed += [path] + list(path.parent.glob(path.name + '.n3m.*'))
                total = sum(p.stat().st_size for p in set(managed) if p.is_file())
                assert total <= 1048576, total
                records = [json.loads(line) for p in (root/'managed').glob('events.jsonl*')
                           for line in p.read_text().splitlines()]
                events = {record.get('event') for record in records}
                assert {'process_start', 'process_stop', 'backend_state', 'live_state',
                        'map_save_success', 'aggregate'} <= events, events
                assert any(record.get('event') == 'live_state' and
                           'to=invalid_twist_or_body_frame' in record.get('detail', '') for record in records)
                status = json.loads((root/'managed/logger_health.json').read_text())
                assert status['written'] > 0, status
                assert sum(record.get('event') == 'process_start' for record in records) == 2
                assert any(record.get('event') == 'logger_health' and record.get('write_failures', 0) > 0 for record in records)
                assert (root/'managed/runtime.log').stat().st_size > 0
                result.update(unmanaged_node_log_bytes=unmanaged_node_bytes, managed_bytes=total, logging_status=first_status, restarted_logging_status=status, events=sorted(events))
            print(json.dumps(result))
            if args.result:
                Path(args.result).write_text(json.dumps(result, indent=2) + '\n')
        except Exception:
            output.flush()
            print((root/'process.log').read_text(errors='replace')[-6000:])
            if 'statuses' in locals():
                print(json.dumps({'last_statuses': statuses[-5:], 'last_odom_statuses': [item for item in statuses if item.get('projection', {}).get('kind') == 'odom'][-5:], 'logging_health': health[-2:]}))
            for path in (root/'managed').glob('events.jsonl*'):
                if path.is_file():
                    print(path.name, '\n'.join(line for line in path.read_text(errors='replace').splitlines() if '"event":"aggregate"' not in line)[-12000:])
            raise
        finally:
            stop_spin.set()
            if 'spin_thread' in locals():
                spin_thread.join(timeout=2)
            for child in reversed(children):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            if ros:
                if args.ros == 'noetic':
                    ros.signal_shutdown('test done')
                else:
                    observer.destroy_node()
                    ros.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros', choices=['humble', 'noetic'], required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--result')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='n3-logging-e2e-') as temp:
        run(args, Path(temp))
