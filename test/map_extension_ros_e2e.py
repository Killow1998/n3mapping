#!/usr/bin/env python3
"""Native localization -> extension -> save/reload without restarting the LIO source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
import xmlrpc.client

import yaml


def room_cloud(x):
    faces = [(0,-6,-4,4,0,3.5),(0,8,-4,4,0,3.5),
             (1,-4,0,3.5,-6,8),(1,4,0,3.5,-6,8),
             (2,0,-6,8,-4,4),(2,3.5,-6,8,-4,4),
             (0,0,-4,-.6,0,3.5),(0,0,.6,4,0,3.5),(0,0,-.6,.6,2.3,3.5),
             (0,-4,-2.5,-1.5,0,2.1),(1,-1.5,0,2.1,-5,-4),
             (0,5,1,3,0,2.7),(1,1,0,2.7,5,7)]
    origin = (x, 0., 1.)
    points = []
    for pitch in range(-35, 56, 5):
        p = math.radians(pitch)
        for bearing in range(0, 360, 2):
            h = math.radians(bearing)
            ray = (math.cos(p)*math.cos(h), math.cos(p)*math.sin(h), math.sin(p))
            nearest = 30.
            for axis, bound, low_a, high_a, low_b, high_b in faces:
                if abs(ray[axis]) < 1e-8:
                    continue
                distance = (bound-origin[axis])/ray[axis]
                a, b = (axis+1)%3, (axis+2)%3
                if .01 < distance < nearest and low_a <= origin[a]+distance*ray[a] <= high_a and low_b <= origin[b]+distance*ray[b] <= high_b:
                    nearest = distance
            if nearest < 30.:
                points.append(struct.pack('<4f', *(nearest*v for v in ray), 1.))
    return b''.join(points)


def run(args, root):
    os.environ.update(ROS_HOME=str(root/'ros'), ROS_LOG_DIR=str(root/'ros_logs'),
                      ROS_LOCALHOST_ONLY='1', N3MAPPING_LOG_DIR=str(root/'logs'))
    children = []
    log = (root/'processes.log').open('w')
    def launch(command):
        child = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        children.append(child)
        return child
    def stop(child):
        child.send_signal(signal.SIGINT)
        assert child.wait(timeout=30) == 0
    try:
        if args.ros == 'noetic':
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_IP='127.0.0.1')
            os.environ.pop('ROS_HOSTNAME', None)
            launch(['roscore', '-p', str(port)])
            api = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
            for _ in range(150):
                try:
                    if api.getPid('/extension_probe')[0] == 1: break
                except OSError:
                    pass
                time.sleep(.02)
            import rospy as ros
            import rosparam
            ros.init_node('extension_probe', disable_signals=True)
        else:
            import rclpy as ros
            ros.init()
            observer = ros.create_node('extension_probe')
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import String, UInt32
        from std_srvs.srv import Trigger
        latest, truth, errors, locks = {}, {}, [], []
        global_maps = []
        current_x, transformed = -4., False
        def receive_status(msg):
            latest.clear(); latest.update(json.loads(msg.data))
        def stamp_ns(stamp):
            return stamp.to_nsec() if args.ros == 'noetic' else stamp.sec*10**9+stamp.nanosec
        def receive_pose(msg):
            expected = truth.get(stamp_ns(msg.header.stamp))
            if expected is not None:
                p = msg.pose.pose.position
                errors.append(math.sqrt((p.x-expected)**2+p.y*p.y+(p.z-1.)**2))
        if args.ros == 'noetic':
            subscriptions = [ros.Subscriber('/switch/status', String, receive_status, queue_size=100),
                             ros.Subscriber('/switch/output', Odometry, receive_pose, queue_size=100),
                             ros.Subscriber('/n3mapping/relocalization_lock', UInt32, lambda m: locks.append(m.data), queue_size=10)]
            odom_pub = ros.Publisher('/switch/odom', Odometry, queue_size=100)
            cloud_pub = ros.Publisher('/switch/cloud', PointCloud2, queue_size=10)
        else:
            subscriptions = [observer.create_subscription(String, '/switch/status', receive_status, 100),
                             observer.create_subscription(Odometry, '/switch/output', receive_pose, 100),
                             observer.create_subscription(UInt32, '/n3mapping/relocalization_lock', lambda m: locks.append(m.data), 10)]
            odom_pub = observer.create_publisher(Odometry, '/switch/odom', 100)
            cloud_pub = observer.create_publisher(PointCloud2, '/switch/cloud', 10)
        def emit_odom():
            msg = Odometry()
            msg.header.stamp = ros.Time.now() if args.ros == 'noetic' else observer.get_clock().now().to_msg()
            msg.header.frame_id = 'lio_odom'; msg.child_frame_id = 'lidar'
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            if transformed:
                dx, dy = current_x-12., 7.
                p.x = math.cos(.7)*dx+math.sin(.7)*dy
                p.y = -math.sin(.7)*dx+math.cos(.7)*dy
                p.z = .7; q.z = math.sin(-.35); q.w = math.cos(-.35)
            else:
                p.x = current_x; p.z = 1.; q.w = 1.
            truth[stamp_ns(msg.header.stamp)] = current_x
            odom_pub.publish(msg)
            return msg.header.stamp
        def wait(predicate, timeout=20.):
            end = time.monotonic()+timeout
            while not predicate():
                assert time.monotonic() < end, ('timeout', latest)
                emit_odom()
                if args.ros == 'humble': ros.spin_once(observer, timeout_sec=.005)
                else: time.sleep(.005)
        def call(name):
            if args.ros == 'noetic':
                ros.wait_for_service(name, timeout=5)
                return ros.ServiceProxy(name, Trigger)()
            client = observer.create_client(Trigger, name)
            assert client.wait_for_service(timeout_sec=5), name
            future = client.call_async(Trigger.Request())
            wait(future.done)
            response = future.result(); observer.destroy_client(client)
            return response
        def start(mode, source, destination):
            previous_session = latest.get('projection', {}).get('session_id')
            latest.clear(); locks.clear(); errors.clear()
            settings = dict(mode=mode, map_path=str(source), map_save_path=str(destination),
                            num_threads=4, mapping_static_start_guard_enable=False,
                            save_global_map_on_shutdown=False,
                            cloud_topic='/switch/cloud', odom_topic='/switch/odom',
                            output_status_topic='/switch/status', output_odom_topic='/switch/output',
                            world_frame='map', body_frame='lidar')
            if args.ros == 'noetic':
                rosparam.upload_params('/switch_node', settings)
                node = launch([str(args.node), '__name:=switch_node'])
            else:
                parameters = root/(mode+'_'+destination.name+'.yaml')
                parameters.write_text(yaml.safe_dump({'/**': {'ros__parameters': settings}}))
                node = launch([str(args.node), '--ros-args', '-r', '__node:=switch_node', '--params-file', str(parameters)])
            connected = 'get_num_connections' if args.ros == 'noetic' else 'get_subscription_count'
            wait(lambda: getattr(odom_pub,connected)()>0 and getattr(cloud_pub,connected)()>0
                 and latest.get('projection', {}).get('session_id') not in (None, previous_session))
            return node
        def frame(x):
            nonlocal current_x
            # Keep authored translation steps below the normal odometry speed gate.
            time.sleep(.2); current_x = x
            msg = PointCloud2(); msg.header.stamp = emit_odom(); msg.header.frame_id = 'lidar'
            msg.data = room_cloud(x); msg.height=1; msg.width=len(msg.data)//16
            msg.fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z','intensity'])]
            msg.point_step=16; msg.row_step=len(msg.data); msg.is_dense=True
            cloud_pub.publish(msg); stamp=stamp_ns(msg.header.stamp)/1e9
            wait(lambda: abs(latest.get('observation_stamp',0.)-stamp)<1e-5)
        original, extended = root/'original', root/'extended'
        node = start('mapping', '', original)
        for x in [-4.,-3.,-2.]: frame(x)
        assert call('/n3mapping/save_map').success
        stop(node)
        source = original/'n3map.pbstream'; source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        source_atlas = Path(str(source)+'.localization_atlas.pb')
        assert source_atlas.stat().st_size > 0
        source_atlas.unlink()  # Exercise first-load preparation of an older map.
        current_x, transformed = -3., True
        node = start('localization', source, extended)
        denied = call('/n3mapping/start_map_extension')
        assert not denied.success and denied.message=='localization_not_authoritative', denied
        assert not call('/n3mapping/save_map').success
        for _ in range(24):
            frame(-3.)
            if latest.get('state')=='tracking': break
        assert latest.get('state')=='tracking',latest
        atlas_sha = hashlib.sha256(source_atlas.read_bytes()).hexdigest()
        wait(lambda:len(locks)==1)
        # Subscribe only after localization: headless startup must not be needed
        # to populate the map, and a later subscriber must receive the latched map.
        def subscribe_map(receive):
            if args.ros == 'noetic':
                return ros.Subscriber('/n3mapping/global_map', PointCloud2, receive, queue_size=1)
            from rclpy.qos import QoSProfile, DurabilityPolicy
            return observer.create_subscription(PointCloud2, '/n3mapping/global_map', receive,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        subscriptions.append(subscribe_map(lambda msg: global_maps.append(bytes(msg.data))))
        wait(lambda: len(global_maps) > 0)
        assert global_maps[-1]
        late_maps = []
        subscriptions.append(subscribe_map(lambda msg: late_maps.append(bytes(msg.data))))
        wait(lambda: len(late_maps) > 0)
        assert late_maps[-1] == global_maps[-1]
        initial_global_map = global_maps[-1]
        session = latest['projection']['session_id']
        started=time.monotonic(); switched=call('/n3mapping/start_map_extension')
        switch_ms=(time.monotonic()-started)*1000
        assert switched.success, switched
        assert call('/n3mapping/start_map_extension').success
        for x in [-3.,-2.,-1.,0.,1.,0.,-1.,-2.,-3.]:
            frame(x)
            assert latest.get('mode')=='map_extension',latest
            assert latest['projection']['session_id']==session
            assert 'relocalization_performance' not in latest,latest
        assert node.poll() is None
        assert len(locks)==1,locks
        assert errors and max(errors)<.1,max(errors,default=-1)
        assert call('/n3mapping/save_map').success
        wait(lambda: global_maps[-1] != initial_global_map)
        maximum_error=max(errors)
        stop(node)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
        assert hashlib.sha256(source_atlas.read_bytes()).hexdigest()==atlas_sha
        assert (extended/'n3map.pbstream.localization_atlas.pb').stat().st_size > 0
        node=start('localization',extended/'n3map.pbstream',root/'reload_readonly')
        for _ in range(24):
            frame(-3.)
            if latest.get('state')=='tracking': break
        assert latest.get('state')=='tracking',latest
        assert errors and max(errors)<.1,max(errors,default=-1)
        stop(node)
        result=dict(ros=args.ros,switch_ms=switch_ms,extension_max_position_error_m=maximum_error,
                    reload_max_position_error_m=max(errors),original_map_sha256=source_sha,
                    restart_at_switch=False,additional_lock_events=0,
                    atlas_enabled_by_default=True,
                    missing_atlas_created_at_first_load=True,
                    late_global_map_received=True, extension_global_map_updated=True)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros',choices=['humble','noetic'],required=True)
    parser.add_argument('--node',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.output or Path(tempfile.mkdtemp(prefix='n3mapping-extension-switch-'))
    root.mkdir(parents=True,exist_ok=True)
    try: run(args,root)
    except BaseException:
        print('Preserved failed E2E:',root)
        print((root/'processes.log').read_text()[-6000:])
        raise
    if not args.output: shutil.rmtree(root)

if __name__=='__main__': main()
