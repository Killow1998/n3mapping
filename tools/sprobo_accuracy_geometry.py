"""Geometry helpers for offline SpRobo accuracy diagnosis.

Native poses and authored boxes are diagnostic inputs, never estimator inputs.
The native CSV must be independently checked against the recorded truth stream.
"""
from pathlib import Path
import sqlite3

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pose_matrix(position, rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = position
    return pose


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def read_binary_pcd(path):
    with Path(path).open("rb") as stream:
        header = {}
        while line := stream.readline():
            words = line.split()
            if words:
                header[words[0]] = words[1:]
            if line.startswith(b"DATA "):
                expected = {
                    b"FIELDS": [b"x", b"y", b"z", b"intensity"],
                    b"SIZE": [b"4"]*4, b"TYPE": [b"F"]*4,
                    b"COUNT": [b"1"]*4, b"DATA": [b"binary"]}
                if any(header.get(key) != value for key, value in expected.items()):
                    raise ValueError("Expected manifest binary XYZI float32 PCD: " + str(path))
                cloud = np.frombuffer(stream.read(), dtype="<f4").reshape(-1, 4).copy()
                if len(cloud) != int(header[b"POINTS"][0]) or not np.isfinite(cloud).all():
                    raise ValueError("Invalid PCD point payload: " + str(path))
                return cloud
    raise ValueError("Missing PCD data header: " + str(path))


class NativeTrajectory:
    """Use the same 5 ms pose interpolation as the rolling-scan simulator."""

    def __init__(self, path, extrinsics):
        rows = np.genfromtxt(path, delimiter=",", names=True)
        self.times = rows["stamp_ns"] / 1e9
        if len(rows) < 2 or not np.allclose(np.diff(self.times), .005, rtol=0, atol=1e-9):
            raise ValueError("Expected complete native poses at a 5 ms period")
        base_position = np.column_stack([rows[key] for key in ("bx", "by", "bz")])
        if not np.isfinite(base_position).all():
            raise ValueError("Native trajectory contains non-finite positions")
        base_rotation = Rotation.from_quat(
            np.column_stack([rows[key] for key in ("bqx", "bqy", "bqz", "bqw")]))
        self.positions, self.rotations = {}, {}
        for name in ("imu_frame", "livox_frame"):
            mount = extrinsics[name]
            if mount["parent"] != "base_link":
                raise ValueError("Expected explicit base_link sensor extrinsics")
            self.positions[name] = base_position + base_rotation.apply(mount["position"])
            rotation = base_rotation * Rotation.from_quat(mount["quaternion_xyzw"])
            self.rotations[name] = Slerp(self.times, rotation)

    def at(self, stamps_ns, frame):
        times = np.atleast_1d(stamps_ns) / 1e9
        # Slerp rejects out-of-range inputs; do not extrapolate future truth.
        rotations = self.rotations[frame](times)
        positions = np.column_stack([
            np.interp(times, self.times, self.positions[frame][:, axis])
            for axis in range(3)])
        return positions, rotations

    def deskew(self, points, stamps_ns, end_ns):
        positions, rotations = self.at(stamps_ns, "livox_frame")
        world = rotations.apply(points) + positions
        end_position, end_rotation = self.at(end_ns, "imu_frame")
        return end_rotation[0].inv().apply(world - end_position[0])


class RawScans:
    """Read recorded Livox scans without starting ROS or subscribing to truth."""

    def __init__(self, bag, msg_dir):
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg

        files = list(Path(bag).glob("*.db3"))
        if len(files) != 1:
            raise ValueError("This diagnostic expects one SQLite bag file")
        self.connection = sqlite3.connect(files[0].resolve().as_uri() + "?mode=ro", uri=True)
        row = self.connection.execute(
            "SELECT id FROM topics WHERE name='/livox/lidar'").fetchone()
        if row is None:
            raise ValueError("Bag is missing /livox/lidar")
        self.topic = row[0]
        self.types = get_typestore(Stores.ROS2_HUMBLE)
        for name in ("CustomPoint", "CustomMsg"):
            definition = (Path(msg_dir) / (name + ".msg")).read_text()
            self.types.register(get_types_from_msg(definition, "livox_ros_driver2/msg/" + name))

    def close(self):
        self.connection.close()

    def preprocessed(self, end_ns):
        # The frozen fixture is 10 Hz. Check the message's own scan clock,
        # since the recorder timestamp includes delivery delay.
        start = end_ns // 100_000_000 * 100_000_000
        messages = self.connection.execute(
            "SELECT data FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp", (self.topic, start, start + 160_000_000))
        for (data,) in messages:
            message = self.types.deserialize_cdr(data, "livox_ros_driver2/msg/CustomMsg")
            if message.timebase != start:
                continue
            # Match FAST_LIO's non-feature Livox preprocessing order exactly.
            eligible = [p for p in message.points[1:]
                        if p.line < 4 and (p.tag & 0x30) in (0, 0x10)]
            selected = eligible[1::2]
            selected = [p for p in selected
                        if p.x*p.x + p.y*p.y + p.z*p.z > .2*.2]
            points = np.array([[p.x, p.y, p.z] for p in selected])
            stamps = message.timebase + np.array([p.offset_time for p in selected])
            intensity = np.array([p.reflectivity for p in selected])
            return points, stamps, intensity
        raise ValueError("Missing raw scan with timebase " + str(start))


def authored_surface_cloud(objects, world_rotation, world_translation, spacing=.1):
    """Sample the independent authored boxes, including their orientations."""
    if spacing <= 0:
        raise ValueError("Surface spacing must be positive")
    surfaces = []
    for box in objects:
        half = np.asarray(box["size"]) / 2
        rotation = Rotation.from_quat(np.roll(box["quaternion"], -1))
        for axis in range(3):
            other_axes = [i for i in range(3) if i != axis]
            grids = np.meshgrid(*[
                np.linspace(-half[i], half[i], int(np.ceil(2*half[i]/spacing)) + 1)
                for i in other_axes], indexing="ij")
            for sign in (-1, 1):
                points = np.zeros((grids[0].size, 3))
                points[:, axis] = sign * half[axis]
                for grid, other_axis in zip(grids, other_axes):
                    points[:, other_axis] = grid.ravel()
                world = rotation.apply(points) + box["position"]
                surfaces.append(world_rotation.inv().apply(world - world_translation))
    return np.concatenate(surfaces)
