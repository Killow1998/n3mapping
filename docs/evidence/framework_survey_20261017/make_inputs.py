"""Recreate the survey fixtures from the identified binary XYZI PCD (not a bag)."""
import hashlib
import pathlib
import sys

import numpy as np

pcd = pathlib.Path(sys.argv[1]).read_bytes()
expected = "8d949ec3adcb143aa526f0ad94771ceb67fbddf7fa7e4c66d89bea8a26f6ae0e"
if hashlib.sha256(pcd).hexdigest() != expected:
    raise SystemExit("This recipe requires the survey's exact PCD; see results.json.")
output = pathlib.Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
payload = pcd.split(b"DATA binary\n", 1)[1]
points = np.frombuffer(payload, dtype="<f4").reshape(-1, 4)[:, :3].astype("<f8")
points = points[np.isfinite(points).all(axis=1)]
points.tofile(output / "map.bin")
yaw = 0.7
rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
for i, quantile in enumerate([0.25, 0.5, 0.75]):
    center = np.quantile(points, quantile, axis=0)
    crop = points[np.sum((points[:, :2] - center[:2]) ** 2, axis=1) <= 64]
    _, indices = np.unique(np.floor(crop / 0.5).astype(np.int64),
                           axis=0, return_index=True)
    query = (crop[np.sort(indices)] - center) @ rotation
    query.tofile(output / f"query{i}.bin")
negative = np.random.default_rng(736).normal(size=(1500, 3)) * [10, 10, 5]
negative.tofile(output / "negative.bin")
