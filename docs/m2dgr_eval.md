# M2DGR Offline Evaluation Adapter

`n3mapping_m2dgr_eval` is the first indoor-capable dataset adapter for the
N3Mapping evaluation matrix. It does not read ROS bags directly. It reads:

- extracted lidar clouds: `.pcd` or KITTI-style `.bin` XYZI files
- ground truth trajectory: TUM format

The expected TUM ground-truth format is:

```text
timestamp x y z qx qy qz qw
```

Lidar cloud filenames must contain a numeric timestamp in the stem, for example:

```text
1000.000000000.pcd
1000.100000000.pcd
```

The tool aligns lidar clouds and GT by nearest timestamp, controlled by
`--max_time_diff`.

## Why Extracted Clouds First

M2DGR is distributed as ROS bags plus GT. Reading ROS1 bags directly would make
the offline evaluator depend on ROS bag APIs and would be awkward for the ROS 2
Humble build. Keeping this tool on extracted clouds preserves the same core
contract as KITTI360 eval and keeps the adapter usable in both Humble and
Noetic builds.

## Mapping Loop Smoke

```bash
ros2 run n3mapping n3mapping_m2dgr_eval \
  --m2dgr_root /path/to/M2DGR \
  --sequence hall_03 \
  --lidar_dir /path/to/M2DGR/hall_03/velodyne_points \
  --gt /path/to/M2DGR/hall_03/groundtruth.txt \
  --mode mapping_loop \
  --max_frames 300 \
  --stride 1 \
  --max_time_diff 0.05 \
  --output /tmp/n3mapping_m2dgr_hall03_mapping
```

Outputs:

- `metrics.json`
- `trajectory_est.txt`
- `trajectory_gt.txt`
- `keyframes_gt.csv`
- `accepted_loops.csv`
- `loop_debug.jsonl`

## Relocalization Smoke

```bash
ros2 run n3mapping n3mapping_m2dgr_eval \
  --m2dgr_root /path/to/M2DGR \
  --sequence hall_03 \
  --lidar_dir /path/to/M2DGR/hall_03/velodyne_points \
  --gt /path/to/M2DGR/hall_03/groundtruth.txt \
  --mode relocalization \
  --build_map_frames 100 \
  --max_frames 250 \
  --dropout 0.1 \
  --noise 0.01 \
  --fake_yaw 15 \
  --output /tmp/n3mapping_m2dgr_hall03_reloc
```

Outputs:

- `metrics.json`
- `relocalization_debug.jsonl`
- `relocalization_queries.csv`

## Matrix Summary

```bash
ros2 run n3mapping n3mapping_eval_matrix.py \
  --run kitti360_drive0005=/tmp/n3mapping_kitti360_drive0005 \
  --run m2dgr_hall03=/tmp/n3mapping_m2dgr_hall03_mapping \
  --output /tmp/n3mapping_indoor_outdoor_matrix
```

The 80% target should be judged from the matrix, not from a single visual run.
