# KITTI360 Reader Smoke Tool

`n3mapping_kitti360_reader` is an offline input inspection tool for KITTI360.
It is only a reader/alignment smoke stage: it does not run N3MappingCore, does
not perform mapping or relocalization evaluation, and does not write
`n3map.pbstream`.

KITTI360 is useful as an offline evaluation input source, but it does not
replace real robot bags. Real deployment still needs bags from the target
sensor stack, mounting geometry, timing, and environment.

## Inputs

```text
<kitti_root>/data_3d_raw/<sequence>/velodyne_points/data/*.bin
<kitti_root>/data_poses/<sequence>/poses.txt
<kitti_root>/calibration
```

The tool aligns lidar bins and poses by frame id. The lidar frame id is parsed
from the numeric `.bin` filename stem, and the pose frame id is parsed from the
first column of `poses.txt`. It intentionally does not align by line number.

## Example

```bash
ros2 run n3mapping n3mapping_kitti360_reader \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --output /tmp/n3mapping_kitti360_reader_test \
  --max_frames 50 \
  --dump_sample_pcd
```

Outputs:

```text
<output>/summary.json
<output>/frame_<frame_id>.pcd      # only when --dump_sample_pcd or --dump_first_n is used
<output>/sample.pcd                # only when --dump_sample_pcd is used
```

`summary.json` includes lidar bin count, pose count, selected common frame
count, total common frame count before `--max_frames`, first/last selected
common frame id, missing pose count, missing lidar count, and whether a
calibration directory was readable.

Use `--dump_first_n N` to dump the first N selected frames for manual point
cloud inspection.

## Offline Eval Framework

`n3mapping_kitti360_eval` is the next offline stage. It feeds KITTI360 frames
into `N3MappingCore` and writes evaluation artifacts, but it still does not tune
mapping, loop-closure, or relocalization parameters.

Mapping/loop smoke:

```bash
ros2 run n3mapping n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --mode mapping_loop \
  --max_frames 200 \
  --stride 1 \
  --output /tmp/n3mapping_kitti360_mapping_loop
```

Outputs:

```text
metrics.json
trajectory_est.txt
trajectory_gt.txt
accepted_loops.csv
loop_debug.jsonl
```

Relocalization smoke:

```bash
ros2 run n3mapping n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --mode relocalization \
  --build_map_frames 100 \
  --max_frames 150 \
  --dropout 0.1 \
  --noise 0.01 \
  --fake_yaw 15 \
  --output /tmp/n3mapping_kitti360_relocalization
```

If `--map /path/to/n3map.pbstream` is supplied, relocalization mode uses that
existing map. Otherwise it builds a temporary map from the first
`--build_map_frames` selected frames and queries the remaining frames.

Outputs:

```text
metrics.json
relocalization_queries.csv
relocalization_debug.jsonl
```

Start smoke runs with `2013_05_28_drive_0003_sync`. Do not treat the initial
metrics as algorithm conclusions until the debug JSONL files have been reviewed.
