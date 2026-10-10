# N3Mapping

## 1. What Is N3Mapping?

N3Mapping is a graph-backend SLAM, loop-closure, map-extension, and relocalization package for external LIO frontends.

It consumes deskewed body-frame point clouds and odometry, then produces optimized trajectories, loop-closure constraints, relocalization results, and saved maps.

### Key Features

- ROS-free C++ backend core.
- ROS 2 Humble and ROS 1 Noetic wrappers.
- Mapping, localization, and map extension modes.
- RHPD-primary place recognition with ScanContext auxiliary yaw/rerank/fallback.
- GTSAM pose graph optimization.
- Map save/load with `n3map.pbstream` and `global_map.pcd`.

### Repository Layout

```text
include/ and src/    ROS-free backend core
humble/              ROS 2 Humble wrapper package
noetic/              ROS 1 Noetic wrapper package
config/              shared runtime config
humble/launch/       ROS 2 launch and RViz resources
noetic/launch/       ROS 1 launch and RViz resources
tools/               synthetic relocalization and offline dataset inspection tools
```

## 2. How To Use N3Mapping

### 2.1 Clone

```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone https://github.com/Killow1998/n3mapping.git
```

Place the external LIO frontend package in the same workspace when mapping from live or bagged LIO output.

### 2.2 Select ROS Wrapper

For ROS 2 Humble:

```bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh humble
```

For ROS 1 Noetic:

```bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh noetic
```

Optional:

```bash
src/n3mapping/scripts/select_distro_wrapper.sh status
src/n3mapping/scripts/select_distro_wrapper.sh clear
```

Launch resources are wrapper-local. Humble uses `*.launch.py`; Noetic uses `*.launch`. After selecting a wrapper, use the launch commands for that ROS distribution only.

The mapping, localization, and map-extension launch entries (including Noetic
`development_runtime.launch`) default the N3 process to `OMP_WAIT_POLICY=PASSIVE`
to reduce OpenMP waiting overhead when sharing CPU cores with the frontend and
navigation. An explicit inherited `OMP_WAIT_POLICY` takes precedence. This does
not change thread counts or registration settings;
an external `GOMP_SPINCOUNT` can affect the waiting behavior. Direct `rosrun` or
executable invocation bypasses these launch defaults.

### 2.3 Configuration

Default config:

```text
config/n3mapping.yaml
```

Common fields users may need to modify:

```yaml
n3mapping_node:
  ros__parameters:
    mode: "mapping"              # mapping | localization | map_extension

    # Input topics from external LIO frontend
    cloud_topic: "/cloud_registered_body"
    odom_topic: "/Odometry"

    # Required for localization / map_extension
    map_path: "/path/to/n3map.pbstream"

    # Save directory for mapping / map_extension
    map_save_path: "/path/to/save_dir"
```

`floor_attitude_enable` is disabled in the built-in, default, and product
profiles. The floor factor is retained for research, and the files under
`config/mapping_stages/` opt in explicitly; they are not safe product defaults.

Use a custom config file only when needed.

Humble:

```bash
ros2 launch n3mapping mapping.launch.py config_file:=/path/to/n3mapping.yaml
```

Noetic:

```bash
roslaunch n3mapping mapping.launch config_file:=/path/to/n3mapping.yaml
```

When `map_path` is left empty in `config/n3mapping.yaml`, N3Mapping uses `N3MAPPING_SOURCE_DIR/map/n3map.pbstream`. When `map_save_path` is left empty, it uses `N3MAPPING_SOURCE_DIR/map`.

### 2.4 Runtime Outputs And Services

| Name | Kind | Type | Meaning |
| --- | --- | --- | --- |
| `/n3mapping/odometry` | topic | `nav_msgs/Odometry` | Optimized/relocalized pose and measured body-frame twist |
| `/localization/status` | topic | `std_msgs/String` | Provider-neutral JSON status, configurable with `output_status_topic` |
| `/n3mapping/path` | topic | `nav_msgs/Path` | Output trajectory |
| `/n3mapping/cloud_body` | topic | `sensor_msgs/PointCloud2` | Current cloud in body frame |
| `/n3mapping/cloud_world` | topic | `sensor_msgs/PointCloud2` | Current cloud transformed to world/map frame |
| `/n3mapping/global_map` | topic | `sensor_msgs/PointCloud2` | Published global map |
| `/n3mapping/loop_closure_markers` | topic | `visualization_msgs/MarkerArray` | Loop closure and trajectory markers |
| `/n3mapping/relocalization_lock` | topic | `std_msgs/UInt32` | Relocalization lock event counter |
| `/n3mapping/save_map` | service | `std_srvs/Trigger` / `std_srvs/srv/Trigger` | Save map files |

External consumers do not need N3Mapping's private status message. The generic
status contains `mode`, `state`, numeric `stamp` (current estimate source time),
`observation_stamp` (last processed global scan source time), `correction_stamp`
(last accepted map correction source time), `frame_id`, and `reason`. Times are
seconds; zero observation/correction time means none is available. Forwarding a
current estimate never refreshes an old observation or correction time. In
localization mode, `tracking` requires an authoritative full localization pose.
Other states are `localizing`, `degraded`, `lost`, and `error`.
Mapping reports `initializing` with `waiting_for_static_start_guard` while the
core's startup protection is holding output, then `tracking` when output is
available. Invalid mapping input and diverged odometry report `error`; a started
process alone is not evidence of mapping output. These observations do not grant
navigation authority or identify the consumer's selected map.
Mapping startup accepts a continuous three-second stable scan/odometry window
(pose stays within 2 cm and 0.01 rad of its window anchor, input gaps at most
0.5 s). This allows a settled stationary front end to start a new mapping session
without waiting for physical movement. Unsettled input retains scan-based motion
detection and the existing 120-second fallback. The window is a bounded startup
consistency check, not proof against future estimator drift. No ROS-specific
readiness flag or upstream process-age assumption is used.
The topic is not latched; consumers check current estimate source and receive
freshness instead of treating a historical lock as current readiness. Correction
age alone is not proof of estimator failure or accuracy. For example:

```json
{"mode":"localization","state":"tracking","stamp":123.45,"observation_stamp":123.1,"correction_stamp":123.0,"frame_id":"map","reason":""}
```

`input_linear_velocity_frame` defaults to `child`, as required by standard
Odometry. Select `parent` explicitly for a frontend such as this workspace's
FAST-LIO that reports linear velocity in the odometry parent frame. N3 rotates
that linear velocity and its covariance using the input odometry orientation;
angular velocity stays in the child frame. Global localization corrections do
not create velocity. The input child frame must match `body_frame`. Invalid
twist/covariance or mismatched child frames suppress output odometry/TF rather
than publishing a fabricated zero measurement. These conversions are shared by
the thin Noetic/Humble adapters, not the backend core.

The Noetic wrapper publishes live odometry on an independent input callback in
all three run modes. The ROS-free `core::RealtimeOdometry` combines the latest
raw pose with an authorized map-to-odom correction for the existing global
output. There are no public local-odometry or map-correction topics. Exact
applied correction and paired-pose metadata are exposed by the existing status
JSON contract below; their authority remains N3-owned. The independent
Noetic cloud queue forwards body clouds and transforms world clouds using the
scan's paired odometry, never the newest callback pose. Cloud conversion and
publication use a different single-worker queue from raw odometry; neither runs
under the projection lock or waits for registration.

The first valid odometry or paired scan freezes the map pose and applied
correction stamp for that source odometry timestamp. A later correction cannot
change that observation's geometry. The cache retains at most `sync_queue_size`
poses; evicted observations are rejected, not reconstructed using a newer
correction. An already-reserved in-flight cloud keeps its copied immutable basis
after eviction unless its generation is invalidated. Approximate synchronization
preserves the cloud's original stamp and associates the basis by its paired
odometry stamp. Invalid input cancels
retained bases and in-flight cloud work at its final publication check; ordinary
geometric observation rejection still retains the trusted anchor.
The check admits completed work; it is not an atomic transaction with ROS publish.

Noetic `/localization/status` may include a `projection` object (schema version
1). `kind=odom` accompanies the existing scalar estimate; `kind=cloud` is an
association-only event and must not update scalar state or freshness. Cloud
events retain the latest scalar snapshot without changing its timestamps. Every
available record carries `session_id`, `generation`, `basis_revision`,
`world_frame`, `source_frame`, `body_frame`, integer `odom_stamp_nsec`, and the
finite rigid `map_from_source` and `source_from_body` transforms. Each transform
contains `translation:[x,y,z]` and `quaternion_xyzw:[x,y,z,w]`. Cloud records also
carry integer `cloud_stamp_nsec` and `cloud_frame` (the body-cloud frame);
`world_frame` identifies the corresponding world-cloud output. Both clouds keep
their original source stamp, even when approximate synchronization paired them
with a different odometry stamp. The full paired pose is present even if the
independent raw-odometry subscription dropped that observation.

`available=false` is an odometry invalidation record carrying session,
generation, frames and input stamp, but no invented transform. It cancels old
associations; a newer timestamp alone cannot restore availability. Recovery
requires a genuinely available full-pose record backed by an authorized
correction. Ordinary rejected backend observations do not revoke the anchor.
Consumers must join by the exact integer stamp and declared frames, wait or drop
unpaired inputs within their existing retention/freshness bounds, and never
infer a correction from timestamp differences. A late cloud's frozen correction
only converts that observation into continuous source coordinates; it must not
roll back the current view correction selected by newer valid odometry.
This metadata contract currently applies to the Noetic live-output path.

N3 publishes the direct world-to-body TF with the same source pose as its odometry.
A rejected observation leaves the last trusted correction unchanged. Correction
age alone does not revoke output or imply known drift; invalid timestamps, poses
or source frames still invalidate the estimate. Live output keeps actual input
timestamps. Existing localization JSON separates `stamp` (current estimate),
`observation_stamp` (last processed scan) and `correction_stamp` (the anchor
actually applied to this estimate), in seconds. A cloud-first frozen estimate
can use an older anchor than the newest backend observation. Quality reflects
the backend observation, not a fabricated covariance guarantee. These independent
output queues are implemented by the Noetic wrapper; Humble still publishes
paired backend odometry/clouds synchronously and does not provide this fast path.

`test/noetic_output_smoke.py /path/to/n3mapping_node` runs the real Noetic wrapper
against synthetic inputs on a temporary local ROS master after sourcing the ROS
and workspace environments. It does not connect to hardware or an existing master.

### 2.5 Map Files And Logs

Mapping and map extension save:

```text
<map_save_path>/n3map.pbstream
<map_save_path>/global_map.pcd
<map_save_path>/optimization.log
```

`optimization.log` is truncated when the node starts and appended after accepted keyframes or loop optimizations.

Terminal output is intentionally lightweight. It keeps warnings, errors, map loading, save-map results, shutdown map-save results, and relocalization-lock events.

For real-device update/build commands, per-run log capture, and local support ZIP
export, see [实机测试与问题反馈](docs/field_testing_zh.md). The support tool
does not launch frontends, upload logs, or include maps/bags. Detailed JSONL
logging is opt-in; default matching and locking behavior is unchanged.

### 2.6 Build, Run, And Test Commands

<details>
<summary><strong>ROS 2 Humble</strong></summary>

### Dependencies

<details>
<summary>Show Dependencies</summary>

```bash
sudo apt-get update
sudo apt-get install -y \
  libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libpcl-dev libeigen3-dev \
  libopencv-dev libboost-all-dev libtbb-dev
```

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws
rosdep install --from-paths src --ignore-src -r -y
```

ROS dependencies:

- ROS 2 Humble
- `ament_cmake`
- `rclcpp`
- `std_msgs`, `std_srvs`
- `sensor_msgs`
- `nav_msgs`
- `geometry_msgs`
- `visualization_msgs`
- `message_filters`
- `tf2`, `tf2_ros`, `tf2_geometry_msgs`
- `pcl_conversions`, `pcl_ros`

Native dependencies:

- PCL
- Eigen3
- OpenCV
- Protobuf
- glog
- OpenMP
- GTSAM
- small_gicp

If GTSAM and small_gicp are not already available in the workspace:

```bash
cd ~/ros_ws/src
git clone https://github.com/borglab/gtsam.git -b 4.1.1
git clone https://github.com/koide3/small_gicp.git
```

</details>

### Build

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh humble

colcon build --packages-up-to n3mapping --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

### Run

Mapping:

```bash
ros2 launch n3mapping mapping.launch.py
```

Localization:

```bash
ros2 launch n3mapping localization.launch.py
```

Map extension:

```bash
ros2 launch n3mapping map_extension.launch.py
```

Headless:

```bash
ros2 launch n3mapping mapping.launch.py rviz:=false
```

### Save Map

```bash
ros2 service call /n3mapping/save_map std_srvs/srv/Trigger {}
```

### Test

```bash
colcon build --packages-up-to n3mapping --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON

ROS_LOG_DIR=/tmp/ros_log colcon test --packages-select n3mapping
colcon test-result --test-result-base build/n3mapping --verbose
```

### Synthetic Relocalization Tools

RViz visualization:

```bash
ros2 launch n3mapping synthetic_relocalization_visualization.launch.py \
  map:=/path/to/n3map.pbstream
```

Headless batch evaluation:

```bash
ros2 run n3mapping n3mapping_synthetic_relocalization_eval \
  --map /path/to/n3map.pbstream \
  --max_queries 100 \
  --query_source local_submap \
  --strict
```

KITTI360 lidar/pose alignment smoke:

```bash
ros2 run n3mapping n3mapping_kitti360_reader \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --output /tmp/n3mapping_kitti360_reader_test \
  --max_frames 50 \
  --dump_sample_pcd
```

This is an offline evaluation input reader only; it does not run mapping or
relocalization and does not replace real robot bags. See the
[dataset evidence contract](docs/industrial_closeout_review_20260814.md#dataset-evidence-contract).

KITTI360 offline mapping-loop smoke:

```bash
ros2 run n3mapping n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --mode mapping_loop \
  --calib_mode auto \
  --max_frames 200 \
  --stride 1 \
  --output /tmp/n3mapping_kitti360_mapping_loop
```

Label KITTI360 loop candidates with keyframe ground truth:

```bash
ros2 run n3mapping n3mapping_loop_debug_analyze.py \
  --loop_debug /tmp/n3mapping_kitti360_mapping_loop/loop_debug.jsonl \
  --keyframes_gt /tmp/n3mapping_kitti360_mapping_loop/keyframes_gt.csv \
  --accepted_loops /tmp/n3mapping_kitti360_mapping_loop/accepted_loops.csv \
  --output /tmp/n3mapping_kitti360_mapping_loop/loop_gt_analysis
```

Summarize one or more eval runs into a metric matrix:

```bash
ros2 run n3mapping n3mapping_eval_matrix.py \
  --run kitti360_drive0003=/tmp/n3mapping_kitti360_mapping_loop \
  --output /tmp/n3mapping_eval_matrix
```

Benchmark interpretation and current acceptance limits are recorded in the
[industrial closeout review](docs/industrial_closeout_review_20260814.md).

M2DGR extracted-cloud offline eval:

```bash
ros2 run n3mapping n3mapping_m2dgr_eval \
  --m2dgr_root /path/to/M2DGR \
  --sequence hall_03 \
  --lidar_dir /path/to/M2DGR/hall_03/velodyne_points \
  --gt /path/to/M2DGR/hall_03/groundtruth.txt \
  --mode mapping_loop \
  --output /tmp/n3mapping_m2dgr_hall03_mapping
```

Details and evidence limitations are in the
[dataset evidence contract](docs/industrial_closeout_review_20260814.md#dataset-evidence-contract).

</details>

---

<details>
<summary><strong>ROS 1 Noetic</strong></summary>

### Dependencies

<details>
<summary>Show Dependencies</summary>

```bash
sudo apt-get update
sudo apt-get install -y \
  libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libpcl-dev libeigen3-dev \
  libopencv-dev libboost-all-dev libtbb-dev
```

```bash
source /opt/ros/noetic/setup.bash
cd ~/ros_ws
rosdep install --from-paths src --ignore-src -r -y
```

ROS dependencies:

- ROS 1 Noetic
- `catkin`
- `roscpp`
- `std_msgs`, `std_srvs`
- `sensor_msgs`
- `nav_msgs`
- `geometry_msgs`
- `visualization_msgs`
- `message_filters`
- `tf2_ros`
- `pcl_conversions`, `pcl_ros`

Native dependencies:

- PCL
- Eigen3
- OpenCV
- Protobuf
- glog
- OpenMP
- GTSAM
- small_gicp

If GTSAM and small_gicp are not already available in the workspace:

```bash
cd ~/ros_ws/src
git clone https://github.com/borglab/gtsam.git -b 4.1.1
git clone https://github.com/koide3/small_gicp.git
```

</details>

### Build

```bash
source /opt/ros/noetic/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh noetic

catkin build n3mapping --no-status -j2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source devel/setup.bash
```

### Run

Mapping:

```bash
roslaunch n3mapping mapping.launch
```

Localization:

```bash
roslaunch n3mapping localization.launch
```

Map extension:

```bash
roslaunch n3mapping map_extension.launch
```

Headless:

```bash
roslaunch n3mapping mapping.launch rviz:=false
```

### Save Map

```bash
rosservice call /n3mapping/save_map "{}"
```

### Test

```bash
catkin build n3mapping --no-status -j2 --catkin-make-args run_tests
source devel/setup.bash
cd build/n3mapping
ctest --output-on-failure
```

### Synthetic Relocalization Tools

RViz visualization:

```bash
roslaunch n3mapping synthetic_relocalization_visualization.launch \
  map:=/path/to/n3map.pbstream
```

Headless batch evaluation:

```bash
rosrun n3mapping n3mapping_synthetic_relocalization_eval \
  --map /path/to/n3map.pbstream \
  --max_queries 100 \
  --query_source local_submap \
  --strict
```

KITTI360 lidar/pose alignment smoke:

```bash
rosrun n3mapping n3mapping_kitti360_reader \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --output /tmp/n3mapping_kitti360_reader_test \
  --max_frames 50 \
  --dump_sample_pcd
```

This is an offline evaluation input reader only; it does not run mapping or
relocalization and does not replace real robot bags. See the
[dataset evidence contract](docs/industrial_closeout_review_20260814.md#dataset-evidence-contract).

KITTI360 offline mapping-loop smoke:

```bash
rosrun n3mapping n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --mode mapping_loop \
  --max_frames 200 \
  --stride 1 \
  --output /tmp/n3mapping_kitti360_mapping_loop
```

Summarize eval runs:

```bash
rosrun n3mapping n3mapping_eval_matrix.py \
  --run kitti360_drive0003=/tmp/n3mapping_kitti360_mapping_loop \
  --output /tmp/n3mapping_eval_matrix
```

M2DGR extracted-cloud offline eval:

```bash
rosrun n3mapping n3mapping_m2dgr_eval \
  --m2dgr_root /path/to/M2DGR \
  --sequence hall_03 \
  --lidar_dir /path/to/M2DGR/hall_03/velodyne_points \
  --gt /path/to/M2DGR/hall_03/groundtruth.txt \
  --mode mapping_loop \
  --output /tmp/n3mapping_m2dgr_hall03_mapping
```

</details>

---

## 3. Additional Information

### Developer Notes

<details>
<summary>Show Developer Notes</summary>

- Keep backend mapping, loop closure, relocalization, graph optimization, and map serialization logic in the ROS-free core.
- Keep ROS wrappers as thin adapters for parameters, messages, topics, services, TF, and launch defaults.
- Keep Humble and Noetic runtime behavior consistent.
- Do not commit local wrapper-selection marker files.
- Do not commit generated maps, logs, build artifacts, or local test configs.

</details>

### TODO

- Long-duration real-world mapping, localization, and map-extension regression.
- More compatibility tests for old `pbstream` files.
- Real-bag evaluation scripts for common workflows.
- Optional frontend integration examples.
