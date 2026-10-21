# N3Mapping

## 1. What Is N3Mapping?

N3Mapping is a graph-backend SLAM, loop-closure, map-extension, and relocalization package for external LIO frontends.

It consumes deskewed body-frame point clouds and odometry, then produces optimized trajectories, loop-closure constraints, relocalization results, and saved maps.

### Key Features

- ROS-free C++ backend core.
- ROS 2 Humble and ROS 1 Noetic wrappers.
- Mapping, localization, and map extension modes.
- Unified RHPD place recognition and direction-preserving heading matching, with sparse descriptor storage.
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
`development_runtime.launch`) and the Product runtime launcher default the N3
process to `OMP_WAIT_POLICY=PASSIVE`
to reduce OpenMP waiting overhead when sharing CPU cores with the frontend and
navigation. An explicit inherited `OMP_WAIT_POLICY` takes precedence. This does
not change thread counts or registration settings;
an external `GOMP_SPINCOUNT` can affect the waiting behavior. Direct `rosrun` or
executable invocation bypasses these launch defaults.

Evidence for local CPU/latency optimization, its workload boundaries and reproducible probes: [runtime CPU measurements](docs/evidence/runtime_cpu_20261017/README.md).

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

The Product localization profile requires a matching Atlas cache. Compile it
from the source map after building and sourcing the workspace:

```bash
ros2 run n3mapping n3mapping_localization_atlas_compile --map /path/to/n3map.pbstream
```

The default output is `<map_path>.localization_atlas.pb`. Atlas format v2 stores
the full double-precision spatial coordinates, normals and 3×3 covariances,
omitting constant homogeneous padding. Old caches must be rebuilt from the
original `pbstream`; the map itself does not need conversion. To preserve an old
cache, use `--output /path/to/new_atlas.pb` and set `reloc_atlas_path` accordingly.
Use `--force` to explicitly replace an existing cache. The 1 GiB cache limit
and matching map/configuration fingerprints still apply.
At runtime, the two point-to-plane levels retain coordinates and normals only;
the fine GICP level retains full covariances. Existing v2 caches remain readable.
Saving reconstructs the unused plane-level covariance from its normal using the
same small_gicp eigenvalue model, without changing the on-disk layout.

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

Both wrappers publish live odometry on an independent input callback in
all three run modes. The ROS-free `core::RealtimeOutput` owns authorization and
combines the latest raw pose with the last trusted map-to-odom correction for
the existing global output. There are no public local-odometry or map-correction topics. Exact
applied correction and paired-pose metadata are exposed by the existing status
JSON contract below; their authority remains N3-owned. Each adapter
forwards body clouds and transforms world clouds using the
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

`/localization/status` in both wrappers may include a `projection` object (schema version
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
Body-only subscriptions still receive the association; without world-cloud
subscribers the adapter skips full cloud conversion and transformation.

N3 publishes the direct world-to-body TF with the same source pose as its odometry.
A rejected observation leaves the last trusted correction unchanged. Correction
age alone does not revoke output or imply known drift; invalid timestamps, poses
or source frames still invalidate the estimate. Live output keeps actual input
timestamps. Existing localization JSON separates `stamp` (current estimate),
`observation_stamp` (last processed scan) and `correction_stamp` (the anchor
actually applied to this estimate), in seconds. A cloud-first frozen estimate
can use an older anchor than the newest backend observation. Quality reflects
the backend observation, not a fabricated covariance guarantee. A `LOST` backend
observation also leaves an existing trusted anchor intact while raw input stays
valid; it grants no new initialization authority or lock epoch.

Humble uses separate callback groups and independently spun serial executors for
live odometry and paired clouds. The backend, dense trajectory recorder, loop
processing and save service keep one serial executor. Raw odometry retains only
one pending sample (reliable, volatile, depth 1); the other subscriptions retain
the configured synchronization queue and existing QoS. Product V1 uses exact
synchronization; other profiles keep `sync_time_tolerance`. Localization drains
received backend pairs into the existing `LatestFrameSlot`, replacing pending
work with the newest complete pair before processing it. Mapping and extension
retain their serial paired backend processing. Shutdown stops and joins both
live workers before saving maps. Path and authority events keep backend cadence;
TF keeps odometry source time and suppresses zero/non-increasing timestamps.

`test_humble_live_output`, registered with Humble's CTest/ament tests, exercises
actual ROS subscriptions and publishers with blocked backend/cloud callbacks,
a coupled-executor control, frozen/invalidation metadata, and shutdown/map save.
`test_humble_conversions` checks source validation and integer nanoseconds.

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

Localization changes should follow the existing ownership boundaries:

| Responsibility | Owner |
| --- | --- |
| Input-source lifecycle and backend orchestration | `N3MappingCore` localization facade |
| Session state, reset and map replacement | `WorldLocalizing` |
| Query history and place retrieval | `RelocalizationQueryBuilder`, `RelocalizationPlaceIndex` |
| Candidate registration and temporal acceptance | Candidate evaluator, hypothesis manager and decision policy |
| Tracking targets, bounded local caches and the immutable loaded-map target | `TrackingTargetCache` |
| Online loop retrieval, isolated geometry/graph work and revision-checked commit | `N3MappingCore` loop orchestration and shared loop-processing implementation |
| Trusted corrections and live pose/cloud projection | `core::RealtimeOutput` |

Search, tracking, map evidence and offline probes have separate implementation
files. Session reset preserves map-derived tracking targets; map replacement
invalidates their generation and queued work. Loaded-map tracking prepares one
registration target per frozen map generation, with a four-entry cache of exact
raw visibility crops. Crop centers belong to that same frozen geometry.
Speculative target-preparation threads have been removed.

Mapping-loop retrieval stays with the serialized owner. Geometry verification
and graph solving use an isolated snapshot; the owner commits only a matching
map generation and revision. Appended keyframes catch up through a conditioned
tail solve. A conflicting graph revision requires readmission and re-solving,
while retaining measured registration evidence. Saving drains pending loop
work. Strict map extension verifies loaded-map geometry and commits its tracking
constraint inline. This append solve preserves the Atlas-verified live pose;
the worker later installs the optimized reference and its coordinate correction
together.

`loop_geometry_cpu_budget` defaults to **0.5** CPU seconds per wall second for
background extension geometry; **0** disables pacing. It leaves ordinary mapping
loop processing unchanged. Successive extension requests coalesce while every
measured constraint remains in the graph. The interval uses process CPU during
the job, including parallel feature work and concurrent foreground work, as a
conservative upper bound. It is not a cap on total node CPU. Pending age retains
the oldest request time; saving bypasses pacing to finish outstanding work.

The deterministic `test_localization_episode` passes actual saved/loaded maps
through the production core and live-output owner. It covers initialization,
moving outside the map, reentry, invalid odometry, source replacement, gradual
yaw-drift recovery, fresh map lifecycles,
ambiguous places, no candidates and strict map-extension tracking. Together
with ROS publication tests, it is the integration check for changes within
these boundaries:

```bash
ctest --test-dir build/n3mapping --output-on-failure \
  -R 'test_(localization_episode|world_localizing|humble_live_output)$'
```

The episode test also accepts `N3MAPPING_EPISODE_ATLAS=1` and
`N3MAPPING_EPISODE_FREE_SPACE_MODE=kill` for the existing algorithm variants.
`N3MAPPING_EPISODE_TRACE=/tmp/episode.csv` records per-frame decisions, poses,
timestamps, work counts and timings for ablations. Synthetic episodes measure
backend behavior; real-sensor drift and robot acceptance require recorded or
live sensor input.

The [2026-10-17 optimization evidence](docs/evidence/localization_optimization_20261017.json)
records 61 passing CTest projects and 36 ablation runs with identical public
outputs. It shows reduced repeated work and a prefetch tail-latency benefit in
these synthetic scenes; whole-process runtime remains essentially unchanged.
The later 2026-10-20 recorded extension study supersedes that synthetic prefetch
result. The initially rejected candidate exposed a pose-selection bug: a valid
ICP result could be replaced by a prediction that failed the quality gate.
After fixing selection and strict tracking recovery, the complete independent
motion recording succeeds on all 2,232 post-initialization inputs. Sharing one
loaded-map target reduces CPU from 896.7 to 186.5 s versus the reliable
per-anchor control, and peak RSS from 2.99 to 1.86 GiB. The comparison includes
the whole route; rejected tails are not counted as cheap successful tracking.
See the [recovery and asynchronous-loop evidence](docs/evidence/runtime_cpu_20261017/README.md#extension-recovery-and-asynchronous-online-loops--2026-10-20)
for timing scope, accuracy changes and remaining limits.

Registration uses the existing correspondence-distance gate to bound exact
nearest-neighbor search. It preserves accepted neighbors, equality at the gate,
tie order and optimizer settings; it does not use approximate search.
Each factor's previous correspondence can tighten that bound for the next
iteration. The old point is only a search hint: traversal still selects the
exact neighbor in the original order, with no extra per-point cache.
Occupied subtree bounds further prune search; outward rounding preserves the
accepted neighbors. Loaded-map tracking shares the geometry without those extra
box checks. Atlas visibility crops use the existing 128 MiB budget rather than
the eight-entry cap intended for independently prepared local targets.

For a CPU budget, compare process CPU time at the same sensor cadence, then
sample the deployed node with `tools/n3mapping_process_resource_sample.py`.
Its `process_cpu_percent` uses 100% for one fully occupied core. A continuously
running test may still occupy one core after it becomes faster. At 10 Hz, a
half-core average budget is 50 CPU ms per frame, including background work;
tracking, recovery, input/output cadence and latency must also pass on the
target CPU and map.

The [exact-search CPU evidence](docs/evidence/registration_cpu_optimization_20261017.json)
records a 13%–15% reduction in process CPU time on the same synthetic episodes,
with identical public outputs. The RK3588 half-core deployment target remains
unverified; target sampling is required.

The [open-source relocalization survey](docs/evidence/framework_survey_20261017/README.md)
reviews 12 pinned projects and records direct component runs from five of them.
A subsequent prepared-cloud memory ablation preserved outputs but increased
process CPU time in these episodes, so it was rejected. The report separates
source review, component measurements and hardware validation; it does not
claim a full-system accuracy comparison or an RK3588 half-core result.
The [subsequent correspondence-reuse evidence](docs/evidence/registration_hint_optimization_20261017.json)
records a further 3.6%–3.8% reduction in process CPU time with identical episode
outputs and 61 passing CTest projects. Pinned map-registration batches also
check one and four threads; individual short measurements and target hardware
remain separate from that measured result.

</details>

### TODO

Current work is updated **2026-10-21**. Maintenance version **1.0.1** closed the
accepted performance changes. Current priorities are **03 search cost and
scan-level motion bias / correlated loop uncertainty**, while preserving
relocalization accuracy and online mapping/extension behavior. Offline replay is a validation method for
that online behavior. Stationary startup remains required by Xian's downstream
workflow. Continuing localization outside the global map is deferred pending
Xian's decision; the highly repetitive out-of-map corridor remains a recorded
limitation. The 202–211 s full-map tracking excursion is now fixed on the complete
recording. Preserve the explicit source-replacement
and invalid-input recovery fixes and the original fixed inputs.
Detailed results and limits are in the [measurement record](docs/evidence/runtime_cpu_20261017/README.md);
the [accepted startup ablation](docs/evidence/runtime_cpu_20261017/README.md#stationary-startup-exact-free-space-queries-and-yaw-work-ablation--2026-10-19)
remains the performance baseline.

| Priority / state | Task and completion evidence |
| --- | --- |
| Reliability — startup and output baseline retained | Original14 keeps all 14 frame-4 locks and 4,088 outputs. Both independent routes preserve all outputs. Fresh-route recovery advances 2002→1938 (6.4 input seconds); backend-loss observations fall 123→59. Previous locks cannot break a geometric tie. Reference coverage gaps and partial-map aliasing remain explicit. |
| Accuracy — recorded simulation cases below 10 cm | Prior mapping RMSE/maximum 1.399/9.289 cm retained. Current original14 0.840/2.481 cm; original/fresh motion maxima 7.933/3.247 cm. Small RMSE and old-route peak regressions are preserved in the [recovery evidence](docs/evidence/runtime_cpu_20261017/README.md#shared-atlas-tracking-and-fresh-recovery-confirmation--2026-10-21). |
| Online loop/map consistency — checkpoints retained; calibration open | Source/target deskew cross-ablation attributes the worst audited mapping-loop bias to both scans: 16.757→0.495 mm with observer-only native deskew. A shared rigid scan bias can cancel around graph cycles. Production weights remain unchanged; the diagnostic is not a runtime correction. [Evidence and limits](docs/evidence/runtime_cpu_20261017/README.md#occupied-search-bounds-and-scan-level-loop-bias--2026-10-21). |
| Online extension — RAM reduced; half-core open | Last fully measured first/second/fresh extensions use 0.873/0.860/0.840 cores and 1.873/1.987/1.813 GiB runtime RSS. The paired fresh-route memory change is 2.088→1.816 GiB with CPU approximately unchanged. All recorded outputs stay below 10 cm; [save peaks and latency tradeoffs](docs/evidence/runtime_cpu_20261017/README.md#recovery-geometry-compact-map-assembly-and-uncertainty-ablation--2026-10-21) remain explicit. |
| Asynchronous online mapping loops — complete causal qualification | Mapping averages 0.413 cores including background drain; maximum live error 9.289 cm. Full-node/hardware and independent Noetic compilation remain open. |
| Search cost — 03 half-core remains open | Exact occupied bounds and Atlas crop retention: paired three-repeat medians 03 0.569→0.552, 14 0.468→0.448 cores. Full original14 0.407; old/new routes 0.244/0.324. All decisions retained; pose differences below 6e-14 matrix units. [Scope, timing ranges and ablations](docs/evidence/runtime_cpu_20261017/README.md#occupied-search-bounds-and-scan-level-loop-bias--2026-10-21). |

For uncertainty audits, `LOOP` identifies a robust graph edge, not its measurement
source. The audited extension additions are loaded-map tracking constraints with
fixed noise; geometric loops use their configured information model. The second
replayed extension's 230 exact scan-copy matches are excluded from independent
calibration. The [source/repeatability audit](docs/evidence/runtime_cpu_20261017/README.md#repeated-input-determinism-and-constraint-source-correction--2026-10-21)
corrects the earlier mixed comparison; startup logs now expose both noise models.

The performance target retains **four physical P cores**, four registration threads,
`OMP_WAIT_POLICY=PASSIVE`, the same map/configuration and MID360-derived **10 Hz**
input. The goal is at most **0.5 CPU s per input second**, including stationary
startup, subsequent tracking and background work; short peaks are allowed.
Outside-map continuation and its recovery costs remain separate deferred evidence.
Do not dilute search cost
with longer stationary recordings or drop inputs to meet the average. Report
lock/recovery latency and peak RSS alongside CPU. Current direct-core CPU measurements
exclude loading/saving, frontend, simulator and DDS. Runtime peak RSS includes
map loading; save peak is reported separately. Full-node deployment and RK3588
qualification remain separate pending checks.

The [frozen spatial-geometry comparison](docs/evidence/runtime_cpu_20261017/README.md#spatial-geometry-ablation-and-motion-peak-attribution--2026-10-20)
retains the current map and localization algorithm. Complete 14-point and
independent-route tests reject both the tested bounded joint-geometry map and
its uncalibrated graph propagation. At the 17.28 cm moving peak, the selected
original-scan fallback itself is 19.75 cm wrong; the same scan and actual
prediction seed reach 6.42 cm against the existing full map. Endpoint logging
now distinguishes optimizer output, visibility selection and publication.
This isolates a target/registration-selection contribution, while other map
bias remains. This motivated the subsequent signed association and online
integration below. A smaller internal residual alone is not acceptance.

The subsequent [oriented shared-plane comparison](docs/evidence/runtime_cpu_20261017/README.md#oriented-shared-plane-geometry-and-directional-information--2026-10-20)
shows a substantially better candidate: map trajectory RMSE 4.44→0.61 cm,
independent surface RMSE 21.69→3.61 mm, complete fixed14 RMSE/maximum
0.74/2.41 cm and moving RMSE/maximum 1.92/15.28 cm. All six causal snapshots
also improve without future data. Association fixes provide the main accuracy
gain; connected-plane aggregation reduces 317,867 observations to 20,242 and
snapshot processing to about 1.38 s / 396 MiB. The maintained research tool
is `n3mapping_planar_geometry_eval`, behind `N3MAPPING_BUILD_RESEARCH_TOOLS`.
It writes CSV/JSON and rejects unsupported planar/session/floor factors.
The default map remains unchanged: 01/08 have small individual regressions,
moving angular maximum increases 0.545→0.601°, and 18 outputs still exceed
10 cm. The one-centimetre geometry noise floor is an explicit model assumption,
not a statistically calibrated covariance. The subsequent
[online integration](docs/evidence/runtime_cpu_20261017/README.md#asynchronous-joint-geometry-mapping-extension-and-checkpoints--2026-10-21)
now persists those constraints and feeds them back through the isolated worker.
`loop_geometry_enable` defaults to true for new background refinement; it never
removes constraints already saved in a loaded map. Map, Atlas and dense trajectory
provenance share the committed checkpoint. Original-map RMSE changes from
0.647 to 0.907 cm after incorporating the new session, so universal metric
improvement is not claimed. The [subsequent refinement](docs/evidence/runtime_cpu_20261017/README.md#tracking-authority-plane-partition-and-extension-budget--2026-10-21) partitions failed plane groups and paces
extension geometry, reducing whole-extension CPU/RAM while preserving every tested
output. The [Atlas refinement](docs/evidence/runtime_cpu_20261017/README.md#exact-atlas-feature-reuse-and-compact-target-covariance--2026-10-21) reuses unchanged neighborhoods and removes
redundant target covariance storage. Full same-input tests retain reliability;
whole-extension half-core remains open; the following stage addresses online mapping peaks.
The [subsequent timely-geometry stage](docs/evidence/runtime_cpu_20261017/README.md#timely-keyframe-geometry-and-repeat-input-information--2026-10-21) removes the eight online mapping
peaks using only geometry already available at keyframe insertion. Original14,
original/fresh independent routes and repeated/fresh extensions are fully replayed.
Fresh-route recovery and small metric regressions remain explicit. Map format
2.10.0 stores per-observation information weights; old maps default to unit weight,
and old binaries reject the new format. Broader noise calibration and full-system
resource targets remain open.

Retain existing interfaces and numeric acceptance gates. Compare actual emitted
poses, lock decisions and input/output continuity on the same full inputs for
release decisions; short prefixes are screening evidence. Use the existing E2E
episodes and frozen bags, with truth restricted to observers and scoring. A pose
localized against the current map is not an independent loop measurement, and
repeated evidence from the same scans must not be counted as new confirmation.
After each substantial stage, review the complete affected path, ablate the
proposed changes, remove experiments without demonstrated overall benefit, and
keep module responsibilities explicit. Prefer small cohesive files over added
layers or mechanical splitting. Run the relevant observable-behavior E2E checks,
record verified results and remaining gaps here and in the existing measurement
record, then make the authorized local commit before continuing. Frontend
selection and unrelated new features stay outside scope. Loop-consistency work
targets the existing online mapping/map-extension behavior; frozen replay must
check both loop constraints and their effect on the graph and map.

Other pending work:

- Long-duration real-world mapping, localization, and map-extension regression.
- More compatibility tests for old `pbstream` files.
- Real-bag evaluation scripts for common workflows.
- Optional frontend integration examples.
