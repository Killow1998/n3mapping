# N3Mapping

N3Mapping is a graph-backend SLAM and relocalization package with a ROS-free C++ core and thin ROS wrappers. It consumes external LIO output as an undistorted body-frame point cloud plus odometry, performs RHPD-primary loop/relocalization retrieval with ScanContext as auxiliary yaw/rerank/fallback, verifies candidates with small_gicp, optimizes a GTSAM pose graph, and saves maps as `pbstream` files.

The `ros_wrapper` branch is the intended common development line for both ROS 2 Humble and ROS 1 Noetic. Backend changes should land in `n3mapping_core`; ROS wrappers should stay thin adapters for parameters, messages, topics, services, TF, and launch files.

## Architecture

- `n3mapping_core`: ROS-free C++ backend library.
- `noetic/`: ROS 1 Noetic catkin package. Package name stays `n3mapping`.
- `humble/`: ROS 2 Humble ament package. Package name stays `n3mapping`.
- `config/` and `launch/`: shared runtime resources. Wrapper package directories expose these through symlinks and install rules; the files themselves are not forked.
- Wrapper-specific source stays inside each wrapper package directory (`noetic/src`, `noetic/include`, `humble/src`, `humble/include`). The root `src/` and `include/n3mapping/` trees are reserved for ROS-free core code.

The core API boundary is an external-LIO frame: point cloud in body/lidar frame and odometry pose. N3Mapping does not currently embed FAST-LIO, DLIO, or SuperLIO in this branch.

Run-mode semantics are shared by the core API: wrappers parse `mode` through `CoreRunMode` and call `N3MappingCore::processFrame()`. Keep backend behavior, map snapshot policy, keyframe decisions, and publish/save map generation in the core; wrapper-specific code should only adapt ROS parameters, messages, topics, services, TF, and launch-time defaults.

The repository root is intentionally not a ROS package. `noetic/` and `humble/` are sibling wrapper packages over the same core sources. Use `scripts/select_distro_wrapper.sh` to generate local profile markers before building in a mixed workspace:

```bash
scripts/select_distro_wrapper.sh        # auto: choose from ROS_VERSION
scripts/select_distro_wrapper.sh noetic # hide humble/ from catkin
scripts/select_distro_wrapper.sh humble # hide noetic/ from colcon/ament
scripts/select_distro_wrapper.sh status # inspect local profile markers
scripts/select_distro_wrapper.sh clear  # remove generated markers
```

These generated markers are ignored by git. Do not commit a permanent `noetic/COLCON_IGNORE`: Noetic's `catkin_pkg` also honors `COLCON_IGNORE` and would ignore the Noetic wrapper.

## Dependencies

ROS 2 dependencies:

- ROS 2 Humble
- `ament_cmake`
- `rclcpp`
- `std_msgs`
- `sensor_msgs`
- `nav_msgs`
- `geometry_msgs`
- `visualization_msgs`
- `message_filters`
- `tf2`, `tf2_ros`, `tf2_geometry_msgs`
- `pcl_conversions`, `pcl_ros`

ROS 1 dependencies:

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

Native/library dependencies:

- PCL
- Eigen3
- OpenCV
- Protobuf
- glog
- OpenMP
- GTSAM
- small_gicp

Default external LIO topics:

- point cloud: `/cloud_registered_body` (`sensor_msgs/PointCloud2`)
- odometry: `/Odometry` (`nav_msgs/Odometry`)

The input cloud is expected to be deskewed in lidar/body frame, not pre-transformed into the saved map frame.

## Install System Packages

```bash
sudo apt-get update
sudo apt-get install -y \
  libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libpcl-dev libeigen3-dev \
  libopencv-dev libboost-all-dev libtbb-dev
```

Prepare the ROS 2 workspace and install ROS dependencies:

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone https://github.com/borglab/gtsam.git -b 4.1.1
git clone https://github.com/koide3/small_gicp.git
git clone https://github.com/Killow1998/n3mapping.git

cd ~/ros_ws
rosdep install --from-paths src --ignore-src -r -y
```

Place an external LIO frontend package in the same workspace when mapping from live or bagged LIO output. N3Mapping subscribes to the frontend cloud and odometry topics; the frontend is not built into this branch.

## Build

`noetic/` and `humble/` are separate wrapper packages with the same package name, `n3mapping`. Build from the matching ROS workspace; the core sources are shared from the repository root.

### ROS 2 Humble

Build GTSAM first:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws
colcon build --packages-select gtsam --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
```

Then build `small_gicp` and `n3mapping`:

```bash
cd ~/ros_ws
colcon build --packages-up-to n3mapping --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

If the whole repository is checked out into a mixed ROS 1 / ROS 2 source tree, ensure the Humble workspace discovers `humble/` as the `n3mapping` package and does not also discover `noetic/` with the same package name.

### ROS 1 Noetic

Build GTSAM first:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
src/n3mapping/scripts/select_distro_wrapper.sh noetic
catkin build gtsam --no-status -j2 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
```

Then build `small_gicp` and `n3mapping`:

```bash
cd ~/catkin_ws
src/n3mapping/scripts/select_distro_wrapper.sh
catkin build small_gicp n3mapping --no-status -j2 --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

The Noetic build discovers `noetic/package.xml`, produces the package name `n3mapping`, and builds an executable named `n3mapping_node`. It compiles `n3mapping_core` from the shared root sources plus the thin wrapper sources under `noetic/`.

## Configuration

Default config:

```text
config/n3mapping.yaml
```

Important parameters:

- `mode`: `mapping`, `localization`, or `map_extension`
- `cloud_topic`: default `/cloud_registered_body`
- `odom_topic`: default `/Odometry`
- `map_path`: pbstream file to load in localization/map extension
- `map_save_path`: directory for saved maps in mapping mode
- `global_map_publish_hz`: global map publish frequency for wrappers
- `global_map_voxel_size`: voxel size for published global map
- `save_global_map_voxel_size`: voxel size for saved `global_map.pcd`; `0.0` keeps full resolution
- `rhpd_enabled`: default `true`
- `rhpd_num_candidates`, `rhpd_preselect_candidates`: RHPD retrieval count and coarse preselect count
- `rhpd_use_sc_yaw`: use ScanContext yaw as an auxiliary yaw estimate
- `sc_aux_weight`, `sc_aux_veto_*`: ScanContext auxiliary rerank/veto settings
- `reloc_temporal_window_size`, `reloc_lock_*`: relocalization lock stability gate

RHPD is the primary retrieval descriptor. ScanContext is auxiliary only: yaw estimation, weak rerank/veto, or fallback when RHPD is disabled/unavailable.

For topic or map-path overrides, keep the shared launch/config files unchanged and pass a separate config file through the launch `config_file` argument. This is the same workflow for Humble and Noetic; only the launch command syntax differs.

Example external config fragment:

```yaml
n3mapping_node:
  ros__parameters:
    cloud_topic: "/scout/slam/cloud_registered_body"
    odom_topic: "/scout/slam/odometry"
    map_path: "/path/to/n3map.pbstream"
```

## Run

### ROS 2 Humble

Mapping:

```bash
source ~/ros_ws/install/setup.bash
ros2 launch n3mapping mapping.launch.py config_file:=/path/to/n3mapping.yaml
```

Localization:

```bash
source ~/ros_ws/install/setup.bash
ros2 launch n3mapping localization.launch.py config_file:=/path/to/n3mapping.yaml
```

Map extension:

```bash
source ~/ros_ws/install/setup.bash
ros2 launch n3mapping map_extension.launch.py config_file:=/path/to/n3mapping.yaml
```

### ROS 1 Noetic

Mapping:

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch n3mapping mapping.launch \
  config_file:=/path/to/n3mapping.yaml \
  rviz:=true
```

Localization:

```bash
roslaunch n3mapping localization.launch \
  config_file:=/path/to/n3mapping.yaml
```

Map extension:

```bash
roslaunch n3mapping map_extension.launch \
  config_file:=/path/to/n3mapping.yaml
```

The launch files start RViz with the package RViz configs. The wrapper publishes:

Noetic `mapping`, `localization`, and `map_extension` launch files share `launch/n3_noetic.rviz`. Humble `mapping`, `localization`, and `map_extension` launch files share `launch/n3.rviz`.

- `/n3mapping/odometry`
- `/n3mapping/path`
- `/n3mapping/cloud_body`
- `/n3mapping/cloud_world`
- `/n3mapping/loop_closure_markers`
- `/n3mapping/global_map`
- `/n3mapping/relocalization_lock`
- `/n3mapping/save_map` (`std_srvs/Trigger`, ROS 1)

## Map Files

Mapping mode saves:

- `n3map.pbstream`
- `global_map.pcd`

RHPD descriptors are schema-checked on load. If an old map has missing, invalid, or incompatible RHPD descriptors, N3Mapping rebuilds them from saved keyframes when possible.

## Tests

### ROS 2 Humble

Run the Humble test suite:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws
colcon build --packages-up-to n3mapping --symlink-install --cmake-args -DBUILD_TESTING=ON
ROS_LOG_DIR=/tmp/ros_log colcon test --packages-select n3mapping
colcon test-result --test-result-base build/n3mapping --verbose
```

The Humble suite includes core mapping/relocalization/loop tests, Humble conversion tests, and wrapper boundary checks.

### ROS 1 Noetic

Run the Noetic test suite:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
src/n3mapping/scripts/select_distro_wrapper.sh noetic
catkin build n3mapping --no-status -j2 --catkin-make-args run_tests
source devel/setup.bash
cd build/n3mapping
ctest --output-on-failure
```

The Noetic suite includes the shared core loop/relocalization tests (`test_loop_detector`, `test_world_localizing`, `test_synthetic_relocalization`) plus wrapper/core boundary checks. Noetic does not run Humble message-conversion tests.

## Synthetic Relocalization Visualization

Use these tools to inspect whether a synthetic query starts misaligned and then relocalizes onto the saved map.

### ROS 2 Humble

```bash
source ~/ros_ws/install/setup.bash
ros2 launch n3mapping synthetic_relocalization_visualization.launch.py \
  map:=$HOME/ros_ws/src/n3mapping/map/n3map.pbstream \
  max_tests:=20 \
  interval_sec:=20 \
  random_seed:=-1 \
  dropout:=0.3 \
  noise_sigma:=0.02 \
  fake_odom_yaw_deg:=90
```

### ROS 1 Noetic

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch n3mapping synthetic_relocalization_visualization.launch \
  map:=/path/to/n3map.pbstream \
  max_tests:=20 \
  interval_sec:=20 \
  random_seed:=-1 \
  dropout:=0.3 \
  noise_sigma:=0.02 \
  fake_odom_yaw_deg:=90
```

For a non-RViz batch evaluation, both wrappers build `n3mapping_synthetic_relocalization_eval`:

```bash
rosrun n3mapping n3mapping_synthetic_relocalization_eval \
  --map /path/to/n3map.pbstream \
  --max_queries 100 \
  --query_source local_submap \
  --dropout 0.3 \
  --noise_sigma 0.02 \
  --strict
```

RViz topics:

- gray: `/n3mapping/synthetic/global_map`
- red: `/n3mapping/synthetic/query_before`
- green: `/n3mapping/synthetic/query_after`
- blue: `/n3mapping/synthetic/query_gt`
- markers/text: `/n3mapping/synthetic/relocalization_markers`

## Debugging Loop Closure and Optimization

Mapping mode publishes loop closure markers on:

```text
/n3mapping/loop_closure_markers
```

Optimization diagnostics are appended under the configured map save directory:

```text
optimization.log
```

This log records accepted loop count and pose update statistics so you can inspect whether loop closure actually changed the trajectory.

## Architecture Guardrails

These are design constraints for future development, not extra runtime steps:

- RHPD remains the main descriptor path for loop and relocalization candidate retrieval.
- ScanContext should stay auxiliary only: yaw hint, weak rerank/veto, or fallback when RHPD is disabled or unavailable.
- ROS wrappers should stay thin. Backend mapping, loop closure, relocalization, graph optimization, and map serialization logic should live in `n3mapping_core`.
