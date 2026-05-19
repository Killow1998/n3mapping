# N3Mapping

N3Mapping is a graph-backend SLAM and relocalization package with a ROS-free C++ core and thin ROS wrappers. It consumes external LIO output as an undistorted body-frame point cloud plus odometry, performs RHPD-primary loop/relocalization retrieval with ScanContext as auxiliary yaw/rerank/fallback, verifies candidates with small_gicp, optimizes a GTSAM pose graph, and saves maps as `pbstream` files.

Current primary ROS target: ROS 2 Humble. A ROS 1 Noetic wrapper skeleton exists under `ros1/`, but the actively verified workflow in this branch is Humble.

## Architecture

- `n3mapping_core`: ROS-free C++ backend library.
- `n3mapping_ros2_wrapper`: ROS 2 message/config conversion wrapper.
- `n3mapping_node`: ROS 2 executable for mapping, localization, and map extension.
- `ros1/`: Noetic wrapper skeleton for future validation.

The core API boundary is an external-LIO frame: point cloud in body/lidar frame and odometry pose. N3Mapping does not currently embed FAST-LIO, DLIO, or SuperLIO in this branch.

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

Install ROS dependencies for the workspace in the usual way:

```bash
cd ~/ros_ws/to_migrate_ws
rosdep install --from-paths src --ignore-src -r -y
```

If your environment does not provide `gtsam` or `small_gicp` as ROS/CMake packages, put them in the same workspace and build them before or together with `n3mapping`.

## Build

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws/to_migrate_ws
colcon build --packages-select n3mapping --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

For tests:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws/to_migrate_ws
colcon build --packages-select n3mapping --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
```

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
- `rhpd_enabled`: default `true`
- `rhpd_num_candidates`, `rhpd_preselect_candidates`: RHPD retrieval count and coarse preselect count
- `rhpd_use_sc_yaw`: use ScanContext yaw as an auxiliary yaw estimate
- `sc_aux_weight`, `sc_aux_veto_*`: ScanContext auxiliary rerank/veto settings
- `reloc_temporal_window_size`, `reloc_lock_*`: relocalization lock stability gate

RHPD is the primary retrieval descriptor. ScanContext is auxiliary only: yaw estimation, weak rerank/veto, or fallback when RHPD is disabled/unavailable.

## Run

Mapping:

```bash
source ~/ros_ws/to_migrate_ws/install/setup.bash
ros2 launch n3mapping mapping.launch.py \
  config_file:=/home/user/ros_ws/to_migrate_ws/src/n3mapping/config/n3mapping.yaml
```

Localization:

```bash
source ~/ros_ws/to_migrate_ws/install/setup.bash
ros2 launch n3mapping localization.launch.py \
  config_file:=/home/user/ros_ws/to_migrate_ws/src/n3mapping/config/n3mapping.yaml
```

Map extension:

```bash
source ~/ros_ws/to_migrate_ws/install/setup.bash
ros2 launch n3mapping map_extension.launch.py \
  config_file:=/home/user/ros_ws/to_migrate_ws/src/n3mapping/config/n3mapping.yaml
```

The launch files start RViz with the package RViz configs. The wrapper publishes:

- `/n3mapping/odometry`
- `/n3mapping/path`
- `/n3mapping/cloud_body`
- `/n3mapping/cloud_world`
- `/n3mapping/loop_closure_markers`
- `/n3mapping/global_map`
- `/n3mapping/relocalization_lock`

## Map Files

Mapping mode saves:

- `n3map.pbstream`
- `global_map.pcd`

RHPD descriptors are schema-checked on load. If an old map has missing, invalid, or incompatible RHPD descriptors, N3Mapping rebuilds them from saved keyframes when possible.

## Tests

Run all tests:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws/to_migrate_ws
colcon build --packages-select n3mapping --cmake-args -DBUILD_TESTING=ON
ROS_LOG_DIR=/tmp/ros_log colcon test --packages-select n3mapping
colcon test-result --test-result-base build/n3mapping --verbose
```

## Synthetic Relocalization Visualization

Use this to inspect whether a synthetic query starts misaligned and then relocalizes onto the saved map:

```bash
source ~/ros_ws/to_migrate_ws/install/setup.bash
ros2 launch n3mapping synthetic_relocalization_visualization.launch.py \
  map:=/home/user/ros_ws/to_migrate_ws/src/n3mapping/map/n3map.pbstream \
  max_tests:=20 \
  interval_sec:=20 \
  random_seed:=-1 \
  dropout:=0.3 \
  noise_sigma:=0.02 \
  fake_odom_yaw_deg:=90
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
