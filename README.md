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
launch/              shared launch and RViz resources
tools/               synthetic relocalization tools
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
| `/n3mapping/odometry` | topic | `nav_msgs/Odometry` | Optimized or relocalized output pose |
| `/n3mapping/path` | topic | `nav_msgs/Path` | Output trajectory |
| `/n3mapping/cloud_body` | topic | `sensor_msgs/PointCloud2` | Current cloud in body frame |
| `/n3mapping/cloud_world` | topic | `sensor_msgs/PointCloud2` | Current cloud transformed to world/map frame |
| `/n3mapping/global_map` | topic | `sensor_msgs/PointCloud2` | Published global map |
| `/n3mapping/loop_closure_markers` | topic | `visualization_msgs/MarkerArray` | Loop closure and trajectory markers |
| `/n3mapping/relocalization_lock` | topic | `std_msgs/UInt32` | Relocalization lock event counter |
| `/n3mapping/save_map` | service | `std_srvs/Trigger` / `std_srvs/srv/Trigger` | Save map files |

### 2.5 Map Files And Logs

Mapping and map extension save:

```text
<map_save_path>/n3map.pbstream
<map_save_path>/global_map.pcd
<map_save_path>/optimization.log
```

`optimization.log` is truncated when the node starts and appended after accepted keyframes or loop optimizations.

Terminal output is intentionally lightweight. It keeps warnings, errors, map loading, save-map results, shutdown map-save results, and relocalization-lock events.

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
