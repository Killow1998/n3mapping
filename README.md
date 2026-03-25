# N3Mapping

N3Mapping is a ROS 2 backend SLAM package. It consumes undistorted point clouds and odometry from FAST-LIO2, performs loop detection and verification, optimizes a pose graph with GTSAM, and saves maps as `pbstream` files for relocalization and map extension.

## Dependencies

- ROS 2 Humble (`ament_cmake`, `rclcpp`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `message_filters`, `tf2`, `tf2_ros`, `pcl_conversions`, `cv_bridge`)
- PCL, Eigen3, OpenCV
- Protobuf
- glog, OpenMP
- small_gicp
- GTSAM
- FAST-LIO2 frontend (default inputs: `/cloud_registered_body`, `/Odometry`)

## System Packages (Ubuntu)

```bash
sudo apt-get install -y \
  libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libpcl-dev libeigen3-dev \
  libopencv-dev libboost-all-dev libtbb-dev
```

## Build

Put `gtsam`, `small_gicp`, `n3mapping`, and your frontend package in the same ROS 2 workspace.

```bash
source /opt/ros/humble/setup.bash
cd ~/<your ros workspace>/src
git clone https://github.com/borglab/gtsam.git -b 4.1.1
git clone https://github.com/koide3/small_gicp.git
```

Build `gtsam` first with noetic-aligned CMake options:

```bash
cd ~/<your ros workspace>
colcon build --packages-select gtsam --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
```

Then build `small_gicp` and `n3mapping`:

```bash
cd ~/<your ros workspace>
colcon build --packages-up-to n3mapping --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Configuration

Default config file: `config/n3mapping.yaml`

Key groups:

- runtime mode: `mapping`, `localization`, `map_extension`
- topics and frames
- ScanContext / Hybrid ScanContext parameters
- RHPD relocalization parameters
- loop verification thresholds
- map save/load paths

## Run

After `source install/setup.bash`:

- Mapping
  ```bash
  ros2 launch n3mapping mapping.launch.py config_file:=/path/to/n3mapping.yaml
  ```

- Relocalization
  ```bash
  ros2 launch n3mapping localization.launch.py config_file:=/path/to/n3mapping.yaml
  ```

- Map extension
  ```bash
  ros2 launch n3mapping map_extension.launch.py config_file:=/path/to/n3mapping.yaml
  ```

## Topics

- Subscribed:
  - `cloud_topic` (`sensor_msgs/PointCloud2`, default `/cloud_registered_body`)
  - `odom_topic` (`nav_msgs/Odometry`, default `/Odometry`)

- Published:
  - `/n3mapping/odometry`
  - `/n3mapping/path`
  - `/n3mapping/cloud_body`
  - `/n3mapping/cloud_world`
  - `/n3mapping/loop_closure_markers`
  - `/n3mapping/global_map`

## Map Files and Version Policy

Map files include:

- `n3map.pbstream`
- `global_map.pcd`

`pbstream` metadata version is `2.0.0`.

Load policy:

- file version `< 2.0.0`: map load is allowed and RHPD descriptors are rebuilt.
- file version `> 2.0.0`: map load is rejected.
- file version `== 2.0.0` but RHPD missing/invalid: RHPD descriptors are rebuilt.

## Tests

```bash
colcon build --packages-select n3mapping --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select n3mapping
```
