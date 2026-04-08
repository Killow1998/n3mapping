# N3Mapping

N3Mapping is a ROS 1 Noetic backend SLAM package. It takes undistorted point clouds and odometry from FAST-LIO2, runs ScanContext loop detection, aligns point clouds with small_gicp, optimizes the pose graph with GTSAM, and saves the result as a `pbstream` for relocalization or map extension.

## Dependencies
- ROS 1 Noetic
- PCL, Eigen3, OpenCV
- Protobuf
- glog, OpenMP
- small_gicp
- GTSAM
- FAST-LIO2 (default inputs: `/cloud_registered_body` and `/Odometry`)

## System packages
```bash
sudo apt-get install -y \
  libprotobuf-dev protobuf-compiler \
  libgoogle-glog-dev libpcl-dev libeigen3-dev \
  libopencv-dev libboost-all-dev libtbb-dev
```

## Build
Put `gtsam`, `small_gicp`, `n3mapping`, and the frontend package in the same workspace.

From the workspace root:

```bash
source /opt/ros/noetic/setup.bash
catkin init
git clone https://github.com/borglab/gtsam.git -b 4.1.1
git clone https://github.com/koide3/small_gicp.git
```

If the workspace is already initialized, skip `catkin init`.

Build `gtsam` with its CMake options:

```bash
catkin build gtsam --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
```

Build `small_gicp` and `n3mapping` in `Release` mode:

```bash
catkin build small_gicp n3mapping --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Configuration
The default config file is `config/n3mapping.yaml`.

Common parameters:

- `mode`: `mapping`, `localization`, `map_extension`
- `cloud_topic`, `odom_topic`
- `map_path`, `map_save_path`
- ScanContext, small_gicp, and GTSAM parameters

Maps are written to `map/` by default. Change the save path if needed.

Also note:
- runtime mode: `mapping`, `localization`, `map_extension`
  - `map_extension` is the external mode name; internally this path is implemented by `MappingResuming`.
- topics and frames
- ScanContext / Hybrid ScanContext parameters
- RHPD relocalization parameters
- loop verification thresholds
- map save/load paths

Loop-candidate retrieval semantics (mapping mode):

- Active path: descriptor retrieval (ScanContext KD-tree + refined descriptor distance) -> ICP verification -> geometric gate -> loop edge filtering.
- `loop_closest_id_th`, `loop_min_id_interval`, and `loop_max_range` are retained as compatibility/legacy parameters.
- These three legacy parameters are currently not used in the active mapping loop-candidate retrieval/verification path.

## Run
After `source devel/setup.bash`:

- Mapping
  ```bash
  roslaunch n3mapping mapping.launch config_file:=/path/to/n3mapping.yaml
  ```
- Relocalization
  ```bash
  roslaunch n3mapping localization.launch config_file:=/path/to/n3mapping.yaml
  ```
- Map extension
  ```bash
  roslaunch n3mapping map_extension.launch config_file:=/path/to/n3mapping.yaml
  ```

## Topics
- Subscribed: `cloud_topic` (`sensor_msgs/PointCloud2`, default `/cloud_registered_body`), `odom_topic` (`nav_msgs/Odometry`, default `/Odometry`)
- Published: `/n3mapping/odometry`, `/n3mapping/path`, `/n3mapping/cloud_body`, `/n3mapping/cloud_world`

## Map files
On shutdown, the node can save `global_map.pcd` and `n3map.pbstream`.

## Tests
```bash
catkin build n3mapping --cmake-args -DBUILD_TESTING=ON
catkin test n3mapping
```
