# N3Mapping

N3Mapping 是基于图优化的 ROS 2 后端 SLAM，接收 FAST-LIO2 去畸变点云与里程计，执行 ScanContext 回环检测、小 GICP 配准、GTSAM 图优化，并将地图序列化为可恢复/续建的 pbstream。

## 依赖
- ROS 2 (ament_cmake、rclcpp、std_msgs、sensor_msgs、nav_msgs、geometry_msgs、message_filters、tf2、tf2_ros、tf2_geometry_msgs、pcl_conversions、pcl_ros、cv_bridge)
- small_gicp（作为 colcon 包编译后被 find_package）
- PCL、Eigen3、OpenCV
- GTSAM、glog、OpenMP
- Protobuf（libprotobuf-dev、protobuf-compiler）
- 前端：FAST-LIO2（默认订阅 `/cloud_registered_body` 与 `/Odometry`）

## 构建
1) 将本包与依赖（如 small_gicp、FAST-LIO2）放在同一个 ROS 2 workspace 的 `src/` 下。  
2) 构建：
   ```bash
   colcon build --packages-select n3mapping --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
   source install/setup.bash
   ```
   Proto 代码会在构建时自动生成到 `build/n3mapping/proto_gen/`。

## 配置
- 默认配置位于 `config/n3mapping.yaml`，可按模式修改 `mode`（mapping/localization/map_extension）、`map_path`、话题名、ScanContext/GICP/GTSAM 参数等。
- 地图默认保存在源目录的 `map/`（由 `N3MAPPING_SOURCE_DIR` 宏设置）；如需避免写入仓库，可在配置中将 `map_save_path`、`map_path` 指向工作盘的其他目录。

## 运行
在终端 `source install/setup.bash` 后：
- 建图：
  ```bash
  ros2 launch n3mapping mapping.launch.py config_file:=/path/to/n3mapping.yaml
  ```
- 重定位（需已有地图 pbstream）：
  ```bash
  ros2 launch n3mapping localization.launch.py config_file:=/path/to/n3mapping.yaml
  ```
- 地图续建：
  ```bash
  ros2 launch n3mapping map_extension.launch.py config_file:=/path/to/n3mapping.yaml
  ```

## 话题
- 订阅：`cloud_topic` (PointCloud2，默认 `/cloud_registered_body`)，`odom_topic` (nav_msgs/Odometry，默认 `/Odometry`)
- 发布：`/n3mapping/odometry`、`/n3mapping/path`、`/n3mapping/cloud_body`、`/n3mapping/cloud_world`

## 数据与地图
- 运行结束时可保存 `global_map.pcd` 与 `n3map.pbstream` 到 `map/` 或自定义 `map_save_path`。
- 大体积地图文件默认被 `.gitignore` 忽略，避免上传仓库。

## 测试
启用测试后运行：
```bash
colcon build --packages-select n3mapping --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select n3mapping
```
