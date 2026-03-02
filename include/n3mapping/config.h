#ifndef N3MAPPING_CONFIG_H
#define N3MAPPING_CONFIG_H

#include <glog/logging.h>
#include <ros/ros.h>
#include <string>

namespace n3mapping {

/**
 * @brief N3Mapping 配置参数结构
 *
 * 包含所有可配置参数，支持从 ROS1 参数服务器加载
 */
struct Config
{
    // ==================== 运行模式 ====================
    std::string mode = "mapping"; // mapping, localization, map_extension
#ifdef N3MAPPING_SOURCE_DIR
    std::string map_path = std::string(N3MAPPING_SOURCE_DIR) + "/map/n3map.pbstream"; // 默认地图文件路径
#else
    std::string map_path = "";           // 地图文件路径 (用于 localization 和 map_extension)
#endif

    // ==================== 话题配置 ====================
    std::string cloud_topic = "/cloud_registered_body";
    std::string odom_topic = "/Odometry";
    std::string output_odom_topic = "/n3mapping/odometry";
    std::string output_path_topic = "/n3mapping/path";
    std::string output_cloud_body_topic = "/n3mapping/cloud_body";
    std::string output_cloud_world_topic = "/n3mapping/cloud_world";

    // ==================== 坐标系配置 ====================
    std::string world_frame = "map"; // 全局地图坐标系
    std::string body_frame = "body"; // 机器人本体坐标系

    // ==================== 关键帧选择 ====================
    double keyframe_distance_threshold = 1.0; // 米
    double keyframe_angle_threshold = 0.5;    // 弧度 (~28.6度)

    // ==================== 点云配准 (small_gicp) ====================
    double gicp_downsampling_resolution = 0.5;           // 下采样分辨率 (米)
    double gicp_max_correspondence_distance = 2.0;       // 最大对应点距离
    int gicp_max_iterations = 30;                        // 最大迭代次数
    double gicp_transformation_epsilon = 1e-6;           // 变换收敛阈值
    double gicp_rotation_epsilon_deg = 1.0;              // 旋转收敛阈值 (度)
    double gicp_fitness_threshold = 0.3;                 // 配准得分阈值
    int gicp_num_neighbors = 20;                         // 法向量估计邻居数
    int gicp_submap_size = 5;                            // 局部子图帧数 (前后各 N 帧，共 2N+1 帧)
    bool icp_refine_use_gicp = true;                     // 是否在粗对齐后使用 GICP 精修
    int icp_refine_max_iterations = 20;                  // 精修迭代次数
    double icp_refine_max_correspondence_distance = 1.0; // 精修最大对应距离
    double icp_refine_downsampling_resolution = 0.1;     // 精修下采样分辨率
    double icp_refine_fitness_gate = 0.5;                // 触发精修的 fitness 上限
    double icp_refine_delta_translation_gate = 3.0;      // 触发精修的平移门限 (米)
    double icp_refine_delta_rotation_gate = 0.5;         // 触发精修的旋转门限 (弧度)

    // ==================== 回环检测 (ScanContext) ====================
    double sc_dist_threshold = 0.2; // ScanContext 距离阈值
    int sc_num_exclude_recent = 50; // 排除最近 N 帧
    int sc_num_candidates = 10;     // 候选帧数量 (Top-K)
    double sc_max_radius = 80.0;    // ScanContext 最大半径
    int sc_num_rings = 20;          // 环数
    int sc_num_sectors = 60;        // 扇区数

    // ==================== KD-Tree 缓存 (LRU 策略) ====================
    size_t kdtree_cache_size = 20; // 缓存最近使用的 N 个关键帧 KD-Tree

    // ==================== 图优化 (GTSAM) ====================
    int optimization_iterations = 10;            // 优化迭代次数
    double prior_noise_position = 0.01;          // 先验位置噪声 (米)
    double prior_noise_rotation = 0.01;          // 先验旋转噪声 (弧度)
    double odom_noise_position = 0.1;            // 里程计位置噪声
    double odom_noise_rotation = 0.1;            // 里程计旋转噪声
    double loop_noise_position = 0.1;            // 回环位置噪声
    double loop_noise_rotation = 0.1;            // 回环旋转噪声
    bool use_robust_kernel = true;               // 回环边使用鲁棒核函数
    std::string robust_kernel_type = "Huber";    // Huber, Cauchy, DCS
    double robust_kernel_delta = 1.0;            // 鲁棒核函数阈值
    double loop_min_inlier_ratio = 0.5;          // 回环验证最小内点比例
    double loop_fitness_threshold = 0.2;         // 回环验证的最大 fitness
    double loop_max_pre_translation_error = 5.0; // 回环一致性门控: 最大平移偏差 (米)
    double loop_max_pre_rotation_error = 1.0;    // 回环一致性门控: 最大旋转偏差 (弧度)
    double loop_max_z_diff = 1.0;                // 回环测量 Z 轴最大偏差 (米)，防止地图分层
    bool loop_use_icp_information = false;       // 是否使用 ICP Hessian 作为回环信息矩阵

    // ==================== 点云输出 ====================
    double output_cloud_voxel_size = 0.2; // 输出点云下采样分辨率 (0 表示不下采样)

    // ==================== 地图序列化 ====================
#ifdef N3MAPPING_SOURCE_DIR
    std::string map_save_path = std::string(N3MAPPING_SOURCE_DIR) + "/map"; // 默认保存到源目录
#else
    std::string map_save_path = "./map"; // 地图保存路径
#endif

    // ==================== 全局地图 ====================
    double global_map_voxel_size = 0.1;      // 全局地图下采样分辨率
    bool save_global_map_on_shutdown = true; // 关闭时保存全局地图

    // ==================== 并行计算 ====================
    int num_threads = 4; // OpenMP 线程数

    // ==================== 时间同步 ====================
    double sync_time_tolerance = 0.1; // 时间同步容差 (秒)

    // ==================== 重定位 ====================
    int reloc_num_candidates = 10;            // 重定位候选帧数量
    double reloc_sc_dist_threshold = 0.3;     // 重定位 ScanContext 距离阈值
    double reloc_min_confidence = 0.3;        // 最小置信度
    double reloc_min_inlier_ratio = 0.03;     // 最小内点比例（相对于 downsampled source 点数）
    double reloc_search_radius = 20.0;        // 跟踪定位搜索半径 (米)
    int reloc_max_track_failures = 20;        // 最大连续跟踪失败次数
    double reloc_track_max_translation = 3.0; // 跟踪阶段允许的最大位移修正 (米)
    double reloc_track_max_rotation = 1.0;    // 跟踪阶段允许的最大旋转修正 (弧度)

    /**
    * @brief 从 ROS1 节点加载配置参数
    * @param node ROS1 节点句柄
     */
    void loadFromROS(ros::NodeHandle& node);

    /**
     * @brief 打印配置信息
     */
    void print() const;
};

} // namespace n3mapping

#endif // N3MAPPING_CONFIG_H
