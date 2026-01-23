#include "n3mapping/config.h"

namespace n3mapping {

void
Config::loadFromROS(rclcpp::Node* node)
{
    // 声明并获取参数
    auto declare_and_get = [&node](const std::string& name, auto& value) {
        node->declare_parameter(name, value);
        node->get_parameter(name, value);
    };

    // 声明并获取字符串参数，空字符串时保留默认值
    auto declare_and_get_string = [&node](const std::string& name, std::string& value) {
        std::string default_value = value; // 保存编译时默认值
        node->declare_parameter(name, value);
        std::string param_value;
        node->get_parameter(name, param_value);
        if (!param_value.empty()) {
            value = param_value; // 只有非空时才覆盖
        }
        // 空字符串时保留 default_value (即编译时的默认值)
    };

    // 运行模式
    declare_and_get("mode", mode);
    declare_and_get_string("map_path", map_path);

    // 话题配置
    declare_and_get("cloud_topic", cloud_topic);
    declare_and_get("odom_topic", odom_topic);
    declare_and_get("output_odom_topic", output_odom_topic);
    declare_and_get("output_path_topic", output_path_topic);
    declare_and_get("output_cloud_body_topic", output_cloud_body_topic);
    declare_and_get("output_cloud_world_topic", output_cloud_world_topic);

    // 坐标系配置
    declare_and_get("world_frame", world_frame);
    declare_and_get("body_frame", body_frame);

    // 关键帧选择
    declare_and_get("keyframe_distance_threshold", keyframe_distance_threshold);
    declare_and_get("keyframe_angle_threshold", keyframe_angle_threshold);

    // 点云配准
    declare_and_get("gicp_downsampling_resolution", gicp_downsampling_resolution);
    declare_and_get("gicp_max_correspondence_distance", gicp_max_correspondence_distance);
    declare_and_get("gicp_max_iterations", gicp_max_iterations);
    declare_and_get("gicp_transformation_epsilon", gicp_transformation_epsilon);
    declare_and_get("gicp_rotation_epsilon_deg", gicp_rotation_epsilon_deg);
    declare_and_get("gicp_fitness_threshold", gicp_fitness_threshold);
    declare_and_get("gicp_num_neighbors", gicp_num_neighbors);
    declare_and_get("gicp_submap_size", gicp_submap_size);
    declare_and_get("icp_refine_use_gicp", icp_refine_use_gicp);
    declare_and_get("icp_refine_max_iterations", icp_refine_max_iterations);
    declare_and_get("icp_refine_max_correspondence_distance", icp_refine_max_correspondence_distance);
    declare_and_get("icp_refine_downsampling_resolution", icp_refine_downsampling_resolution);
    declare_and_get("icp_refine_fitness_gate", icp_refine_fitness_gate);
    declare_and_get("icp_refine_delta_translation_gate", icp_refine_delta_translation_gate);
    declare_and_get("icp_refine_delta_rotation_gate", icp_refine_delta_rotation_gate);

    // 回环检测
    declare_and_get("sc_dist_threshold", sc_dist_threshold);
    declare_and_get("sc_num_exclude_recent", sc_num_exclude_recent);
    declare_and_get("sc_num_candidates", sc_num_candidates);
    declare_and_get("sc_max_radius", sc_max_radius);
    declare_and_get("sc_num_rings", sc_num_rings);
    declare_and_get("sc_num_sectors", sc_num_sectors);

    // 图优化
    declare_and_get("optimization_iterations", optimization_iterations);
    declare_and_get("prior_noise_position", prior_noise_position);
    declare_and_get("prior_noise_rotation", prior_noise_rotation);
    declare_and_get("odom_noise_position", odom_noise_position);
    declare_and_get("odom_noise_rotation", odom_noise_rotation);
    declare_and_get("loop_noise_position", loop_noise_position);
    declare_and_get("loop_noise_rotation", loop_noise_rotation);

    // 点云输出
    declare_and_get("output_cloud_voxel_size", output_cloud_voxel_size);

    // 地图序列化
    declare_and_get_string("map_save_path", map_save_path);

    // 全局地图
    declare_and_get("global_map_voxel_size", global_map_voxel_size);
    declare_and_get("save_global_map_on_shutdown", save_global_map_on_shutdown);

    // 并行计算
    declare_and_get("num_threads", num_threads);

    // 时间同步
    declare_and_get("sync_time_tolerance", sync_time_tolerance);

    // 重定位
    declare_and_get("reloc_num_candidates", reloc_num_candidates);
    declare_and_get("reloc_sc_dist_threshold", reloc_sc_dist_threshold);
    declare_and_get("reloc_min_confidence", reloc_min_confidence);
    declare_and_get("reloc_min_inlier_ratio", reloc_min_inlier_ratio);
    declare_and_get("reloc_search_radius", reloc_search_radius);
    declare_and_get("reloc_max_track_failures", reloc_max_track_failures);
    declare_and_get("reloc_track_max_translation", reloc_track_max_translation);
    declare_and_get("reloc_track_max_rotation", reloc_track_max_rotation);
}

void
Config::print(const rclcpp::Logger& logger) const
{
    RCLCPP_INFO(logger, "========== N3Mapping Configuration ==========");

    // --- General ---
    RCLCPP_INFO(logger, "Mode: %s", mode.c_str());
    RCLCPP_INFO(logger, "Map path: %s", map_path.c_str());
    RCLCPP_INFO(logger, "Map save path: %s", map_save_path.c_str());

    // --- Frames ---
    RCLCPP_INFO(logger, "--- Frames ---");
    RCLCPP_INFO(logger, "  World frame: %s", world_frame.c_str());
    RCLCPP_INFO(logger, "  Body frame: %s", body_frame.c_str());

    // --- Topics ---
    RCLCPP_INFO(logger, "--- Topics ---");
    RCLCPP_INFO(logger, "  Cloud topic: %s", cloud_topic.c_str());
    RCLCPP_INFO(logger, "  Odom topic: %s", odom_topic.c_str());
    RCLCPP_INFO(logger, "  Output odom topic: %s", output_odom_topic.c_str());
    RCLCPP_INFO(logger, "  Output path topic: %s", output_path_topic.c_str());
    RCLCPP_INFO(logger, "  Output cloud body: %s", output_cloud_body_topic.c_str());
    RCLCPP_INFO(logger, "  Output cloud world: %s", output_cloud_world_topic.c_str());

    // --- Keyframe Selection ---
    RCLCPP_INFO(logger, "--- Keyframe Selection ---");
    RCLCPP_INFO(logger, "  Distance threshold: %.2f m", keyframe_distance_threshold);
    RCLCPP_INFO(logger, "  Angle threshold: %.2f rad", keyframe_angle_threshold);

    // --- Point Cloud Registration ---
    RCLCPP_INFO(logger, "--- Point Cloud Registration (GICP) ---");
    RCLCPP_INFO(logger, "  Downsampling resolution: %.2f m", gicp_downsampling_resolution);
    RCLCPP_INFO(logger, "  Max correspondence dist: %.2f m", gicp_max_correspondence_distance);
    RCLCPP_INFO(logger, "  Max iterations: %d", gicp_max_iterations);
    RCLCPP_INFO(logger, "  Transformation epsilon: %.1e", gicp_transformation_epsilon);
    RCLCPP_INFO(logger, "  Rotation epsilon: %.2f deg", gicp_rotation_epsilon_deg);
    RCLCPP_INFO(logger, "  Fitness threshold: %.3f", gicp_fitness_threshold);
    RCLCPP_INFO(logger, "  Num neighbors: %d", gicp_num_neighbors);
    RCLCPP_INFO(logger, "  Submap size: %d (2N+1 frames)", gicp_submap_size);
    RCLCPP_INFO(logger, "  Refine use GICP: %s", icp_refine_use_gicp ? "true" : "false");
    RCLCPP_INFO(logger, "  Refine max iterations: %d", icp_refine_max_iterations);
    RCLCPP_INFO(logger, "  Refine max correspondence dist: %.2f m", icp_refine_max_correspondence_distance);
    RCLCPP_INFO(logger, "  Refine downsampling resolution: %.2f m", icp_refine_downsampling_resolution);
    RCLCPP_INFO(logger, "  Refine fitness gate: %.3f", icp_refine_fitness_gate);
    RCLCPP_INFO(logger, "  Refine delta translation gate: %.2f m", icp_refine_delta_translation_gate);
    RCLCPP_INFO(logger, "  Refine delta rotation gate: %.2f rad", icp_refine_delta_rotation_gate);

    // --- Loop Detection ---
    RCLCPP_INFO(logger, "--- Loop Detection (ScanContext) ---");
    RCLCPP_INFO(logger, "  SC dist threshold: %.3f", sc_dist_threshold);
    RCLCPP_INFO(logger, "  Exclude recent frames: %d", sc_num_exclude_recent);
    RCLCPP_INFO(logger, "  Num candidates: %d", sc_num_candidates);
    RCLCPP_INFO(logger, "  Max radius: %.2f m", sc_max_radius);
    RCLCPP_INFO(logger, "  Num rings: %d", sc_num_rings);
    RCLCPP_INFO(logger, "  Num sectors: %d", sc_num_sectors);

    // --- Graph Optimization ---
    RCLCPP_INFO(logger, "--- Graph Optimization ---");
    RCLCPP_INFO(logger, "  Optimization iterations: %d", optimization_iterations);
    RCLCPP_INFO(logger, "  Prior noise (pos/rot): %.4f / %.4f", prior_noise_position, prior_noise_rotation);
    RCLCPP_INFO(logger, "  Odom noise (pos/rot): %.4f / %.4f", odom_noise_position, odom_noise_rotation);
    RCLCPP_INFO(logger, "  Loop noise (pos/rot): %.4f / %.4f", loop_noise_position, loop_noise_rotation);

    // --- Relocalization ---
    RCLCPP_INFO(logger, "--- Relocalization ---");
    RCLCPP_INFO(logger, "  Num candidates: %d", reloc_num_candidates);
    RCLCPP_INFO(logger, "  SC dist threshold: %.3f", reloc_sc_dist_threshold);
    RCLCPP_INFO(logger, "  Min confidence: %.2f", reloc_min_confidence);
    RCLCPP_INFO(logger, "  Min inlier ratio: %.2f", reloc_min_inlier_ratio);
    RCLCPP_INFO(logger, "  Search radius: %.2f m", reloc_search_radius);
    RCLCPP_INFO(logger, "  Max track failures: %d", reloc_max_track_failures);
    RCLCPP_INFO(logger, "  Track max translation: %.2f m", reloc_track_max_translation);
    RCLCPP_INFO(logger, "  Track max rotation: %.2f rad", reloc_track_max_rotation);

    // --- Map Output & System ---
    RCLCPP_INFO(logger, "--- Map Output & System ---");
    RCLCPP_INFO(logger, "  Output cloud voxel size: %.2f m", output_cloud_voxel_size);
    RCLCPP_INFO(logger, "  Global map voxel size: %.2f m", global_map_voxel_size);
    RCLCPP_INFO(logger, "  Save global map on shutdown: %s", save_global_map_on_shutdown ? "true" : "false");
    RCLCPP_INFO(logger, "  Num threads: %d", num_threads);
    RCLCPP_INFO(logger, "  Sync time tolerance: %.3f s", sync_time_tolerance);

    RCLCPP_INFO(logger, "==============================================");
}

} // namespace n3mapping
