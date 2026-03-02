#include "n3mapping/config.h"

namespace n3mapping {

void
Config::loadFromROS(ros::NodeHandle& node)
{
    auto get_param = [&node](const std::string& name, auto& value) {
        node.param(name, value, value);
    };

    auto get_param_string = [&node](const std::string& name, std::string& value) {
        std::string param_value;
        if (node.getParam(name, param_value)) {
            if (!param_value.empty()) {
                value = param_value;
            }
        }
    };

    // 运行模式
    get_param("mode", mode);
    get_param_string("map_path", map_path);

    // 话题配置
    get_param("cloud_topic", cloud_topic);
    get_param("odom_topic", odom_topic);
    get_param("output_odom_topic", output_odom_topic);
    get_param("output_path_topic", output_path_topic);
    get_param("output_cloud_body_topic", output_cloud_body_topic);
    get_param("output_cloud_world_topic", output_cloud_world_topic);

    // 坐标系配置
    get_param("world_frame", world_frame);
    get_param("body_frame", body_frame);

    // 关键帧选择
    get_param("keyframe_distance_threshold", keyframe_distance_threshold);
    get_param("keyframe_angle_threshold", keyframe_angle_threshold);

    // 点云配准
    get_param("gicp_downsampling_resolution", gicp_downsampling_resolution);
    get_param("gicp_max_correspondence_distance", gicp_max_correspondence_distance);
    get_param("gicp_max_iterations", gicp_max_iterations);
    get_param("gicp_transformation_epsilon", gicp_transformation_epsilon);
    get_param("gicp_rotation_epsilon_deg", gicp_rotation_epsilon_deg);
    get_param("gicp_fitness_threshold", gicp_fitness_threshold);
    get_param("gicp_num_neighbors", gicp_num_neighbors);
    get_param("gicp_submap_size", gicp_submap_size);
    get_param("icp_refine_use_gicp", icp_refine_use_gicp);
    get_param("icp_refine_max_iterations", icp_refine_max_iterations);
    get_param("icp_refine_max_correspondence_distance", icp_refine_max_correspondence_distance);
    get_param("icp_refine_downsampling_resolution", icp_refine_downsampling_resolution);
    get_param("icp_refine_fitness_gate", icp_refine_fitness_gate);
    get_param("icp_refine_delta_translation_gate", icp_refine_delta_translation_gate);
    get_param("icp_refine_delta_rotation_gate", icp_refine_delta_rotation_gate);

    // 回环检测
    get_param("sc_dist_threshold", sc_dist_threshold);
    get_param("sc_num_exclude_recent", sc_num_exclude_recent);
    get_param("sc_num_candidates", sc_num_candidates);
    get_param("sc_max_radius", sc_max_radius);
    get_param("sc_num_rings", sc_num_rings);
    get_param("sc_num_sectors", sc_num_sectors);

    // 图优化
    get_param("optimization_iterations", optimization_iterations);
    get_param("prior_noise_position", prior_noise_position);
    get_param("prior_noise_rotation", prior_noise_rotation);
    get_param("odom_noise_position", odom_noise_position);
    get_param("odom_noise_rotation", odom_noise_rotation);
    get_param("loop_noise_position", loop_noise_position);
    get_param("loop_noise_rotation", loop_noise_rotation);
    get_param("loop_min_inlier_ratio", loop_min_inlier_ratio);
    get_param("loop_fitness_threshold", loop_fitness_threshold);
    get_param("loop_max_pre_translation_error", loop_max_pre_translation_error);
    get_param("loop_max_pre_rotation_error", loop_max_pre_rotation_error);
    get_param("loop_max_z_diff", loop_max_z_diff);
    get_param("loop_use_icp_information", loop_use_icp_information);

    // 点云输出
    get_param("output_cloud_voxel_size", output_cloud_voxel_size);

    // 地图序列化
    get_param_string("map_save_path", map_save_path);

    // 全局地图
    get_param("global_map_voxel_size", global_map_voxel_size);
    get_param("save_global_map_on_shutdown", save_global_map_on_shutdown);

    // 并行计算
    get_param("num_threads", num_threads);

    // 时间同步
    get_param("sync_time_tolerance", sync_time_tolerance);

    // 重定位
    get_param("reloc_num_candidates", reloc_num_candidates);
    get_param("reloc_sc_dist_threshold", reloc_sc_dist_threshold);
    get_param("reloc_min_confidence", reloc_min_confidence);
    get_param("reloc_min_inlier_ratio", reloc_min_inlier_ratio);
    get_param("reloc_search_radius", reloc_search_radius);
    get_param("reloc_max_track_failures", reloc_max_track_failures);
    get_param("reloc_track_max_translation", reloc_track_max_translation);
    get_param("reloc_track_max_rotation", reloc_track_max_rotation);
}

void
Config::print() const
{
    ROS_INFO("========== N3Mapping Configuration ==========");

    // --- General ---
    ROS_INFO("Mode: %s", mode.c_str());
    ROS_INFO("Map path: %s", map_path.c_str());
    ROS_INFO("Map save path: %s", map_save_path.c_str());

    // --- Frames ---
    ROS_INFO("--- Frames ---");
    ROS_INFO("  World frame: %s", world_frame.c_str());
    ROS_INFO("  Body frame: %s", body_frame.c_str());

    // --- Topics ---
    ROS_INFO("--- Topics ---");
    ROS_INFO("  Cloud topic: %s", cloud_topic.c_str());
    ROS_INFO("  Odom topic: %s", odom_topic.c_str());
    ROS_INFO("  Output odom topic: %s", output_odom_topic.c_str());
    ROS_INFO("  Output path topic: %s", output_path_topic.c_str());
    ROS_INFO("  Output cloud body: %s", output_cloud_body_topic.c_str());
    ROS_INFO("  Output cloud world: %s", output_cloud_world_topic.c_str());

    // --- Keyframe Selection ---
    ROS_INFO("--- Keyframe Selection ---");
    ROS_INFO("  Distance threshold: %.2f m", keyframe_distance_threshold);
    ROS_INFO("  Angle threshold: %.2f rad", keyframe_angle_threshold);

    // --- Point Cloud Registration ---
    ROS_INFO("--- Point Cloud Registration (GICP) ---");
    ROS_INFO("  Downsampling resolution: %.2f m", gicp_downsampling_resolution);
    ROS_INFO("  Max correspondence dist: %.2f m", gicp_max_correspondence_distance);
    ROS_INFO("  Max iterations: %d", gicp_max_iterations);
    ROS_INFO("  Transformation epsilon: %.1e", gicp_transformation_epsilon);
    ROS_INFO("  Rotation epsilon: %.2f deg", gicp_rotation_epsilon_deg);
    ROS_INFO("  Fitness threshold: %.3f", gicp_fitness_threshold);
    ROS_INFO("  Num neighbors: %d", gicp_num_neighbors);
    ROS_INFO("  Submap size: %d (2N+1 frames)", gicp_submap_size);
    ROS_INFO("  Refine use GICP: %s", icp_refine_use_gicp ? "true" : "false");
    ROS_INFO("  Refine max iterations: %d", icp_refine_max_iterations);
    ROS_INFO("  Refine max correspondence dist: %.2f m", icp_refine_max_correspondence_distance);
    ROS_INFO("  Refine downsampling resolution: %.2f m", icp_refine_downsampling_resolution);
    ROS_INFO("  Refine fitness gate: %.3f", icp_refine_fitness_gate);
    ROS_INFO("  Refine delta translation gate: %.2f m", icp_refine_delta_translation_gate);
    ROS_INFO("  Refine delta rotation gate: %.2f rad", icp_refine_delta_rotation_gate);

    // --- Loop Detection ---
    ROS_INFO("--- Loop Detection (ScanContext) ---");
    ROS_INFO("  SC dist threshold: %.3f", sc_dist_threshold);
    ROS_INFO("  Exclude recent frames: %d", sc_num_exclude_recent);
    ROS_INFO("  Num candidates: %d", sc_num_candidates);
    ROS_INFO("  Max radius: %.2f m", sc_max_radius);
    ROS_INFO("  Num rings: %d", sc_num_rings);
    ROS_INFO("  Num sectors: %d", sc_num_sectors);

    // --- Graph Optimization ---
    ROS_INFO("--- Graph Optimization ---");
    ROS_INFO("  Optimization iterations: %d", optimization_iterations);
    ROS_INFO("  Prior noise (pos/rot): %.4f / %.4f", prior_noise_position, prior_noise_rotation);
    ROS_INFO("  Odom noise (pos/rot): %.4f / %.4f", odom_noise_position, odom_noise_rotation);
    ROS_INFO("  Loop noise (pos/rot): %.4f / %.4f", loop_noise_position, loop_noise_rotation);
    ROS_INFO("  Loop min inlier ratio: %.2f", loop_min_inlier_ratio);
    ROS_INFO("  Loop fitness threshold: %.3f", loop_fitness_threshold);
    ROS_INFO("  Loop min inlier ratio: %.2f", loop_min_inlier_ratio);

    // --- Relocalization ---
    ROS_INFO("--- Relocalization ---");
    ROS_INFO("  Num candidates: %d", reloc_num_candidates);
    ROS_INFO("  SC dist threshold: %.3f", reloc_sc_dist_threshold);
    ROS_INFO("  Min confidence: %.2f", reloc_min_confidence);
    ROS_INFO("  Min inlier ratio: %.2f", reloc_min_inlier_ratio);
    ROS_INFO("  Search radius: %.2f m", reloc_search_radius);
    ROS_INFO("  Max track failures: %d", reloc_max_track_failures);
    ROS_INFO("  Track max translation: %.2f m", reloc_track_max_translation);
    ROS_INFO("  Track max rotation: %.2f rad", reloc_track_max_rotation);

    // --- Map Output & System ---
    ROS_INFO("--- Map Output & System ---");
    ROS_INFO("  Output cloud voxel size: %.2f m", output_cloud_voxel_size);
    ROS_INFO("  Global map voxel size: %.2f m", global_map_voxel_size);
    ROS_INFO("  Save global map on shutdown: %s", save_global_map_on_shutdown ? "true" : "false");
    ROS_INFO("  Num threads: %d", num_threads);
    ROS_INFO("  Sync time tolerance: %.3f s", sync_time_tolerance);

    ROS_INFO("==============================================");
}

} // namespace n3mapping
