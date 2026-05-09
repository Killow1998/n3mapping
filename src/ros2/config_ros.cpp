#include "n3mapping/ros2/config_ros.h"

#include <rclcpp/rclcpp.hpp>

namespace n3mapping {

void loadConfigFromROS(rclcpp::Node* node, Config& config) {
    auto get = [&node](const std::string& name, auto& value) {
        node->declare_parameter(name, value);
        node->get_parameter(name, value);
    };
    auto gets = [&node](const std::string& name, std::string& value) {
        std::string v;
        node->declare_parameter(name, value);
        node->get_parameter(name, v);
        if (!v.empty()) value = v;
    };

    get("mode", config.mode);
    gets("map_path", config.map_path);

    get("cloud_topic", config.cloud_topic);
    get("odom_topic", config.odom_topic);
    get("frontend_mode", config.frontend_mode);
    get("raw_lidar_topic", config.raw_lidar_topic);
    get("raw_lidar_msg_type", config.raw_lidar_msg_type);
    get("imu_topic", config.imu_topic);
    get("lidar_type", config.lidar_type);
    get("frontend_time_offset", config.frontend_time_offset);
    get("frontend_publish_debug", config.frontend_publish_debug);
    get("frontend_debug_publish_odom", config.frontend_debug_publish_odom);
    get("frontend_debug_publish_deskewed_cloud", config.frontend_debug_publish_deskewed_cloud);
    get("frontend_debug_publish_local_map", config.frontend_debug_publish_local_map);
    get("frontend_debug_publish_timing", config.frontend_debug_publish_timing);
    get("frontend_imu_buffer_max_samples", config.frontend_imu_buffer_max_samples);
    get("frontend_point_filter_num", config.frontend_point_filter_num);
    get("frontend_scan_lines", config.frontend_scan_lines);
    get("frontend_blind", config.frontend_blind);
    get("frontend_max_abs_coordinate", config.frontend_max_abs_coordinate);
    get("frontend_prediction_only_output", config.frontend_prediction_only_output);
    get("dlio_time_encoding", config.dlio_time_encoding);
    get("frontend_lidar_to_body_tx", config.frontend_lidar_to_body_tx);
    get("frontend_lidar_to_body_ty", config.frontend_lidar_to_body_ty);
    get("frontend_lidar_to_body_tz", config.frontend_lidar_to_body_tz);
    get("frontend_lidar_to_body_roll", config.frontend_lidar_to_body_roll);
    get("frontend_lidar_to_body_pitch", config.frontend_lidar_to_body_pitch);
    get("frontend_lidar_to_body_yaw", config.frontend_lidar_to_body_yaw);
    get("frontend_imu_to_body_tx", config.frontend_imu_to_body_tx);
    get("frontend_imu_to_body_ty", config.frontend_imu_to_body_ty);
    get("frontend_imu_to_body_tz", config.frontend_imu_to_body_tz);
    get("frontend_imu_to_body_roll", config.frontend_imu_to_body_roll);
    get("frontend_imu_to_body_pitch", config.frontend_imu_to_body_pitch);
    get("frontend_imu_to_body_yaw", config.frontend_imu_to_body_yaw);
    get("output_odom_topic", config.output_odom_topic);
    get("output_path_topic", config.output_path_topic);
    get("output_cloud_body_topic", config.output_cloud_body_topic);
    get("output_cloud_world_topic", config.output_cloud_world_topic);

    get("world_frame", config.world_frame);
    get("body_frame", config.body_frame);

    get("keyframe_distance_threshold", config.keyframe_distance_threshold);
    get("keyframe_angle_threshold", config.keyframe_angle_threshold);

    get("gicp_downsampling_resolution", config.gicp_downsampling_resolution);
    get("gicp_max_correspondence_distance", config.gicp_max_correspondence_distance);
    get("gicp_max_iterations", config.gicp_max_iterations);
    get("gicp_transformation_epsilon", config.gicp_transformation_epsilon);
    get("gicp_rotation_epsilon_deg", config.gicp_rotation_epsilon_deg);
    get("gicp_fitness_threshold", config.gicp_fitness_threshold);
    get("gicp_num_neighbors", config.gicp_num_neighbors);
    get("gicp_submap_size", config.gicp_submap_size);
    get("icp_refine_use_gicp", config.icp_refine_use_gicp);
    get("icp_refine_max_iterations", config.icp_refine_max_iterations);
    get("icp_refine_max_correspondence_distance", config.icp_refine_max_correspondence_distance);
    get("icp_refine_downsampling_resolution", config.icp_refine_downsampling_resolution);
    get("icp_refine_fitness_gate", config.icp_refine_fitness_gate);
    get("icp_refine_delta_translation_gate", config.icp_refine_delta_translation_gate);
    get("icp_refine_delta_rotation_gate", config.icp_refine_delta_rotation_gate);

    get("sc_dist_threshold", config.sc_dist_threshold);
    get("sc_num_exclude_recent", config.sc_num_exclude_recent);
    get("sc_num_candidates", config.sc_num_candidates);
    get("sc_max_radius", config.sc_max_radius);
    get("sc_num_rings", config.sc_num_rings);
    get("sc_num_sectors", config.sc_num_sectors);

    get("kdtree_cache_size", config.kdtree_cache_size);

    get("optimization_iterations", config.optimization_iterations);
    get("prior_noise_position", config.prior_noise_position);
    get("prior_noise_rotation", config.prior_noise_rotation);
    get("odom_noise_position", config.odom_noise_position);
    get("odom_noise_rotation", config.odom_noise_rotation);
    get("loop_noise_position", config.loop_noise_position);
    get("loop_noise_rotation", config.loop_noise_rotation);
    get("use_robust_kernel", config.use_robust_kernel);
    get("robust_kernel_type", config.robust_kernel_type);
    get("robust_kernel_delta", config.robust_kernel_delta);
    get("loop_min_inlier_ratio", config.loop_min_inlier_ratio);
    get("loop_fitness_threshold", config.loop_fitness_threshold);
    get("loop_max_icp_translation", config.loop_max_icp_translation);
    get("loop_max_icp_rotation", config.loop_max_icp_rotation);
    get("loop_use_icp_information", config.loop_use_icp_information);
    get("loop_kf_gap", config.loop_kf_gap);
    get("loop_closest_id_th", config.loop_closest_id_th);
    get("loop_min_id_interval", config.loop_min_id_interval);
    get("loop_max_range", config.loop_max_range);

    get("output_cloud_voxel_size", config.output_cloud_voxel_size);
    gets("map_save_path", config.map_save_path);
    get("global_map_voxel_size", config.global_map_voxel_size);
    get("save_global_map_on_shutdown", config.save_global_map_on_shutdown);
    get("num_threads", config.num_threads);
    get("sync_time_tolerance", config.sync_time_tolerance);

    get("reloc_num_candidates", config.reloc_num_candidates);
    get("reloc_sc_dist_threshold", config.reloc_sc_dist_threshold);
    get("reloc_min_confidence", config.reloc_min_confidence);
    get("reloc_min_inlier_ratio", config.reloc_min_inlier_ratio);
    get("reloc_search_radius", config.reloc_search_radius);
    get("reloc_max_track_failures", config.reloc_max_track_failures);
    get("reloc_track_max_translation", config.reloc_track_max_translation);
    get("reloc_track_max_rotation", config.reloc_track_max_rotation);
    get("reloc_temporal_window_size", config.reloc_temporal_window_size);
    get("reloc_lock_log_likelihood_threshold", config.reloc_lock_log_likelihood_threshold);
    get("reloc_lock_min_winner_streak", config.reloc_lock_min_winner_streak);
    get("reloc_lock_min_converged_updates", config.reloc_lock_min_converged_updates);
    get("reloc_lock_min_margin", config.reloc_lock_min_margin);
    get("reloc_hypothesis_miss_penalty", config.reloc_hypothesis_miss_penalty);
    get("reloc_hypothesis_not_converged_penalty", config.reloc_hypothesis_not_converged_penalty);
    get("reloc_reloc_inlier_weight", config.reloc_reloc_inlier_weight);
    get("reloc_reloc_desc_dist_weight", config.reloc_reloc_desc_dist_weight);
    get("reloc_track_motion_weight", config.reloc_track_motion_weight);
    get("reloc_track_retry_max_failures", config.reloc_track_retry_max_failures);
    get("reloc_track_retry_corr_scale", config.reloc_track_retry_corr_scale);
    get("reloc_track_retry_max_iterations", config.reloc_track_retry_max_iterations);
    get("reloc_track_unstable_submap_size", config.reloc_track_unstable_submap_size);
    get("reloc_static_agg_enable", config.reloc_static_agg_enable);
    get("reloc_static_agg_max_frames", config.reloc_static_agg_max_frames);
    get("reloc_static_agg_min_frames", config.reloc_static_agg_min_frames);
    get("reloc_static_agg_max_translation", config.reloc_static_agg_max_translation);
    get("reloc_static_agg_max_rotation", config.reloc_static_agg_max_rotation);
    get("reloc_static_agg_voxel_size", config.reloc_static_agg_voxel_size);
    get("reloc_ambiguity_min_margin", config.reloc_ambiguity_min_margin);
    get("reloc_ambiguity_min_ratio", config.reloc_ambiguity_min_ratio);
    get("reloc_ambiguity_min_basin_separation", config.reloc_ambiguity_min_basin_separation);

    get("rhpd_enabled", config.rhpd_enabled);
    get("rhpd_v2_enable", config.rhpd_v2_enable);
    get("rhpd_v3_enable", config.rhpd_v3_enable);
    get("rhpd_max_range", config.rhpd_max_range);
    get("rhpd_z_min", config.rhpd_z_min);
    get("rhpd_z_max", config.rhpd_z_max);
    get("rhpd_dist_threshold", config.rhpd_dist_threshold);
    get("rhpd_num_candidates", config.rhpd_num_candidates);
    get("rhpd_preselect_candidates", config.rhpd_preselect_candidates);
    get("rhpd_submap_kf_radius", config.rhpd_submap_kf_radius);
    get("rhpd_submap_voxel_size", config.rhpd_submap_voxel_size);
    get("rhpd_primary_weight", config.rhpd_primary_weight);
    get("sc_aux_weight", config.sc_aux_weight);
    get("sc_aux_veto_enabled", config.sc_aux_veto_enabled);
    get("sc_aux_veto_threshold", config.sc_aux_veto_threshold);
    get("rhpd_use_sc_yaw", config.rhpd_use_sc_yaw);
    get("rhpd_yaw_hypotheses", config.rhpd_yaw_hypotheses);
    get("rhpd_enable_negative_space", config.rhpd_enable_negative_space);
    get("rhpd_enable_vertical_tokens", config.rhpd_enable_vertical_tokens);
    get("rhpd_enable_pca_confidence", config.rhpd_enable_pca_confidence);
}

void printConfigToROS(const Config& config, const rclcpp::Logger& logger) {
    RCLCPP_INFO(logger, "========== N3Mapping Configuration ==========");
    RCLCPP_INFO(logger, "Mode: %s | Map path: %s", config.mode.c_str(), config.map_path.c_str());
    RCLCPP_INFO(logger,
                "Frontend: mode=%s, raw_lidar=%s, raw_type=%s, imu=%s, lidar_type=%s, time_offset=%.6f, debug_publish=%s",
                config.frontend_mode.c_str(),
                config.raw_lidar_topic.c_str(),
                config.raw_lidar_msg_type.c_str(),
                config.imu_topic.c_str(),
                config.lidar_type.c_str(),
                config.frontend_time_offset,
                config.frontend_publish_debug ? "YES" : "NO");
    RCLCPP_INFO(logger,
                "Frontend debug outputs: odom=%s deskewed_cloud=%s local_map=%s timing=%s",
                config.frontend_debug_publish_odom ? "YES" : "NO",
                config.frontend_debug_publish_deskewed_cloud ? "YES" : "NO",
                config.frontend_debug_publish_local_map ? "YES" : "NO",
                config.frontend_debug_publish_timing ? "YES" : "NO");
    RCLCPP_INFO(logger,
                "Frontend preprocessing: imu_buffer=%d point_filter=%d scan_lines=%d blind=%.3f max_abs_coord=%.1f prediction_only=%s dlio_time_encoding=%s",
                config.frontend_imu_buffer_max_samples,
                config.frontend_point_filter_num,
                config.frontend_scan_lines,
                config.frontend_blind,
                config.frontend_max_abs_coordinate,
                config.frontend_prediction_only_output ? "YES" : "NO",
                config.dlio_time_encoding.c_str());
    RCLCPP_INFO(logger,
                "Frontend extrinsics: T_body_lidar xyz=(%.3f %.3f %.3f) rpy=(%.3f %.3f %.3f) | T_body_imu xyz=(%.3f %.3f %.3f) rpy=(%.3f %.3f %.3f)",
                config.frontend_lidar_to_body_tx,
                config.frontend_lidar_to_body_ty,
                config.frontend_lidar_to_body_tz,
                config.frontend_lidar_to_body_roll,
                config.frontend_lidar_to_body_pitch,
                config.frontend_lidar_to_body_yaw,
                config.frontend_imu_to_body_tx,
                config.frontend_imu_to_body_ty,
                config.frontend_imu_to_body_tz,
                config.frontend_imu_to_body_roll,
                config.frontend_imu_to_body_pitch,
                config.frontend_imu_to_body_yaw);
    RCLCPP_INFO(logger, "Frames: world=%s body=%s", config.world_frame.c_str(), config.body_frame.c_str());
    RCLCPP_INFO(logger, "Keyframe: dist=%.2f m, angle=%.2f rad", config.keyframe_distance_threshold, config.keyframe_angle_threshold);
    RCLCPP_INFO(logger,
                "GICP: res=%.2f, corr=%.2f, iter=%d, fitness_thr=%.3f, submap=%d",
                config.gicp_downsampling_resolution,
                config.gicp_max_correspondence_distance,
                config.gicp_max_iterations,
                config.gicp_fitness_threshold,
                config.gicp_submap_size);
    RCLCPP_INFO(logger,
                "SC: dist_thr=%.3f, exclude=%d, candidates=%d",
                config.sc_dist_threshold,
                config.sc_num_exclude_recent,
                config.sc_num_candidates);
    RCLCPP_INFO(logger,
                "Odom noise: pos=%.4f rot=%.4f | Loop noise: pos=%.4f rot=%.4f",
                config.odom_noise_position,
                config.odom_noise_rotation,
                config.loop_noise_position,
                config.loop_noise_rotation);
    RCLCPP_INFO(logger,
                "Robust kernel: %s type=%s delta=%.2f",
                config.use_robust_kernel ? "ON" : "OFF",
                config.robust_kernel_type.c_str(),
                config.robust_kernel_delta);
    RCLCPP_INFO(logger,
                "Loop: fitness_thr=%.3f, min_inlier=%.2f, max_icp_t=%.2f, max_icp_r=%.2f",
                config.loop_fitness_threshold,
                config.loop_min_inlier_ratio,
                config.loop_max_icp_translation,
                config.loop_max_icp_rotation);
    RCLCPP_INFO(logger,
                "Loop candidate pipeline: RHPD primary retrieval -> optional SC yaw/weak rerank/veto -> ICP -> geom gate -> LoopClosureManager filter/select");
    RCLCPP_WARN(logger,
                "loop_closest_id_th/min_id_interval/max_range are retained for compatibility/logging only; they are not used in the active mapping loop-candidate retrieval/verification path.");
    RCLCPP_INFO(logger,
                "Loop timing: loop_kf_gap=%d (active)",
                config.loop_kf_gap);
    RCLCPP_INFO(logger,
                "Legacy loop distance params (inactive, compatibility/logging only): closest_id_th=%d, min_id_interval=%d, max_range=%.1f",
                config.loop_closest_id_th,
                config.loop_min_id_interval,
                config.loop_max_range);
    RCLCPP_INFO(logger,
                "Reloc: candidates=%d, sc_thr=%.3f, min_conf=%.2f",
                config.reloc_num_candidates,
                config.reloc_sc_dist_threshold,
                config.reloc_min_confidence);
    RCLCPP_INFO(logger,
                "Reloc temporal: window=%d, lock_ll=%.2f, min_streak=%d, min_conv=%d, min_margin=%.2f, miss_pen=%.2f, nonconv_pen=%.2f",
                config.reloc_temporal_window_size,
                config.reloc_lock_log_likelihood_threshold,
                config.reloc_lock_min_winner_streak,
                config.reloc_lock_min_converged_updates,
                config.reloc_lock_min_margin,
                config.reloc_hypothesis_miss_penalty,
                config.reloc_hypothesis_not_converged_penalty);
    RCLCPP_INFO(logger,
                "Reloc score weights: inlier=%.2f, desc=%.2f, motion=%.2f",
                config.reloc_reloc_inlier_weight,
                config.reloc_reloc_desc_dist_weight,
                config.reloc_track_motion_weight);
    RCLCPP_INFO(logger,
                "Reloc retry: max_fail=%d, corr_scale=%.2f, max_iter=%d, unstable_submap=%d",
                config.reloc_track_retry_max_failures,
                config.reloc_track_retry_corr_scale,
                config.reloc_track_retry_max_iterations,
                config.reloc_track_unstable_submap_size);
    RCLCPP_INFO(logger,
                "Reloc static aggregation: enable=%s, frames=%d(min=%d), motion_gate=(%.3fm, %.3frad), voxel=%.3f",
                config.reloc_static_agg_enable ? "YES" : "NO",
                config.reloc_static_agg_max_frames,
                config.reloc_static_agg_min_frames,
                config.reloc_static_agg_max_translation,
                config.reloc_static_agg_max_rotation,
                config.reloc_static_agg_voxel_size);
    RCLCPP_INFO(logger,
                "Reloc ambiguity guard: margin>=%.3f, ratio>=%.3f, basin_sep>=%.2fm",
                config.reloc_ambiguity_min_margin,
                config.reloc_ambiguity_min_ratio,
                config.reloc_ambiguity_min_basin_separation);
    RCLCPP_INFO(logger,
                "RHPD: enabled=%s, v2=%s, v3=%s, max_range=%.1f, z=[%.1f,%.1f], dist_thr=%.1f, candidates=%d, preselect=%d, submap_radius=%d, submap_voxel=%.2f",
                config.rhpd_enabled ? "YES" : "NO",
                config.rhpd_v2_enable ? "YES" : "NO",
                config.rhpd_v3_enable ? "YES" : "NO",
                config.rhpd_max_range,
                config.rhpd_z_min,
                config.rhpd_z_max,
                config.rhpd_dist_threshold,
                config.rhpd_num_candidates,
                config.rhpd_preselect_candidates,
                config.rhpd_submap_kf_radius,
                config.rhpd_submap_voxel_size);
    RCLCPP_INFO(logger,
                "RHPD primary retrieval: weight=%.2f, sc_aux_weight=%.2f, sc_aux_veto=%s(thr=%.3f), use_sc_yaw=%s, yaw_hyp=%d, aug(neg=%s, vert=%s, pca_conf=%s)",
                config.rhpd_primary_weight,
                config.sc_aux_weight,
                config.sc_aux_veto_enabled ? "YES" : "NO",
                config.sc_aux_veto_threshold,
                config.rhpd_use_sc_yaw ? "YES" : "NO",
                config.rhpd_yaw_hypotheses,
                config.rhpd_enable_negative_space ? "YES" : "NO",
                config.rhpd_enable_vertical_tokens ? "YES" : "NO",
                config.rhpd_enable_pca_confidence ? "YES" : "NO");
    RCLCPP_INFO(logger, "Threads: %d | Save path: %s", config.num_threads, config.map_save_path.c_str());
    RCLCPP_INFO(logger, "==============================================");
}

} // namespace n3mapping
