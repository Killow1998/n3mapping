// Config parameter loading from ROS and printing.
#include "n3mapping/config.h"

namespace n3mapping {

void Config::loadFromROS(ros::NodeHandle& node) {
    auto get = [&node](const std::string& name, auto& value) { node.param(name, value, value); };
    auto gets = [&node](const std::string& name, std::string& value) {
        std::string v; if (node.getParam(name, v) && !v.empty()) value = v;
    };

    get("mode", mode);
    gets("map_path", map_path);

    get("cloud_topic", cloud_topic);
    get("odom_topic", odom_topic);
    get("output_odom_topic", output_odom_topic);
    get("output_path_topic", output_path_topic);
    get("output_cloud_body_topic", output_cloud_body_topic);
    get("output_cloud_world_topic", output_cloud_world_topic);

    get("world_frame", world_frame);
    get("body_frame", body_frame);

    get("keyframe_distance_threshold", keyframe_distance_threshold);
    get("keyframe_angle_threshold", keyframe_angle_threshold);

    get("gicp_downsampling_resolution", gicp_downsampling_resolution);
    get("gicp_max_correspondence_distance", gicp_max_correspondence_distance);
    get("gicp_max_iterations", gicp_max_iterations);
    get("gicp_transformation_epsilon", gicp_transformation_epsilon);
    get("gicp_rotation_epsilon_deg", gicp_rotation_epsilon_deg);
    get("gicp_fitness_threshold", gicp_fitness_threshold);
    get("gicp_num_neighbors", gicp_num_neighbors);
    get("gicp_submap_size", gicp_submap_size);
    get("icp_refine_use_gicp", icp_refine_use_gicp);
    get("icp_refine_max_iterations", icp_refine_max_iterations);
    get("icp_refine_max_correspondence_distance", icp_refine_max_correspondence_distance);
    get("icp_refine_downsampling_resolution", icp_refine_downsampling_resolution);
    get("icp_refine_fitness_gate", icp_refine_fitness_gate);
    get("icp_refine_delta_translation_gate", icp_refine_delta_translation_gate);
    get("icp_refine_delta_rotation_gate", icp_refine_delta_rotation_gate);

    get("sc_dist_threshold", sc_dist_threshold);
    get("sc_num_exclude_recent", sc_num_exclude_recent);
    get("sc_num_candidates", sc_num_candidates);
    get("sc_max_radius", sc_max_radius);
    get("sc_num_rings", sc_num_rings);
    get("sc_num_sectors", sc_num_sectors);

    get("optimization_iterations", optimization_iterations);
    get("prior_noise_position", prior_noise_position);
    get("prior_noise_rotation", prior_noise_rotation);
    get("odom_noise_position", odom_noise_position);
    get("odom_noise_rotation", odom_noise_rotation);
    get("loop_noise_position", loop_noise_position);
    get("loop_noise_rotation", loop_noise_rotation);
    get("use_robust_kernel", use_robust_kernel);
    get("robust_kernel_type", robust_kernel_type);
    get("robust_kernel_delta", robust_kernel_delta);
    get("loop_min_inlier_ratio", loop_min_inlier_ratio);
    get("loop_fitness_threshold", loop_fitness_threshold);
    get("loop_max_icp_translation", loop_max_icp_translation);
    get("loop_max_icp_rotation", loop_max_icp_rotation);
    get("loop_use_icp_information", loop_use_icp_information);
    get("loop_kf_gap", loop_kf_gap);
    get("loop_closest_id_th", loop_closest_id_th);
    get("loop_min_id_interval", loop_min_id_interval);
    get("loop_max_range", loop_max_range);

    get("output_cloud_voxel_size", output_cloud_voxel_size);
    gets("map_save_path", map_save_path);
    get("global_map_voxel_size", global_map_voxel_size);
    get("save_global_map_on_shutdown", save_global_map_on_shutdown);
    get("num_threads", num_threads);
    get("sync_time_tolerance", sync_time_tolerance);

    get("reloc_num_candidates", reloc_num_candidates);
    get("reloc_sc_dist_threshold", reloc_sc_dist_threshold);
    get("reloc_min_confidence", reloc_min_confidence);
    get("reloc_min_inlier_ratio", reloc_min_inlier_ratio);
    get("reloc_search_radius", reloc_search_radius);
    get("reloc_max_track_failures", reloc_max_track_failures);
    get("reloc_track_max_translation", reloc_track_max_translation);
    get("reloc_track_max_rotation", reloc_track_max_rotation);
    get("reloc_temporal_window_size", reloc_temporal_window_size);
    get("reloc_lock_log_likelihood_threshold", reloc_lock_log_likelihood_threshold);
    get("reloc_hypothesis_miss_penalty", reloc_hypothesis_miss_penalty);
    get("reloc_hypothesis_not_converged_penalty", reloc_hypothesis_not_converged_penalty);
    get("reloc_reloc_inlier_weight", reloc_reloc_inlier_weight);
    get("reloc_reloc_desc_dist_weight", reloc_reloc_desc_dist_weight);
    get("reloc_track_motion_weight", reloc_track_motion_weight);
    get("reloc_track_retry_max_failures", reloc_track_retry_max_failures);
    get("reloc_track_retry_corr_scale", reloc_track_retry_corr_scale);
    get("reloc_track_retry_max_iterations", reloc_track_retry_max_iterations);
    get("reloc_track_unstable_submap_size", reloc_track_unstable_submap_size);

    get("rhpd_enabled", rhpd_enabled);
    get("rhpd_max_range", rhpd_max_range);
    get("rhpd_z_min", rhpd_z_min);
    get("rhpd_z_max", rhpd_z_max);
    get("rhpd_dist_threshold", rhpd_dist_threshold);
    get("rhpd_num_candidates", rhpd_num_candidates);
}

void Config::print() const {
    ROS_INFO("========== N3Mapping Configuration ==========");
    ROS_INFO("Mode: %s | Map path: %s", mode.c_str(), map_path.c_str());
    ROS_INFO("Frames: world=%s body=%s", world_frame.c_str(), body_frame.c_str());
    ROS_INFO("Keyframe: dist=%.2f m, angle=%.2f rad", keyframe_distance_threshold, keyframe_angle_threshold);
    ROS_INFO("GICP: res=%.2f, corr=%.2f, iter=%d, fitness_thr=%.3f, submap=%d",
             gicp_downsampling_resolution, gicp_max_correspondence_distance, gicp_max_iterations,
             gicp_fitness_threshold, gicp_submap_size);
    ROS_INFO("SC: dist_thr=%.3f, exclude=%d, candidates=%d", sc_dist_threshold, sc_num_exclude_recent, sc_num_candidates);
    ROS_INFO("Odom noise: pos=%.4f rot=%.4f | Loop noise: pos=%.4f rot=%.4f",
             odom_noise_position, odom_noise_rotation, loop_noise_position, loop_noise_rotation);
    ROS_INFO("Robust kernel: %s type=%s delta=%.2f", use_robust_kernel ? "ON" : "OFF",
             robust_kernel_type.c_str(), robust_kernel_delta);
    ROS_INFO("Loop: fitness_thr=%.3f, min_inlier=%.2f, max_icp_t=%.2f, max_icp_r=%.2f",
             loop_fitness_threshold, loop_min_inlier_ratio, loop_max_icp_translation, loop_max_icp_rotation);
    ROS_INFO("Loop candidate pipeline: descriptor retrieval (SC KD-tree + refined distance) -> ICP -> geom gate -> LoopClosureManager filter/select");
    ROS_WARN("loop_closest_id_th/min_id_interval/max_range are retained for compatibility/logging only; they are not used in the active mapping loop-candidate retrieval/verification path.");
    ROS_INFO("Loop timing: loop_kf_gap=%d (active)", loop_kf_gap);
    ROS_INFO("Legacy loop distance params (inactive, compatibility/logging only): closest_id_th=%d, min_id_interval=%d, max_range=%.1f",
             loop_closest_id_th, loop_min_id_interval, loop_max_range);
    ROS_INFO("Reloc: candidates=%d, sc_thr=%.3f, min_conf=%.2f", reloc_num_candidates, reloc_sc_dist_threshold, reloc_min_confidence);
    ROS_INFO("Reloc temporal: window=%d, lock_ll=%.2f, miss_pen=%.2f, nonconv_pen=%.2f",
             reloc_temporal_window_size, reloc_lock_log_likelihood_threshold,
             reloc_hypothesis_miss_penalty, reloc_hypothesis_not_converged_penalty);
    ROS_INFO("Reloc score weights: inlier=%.2f, desc=%.2f, motion=%.2f",
             reloc_reloc_inlier_weight, reloc_reloc_desc_dist_weight, reloc_track_motion_weight);
    ROS_INFO("Reloc retry: max_fail=%d, corr_scale=%.2f, max_iter=%d, unstable_submap=%d",
             reloc_track_retry_max_failures, reloc_track_retry_corr_scale,
             reloc_track_retry_max_iterations, reloc_track_unstable_submap_size);
    ROS_INFO("RHPD: enabled=%s, max_range=%.1f, z=[%.1f,%.1f], dist_thr=%.1f, candidates=%d",
             rhpd_enabled ? "YES" : "NO", rhpd_max_range, rhpd_z_min, rhpd_z_max,
             rhpd_dist_threshold, rhpd_num_candidates);
    ROS_INFO("Threads: %d | Save path: %s", num_threads, map_save_path.c_str());
    ROS_INFO("==============================================");
}

} // namespace n3mapping
