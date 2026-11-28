#ifndef N3MAPPING_CONFIG_H
#define N3MAPPING_CONFIG_H

#include <cstdint>
#include <string>

namespace n3mapping {

struct Config {
    Config();

    std::string mode;
    std::string map_path;

    std::string cloud_topic;
    std::string odom_topic;
    std::string output_odom_topic;
    std::string output_path_topic;
    std::string output_cloud_body_topic;
    std::string output_cloud_world_topic;

    std::string world_frame;
    std::string body_frame;

    double keyframe_distance_threshold;
    double keyframe_angle_threshold;

    bool submap_shadow_enable;
    int submap_max_keyframes;
    int submap_cloud_max_bytes;

    double gicp_downsampling_resolution;
    double gicp_max_correspondence_distance;
    int gicp_max_iterations;
    double gicp_transformation_epsilon;
    double gicp_rotation_epsilon_deg;
    double gicp_fitness_threshold;
    int gicp_num_neighbors;
    int gicp_submap_size;
    bool icp_refine_use_gicp;
    int icp_refine_max_iterations;
    double icp_refine_max_correspondence_distance;
    double icp_refine_downsampling_resolution;
    double icp_refine_fitness_gate;
    double icp_refine_delta_translation_gate;
    double icp_refine_delta_rotation_gate;

    double heading_dist_threshold;
    int heading_num_candidates;
    double heading_max_radius;
    int heading_num_rings;
    int heading_num_sectors;

    int optimization_iterations;
    double prior_noise_position;
    double prior_noise_rotation;
    double odom_noise_position;
    double odom_noise_rotation;

    double loop_noise_position;
    double loop_noise_rotation;

    double loaded_map_tracking_noise_position;
    double loaded_map_tracking_noise_rotation;

    double loop_noise_position_z;

    bool loop_axis_weighting_enable;

    double loop_axis_weighting_max;
    bool use_robust_kernel;
    std::string robust_kernel_type;
    double robust_kernel_delta;
    double loop_min_inlier_ratio;
    double loop_fitness_threshold;
    double loop_max_icp_translation;
    double loop_max_icp_rotation;
    bool loop_use_icp_information;
    double loop_icp_prefilter_voxel_size;
    int loop_icp_max_points;
    bool loop_debug_enable;
    bool loop_debug_vertical_hypotheses_enable;
    std::string loop_debug_path;
    bool loop_geometry_enable;

    double loop_geometry_cpu_budget;
    bool continuation_penetration_cleaning_enable;
    double continuation_penetration_margin_m;
    int continuation_penetration_min_views;
    double continuation_penetration_resolution_deg;
    double continuation_penetration_max_range_m;
    bool loop_spatial_candidates_enable;
    double loop_spatial_candidate_radius;
    int loop_spatial_candidate_max_candidates;

    int loop_kf_gap;

    double loop_min_path_length_m;

    bool loop_keep_all_verified;

    double loop_same_query_consensus_translation_m;
    double loop_same_query_consensus_rotation_rad;

    bool mapping_static_start_guard_enable;
    double mapping_static_voxel_m;
    double mapping_static_moved_overlap;
    int mapping_static_moved_consecutive;
    double mapping_static_max_wait_s;

    bool odom_sanity_enable;
    double odom_sanity_max_speed_mps;
    double odom_sanity_max_angular_rate_dps;
    int odom_sanity_max_consecutive;

    bool floor_attitude_enable;

    double floor_attitude_noise_deg;
    double floor_attitude_max_radius_m;
    int floor_attitude_min_points;

    double loop_max_range;

    std::string map_save_path;

    double global_map_voxel_size;
    double save_global_map_voxel_size;
    double global_map_publish_hz;
    bool save_global_map_on_shutdown;

    int num_threads;
    int sync_queue_size;
    double sync_time_tolerance;

    int reloc_num_candidates;
    double reloc_heading_dist_threshold;
    double reloc_min_confidence;
    double reloc_min_inlier_ratio;
    double reloc_search_radius;
    bool reloc_initial_prior_enabled;
    double reloc_initial_prior_x;
    double reloc_initial_prior_y;
    double reloc_initial_prior_z;
    double reloc_initial_prior_xy_radius;
    double reloc_initial_prior_z_radius;
    double reloc_initial_prior_yaw;
    double reloc_initial_prior_yaw_tolerance;
    int reloc_max_track_failures;
    double reloc_track_max_translation;
    double reloc_track_max_rotation;
    int reloc_temporal_window_size;
    double reloc_lock_log_likelihood_threshold;
    int reloc_lock_min_winner_streak;
    int reloc_lock_min_converged_updates;
    double reloc_lock_min_margin;
    double reloc_hypothesis_miss_penalty;
    double reloc_hypothesis_not_converged_penalty;
    double reloc_reloc_inlier_weight;
    double reloc_reloc_desc_dist_weight;
    double reloc_track_motion_weight;
    int reloc_track_retry_max_failures;
    double reloc_track_retry_corr_scale;
    int reloc_track_retry_max_iterations;
    int reloc_track_unstable_submap_size;
    bool reloc_static_agg_enable;
    int reloc_static_agg_max_frames;
    int reloc_static_agg_min_frames;
    double reloc_static_agg_max_translation;
    double reloc_static_agg_max_rotation;
    double reloc_static_agg_voxel_size;
    double reloc_ambiguity_min_margin;
    double reloc_ambiguity_min_ratio;

    double reloc_ambiguity_min_consistency_margin;

    bool reloc_ambiguity_ignore_basin_separation;
    double reloc_ambiguity_min_basin_separation;

    bool reloc_visibility_occlusion_aware;

    bool loaded_map_visibility_endpoint_fast_enable;
    bool reloc_debug_enable;
    std::string reloc_debug_path;
    bool reloc_atlas_enable;
    std::string reloc_atlas_path;


    int reloc_target_cache_max_bytes;
    int reloc_target_cache_max_entries;

    int localization_tracking_target_cache_max_bytes;
    int localization_tracking_target_cache_max_entries;

    bool reloc_free_space_enable;
    std::string reloc_free_space_mode;
    double reloc_free_space_resolution;
    double reloc_free_space_max_ray_length;
    int reloc_free_space_occupied_min_points;
    double reloc_free_space_kill_sigmas;
    std::string reloc_free_space_map_pcd;
    bool reloc_persist_hypotheses;
    int reloc_persist_max_frames;

    bool rhpd_enabled;
    double rhpd_max_range;
    double rhpd_z_min;
    double rhpd_z_max;
    double rhpd_dist_threshold;
    int rhpd_num_candidates;
    int rhpd_preselect_candidates;
    int rhpd_submap_kf_radius;
    double rhpd_submap_voxel_size;
    double rhpd_primary_weight;
    double heading_aux_weight;
    bool heading_aux_veto_enabled;
    double heading_aux_veto_threshold;
    bool rhpd_use_heading_yaw;
    int rhpd_yaw_hypotheses;
    bool rhpd_enable_negative_space;
    bool rhpd_enable_pca_confidence;


    std::string toString() const;
    bool validate(std::string* error = nullptr) const;
};

Config makeProductLocalizationConfig(const std::string& map_path,
                                     const std::string& atlas_path);

std::string runtimeConfigCanonical(const Config& config);

}

#endif
