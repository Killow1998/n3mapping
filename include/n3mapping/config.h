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

    // Shadow-only submap scaffold. Disabled by default and never consumed by
    // localization, loop closure, or the pose graph.
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

    double sc_dist_threshold;
    int sc_num_exclude_recent;
    int sc_num_candidates;
    double sc_max_radius;
    int sc_num_rings;
    int sc_num_sectors;


    int optimization_iterations;
    double prior_noise_position;
    double prior_noise_rotation;
    double odom_noise_position;
    double odom_noise_rotation;
    // Standard deviations of verified relative-pose measurements: metres and
    // radians, respectively. Retrieval/yaw ambiguity is an admission concern;
    // it is not the covariance of an accepted geometric registration.
    double loop_noise_position;
    double loop_noise_rotation;
    // Continuous extension tracking is local, initialized from the preceding
    // frame, and accepted only when immutable loaded-map geometry agrees. Its
    // horizontal and orientation sigmas are calibrated separately from loops.
    // The vertical sigma is still shared below.
    double loaded_map_tracking_noise_position;
    double loaded_map_tracking_noise_rotation;
    // Retained from the 0723 floor audit (median z error 0.407 m). This is a
    // dataset calibration, not a universal LiDAR limitation: the enclosed
    // SpRobo fixture has strong vertical geometry. Do not transfer its ideal
    // sensor calibration to real bags without an independent validation.
    // Non-positive means "use loop_noise_position". Strict extension currently
    // also uses this vertical sigma; changing it requires extension replay.
    double loop_noise_position_z;
    // Redistributes each loop edge's weight between axes according to its own
    // registration Hessian, keeping the geometric mean where the configured
    // noise puts it. Off leaves every loop with the configured sigmas, which is
    // what everything before this was measured against.
    bool loop_axis_weighting_enable;
    // How far a single axis may depart from the configured stiffness. The
    // measured vertical-to-horizontal ratio reaches 5.27, so 4 lets the extreme
    // cases move by a factor of two in sigma without letting a degenerate
    // Hessian dominate.
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
    // Extension-only background CPU share; 0 disables pacing. Save drains all work.
    double loop_geometry_cpu_budget;
    bool loop_spatial_candidates_enable;
    double loop_spatial_candidate_radius;
    int loop_spatial_candidate_max_candidates;

    int loop_kf_gap;
    // Minimum odometry path travelled between loop endpoints [m].
    double loop_min_path_length_m;

    // Keep every candidate in the accepted same-query consensus set.
    bool loop_keep_all_verified;
    // When several verified candidates belong to the same query, compare the
    // global query poses implied by each match and measurement.  Keep the
    // strict-majority consensus set instead of allowing one corridor alias to
    // enter the graph alongside several agreeing measurements.  These bounds
    // are candidate-to-candidate checks; no ground truth is used at runtime.
    double loop_same_query_consensus_translation_m;
    double loop_same_query_consensus_rotation_rad;

    // Admit mapping after stable stationary input, observed motion or timeout.
    bool mapping_static_start_guard_enable;
    double mapping_static_voxel_m;
    double mapping_static_moved_overlap;
    int mapping_static_moved_consecutive;
    double mapping_static_max_wait_s;

    bool odom_sanity_enable;
    double odom_sanity_max_speed_mps;
    double odom_sanity_max_angular_rate_dps;
    int odom_sanity_max_consecutive;

    // Experimental absolute roll/pitch observation. Disabled by default after
    // floor-on runs with a tilted LiDAR produced systematic z drift; the
    // implementation and serialization remain available for controlled study.
    bool floor_attitude_enable;
    // How level a building floor is, not how precisely the plane fits.
    double floor_attitude_noise_deg;
    double floor_attitude_max_radius_m;
    int floor_attitude_min_points;
    // Maximum candidate separation in current odometry coordinates [m].
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
    double reloc_sc_dist_threshold;
    double reloc_min_confidence;
    double reloc_min_inlier_ratio;
    double reloc_search_radius;
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
    // Positive uses a consistency-ratio gap; zero retains the log-odds margin.
    double reloc_ambiguity_min_consistency_margin;
    // Apply ambiguity checks even when competing poses are close in translation.
    bool reloc_ambiguity_ignore_basin_separation;
    double reloc_ambiguity_min_basin_separation;
    // Experimental range matching; see VisibilityConsistencyOptions. Keep off
    // until independent accuracy and foreground-occlusion cases qualify it.
    // Agreement on recorded poses alone does not establish absolute accuracy.
    bool reloc_visibility_occlusion_aware;
    // Experimental loaded-map tracking cascade. When enabled, a registration
    // endpoint that already passes every current strict geometry/visibility
    // gate is accepted without evaluating motion-prediction visibility. The
    // maintained product profiles keep this false until same-input trajectory
    // and map-quality evidence supports integration.
    bool loaded_map_visibility_endpoint_fast_enable;
    bool reloc_debug_enable;
    std::string reloc_debug_path;
    bool reloc_atlas_enable;
    std::string reloc_atlas_path;
    // Registration target selection is explicit so Atlas presence cannot
    // silently change semantics during later cache work. The legacy mode
    // preserves target geometry: global Atlas when loaded, otherwise the exact
    // local crop with a search-only 128 MiB / 8-entry preparation working set.
    std::string reloc_target_mode;
    // Explicit local_lru/shadow_local_lru modes require both budgets; zero
    // disables their retention. The legacy fallback owns its separate bound.
    int reloc_target_cache_max_bytes;
    int reloc_target_cache_max_entries;
    // Ordinary localization repeatedly revisits the same local-map anchor.
    // Its prepared registration targets have a separate, explicit LRU budget
    // so target reuse cannot grow without a memory bound. Zero disables it;
    // maintained YAML profiles opt in with measured limits.
    int localization_tracking_target_cache_max_bytes;
    int localization_tracking_target_cache_max_entries;

    // Free-space evidence and hypothesis persistence.
    bool reloc_free_space_enable;
    std::string reloc_free_space_mode;  // veto | kill
    double reloc_free_space_resolution;
    double reloc_free_space_max_ray_length;
    int reloc_free_space_occupied_min_points;
    double reloc_free_space_kill_sigmas;
    std::string reloc_free_space_map_pcd;
    bool reloc_persist_hypotheses;
    int reloc_persist_max_frames;

    bool rhpd_enabled;
    bool rhpd_v2_enable;
    bool rhpd_v3_enable;
    double rhpd_max_range;
    double rhpd_z_min;
    double rhpd_z_max;
    double rhpd_dist_threshold;
    int rhpd_num_candidates;
    int rhpd_preselect_candidates;
    int rhpd_submap_kf_radius;
    double rhpd_submap_voxel_size;
    double rhpd_primary_weight;
    double sc_aux_weight;
    bool sc_aux_veto_enabled;
    double sc_aux_veto_threshold;
    bool rhpd_use_sc_yaw;
    int rhpd_yaw_hypotheses;
    bool rhpd_enable_negative_space;
    bool rhpd_enable_pca_confidence;
    // See RHPDescriptor::Params. One reproduces the shipped distance exactly.
    double rhpd_part_a_scale;
    double rhpd_aux_scale;

    std::string toString() const;
    bool validate(std::string* error = nullptr) const;
};

// The offline product Gate must exercise the same frozen localization profile
// as the ROS runtime. Deployment paths are explicit inputs; no case may
// override algorithm fields.
Config makeProductLocalizationConfig(const std::string& map_path,
                                     const std::string& atlas_path);

// Stable, complete field serialization used to prove that a ROS-loaded
// product_v1.yaml and makeProductLocalizationConfig() are equivalent.
std::string runtimeConfigCanonical(const Config& config);

} // namespace n3mapping

#endif // N3MAPPING_CONFIG_H
