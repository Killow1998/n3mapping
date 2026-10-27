#ifndef N3MAPPING_CONFIG_H
#define N3MAPPING_CONFIG_H

#include <cstdint>
#include <string>

namespace n3mapping {

struct Config {
    std::string mode = "mapping";
    std::string map_path = "";

    std::string cloud_topic = "/cloud_registered_body";
    std::string odom_topic = "/Odometry";
    std::string output_odom_topic = "/n3mapping/odometry";
    std::string output_path_topic = "/n3mapping/path";
    std::string output_cloud_body_topic = "/n3mapping/cloud_body";
    std::string output_cloud_world_topic = "/n3mapping/cloud_world";

    std::string world_frame = "map";
    std::string body_frame = "body";

    double keyframe_distance_threshold = 1.0;
    double keyframe_angle_threshold = 0.5;

    // Shadow-only submap scaffold. Disabled by default and never consumed by
    // localization, loop closure, or the pose graph.
    bool submap_shadow_enable = false;
    int submap_max_keyframes = 20;
    int submap_cloud_max_bytes = 16 * 1024 * 1024;

    double gicp_downsampling_resolution = 0.1;
    double gicp_max_correspondence_distance = 2.0;
    int gicp_max_iterations = 30;
    double gicp_transformation_epsilon = 1e-6;
    double gicp_rotation_epsilon_deg = 1.0;
    double gicp_fitness_threshold = 0.3;
    int gicp_num_neighbors = 20;
    int gicp_submap_size = 5;
    bool icp_refine_use_gicp = true;
    int icp_refine_max_iterations = 20;
    double icp_refine_max_correspondence_distance = 0.3;
    double icp_refine_downsampling_resolution = 0.05;
    double icp_refine_fitness_gate = 0.5;
    double icp_refine_delta_translation_gate = 3.0;
    double icp_refine_delta_rotation_gate = 0.5;

    double sc_dist_threshold = 0.2;
    int sc_num_exclude_recent = 50;
    int sc_num_candidates = 10;
    double sc_max_radius = 80.0;
    int sc_num_rings = 20;
    int sc_num_sectors = 60;


    int optimization_iterations = 10;
    double prior_noise_position = 0.01;
    double prior_noise_rotation = 0.01;
    double odom_noise_position = 0.01;
    double odom_noise_rotation = 0.001;
    // Standard deviations of verified relative-pose measurements: metres and
    // radians, respectively. Retrieval/yaw ambiguity is an admission concern;
    // it is not the covariance of an accepted geometric registration.
    double loop_noise_position = 0.05;
    double loop_noise_rotation = 0.5;
    // Continuous extension tracking is local, initialized from the preceding
    // frame, and accepted only when immutable loaded-map geometry agrees. Its
    // horizontal and orientation sigmas are calibrated separately from loops.
    // The vertical sigma is still shared below.
    double loaded_map_tracking_noise_position = 0.05;
    double loaded_map_tracking_noise_rotation = 0.01;
    // Retained from the 0723 floor audit (median z error 0.407 m). This is a
    // dataset calibration, not a universal LiDAR limitation: the enclosed
    // SpRobo fixture has strong vertical geometry. Do not transfer its ideal
    // sensor calibration to real bags without an independent validation.
    // Non-positive means "use loop_noise_position". Strict extension currently
    // also uses this vertical sigma; changing it requires extension replay.
    double loop_noise_position_z = 0.4;
    // Redistributes each loop edge's weight between axes according to its own
    // registration Hessian, keeping the geometric mean where the configured
    // noise puts it. Off leaves every loop with the configured sigmas, which is
    // what everything before this was measured against.
    bool loop_axis_weighting_enable = false;
    // How far a single axis may depart from the configured stiffness. The
    // measured vertical-to-horizontal ratio reaches 5.27, so 4 lets the extreme
    // cases move by a factor of two in sigma without letting a degenerate
    // Hessian dominate.
    double loop_axis_weighting_max = 4.0;
    bool use_robust_kernel = true;
    std::string robust_kernel_type = "Cauchy";
    double robust_kernel_delta = 1.0;
    double loop_min_inlier_ratio = 0.5;
    double loop_fitness_threshold = 0.3;
    double loop_max_icp_translation = 5.0;
    double loop_max_icp_rotation = 0.5;
    bool loop_use_icp_information = false;
    double loop_icp_prefilter_voxel_size = 0.2;
    int loop_icp_max_points = 50000;
    bool loop_debug_enable = false;
    bool loop_debug_vertical_hypotheses_enable = false;
    std::string loop_debug_path = "";
    bool loop_geometry_enable = true;
    // Extension-only background CPU share; 0 disables pacing. Save drains all work.
    double loop_geometry_cpu_budget = 0.5;
    bool loop_spatial_candidates_enable = true;
    double loop_spatial_candidate_radius = 15.0;
    int loop_spatial_candidate_max_candidates = 5;

    int loop_kf_gap = 5;
    // Minimum odometry path travelled between loop endpoints [m].
    double loop_min_path_length_m = 5.0;

    // Keep every candidate in the accepted same-query consensus set.
    bool loop_keep_all_verified = true;
    // When several verified candidates belong to the same query, compare the
    // global query poses implied by each match and measurement.  Keep the
    // strict-majority consensus set instead of allowing one corridor alias to
    // enter the graph alongside several agreeing measurements.  These bounds
    // are candidate-to-candidate checks; no ground truth is used at runtime.
    double loop_same_query_consensus_translation_m = 1.0;
    double loop_same_query_consensus_rotation_rad = 0.17453292519943295;

    // Admit mapping after stable stationary input, observed motion or timeout.
    bool mapping_static_start_guard_enable = true;
    double mapping_static_voxel_m = 0.3;
    double mapping_static_moved_overlap = 0.6;
    int mapping_static_moved_consecutive = 3;
    double mapping_static_max_wait_s = 120.0;

    bool odom_sanity_enable = true;
    double odom_sanity_max_speed_mps = 10.0;
    double odom_sanity_max_angular_rate_dps = 720.0;
    int odom_sanity_max_consecutive = 5;

    // Experimental absolute roll/pitch observation. Disabled by default after
    // floor-on runs with a tilted LiDAR produced systematic z drift; the
    // implementation and serialization remain available for controlled study.
    bool floor_attitude_enable = false;
    // How level a building floor is, not how precisely the plane fits.
    double floor_attitude_noise_deg = 1.0;
    double floor_attitude_max_radius_m = 8.0;
    int floor_attitude_min_points = 400;
    // Maximum candidate separation in current odometry coordinates [m].
    double loop_max_range = 6.0;


#ifdef N3MAPPING_SOURCE_DIR
    std::string map_save_path = std::string(N3MAPPING_SOURCE_DIR) + "/map";
#else
    std::string map_save_path = "./map";
#endif

    double global_map_voxel_size = 0.1;
    double save_global_map_voxel_size = 0.1;
    double global_map_publish_hz = 1.0;
    bool save_global_map_on_shutdown = true;

    int num_threads = 4;
    int sync_queue_size = 100;
    double sync_time_tolerance = 0.1;

    int reloc_num_candidates = 10;
    double reloc_sc_dist_threshold = 0.3;
    double reloc_min_confidence = 0.3;
    double reloc_min_inlier_ratio = 0.03;
    double reloc_search_radius = 20.0;
    int reloc_max_track_failures = 5;
    double reloc_track_max_translation = 3.0;
    double reloc_track_max_rotation = 1.0;
    int reloc_temporal_window_size = 5;
    double reloc_lock_log_likelihood_threshold = 2.0;
    int reloc_lock_min_winner_streak = 3;
    int reloc_lock_min_converged_updates = 3;
    double reloc_lock_min_margin = 0.35;
    double reloc_hypothesis_miss_penalty = 8.0;
    double reloc_hypothesis_not_converged_penalty = 8.0;
    double reloc_reloc_inlier_weight = 2.0;
    double reloc_reloc_desc_dist_weight = 0.2;
    double reloc_track_motion_weight = 0.5;
    int reloc_track_retry_max_failures = 5;
    double reloc_track_retry_corr_scale = 5.0;
    int reloc_track_retry_max_iterations = 50;
    int reloc_track_unstable_submap_size = 10;
    bool reloc_static_agg_enable = true;
    int reloc_static_agg_max_frames = 5;
    int reloc_static_agg_min_frames = 3;
    double reloc_static_agg_max_translation = 0.25;
    double reloc_static_agg_max_rotation = 0.20;
    double reloc_static_agg_voxel_size = 0.12;
    double reloc_ambiguity_min_margin = 0.35;
    double reloc_ambiguity_min_ratio = 1.05;
    // Positive uses a consistency-ratio gap; zero retains the log-odds margin.
    double reloc_ambiguity_min_consistency_margin = 0.0;
    // Apply ambiguity checks even when competing poses are close in translation.
    bool reloc_ambiguity_ignore_basin_separation = false;
    double reloc_ambiguity_min_basin_separation = 3.0;
    // Experimental range matching; see VisibilityConsistencyOptions. Keep off
    // until independent accuracy and foreground-occlusion cases qualify it.
    // Agreement on recorded poses alone does not establish absolute accuracy.
    bool reloc_visibility_occlusion_aware = false;
    // Experimental loaded-map tracking cascade. When enabled, a registration
    // endpoint that already passes every current strict geometry/visibility
    // gate is accepted without evaluating motion-prediction visibility. The
    // maintained product profiles keep this false until same-input trajectory
    // and map-quality evidence supports integration.
    bool loaded_map_visibility_endpoint_fast_enable = false;
    bool reloc_debug_enable = false;
    std::string reloc_debug_path = "";
    bool reloc_atlas_enable = false;
    std::string reloc_atlas_path = "";
    // Registration target selection is explicit so Atlas presence cannot
    // silently change semantics during later cache work. The legacy mode
    // preserves target geometry: global Atlas when loaded, otherwise the exact
    // local crop with a search-only 128 MiB / 8-entry preparation working set.
    std::string reloc_target_mode = "legacy_global_atlas";
    // Explicit local_lru/shadow_local_lru modes require both budgets; zero
    // disables their retention. The legacy fallback owns its separate bound.
    int reloc_target_cache_max_bytes = 0;
    int reloc_target_cache_max_entries = 0;
    // Ordinary localization repeatedly revisits the same local-map anchor.
    // Its prepared registration targets have a separate, explicit LRU budget
    // so target reuse cannot grow without a memory bound. Zero disables it;
    // maintained YAML profiles opt in with measured limits.
    int localization_tracking_target_cache_max_bytes = 0;
    int localization_tracking_target_cache_max_entries = 0;

    // Free-space evidence and hypothesis persistence.
    bool reloc_free_space_enable = true;
    std::string reloc_free_space_mode = "veto";  // veto | kill
    double reloc_free_space_resolution = 0.20;
    double reloc_free_space_max_ray_length = 30.0;
    int reloc_free_space_occupied_min_points = 2;
    double reloc_free_space_kill_sigmas = 5.0;
    std::string reloc_free_space_map_pcd = "";
    bool reloc_persist_hypotheses = false;
    int reloc_persist_max_frames = 300;

    bool rhpd_enabled = true;
    bool rhpd_v2_enable = true;
    bool rhpd_v3_enable = false;
    double rhpd_max_range = 30.0;
    double rhpd_z_min = -2.0;
    double rhpd_z_max = 6.0;
    double rhpd_dist_threshold = 25.0;
    int rhpd_num_candidates = 10;
    int rhpd_preselect_candidates = 100;
    int rhpd_submap_kf_radius = 3;
    double rhpd_submap_voxel_size = 0.15;
    double rhpd_primary_weight = 1.0;
    double sc_aux_weight = 0.15;
    bool sc_aux_veto_enabled = false;
    double sc_aux_veto_threshold = 0.6;
    bool rhpd_use_sc_yaw = true;
    int rhpd_yaw_hypotheses = 4;
    bool rhpd_enable_negative_space = true;
    bool rhpd_enable_pca_confidence = true;
    // See RHPDescriptor::Params. One reproduces the shipped distance exactly.
    double rhpd_part_a_scale = 1.0;
    double rhpd_aux_scale = 1.0;

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
