#ifndef N3MAPPING_CONFIG_H
#define N3MAPPING_CONFIG_H

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
    double loop_noise_position = 0.05;
    double loop_noise_rotation = 0.5;
    // Continuous extension tracking is local, initialized from the preceding
    // frame, and accepted only when immutable loaded-map geometry agrees. Its
    // orientation constraint therefore needs a tighter model than a global
    // descriptor loop, whose yaw can be ambiguous by design.
    double loaded_map_tracking_noise_position = 0.05;
    double loaded_map_tracking_noise_rotation = 0.01;
    // Height is the axis a loop registers worst. Measured against the floor on
    // 0723, a loop's z correction carries a median error of 0.407 m while the
    // same edge's horizontal agreement has no reason to be as poor: a floor is
    // one plane seen edge-on, whereas walls and furniture pin x and y from
    // several directions. Non-positive means "use loop_noise_position", which
    // is the isotropic behaviour everything was measured against.
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
    bool loop_spatial_candidates_enable = true;
    double loop_spatial_candidate_radius = 15.0;
    int loop_spatial_candidate_max_candidates = 5;

    int loop_kf_gap = 5;
    // Unused. Both of these are read from the YAML, validated at startup and
    // printed in the configuration summary, and nothing reads them: the only
    // separation the detector applies is loop_min_path_length_m, plus
    // loop_kf_gap throttling how often detection runs at all. Left in place
    // rather than removed alongside a measured change; deleting them is its own
    // cleanup.
    // Odometry path travelled between two keyframes, below which a loop is not
    // worth forming because the odometry chain already constrains the pair far
    // more tightly than registration could. Replaces the keyframe-count
    // exclusion for RHPD candidates: a count is not a distance, and at 50
    // counts it spanned about 50 m of travel in a building that loops back
    // within 30, which excluded the only revisit that showed the drift.
    double loop_min_path_length_m = 5.0;

    // Absolute roll/pitch from the floor under each scan. Without it the pose
    // graph has no observation of gravity at all: every edge is relative, so a
    // world frame that tilts while running is invisible and the height error it
    // causes cannot be recovered.
    // Commit every loop that passed verification rather than one per query.
    // Selecting one made sense when the search window was wide enough to admit
    // corridor aliases; with the window sized to the drift it only throws away
    // constraints, and it ranks by a fitness that favours near neighbours.
    bool loop_keep_all_verified = true;

    // Stops mapping once the front end's pose has run away. FAST_LIO emits no
    // divergence signal, so without this a stairwell that breaks scan matching
    // turns a usable partial map into a 3,000,000 m one. Limits are set at the
    // physically impossible, not at the merely poor.
    // Holds mapping back until the platform is seen to move. While it stands
    // still the estimator has nothing to separate a tilted body from a
    // misplaced gravity vector, and the moving part of that error goes into the
    // odometry edges where nothing downstream can take it out. Skipping those
    // frames costs nothing, because a stationary opening carries no mapping
    // information; doing it by hand took the worst revisit pair from 2.3275 m
    // to 0.5865 and the floor residual span from 0.434 to 0.255.
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
    // How far apart two keyframes may be, by the drifted poses, and still be
    // worth registering. The comment here used to argue it must exceed the
    // map's extent so it could not act as a correctness gate; that was written
    // before the aliases were understood. A window sized to the drift rather
    // than to the map is what separates a true revisit from a corridor that
    // looks like one: 150 admitted matches from 14.766 m of cycle-closure
    // error, 3.0 brought it to 0.887. This is the value the product ships and
    // the built-in default has to agree with it -- a caller that does not load
    // the YAML was silently getting the aliasing one.
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
    // Ambiguity measured on the consistency ratio instead of on its logit, so
    // the gate does not change meaning when the ratio's operating point moves.
    // Zero keeps the log-odds gate above. 0.085 is the first-order equivalent
    // of reloc_ambiguity_min_margin at the measured baseline operating point:
    // d(logit)/dc is 1/(c(1-c)), which is 4.10 at c = 0.424, and 0.35 / 4.10 is
    // 0.085. It is a conversion of the existing threshold, not a new number to
    // tune.
    double reloc_ambiguity_min_consistency_margin = 0.0;
    // The ambiguity gate currently only applies when the two leading
    // hypotheses are far apart in translation. But the runner-up is already
    // chosen as the first hypothesis that is not the same physical pose as the
    // leader -- a test that counts a large rotation as different -- so
    // re-deriving separation from translation alone exempts exactly the pair
    // that sits in one place facing two ways. 11-58-53 locks 33 m from truth
    // that way, with 2.08 m between the hypotheses and 42 degrees of yaw.
    // True makes the gate apply whenever a competing hypothesis exists.
    bool reloc_ambiguity_ignore_basin_separation = false;
    double reloc_ambiguity_min_basin_separation = 3.0;
    // See VisibilityConsistencyOptions::occlusion_aware. False reproduces the
    // published evidence exactly; it decides every lock, so it stays off until
    // the twenty-query evaluation says otherwise.
    bool reloc_visibility_occlusion_aware = false;
    bool reloc_debug_enable = false;
    std::string reloc_debug_path = "";
    bool reloc_atlas_enable = false;
    std::string reloc_atlas_path = "";

    // Free-space evidence and hypothesis persistence. Until 2026-08 these were
    // process-environment switches read inside WorldLocalizing
    // (N3MAPPING_FREESPACE_*, N3MAPPING_RELOC_PERSIST). The defaults below are
    // exactly the env-unset behaviour, so a default YAML run is unchanged;
    // setting the old environment variables no longer has any effect.
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
    bool rhpd_enable_vertical_tokens = true;
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
