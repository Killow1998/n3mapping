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
    double icp_refine_max_correspondence_distance = 1.0;
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

    int kdtree_cache_size = 20;

    int optimization_iterations = 10;
    double prior_noise_position = 0.01;
    double prior_noise_rotation = 0.01;
    double odom_noise_position = 0.01;
    double odom_noise_rotation = 0.001;
    double loop_noise_position = 0.05;
    double loop_noise_rotation = 0.5;
    // Height is the axis a loop registers worst. Measured against the floor on
    // 0723, a loop's z correction carries a median error of 0.407 m while the
    // same edge's horizontal agreement has no reason to be as poor: a floor is
    // one plane seen edge-on, whereas walls and furniture pin x and y from
    // several directions. Non-positive means "use loop_noise_position", which
    // is the isotropic behaviour everything was measured against.
    double loop_noise_position_z = 0.4;
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
    int loop_spatial_candidate_min_id_gap = 50;
    int loop_spatial_candidate_max_candidates = 5;

    int loop_kf_gap = 5;
    int loop_closest_id_th = 50;
    int loop_min_id_interval = 20;
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
    bool odom_sanity_enable = true;
    double odom_sanity_max_speed_mps = 10.0;
    double odom_sanity_max_angular_rate_dps = 720.0;
    int odom_sanity_max_consecutive = 5;

    bool floor_attitude_enable = true;
    // How level a building floor is, not how precisely the plane fits. The fit
    // is sub-degree over several hundred points; the horizontality assumption is
    // the looser of the two and is what belongs in the noise model.
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

    double output_cloud_voxel_size = 0.1;

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
    double reloc_ambiguity_min_basin_separation = 3.0;
    bool reloc_debug_enable = false;
    std::string reloc_debug_path = "";
    bool reloc_atlas_enable = false;
    std::string reloc_atlas_path = "";

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
