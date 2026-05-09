#ifndef N3MAPPING_CONFIG_H
#define N3MAPPING_CONFIG_H

#include <string>

namespace n3mapping {

struct Config {
    std::string mode = "mapping";
#ifdef N3MAPPING_SOURCE_DIR
    std::string map_path = std::string(N3MAPPING_SOURCE_DIR) + "/map/n3map.pbstream";
#else
    std::string map_path = "";
#endif

    std::string cloud_topic = "/cloud_registered_body";
    std::string odom_topic = "/Odometry";
    std::string frontend_mode = "external";
    std::string raw_lidar_topic = "/points_raw";
    std::string raw_lidar_msg_type = "pointcloud2";
    std::string imu_topic = "/imu";
    std::string lidar_type = "generic";
    double frontend_time_offset = 0.0;
    bool frontend_publish_debug = false;
    bool frontend_debug_publish_odom = false;
    bool frontend_debug_publish_deskewed_cloud = false;
    bool frontend_debug_publish_local_map = false;
    bool frontend_debug_publish_timing = false;
    int frontend_imu_buffer_max_samples = 2000;
    int frontend_point_filter_num = 1;
    int frontend_scan_lines = 128;
    double frontend_blind = 0.0;
    double frontend_max_abs_coordinate = 1.0e8;
    bool frontend_prediction_only_output = false;
    std::string dlio_time_encoding = "auto";
    double frontend_lidar_to_body_tx = 0.0;
    double frontend_lidar_to_body_ty = 0.0;
    double frontend_lidar_to_body_tz = 0.0;
    double frontend_lidar_to_body_roll = 0.0;
    double frontend_lidar_to_body_pitch = 0.0;
    double frontend_lidar_to_body_yaw = 0.0;
    double frontend_imu_to_body_tx = 0.0;
    double frontend_imu_to_body_ty = 0.0;
    double frontend_imu_to_body_tz = 0.0;
    double frontend_imu_to_body_roll = 0.0;
    double frontend_imu_to_body_pitch = 0.0;
    double frontend_imu_to_body_yaw = 0.0;
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
    double loop_noise_position = 0.5;
    double loop_noise_rotation = 0.5;
    bool use_robust_kernel = true;
    std::string robust_kernel_type = "Cauchy";
    double robust_kernel_delta = 1.0;
    double loop_min_inlier_ratio = 0.5;
    double loop_fitness_threshold = 0.3;
    double loop_max_icp_translation = 5.0;
    double loop_max_icp_rotation = 0.5;
    bool loop_use_icp_information = false;

    int loop_kf_gap = 5;
    int loop_closest_id_th = 50;
    int loop_min_id_interval = 20;
    double loop_max_range = 30.0;

    double output_cloud_voxel_size = 0.2;

#ifdef N3MAPPING_SOURCE_DIR
    std::string map_save_path = std::string(N3MAPPING_SOURCE_DIR) + "/map";
#else
    std::string map_save_path = "./map";
#endif

    double global_map_voxel_size = 0.1;
    bool save_global_map_on_shutdown = true;

    int num_threads = 4;
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
};

} // namespace n3mapping

#endif // N3MAPPING_CONFIG_H
