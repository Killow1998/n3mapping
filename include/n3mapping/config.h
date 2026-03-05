// Config structure for n3mapping. All configurable parameters with ROS param loading.
#pragma once

#include <ros/ros.h>
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

    size_t kdtree_cache_size = 20;

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

    // Distance-based loop detection (used in mapping mode instead of SC)
    int loop_kf_gap = 5;             // check loop every N keyframes
    int loop_closest_id_th = 50;     // min ID gap between query and candidate
    int loop_min_id_interval = 20;   // min ID gap between consecutive candidates
    double loop_max_range = 30.0;    // max XY distance for candidate search

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
    double reloc_hypothesis_miss_penalty = 8.0;
    double reloc_hypothesis_not_converged_penalty = 8.0;
    double reloc_reloc_inlier_weight = 2.0;
    double reloc_reloc_desc_dist_weight = 0.2;
    double reloc_track_motion_weight = 0.5;
    int reloc_track_retry_max_failures = 5;
    double reloc_track_retry_corr_scale = 5.0;
    int reloc_track_retry_max_iterations = 50;
    int reloc_track_unstable_submap_size = 10;

    // RHPD (Ring-Height + Planar Descriptor) for relocalization
    bool rhpd_enabled = true;             // use RHPD instead of SC for relocalization
    double rhpd_max_range = 30.0;         // max radial distance [m]
    double rhpd_z_min = -2.0;             // min height relative to sensor [m]
    double rhpd_z_max = 6.0;              // max height relative to sensor [m]
    double rhpd_dist_threshold = 25.0;    // L2 distance threshold for candidate acceptance
    int rhpd_num_candidates = 10;         // top-k candidates to verify with ICP

    void loadFromROS(ros::NodeHandle& node);
    void print() const;
};

} // namespace n3mapping
