#include "n3mapping/config.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace n3mapping {

Config makeProductLocalizationConfig(const std::string& map_path,
                                     const std::string& atlas_path) {
    Config config;
    config.mode = "localization";
    config.map_path = map_path;

    // These values differ from the C++ fallback defaults and are frozen in
    // config/product_v1.yaml. Most are mapping/output-only, but assigning the
    // complete profile keeps the Gate and deployed ROS node equivalent.
    config.sc_num_candidates = 5;
    config.loop_min_inlier_ratio = 0.7;
    config.loop_fitness_threshold = 0.2;
    config.loop_max_icp_translation = 2.0;
    config.save_global_map_on_shutdown = false;

    config.reloc_atlas_enable = true;
    config.reloc_atlas_path = atlas_path;

    // Frozen free-space/persistence profile, mirrored in config/product_v1.yaml.
    config.reloc_free_space_enable = true;
    config.reloc_free_space_mode = "veto";
    config.reloc_free_space_resolution = 0.20;
    config.reloc_free_space_max_ray_length = 30.0;
    config.reloc_free_space_occupied_min_points = 2;
    config.reloc_free_space_kill_sigmas = 5.0;
    config.reloc_free_space_map_pcd = "";
    config.reloc_persist_hypotheses = false;
    config.reloc_persist_max_frames = 300;
    return config;
}

std::string runtimeConfigCanonical(const Config& config) {
    std::ostringstream oss;
    oss << std::setprecision(17) << std::boolalpha;
#define N3MAPPING_CONFIG_FIELD(name) oss << #name << '=' << config.name << '\n'
    N3MAPPING_CONFIG_FIELD(mode);
    N3MAPPING_CONFIG_FIELD(map_path);
    N3MAPPING_CONFIG_FIELD(cloud_topic);
    N3MAPPING_CONFIG_FIELD(odom_topic);
    N3MAPPING_CONFIG_FIELD(output_odom_topic);
    N3MAPPING_CONFIG_FIELD(output_path_topic);
    N3MAPPING_CONFIG_FIELD(output_cloud_body_topic);
    N3MAPPING_CONFIG_FIELD(output_cloud_world_topic);
    N3MAPPING_CONFIG_FIELD(world_frame);
    N3MAPPING_CONFIG_FIELD(body_frame);
    N3MAPPING_CONFIG_FIELD(keyframe_distance_threshold);
    N3MAPPING_CONFIG_FIELD(keyframe_angle_threshold);
    N3MAPPING_CONFIG_FIELD(submap_shadow_enable);
    N3MAPPING_CONFIG_FIELD(submap_max_keyframes);
    N3MAPPING_CONFIG_FIELD(submap_cloud_max_bytes);
    N3MAPPING_CONFIG_FIELD(gicp_downsampling_resolution);
    N3MAPPING_CONFIG_FIELD(gicp_max_correspondence_distance);
    N3MAPPING_CONFIG_FIELD(gicp_max_iterations);
    N3MAPPING_CONFIG_FIELD(gicp_transformation_epsilon);
    N3MAPPING_CONFIG_FIELD(gicp_rotation_epsilon_deg);
    N3MAPPING_CONFIG_FIELD(gicp_fitness_threshold);
    N3MAPPING_CONFIG_FIELD(gicp_num_neighbors);
    N3MAPPING_CONFIG_FIELD(gicp_submap_size);
    N3MAPPING_CONFIG_FIELD(icp_refine_use_gicp);
    N3MAPPING_CONFIG_FIELD(icp_refine_max_iterations);
    N3MAPPING_CONFIG_FIELD(icp_refine_max_correspondence_distance);
    N3MAPPING_CONFIG_FIELD(icp_refine_downsampling_resolution);
    N3MAPPING_CONFIG_FIELD(icp_refine_fitness_gate);
    N3MAPPING_CONFIG_FIELD(icp_refine_delta_translation_gate);
    N3MAPPING_CONFIG_FIELD(icp_refine_delta_rotation_gate);
    N3MAPPING_CONFIG_FIELD(sc_dist_threshold);
    N3MAPPING_CONFIG_FIELD(sc_num_exclude_recent);
    N3MAPPING_CONFIG_FIELD(sc_num_candidates);
    N3MAPPING_CONFIG_FIELD(sc_max_radius);
    N3MAPPING_CONFIG_FIELD(sc_num_rings);
    N3MAPPING_CONFIG_FIELD(sc_num_sectors);
    N3MAPPING_CONFIG_FIELD(optimization_iterations);
    N3MAPPING_CONFIG_FIELD(prior_noise_position);
    N3MAPPING_CONFIG_FIELD(prior_noise_rotation);
    N3MAPPING_CONFIG_FIELD(odom_noise_position);
    N3MAPPING_CONFIG_FIELD(odom_noise_rotation);
    N3MAPPING_CONFIG_FIELD(loop_noise_position);
    N3MAPPING_CONFIG_FIELD(loop_noise_rotation);
    N3MAPPING_CONFIG_FIELD(loaded_map_tracking_noise_position);
    N3MAPPING_CONFIG_FIELD(loaded_map_tracking_noise_rotation);
    N3MAPPING_CONFIG_FIELD(loop_noise_position_z);
    N3MAPPING_CONFIG_FIELD(loop_axis_weighting_enable);
    N3MAPPING_CONFIG_FIELD(loop_axis_weighting_max);
    N3MAPPING_CONFIG_FIELD(use_robust_kernel);
    N3MAPPING_CONFIG_FIELD(robust_kernel_type);
    N3MAPPING_CONFIG_FIELD(robust_kernel_delta);
    N3MAPPING_CONFIG_FIELD(loop_min_inlier_ratio);
    N3MAPPING_CONFIG_FIELD(loop_fitness_threshold);
    N3MAPPING_CONFIG_FIELD(loop_max_icp_translation);
    N3MAPPING_CONFIG_FIELD(loop_max_icp_rotation);
    N3MAPPING_CONFIG_FIELD(loop_use_icp_information);
    N3MAPPING_CONFIG_FIELD(loop_icp_prefilter_voxel_size);
    N3MAPPING_CONFIG_FIELD(loop_icp_max_points);
    N3MAPPING_CONFIG_FIELD(loop_debug_enable);
    N3MAPPING_CONFIG_FIELD(loop_debug_vertical_hypotheses_enable);
    N3MAPPING_CONFIG_FIELD(loop_debug_path);
    N3MAPPING_CONFIG_FIELD(loop_spatial_candidates_enable);
    N3MAPPING_CONFIG_FIELD(loop_spatial_candidate_radius);
    N3MAPPING_CONFIG_FIELD(loop_spatial_candidate_max_candidates);
    N3MAPPING_CONFIG_FIELD(loop_kf_gap);
    N3MAPPING_CONFIG_FIELD(loop_min_path_length_m);
    N3MAPPING_CONFIG_FIELD(loop_keep_all_verified);
    N3MAPPING_CONFIG_FIELD(loop_same_query_consensus_translation_m);
    N3MAPPING_CONFIG_FIELD(loop_same_query_consensus_rotation_rad);
    N3MAPPING_CONFIG_FIELD(mapping_static_start_guard_enable);
    N3MAPPING_CONFIG_FIELD(mapping_static_voxel_m);
    N3MAPPING_CONFIG_FIELD(mapping_static_moved_overlap);
    N3MAPPING_CONFIG_FIELD(mapping_static_moved_consecutive);
    N3MAPPING_CONFIG_FIELD(mapping_static_max_wait_s);
    N3MAPPING_CONFIG_FIELD(odom_sanity_enable);
    N3MAPPING_CONFIG_FIELD(odom_sanity_max_speed_mps);
    N3MAPPING_CONFIG_FIELD(odom_sanity_max_angular_rate_dps);
    N3MAPPING_CONFIG_FIELD(odom_sanity_max_consecutive);
    N3MAPPING_CONFIG_FIELD(floor_attitude_enable);
    N3MAPPING_CONFIG_FIELD(floor_attitude_noise_deg);
    N3MAPPING_CONFIG_FIELD(floor_attitude_max_radius_m);
    N3MAPPING_CONFIG_FIELD(floor_attitude_min_points);
    N3MAPPING_CONFIG_FIELD(loop_max_range);
    N3MAPPING_CONFIG_FIELD(map_save_path);
    N3MAPPING_CONFIG_FIELD(global_map_voxel_size);
    N3MAPPING_CONFIG_FIELD(save_global_map_voxel_size);
    N3MAPPING_CONFIG_FIELD(global_map_publish_hz);
    N3MAPPING_CONFIG_FIELD(save_global_map_on_shutdown);
    N3MAPPING_CONFIG_FIELD(num_threads);
    N3MAPPING_CONFIG_FIELD(sync_queue_size);
    N3MAPPING_CONFIG_FIELD(sync_time_tolerance);
    N3MAPPING_CONFIG_FIELD(reloc_num_candidates);
    N3MAPPING_CONFIG_FIELD(reloc_sc_dist_threshold);
    N3MAPPING_CONFIG_FIELD(reloc_min_confidence);
    N3MAPPING_CONFIG_FIELD(reloc_min_inlier_ratio);
    N3MAPPING_CONFIG_FIELD(reloc_search_radius);
    N3MAPPING_CONFIG_FIELD(reloc_max_track_failures);
    N3MAPPING_CONFIG_FIELD(reloc_track_max_translation);
    N3MAPPING_CONFIG_FIELD(reloc_track_max_rotation);
    N3MAPPING_CONFIG_FIELD(reloc_temporal_window_size);
    N3MAPPING_CONFIG_FIELD(reloc_lock_log_likelihood_threshold);
    N3MAPPING_CONFIG_FIELD(reloc_lock_min_winner_streak);
    N3MAPPING_CONFIG_FIELD(reloc_lock_min_converged_updates);
    N3MAPPING_CONFIG_FIELD(reloc_lock_min_margin);
    N3MAPPING_CONFIG_FIELD(reloc_hypothesis_miss_penalty);
    N3MAPPING_CONFIG_FIELD(reloc_hypothesis_not_converged_penalty);
    N3MAPPING_CONFIG_FIELD(reloc_reloc_inlier_weight);
    N3MAPPING_CONFIG_FIELD(reloc_reloc_desc_dist_weight);
    N3MAPPING_CONFIG_FIELD(reloc_track_motion_weight);
    N3MAPPING_CONFIG_FIELD(reloc_track_retry_max_failures);
    N3MAPPING_CONFIG_FIELD(reloc_track_retry_corr_scale);
    N3MAPPING_CONFIG_FIELD(reloc_track_retry_max_iterations);
    N3MAPPING_CONFIG_FIELD(reloc_track_unstable_submap_size);
    N3MAPPING_CONFIG_FIELD(reloc_static_agg_enable);
    N3MAPPING_CONFIG_FIELD(reloc_static_agg_max_frames);
    N3MAPPING_CONFIG_FIELD(reloc_static_agg_min_frames);
    N3MAPPING_CONFIG_FIELD(reloc_static_agg_max_translation);
    N3MAPPING_CONFIG_FIELD(reloc_static_agg_max_rotation);
    N3MAPPING_CONFIG_FIELD(reloc_static_agg_voxel_size);
    N3MAPPING_CONFIG_FIELD(reloc_ambiguity_min_margin);
    N3MAPPING_CONFIG_FIELD(reloc_ambiguity_min_ratio);
    N3MAPPING_CONFIG_FIELD(reloc_ambiguity_min_consistency_margin);
    N3MAPPING_CONFIG_FIELD(reloc_ambiguity_ignore_basin_separation);
    N3MAPPING_CONFIG_FIELD(reloc_ambiguity_min_basin_separation);
    N3MAPPING_CONFIG_FIELD(reloc_visibility_occlusion_aware);
    N3MAPPING_CONFIG_FIELD(reloc_debug_enable);
    N3MAPPING_CONFIG_FIELD(reloc_debug_path);
    N3MAPPING_CONFIG_FIELD(reloc_atlas_enable);
    N3MAPPING_CONFIG_FIELD(reloc_atlas_path);
    N3MAPPING_CONFIG_FIELD(reloc_target_mode);
    N3MAPPING_CONFIG_FIELD(reloc_target_cache_max_bytes);
    N3MAPPING_CONFIG_FIELD(reloc_target_cache_max_entries);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_enable);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_mode);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_resolution);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_max_ray_length);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_occupied_min_points);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_kill_sigmas);
    N3MAPPING_CONFIG_FIELD(reloc_free_space_map_pcd);
    N3MAPPING_CONFIG_FIELD(reloc_persist_hypotheses);
    N3MAPPING_CONFIG_FIELD(reloc_persist_max_frames);
    N3MAPPING_CONFIG_FIELD(rhpd_enabled);
    N3MAPPING_CONFIG_FIELD(rhpd_v2_enable);
    N3MAPPING_CONFIG_FIELD(rhpd_v3_enable);
    N3MAPPING_CONFIG_FIELD(rhpd_max_range);
    N3MAPPING_CONFIG_FIELD(rhpd_z_min);
    N3MAPPING_CONFIG_FIELD(rhpd_z_max);
    N3MAPPING_CONFIG_FIELD(rhpd_dist_threshold);
    N3MAPPING_CONFIG_FIELD(rhpd_num_candidates);
    N3MAPPING_CONFIG_FIELD(rhpd_preselect_candidates);
    N3MAPPING_CONFIG_FIELD(rhpd_submap_kf_radius);
    N3MAPPING_CONFIG_FIELD(rhpd_submap_voxel_size);
    N3MAPPING_CONFIG_FIELD(rhpd_primary_weight);
    N3MAPPING_CONFIG_FIELD(sc_aux_weight);
    N3MAPPING_CONFIG_FIELD(sc_aux_veto_enabled);
    N3MAPPING_CONFIG_FIELD(sc_aux_veto_threshold);
    N3MAPPING_CONFIG_FIELD(rhpd_use_sc_yaw);
    N3MAPPING_CONFIG_FIELD(rhpd_yaw_hypotheses);
    N3MAPPING_CONFIG_FIELD(rhpd_enable_negative_space);
    N3MAPPING_CONFIG_FIELD(rhpd_enable_vertical_tokens);
    N3MAPPING_CONFIG_FIELD(rhpd_enable_pca_confidence);
    N3MAPPING_CONFIG_FIELD(rhpd_part_a_scale);
    N3MAPPING_CONFIG_FIELD(rhpd_aux_scale);
#undef N3MAPPING_CONFIG_FIELD
    return oss.str();
}

std::string Config::toString() const {
    std::ostringstream oss;
    oss << "========== N3Mapping Configuration ==========\n";
    oss << "Mode: " << mode << " | Map path: " << map_path << "\n";
    oss << "Frames: world=" << world_frame << " body=" << body_frame << "\n";
    oss << "Keyframe: dist=" << keyframe_distance_threshold
        << " m, angle=" << keyframe_angle_threshold << " rad\n";
    oss << "Submap scaffold: "
        << (submap_shadow_enable ? "SHADOW" : "OFF")
        << " max_keyframes=" << submap_max_keyframes
        << " cloud_max_bytes=" << submap_cloud_max_bytes << "\n";
    oss << "GICP: res=" << gicp_downsampling_resolution
        << ", corr=" << gicp_max_correspondence_distance
        << ", iter=" << gicp_max_iterations
        << ", fitness_thr=" << gicp_fitness_threshold
        << ", submap=" << gicp_submap_size << "\n";
    oss << "SC: dist_thr=" << sc_dist_threshold
        << ", exclude=" << sc_num_exclude_recent
        << ", candidates=" << sc_num_candidates << "\n";
    oss << "Odom noise: pos=" << odom_noise_position
        << " rot=" << odom_noise_rotation
        << " | Loop noise: pos=" << loop_noise_position
        << " rot=" << loop_noise_rotation << "\n";
    oss << "Loaded-map tracking noise: pos="
        << loaded_map_tracking_noise_position
        << " rot=" << loaded_map_tracking_noise_rotation
        << " z=" << loop_noise_position_z << "\n";
    oss << "Robust kernel: " << (use_robust_kernel ? "ON" : "OFF")
        << " type=" << robust_kernel_type
        << " delta=" << robust_kernel_delta << "\n";
    oss << "Loop: fitness_thr=" << loop_fitness_threshold
        << ", min_inlier=" << loop_min_inlier_ratio
        << ", max_icp_t=" << loop_max_icp_translation
        << ", max_icp_r=" << loop_max_icp_rotation
        << ", prefilter_voxel=" << loop_icp_prefilter_voxel_size
        << ", max_points=" << loop_icp_max_points << "\n";
    oss << "Loop debug JSONL: " << (loop_debug_enable ? "ON" : "OFF")
        << " vertical_hypotheses=" << (loop_debug_vertical_hypotheses_enable ? "ON" : "OFF")
        << " path=" << (loop_debug_path.empty() ? "<map_save_path>/loop_debug.jsonl" : loop_debug_path) << "\n";
    oss << "Loop candidate pipeline: descriptor + spatial proposals -> ICP consistency features -> LoopReferee -> graph commit\n";
    oss << "Loop spatial candidates: " << (loop_spatial_candidates_enable ? "ON" : "OFF")
        << " radius=" << loop_spatial_candidate_radius
        << " max_candidates=" << loop_spatial_candidate_max_candidates << "\n";
    oss << "Loop prediction range gate: max_range=" << loop_max_range
        << " (pre-ICP candidate filter)\n";
    oss << "Loop timing: loop_kf_gap=" << loop_kf_gap << " (active)\n";
    oss << "Loop same-query consensus: translation="
        << loop_same_query_consensus_translation_m
        << " m, rotation=" << loop_same_query_consensus_rotation_rad
        << " rad\n";
    oss << "Floor attitude (experimental): "
        << (floor_attitude_enable ? "ON" : "OFF")
        << " noise_deg=" << floor_attitude_noise_deg << "\n";
    oss << "Reloc: candidates=" << reloc_num_candidates
        << ", sc_thr=" << reloc_sc_dist_threshold
        << ", min_conf=" << reloc_min_confidence << "\n";
    oss << "Reloc temporal: window=" << reloc_temporal_window_size
        << ", lock_ll=" << reloc_lock_log_likelihood_threshold
        << ", min_streak=" << reloc_lock_min_winner_streak
        << ", min_conv=" << reloc_lock_min_converged_updates
        << ", min_margin=" << reloc_lock_min_margin
        << ", miss_pen=" << reloc_hypothesis_miss_penalty
        << ", nonconv_pen=" << reloc_hypothesis_not_converged_penalty << "\n";
    oss << "Reloc score weights: inlier=" << reloc_reloc_inlier_weight
        << ", desc=" << reloc_reloc_desc_dist_weight
        << ", motion=" << reloc_track_motion_weight << "\n";
    oss << "Reloc retry: max_fail=" << reloc_track_retry_max_failures
        << ", corr_scale=" << reloc_track_retry_corr_scale
        << ", max_iter=" << reloc_track_retry_max_iterations
        << ", unstable_submap=" << reloc_track_unstable_submap_size << "\n";
    oss << "Reloc static aggregation: enable="
        << (reloc_static_agg_enable ? "YES" : "NO")
        << ", frames=" << reloc_static_agg_max_frames
        << "(min=" << reloc_static_agg_min_frames << ")"
        << ", motion_gate=(" << reloc_static_agg_max_translation
        << "m, " << reloc_static_agg_max_rotation
        << "rad), voxel=" << reloc_static_agg_voxel_size << "\n";
    oss << "Reloc ambiguity guard: margin>=" << reloc_ambiguity_min_margin
        << ", ratio>=" << reloc_ambiguity_min_ratio
        << ", basin_sep>=" << reloc_ambiguity_min_basin_separation << "m\n";
    oss << "Reloc debug JSONL: " << (reloc_debug_enable ? "ON" : "OFF")
        << " path=" << (reloc_debug_path.empty() ? "<map_save_path>/relocalization_debug.jsonl" : reloc_debug_path) << "\n";
    oss << "Reloc localization atlas: " << (reloc_atlas_enable ? "ON" : "OFF")
        << " path=" << (reloc_atlas_path.empty() ? "<map_path>.localization_atlas.pb" : reloc_atlas_path) << "\n";
    oss << "Reloc target provider: mode=" << reloc_target_mode
        << " cache_bytes=" << reloc_target_cache_max_bytes
        << " cache_entries=" << reloc_target_cache_max_entries << "\n";
    oss << "Reloc free-space: " << (reloc_free_space_enable ? "ON" : "OFF")
        << " mode=" << reloc_free_space_mode
        << " res=" << reloc_free_space_resolution
        << " max_ray=" << reloc_free_space_max_ray_length
        << " occ_min=" << reloc_free_space_occupied_min_points
        << " kill_sigmas=" << reloc_free_space_kill_sigmas
        << " map_pcd=" << (reloc_free_space_map_pcd.empty() ? "<auto>" : reloc_free_space_map_pcd) << "\n";
    oss << "Reloc hypothesis persistence: "
        << (reloc_persist_hypotheses ? "ON" : "OFF")
        << " max_frames=" << reloc_persist_max_frames << "\n";
    oss << "RHPD: enabled=" << (rhpd_enabled ? "YES" : "NO")
        << ", v2=" << (rhpd_v2_enable ? "YES" : "NO")
        << ", v3=" << (rhpd_v3_enable ? "YES" : "NO")
        << ", max_range=" << rhpd_max_range
        << ", z=[" << rhpd_z_min << "," << rhpd_z_max << "]"
        << ", dist_thr=" << rhpd_dist_threshold
        << ", candidates=" << rhpd_num_candidates
        << ", preselect=" << rhpd_preselect_candidates
        << ", submap_radius=" << rhpd_submap_kf_radius
        << ", submap_voxel=" << rhpd_submap_voxel_size << "\n";
    oss << "RHPD primary retrieval: weight=" << rhpd_primary_weight
        << ", sc_aux_weight=" << sc_aux_weight
        << ", sc_aux_veto=" << (sc_aux_veto_enabled ? "YES" : "NO")
        << "(thr=" << sc_aux_veto_threshold << ")"
        << ", use_sc_yaw=" << (rhpd_use_sc_yaw ? "YES" : "NO")
        << ", yaw_hyp=" << rhpd_yaw_hypotheses
        << ", aug(neg=" << (rhpd_enable_negative_space ? "YES" : "NO")
        << ", vert=" << (rhpd_enable_vertical_tokens ? "YES" : "NO")
        << ", pca_conf=" << (rhpd_enable_pca_confidence ? "YES" : "NO")
        << ")\n";
    oss << "Global map publish: hz=" << global_map_publish_hz
        << " voxel=" << global_map_voxel_size
        << " | save voxel=" << save_global_map_voxel_size
        << " save_on_shutdown=" << (save_global_map_on_shutdown ? "true" : "false") << "\n";
    oss << "Threads: " << num_threads
        << " | Sync: queue=" << sync_queue_size
        << " tolerance=" << sync_time_tolerance
        << " | Save path: " << map_save_path << "\n";
    oss << "==============================================";
    return oss.str();
}

bool Config::validate(std::string* error) const {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    auto positive = [&](double value, const char* name) {
        return std::isfinite(value) && value > 0.0 ? true : fail(std::string(name) + " must be > 0");
    };
    auto non_negative = [&](double value, const char* name) {
        return std::isfinite(value) && value >= 0.0 ? true : fail(std::string(name) + " must be >= 0");
    };
    auto at_least = [&](int value, int minimum, const char* name) {
        return value >= minimum ? true : fail(std::string(name) + " must be >= " + std::to_string(minimum));
    };

    if (mode != "mapping" && mode != "localization" && mode != "map_extension") {
        return fail("mode must be one of: mapping, localization, map_extension");
    }
    if ((mode == "localization" || mode == "map_extension") &&
        map_path.empty()) {
        return fail("map_path is required for localization and map_extension");
    }
    if (!positive(keyframe_distance_threshold, "keyframe_distance_threshold")) return false;
    if (!positive(keyframe_angle_threshold, "keyframe_angle_threshold")) return false;
    if (!at_least(submap_max_keyframes, 1,
                  "submap_max_keyframes")) return false;
    // PCL PointXYZI is 16-byte aligned and occupies 32 bytes in the supported
    // builds; reject budgets that cannot hold even one point.
    if (!at_least(submap_cloud_max_bytes, 32,
                  "submap_cloud_max_bytes")) return false;
    if (!positive(prior_noise_position, "prior_noise_position")) return false;
    if (!positive(prior_noise_rotation, "prior_noise_rotation")) return false;
    if (!positive(odom_noise_position, "odom_noise_position")) return false;
    if (!positive(odom_noise_rotation, "odom_noise_rotation")) return false;
    if (!positive(loop_noise_position, "loop_noise_position")) return false;
    if (!positive(loaded_map_tracking_noise_position,
                  "loaded_map_tracking_noise_position")) return false;
    if (!positive(loaded_map_tracking_noise_rotation,
                  "loaded_map_tracking_noise_rotation")) return false;
    if (!at_least(loop_axis_weighting_max, 1.0, "loop_axis_weighting_max")) return false;
    if (!positive(loop_noise_rotation, "loop_noise_rotation")) return false;
    if (!positive(gicp_downsampling_resolution, "gicp_downsampling_resolution")) return false;
    if (!positive(gicp_max_correspondence_distance, "gicp_max_correspondence_distance")) return false;
    if (!positive(gicp_transformation_epsilon, "gicp_transformation_epsilon")) return false;
    if (!positive(gicp_rotation_epsilon_deg, "gicp_rotation_epsilon_deg")) return false;
    if (!positive(gicp_fitness_threshold, "gicp_fitness_threshold")) return false;
    if (!at_least(gicp_max_iterations, 0, "gicp_max_iterations")) return false;
    if (!at_least(gicp_num_neighbors, 1, "gicp_num_neighbors")) return false;
    if (!at_least(gicp_submap_size, 0, "gicp_submap_size")) return false;
    if (!at_least(icp_refine_max_iterations, 0, "icp_refine_max_iterations")) return false;
    if (!positive(icp_refine_max_correspondence_distance, "icp_refine_max_correspondence_distance")) return false;
    if (!positive(icp_refine_downsampling_resolution, "icp_refine_downsampling_resolution")) return false;
    if (!non_negative(icp_refine_fitness_gate, "icp_refine_fitness_gate")) return false;
    if (!non_negative(icp_refine_delta_translation_gate, "icp_refine_delta_translation_gate")) return false;
    if (!non_negative(icp_refine_delta_rotation_gate, "icp_refine_delta_rotation_gate")) return false;
    if (!positive(sc_dist_threshold, "sc_dist_threshold")) return false;
    if (!at_least(sc_num_exclude_recent, 0, "sc_num_exclude_recent")) return false;
    if (!at_least(sc_num_candidates, 1, "sc_num_candidates")) return false;
    if (!positive(sc_max_radius, "sc_max_radius")) return false;
    if (!at_least(sc_num_rings, 1, "sc_num_rings")) return false;
    if (!at_least(sc_num_sectors, 1, "sc_num_sectors")) return false;
    if (!at_least(optimization_iterations, 0, "optimization_iterations")) return false;
    if (!positive(robust_kernel_delta, "robust_kernel_delta")) return false;
    if (!non_negative(loop_min_inlier_ratio, "loop_min_inlier_ratio")) return false;
    if (!positive(loop_fitness_threshold, "loop_fitness_threshold")) return false;
    if (!non_negative(loop_max_icp_translation, "loop_max_icp_translation")) return false;
    if (!non_negative(loop_max_icp_rotation, "loop_max_icp_rotation")) return false;
    if (!non_negative(loop_icp_prefilter_voxel_size, "loop_icp_prefilter_voxel_size")) return false;
    if (!at_least(loop_icp_max_points, 0, "loop_icp_max_points")) return false;
    if (!positive(loop_spatial_candidate_radius, "loop_spatial_candidate_radius")) return false;
    if (!at_least(loop_spatial_candidate_max_candidates, 1, "loop_spatial_candidate_max_candidates")) return false;
    if (!at_least(loop_kf_gap, 0, "loop_kf_gap")) return false;
    if (!at_least(loop_min_path_length_m, 0.0, "loop_min_path_length_m")) return false;
    if (!positive(loop_same_query_consensus_translation_m,
                  "loop_same_query_consensus_translation_m")) return false;
    if (!positive(loop_same_query_consensus_rotation_rad,
                  "loop_same_query_consensus_rotation_rad")) return false;
    if (!positive(floor_attitude_noise_deg, "floor_attitude_noise_deg")) return false;
    if (!positive(odom_sanity_max_speed_mps, "odom_sanity_max_speed_mps")) return false;
    if (!positive(mapping_static_voxel_m, "mapping_static_voxel_m")) return false;
    if (!positive(mapping_static_moved_overlap, "mapping_static_moved_overlap")) return false;
    if (!at_least(mapping_static_moved_consecutive, 1, "mapping_static_moved_consecutive")) return false;
    if (!positive(odom_sanity_max_angular_rate_dps, "odom_sanity_max_angular_rate_dps")) return false;
    if (!at_least(odom_sanity_max_consecutive, 1, "odom_sanity_max_consecutive")) return false;
    if (!positive(floor_attitude_max_radius_m, "floor_attitude_max_radius_m")) return false;
    if (!at_least(floor_attitude_min_points, 1, "floor_attitude_min_points")) return false;
    if (!positive(loop_max_range, "loop_max_range")) return false;
    if (!non_negative(global_map_voxel_size, "global_map_voxel_size")) return false;
    if (!non_negative(save_global_map_voxel_size, "save_global_map_voxel_size")) return false;
    if (!positive(global_map_publish_hz, "global_map_publish_hz")) return false;
    if (!at_least(num_threads, 1, "num_threads")) return false;
    if (!at_least(sync_queue_size, 1, "sync_queue_size")) return false;
    if (!positive(sync_time_tolerance, "sync_time_tolerance")) return false;
    if (!at_least(reloc_num_candidates, 1, "reloc_num_candidates")) return false;
    if (!positive(reloc_sc_dist_threshold, "reloc_sc_dist_threshold")) return false;
    if (!non_negative(reloc_min_confidence, "reloc_min_confidence")) return false;
    if (!non_negative(reloc_min_inlier_ratio, "reloc_min_inlier_ratio")) return false;
    if (!positive(reloc_search_radius, "reloc_search_radius")) return false;
    if (!at_least(reloc_max_track_failures, 0, "reloc_max_track_failures")) return false;
    if (!positive(reloc_track_max_translation, "reloc_track_max_translation")) return false;
    if (!positive(reloc_track_max_rotation, "reloc_track_max_rotation")) return false;
    if (!at_least(reloc_temporal_window_size, 1, "reloc_temporal_window_size")) return false;
    if (!at_least(reloc_lock_min_winner_streak, 1, "reloc_lock_min_winner_streak")) return false;
    if (!at_least(reloc_lock_min_converged_updates, 1, "reloc_lock_min_converged_updates")) return false;
    if (!non_negative(reloc_lock_min_margin, "reloc_lock_min_margin")) return false;
    if (!non_negative(reloc_hypothesis_miss_penalty, "reloc_hypothesis_miss_penalty")) return false;
    if (!non_negative(reloc_hypothesis_not_converged_penalty, "reloc_hypothesis_not_converged_penalty")) return false;
    if (!non_negative(reloc_reloc_inlier_weight, "reloc_reloc_inlier_weight")) return false;
    if (!non_negative(reloc_reloc_desc_dist_weight, "reloc_reloc_desc_dist_weight")) return false;
    if (!non_negative(reloc_track_motion_weight, "reloc_track_motion_weight")) return false;
    if (!at_least(reloc_track_retry_max_failures, 0, "reloc_track_retry_max_failures")) return false;
    if (!positive(reloc_track_retry_corr_scale, "reloc_track_retry_corr_scale")) return false;
    if (!at_least(reloc_track_retry_max_iterations, 0, "reloc_track_retry_max_iterations")) return false;
    if (!at_least(reloc_track_unstable_submap_size, 1, "reloc_track_unstable_submap_size")) return false;
    if (!at_least(reloc_static_agg_max_frames, 1, "reloc_static_agg_max_frames")) return false;
    if (!at_least(reloc_static_agg_min_frames, 1, "reloc_static_agg_min_frames")) return false;
    if (reloc_static_agg_min_frames > reloc_static_agg_max_frames) return fail("reloc_static_agg_min_frames must be <= reloc_static_agg_max_frames");
    if (!non_negative(reloc_static_agg_max_translation, "reloc_static_agg_max_translation")) return false;
    if (!non_negative(reloc_static_agg_max_rotation, "reloc_static_agg_max_rotation")) return false;
    if (!positive(reloc_static_agg_voxel_size, "reloc_static_agg_voxel_size")) return false;
    if (!non_negative(reloc_ambiguity_min_margin, "reloc_ambiguity_min_margin")) return false;
    if (!positive(reloc_ambiguity_min_ratio, "reloc_ambiguity_min_ratio")) return false;
    if (!non_negative(reloc_ambiguity_min_basin_separation, "reloc_ambiguity_min_basin_separation")) return false;
    if (!non_negative(reloc_ambiguity_min_consistency_margin, "reloc_ambiguity_min_consistency_margin")) return false;
    if (reloc_target_mode != "legacy_global_atlas" &&
        reloc_target_mode != "local_no_cache" &&
        reloc_target_mode != "local_lru" &&
        reloc_target_mode != "shadow_local_lru") {
        return fail("reloc_target_mode must be one of: legacy_global_atlas, local_no_cache, local_lru, shadow_local_lru");
    }
    if (!at_least(reloc_target_cache_max_bytes, 0,
                  "reloc_target_cache_max_bytes")) return false;
    if (!at_least(reloc_target_cache_max_entries, 0,
                  "reloc_target_cache_max_entries")) return false;
    if (!non_negative(rhpd_submap_voxel_size, "rhpd_submap_voxel_size")) return false;
    if (!positive(rhpd_max_range, "rhpd_max_range")) return false;
    if (!std::isfinite(rhpd_z_min) || !std::isfinite(rhpd_z_max) || rhpd_z_max <= rhpd_z_min) {
        return fail("rhpd_z_max must be greater than rhpd_z_min");
    }
    if (!positive(rhpd_dist_threshold, "rhpd_dist_threshold")) return false;
    if (!at_least(rhpd_num_candidates, 1, "rhpd_num_candidates")) return false;
    if (!at_least(rhpd_preselect_candidates, 1, "rhpd_preselect_candidates")) return false;
    if (!at_least(rhpd_submap_kf_radius, 0, "rhpd_submap_kf_radius")) return false;
    if (!non_negative(rhpd_primary_weight, "rhpd_primary_weight")) return false;
    if (!non_negative(sc_aux_weight, "sc_aux_weight")) return false;
    if (!positive(sc_aux_veto_threshold, "sc_aux_veto_threshold")) return false;
    if (!at_least(rhpd_yaw_hypotheses, 1, "rhpd_yaw_hypotheses")) return false;
    if (reloc_free_space_mode != "veto" && reloc_free_space_mode != "kill") {
        return fail("reloc_free_space_mode must be one of: veto, kill");
    }
    if (!positive(reloc_free_space_resolution, "reloc_free_space_resolution")) return false;
    if (!positive(reloc_free_space_max_ray_length, "reloc_free_space_max_ray_length")) return false;
    if (reloc_free_space_occupied_min_points < 1 || reloc_free_space_occupied_min_points > 255) {
        return fail("reloc_free_space_occupied_min_points must be in [1,255]");
    }
    if (!non_negative(reloc_free_space_kill_sigmas, "reloc_free_space_kill_sigmas")) return false;
    if (!at_least(reloc_persist_max_frames, 1, "reloc_persist_max_frames")) return false;
    return true;
}

} // namespace n3mapping
