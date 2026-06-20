#include "n3mapping/config.h"

#include <cmath>
#include <sstream>

namespace n3mapping {

std::string Config::toString() const {
    std::ostringstream oss;
    oss << "========== N3Mapping Configuration ==========\n";
    oss << "Mode: " << mode << " | Map path: " << map_path << "\n";
    oss << "Frames: world=" << world_frame << " body=" << body_frame << "\n";
    oss << "Keyframe: dist=" << keyframe_distance_threshold
        << " m, angle=" << keyframe_angle_threshold << " rad\n";
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
    oss << "Robust kernel: " << (use_robust_kernel ? "ON" : "OFF")
        << " type=" << robust_kernel_type
        << " delta=" << robust_kernel_delta << "\n";
    oss << "Loop: fitness_thr=" << loop_fitness_threshold
        << ", min_inlier=" << loop_min_inlier_ratio
        << ", max_icp_t=" << loop_max_icp_translation
        << ", max_icp_r=" << loop_max_icp_rotation
        << ", max_residual_z=" << loop_max_candidate_residual_z
        << ", edge_model=" << (loop_observability_edge_model_enable ? "observability" : "fixed")
        << ", planar_vertical_weight=" << loop_planar_vertical_weight
        << ", prefilter_voxel=" << loop_icp_prefilter_voxel_size
        << ", max_points=" << loop_icp_max_points << "\n";
    oss << "Loop debug JSONL: " << (loop_debug_enable ? "ON" : "OFF")
        << " vertical_hypotheses=" << (loop_debug_vertical_hypotheses_enable ? "ON" : "OFF")
        << " path=" << (loop_debug_path.empty() ? "<map_save_path>/loop_debug.jsonl" : loop_debug_path) << "\n";
    oss << "Loop graph trial gate: " << (loop_graph_trial_gate_enable ? "ON" : "OFF")
        << " max_residual_z=" << loop_graph_trial_max_residual_z << "\n";
    oss << "Loop candidate pipeline: RHPD primary retrieval + optional spatial radius supplement -> optional SC yaw/weak rerank/veto -> ICP -> geom gate -> LoopClosureManager filter/select\n";
    oss << "Loop spatial candidates: " << (loop_spatial_candidates_enable ? "ON" : "OFF")
        << " radius=" << loop_spatial_candidate_radius
        << " min_id_gap=" << loop_spatial_candidate_min_id_gap
        << " max_candidates=" << loop_spatial_candidate_max_candidates << "\n";
    oss << "loop_closest_id_th/min_id_interval/max_range are retained for compatibility/logging only; they are not used in the active mapping loop-candidate retrieval/verification path.\n";
    oss << "Loop timing: loop_kf_gap=" << loop_kf_gap << " (active)\n";
    oss << "Legacy loop distance params (inactive, compatibility/logging only): closest_id_th="
        << loop_closest_id_th << ", min_id_interval=" << loop_min_id_interval
        << ", max_range=" << loop_max_range << "\n";
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
    if (!positive(keyframe_distance_threshold, "keyframe_distance_threshold")) return false;
    if (!positive(keyframe_angle_threshold, "keyframe_angle_threshold")) return false;
    if (!positive(prior_noise_position, "prior_noise_position")) return false;
    if (!positive(prior_noise_rotation, "prior_noise_rotation")) return false;
    if (!positive(odom_noise_position, "odom_noise_position")) return false;
    if (!positive(odom_noise_rotation, "odom_noise_rotation")) return false;
    if (!positive(loop_noise_position, "loop_noise_position")) return false;
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
    if (!at_least(kdtree_cache_size, 1, "kdtree_cache_size")) return false;
    if (!at_least(optimization_iterations, 0, "optimization_iterations")) return false;
    if (!positive(robust_kernel_delta, "robust_kernel_delta")) return false;
    if (!non_negative(loop_min_inlier_ratio, "loop_min_inlier_ratio")) return false;
    if (!positive(loop_fitness_threshold, "loop_fitness_threshold")) return false;
    if (!non_negative(loop_max_icp_translation, "loop_max_icp_translation")) return false;
    if (!non_negative(loop_max_icp_rotation, "loop_max_icp_rotation")) return false;
    if (!non_negative(loop_max_candidate_residual_z, "loop_max_candidate_residual_z")) return false;
    if (!positive(loop_planar_vertical_weight, "loop_planar_vertical_weight")) return false;
    if (loop_planar_vertical_weight > 1.0) return fail("loop_planar_vertical_weight must be <= 1");
    if (!non_negative(loop_icp_prefilter_voxel_size, "loop_icp_prefilter_voxel_size")) return false;
    if (!at_least(loop_icp_max_points, 0, "loop_icp_max_points")) return false;
    if (!non_negative(loop_graph_trial_max_residual_z, "loop_graph_trial_max_residual_z")) return false;
    if (!positive(loop_spatial_candidate_radius, "loop_spatial_candidate_radius")) return false;
    if (!at_least(loop_spatial_candidate_min_id_gap, 1, "loop_spatial_candidate_min_id_gap")) return false;
    if (!at_least(loop_spatial_candidate_max_candidates, 1, "loop_spatial_candidate_max_candidates")) return false;
    if (!at_least(loop_kf_gap, 0, "loop_kf_gap")) return false;
    if (!at_least(loop_closest_id_th, 0, "loop_closest_id_th")) return false;
    if (!at_least(loop_min_id_interval, 0, "loop_min_id_interval")) return false;
    if (!positive(loop_max_range, "loop_max_range")) return false;
    if (!positive(output_cloud_voxel_size, "output_cloud_voxel_size")) return false;
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
    return true;
}

} // namespace n3mapping
