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
        << ", max_icp_r=" << loop_max_icp_rotation << "\n";
    oss << "Loop candidate pipeline: RHPD primary retrieval -> optional SC yaw/weak rerank/veto -> ICP -> geom gate -> LoopClosureManager filter/select\n";
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
    oss << "Threads: " << num_threads << " | Save path: " << map_save_path << "\n";
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

    if (!positive(prior_noise_position, "prior_noise_position")) return false;
    if (!positive(prior_noise_rotation, "prior_noise_rotation")) return false;
    if (!positive(odom_noise_position, "odom_noise_position")) return false;
    if (!positive(odom_noise_rotation, "odom_noise_rotation")) return false;
    if (!positive(loop_noise_position, "loop_noise_position")) return false;
    if (!positive(loop_noise_rotation, "loop_noise_rotation")) return false;
    if (!positive(gicp_downsampling_resolution, "gicp_downsampling_resolution")) return false;
    if (!positive(icp_refine_downsampling_resolution, "icp_refine_downsampling_resolution")) return false;
    if (!positive(output_cloud_voxel_size, "output_cloud_voxel_size")) return false;
    if (!non_negative(global_map_voxel_size, "global_map_voxel_size")) return false;
    if (!non_negative(save_global_map_voxel_size, "save_global_map_voxel_size")) return false;
    if (!positive(reloc_static_agg_voxel_size, "reloc_static_agg_voxel_size")) return false;
    if (!non_negative(rhpd_submap_voxel_size, "rhpd_submap_voxel_size")) return false;
    if (!positive(rhpd_max_range, "rhpd_max_range")) return false;
    if (!std::isfinite(rhpd_z_min) || !std::isfinite(rhpd_z_max) || rhpd_z_max <= rhpd_z_min) {
        return fail("rhpd_z_max must be greater than rhpd_z_min");
    }
    return true;
}

} // namespace n3mapping
