#include "n3mapping/loop_debug_logger.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace n3mapping {

namespace {

std::string jsonEscape(const std::string& value)
{
    std::ostringstream oss;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    oss << static_cast<char>(ch);
                }
                break;
        }
    }
    return oss.str();
}

void appendComma(std::ostream& os, bool* first)
{
    if (*first) {
        *first = false;
        return;
    }
    os << ',';
}

void appendString(std::ostream& os, bool* first, const char* key, const std::string& value)
{
    appendComma(os, first);
    os << '"' << key << "\":\"" << jsonEscape(value) << '"';
}

void appendBool(std::ostream& os, bool* first, const char* key, bool value)
{
    appendComma(os, first);
    os << '"' << key << "\":" << (value ? "true" : "false");
}

void appendNumber(std::ostream& os, bool* first, const char* key, double value)
{
    appendComma(os, first);
    os << '"' << key << "\":";
    if (std::isfinite(value)) {
        os << std::setprecision(17) << value;
    } else {
        os << "null";
    }
}

void appendInteger(std::ostream& os, bool* first, const char* key, int64_t value)
{
    appendComma(os, first);
    os << '"' << key << "\":" << value;
}

void appendSize(std::ostream& os, bool* first, const char* key, std::size_t value)
{
    appendComma(os, first);
    os << '"' << key << "\":" << value;
}

void appendInt(std::ostream& os, bool* first, const char* key, int value)
{
    appendComma(os, first);
    os << '"' << key << "\":" << value;
}

void appendAcceptedEdges(std::ostream& os,
                         bool* first,
                         const std::vector<std::pair<int64_t, int64_t>>& edges)
{
    appendComma(os, first);
    os << "\"accepted_edges\":[";
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << "{\"from_id\":" << edges[i].first
           << ",\"to_id\":" << edges[i].second << '}';
    }
    os << ']';
}

const char* candidateSourceName(LoopCandidate::Source source)
{
    switch (source) {
        case LoopCandidate::Source::RhpdPrimary:
            return "rhpd_primary";
        case LoopCandidate::Source::ScanContextFallback:
            return "scan_context_fallback";
        case LoopCandidate::Source::RhpdFrame:
            return "rhpd_frame";
        case LoopCandidate::Source::SpatialRadius:
            return "spatial_radius";
        case LoopCandidate::Source::Unknown:
        default:
            return "unknown";
    }
}

Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d& transform)
{
    return transform.rotation().eulerAngles(0, 1, 2);
}

void appendResidual(std::ostream& os, bool* first, const Eigen::Isometry3d& residual)
{
    const Eigen::Vector3d translation = residual.translation();
    const Eigen::Vector3d rpy = rollPitchYaw(residual);
    appendNumber(os, first, "residual_x", translation.x());
    appendNumber(os, first, "residual_y", translation.y());
    appendNumber(os, first, "residual_z", translation.z());
    appendNumber(os, first, "residual_roll", rpy.x());
    appendNumber(os, first, "residual_pitch", rpy.y());
    appendNumber(os, first, "residual_yaw", rpy.z());
}

void appendTransformAxes(std::ostream& os,
                         bool* first,
                         const char* prefix,
                         const Eigen::Isometry3d& transform)
{
    const Eigen::Vector3d translation = transform.translation();
    const Eigen::Vector3d rpy = rollPitchYaw(transform);
    const std::string x_key = std::string(prefix) + "_x";
    const std::string y_key = std::string(prefix) + "_y";
    const std::string z_key = std::string(prefix) + "_z";
    const std::string roll_key = std::string(prefix) + "_roll";
    const std::string pitch_key = std::string(prefix) + "_pitch";
    const std::string yaw_key = std::string(prefix) + "_yaw";
    appendNumber(os, first, x_key.c_str(), translation.x());
    appendNumber(os, first, y_key.c_str(), translation.y());
    appendNumber(os, first, z_key.c_str(), translation.z());
    appendNumber(os, first, roll_key.c_str(), rpy.x());
    appendNumber(os, first, pitch_key.c_str(), rpy.y());
    appendNumber(os, first, yaw_key.c_str(), rpy.z());
}

void appendInformationDiag(std::ostream& os, bool* first, const Eigen::Matrix<double, 6, 6>& information)
{
    appendComma(os, first);
    os << "\"loop_information_diag\":[";
    for (int i = 0; i < 6; ++i) {
        if (i > 0) {
            os << ',';
        }
        const double value = information(i, i);
        if (std::isfinite(value)) {
            os << std::setprecision(17) << value;
        } else {
            os << "null";
        }
    }
    os << ']';
}

bool appendLine(const std::string& path, const std::string& line)
{
    try {
        const std::filesystem::path fs_path(path);
        const auto parent = fs_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        std::ofstream file(path, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            return false;
        }
        file << line << '\n';
        return file.good();
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

std::string LoopDebugLogger::resolvePath(const Config& config)
{
    if (!config.loop_debug_path.empty()) {
        return config.loop_debug_path;
    }
    if (config.map_save_path.empty()) {
        return "loop_debug.jsonl";
    }
    return (std::filesystem::path(config.map_save_path) / "loop_debug.jsonl").string();
}

bool LoopDebugLogger::appendCandidate(const std::string& path, const LoopDebugCandidateEvent& event)
{
    std::ostringstream os;
    bool first = true;
    os << '{';
    appendString(os, &first, "record_type", "candidate");
    appendNumber(os, &first, "processing_time", event.processing_time);
    appendNumber(os, &first, "timestamp", event.query_timestamp);
    appendInteger(os, &first, "query_id", event.candidate.query_id);
    appendInteger(os, &first, "match_id", event.candidate.match_id);
    appendString(os, &first, "candidate_source", candidateSourceName(event.candidate.candidate_source));
    appendNumber(os, &first, "rhpd_distance", event.candidate.rhpd_distance);
    appendNumber(os, &first, "sc_distance", event.candidate.sc_distance);
    appendNumber(os, &first, "fused_score", event.candidate.fused_score);
    appendNumber(os, &first, "yaw_diff_rad", static_cast<double>(event.candidate.yaw_diff_rad));
    appendBool(os, &first, "icp_converged", event.icp_converged);
    appendSize(os, &first, "icp_iterations", event.icp_iterations);
    appendNumber(os, &first, "icp_optimizer_error", event.icp_optimizer_error);
    appendString(os, &first, "icp_termination", event.icp_termination);
    appendNumber(os, &first, "fitness_score", event.fitness_score);
    appendNumber(os, &first, "inlier_ratio", event.inlier_ratio);
    appendNumber(os, &first, "icp_translation_norm", event.icp_translation_norm);
    appendNumber(os, &first, "icp_rotation_norm", event.icp_rotation_norm);
    appendResidual(os, &first, event.residual);
    appendTransformAxes(os, &first, "pred_match_query", event.T_pred_match_query);
    appendTransformAxes(os, &first, "icp_correction_match", event.T_icp_correction_match);
    appendTransformAxes(os, &first, "measured_match_query", event.T_measured_match_query);
    if (event.has_loop_measurement) {
        appendTransformAxes(os, &first, "measurement", event.loop_measurement_match_query);
    }
    appendString(os, &first, "edge_mode", event.edge_mode);
    appendNumber(os, &first, "vertical_observability_score", event.vertical_observability_score);
    appendBool(os, &first, "vertical_downweighted", event.vertical_downweighted);
    appendNumber(os, &first, "source_z_span", event.source_z_span);
    appendNumber(os, &first, "target_z_span", event.target_z_span);
    appendNumber(os, &first, "z_overlap_ratio_before", event.z_overlap_ratio_before);
    appendNumber(os, &first, "z_overlap_ratio_after", event.z_overlap_ratio_after);
    appendNumber(os, &first, "source_z_robust_span", event.source_z_robust_span);
    appendNumber(os, &first, "target_z_robust_span", event.target_z_robust_span);
    appendNumber(os, &first, "z_robust_overlap_ratio_before", event.z_robust_overlap_ratio_before);
    appendNumber(os, &first, "z_robust_overlap_ratio_after", event.z_robust_overlap_ratio_after);
    appendNumber(os, &first, "source_target_z_centroid_delta_before", event.source_target_z_centroid_delta_before);
    appendNumber(os, &first, "source_target_z_centroid_delta_after", event.source_target_z_centroid_delta_after);
    appendNumber(os, &first, "vertical_information_ratio", event.vertical_information_ratio);
    appendInt(os, &first, "vertical_hypothesis_count", event.vertical_hypothesis_count);
    appendNumber(os, &first, "best_z_offset_m", event.best_z_offset_m);
    appendNumber(os, &first, "best_z_offset_fitness", event.best_z_offset_fitness);
    appendNumber(os, &first, "zero_z_fitness", event.zero_z_fitness);
    appendNumber(os, &first, "fitness_gap_zero_vs_best", event.fitness_gap_zero_vs_best);
    appendNumber(os, &first, "z_hypothesis_spread_m", event.z_hypothesis_spread_m);
    appendNumber(os, &first, "vertical_ambiguity_score", event.vertical_ambiguity_score);
    appendString(os, &first, "vertical_hypothesis_edge_recommendation",
                 event.vertical_hypothesis_edge_recommendation);
    appendInt(os, &first, "heightmap_overlap_cell_count", event.heightmap_overlap_cell_count);
    appendNumber(os, &first, "heightmap_overlap_ratio", event.heightmap_overlap_ratio);
    appendNumber(os, &first, "heightmap_ground_dz_median", event.heightmap_ground_dz_median);
    appendNumber(os, &first, "heightmap_ground_dz_p90", event.heightmap_ground_dz_p90);
    appendNumber(os, &first, "heightmap_ground_dz_max", event.heightmap_ground_dz_max);
    appendNumber(os, &first, "heightmap_ground_support_ratio", event.heightmap_ground_support_ratio);
    appendNumber(os, &first, "heightmap_vertical_consistency_score",
                 event.heightmap_vertical_consistency_score);
    appendBool(os, &first, "graph_trial_success", event.graph_trial_success);
    appendNumber(os, &first, "graph_trial_residual_x_after", event.graph_trial_residual_x_after);
    appendNumber(os, &first, "graph_trial_residual_y_after", event.graph_trial_residual_y_after);
    appendNumber(os, &first, "graph_trial_residual_z_after", event.graph_trial_residual_z_after);
    appendNumber(os, &first, "graph_trial_residual_roll_after", event.graph_trial_residual_roll_after);
    appendNumber(os, &first, "graph_trial_residual_pitch_after", event.graph_trial_residual_pitch_after);
    appendNumber(os, &first, "graph_trial_residual_yaw_after", event.graph_trial_residual_yaw_after);
    appendNumber(os, &first, "graph_trial_residual_translation_norm_after",
                 event.graph_trial_residual_translation_norm_after);
    appendNumber(os, &first, "graph_trial_residual_rotation_norm_after",
                 event.graph_trial_residual_rotation_norm_after);
    appendNumber(os, &first, "graph_trial_mean_pose_update_translation",
                 event.graph_trial_mean_pose_update_translation);
    appendNumber(os, &first, "graph_trial_max_pose_update_translation",
                 event.graph_trial_max_pose_update_translation);
    appendNumber(os, &first, "graph_trial_mean_pose_update_rotation",
                 event.graph_trial_mean_pose_update_rotation);
    appendNumber(os, &first, "graph_trial_max_pose_update_rotation",
                 event.graph_trial_max_pose_update_rotation);
    appendNumber(os, &first, "graph_trial_existing_loop_residual_delta",
                 event.graph_trial_existing_loop_residual_delta);
    appendNumber(os, &first, "graph_trial_odom_residual_delta", event.graph_trial_odom_residual_delta);
    appendNumber(os, &first, "graph_trial_consistency_score", event.graph_trial_consistency_score);
    appendString(os, &first, "graph_trial_recommendation", event.graph_trial_recommendation);
    appendInt(os, &first, "segment_pair_count", event.segment_pair_count);
    appendInt(os, &first, "segment_valid_pair_count", event.segment_valid_pair_count);
    appendInt(os, &first, "segment_consensus_inlier_count", event.segment_consensus_inlier_count);
    appendNumber(os, &first, "segment_consensus_ratio", event.segment_consensus_ratio);
    appendNumber(os, &first, "segment_translation_median", event.segment_translation_median);
    appendNumber(os, &first, "segment_translation_std", event.segment_translation_std);
    appendNumber(os, &first, "segment_yaw_median", event.segment_yaw_median);
    appendNumber(os, &first, "segment_yaw_std", event.segment_yaw_std);
    appendNumber(os, &first, "segment_z_std", event.segment_z_std);
    appendNumber(os, &first, "segment_roll_pitch_std", event.segment_roll_pitch_std);
    appendString(os, &first, "segment_direction", event.segment_direction);
    appendString(os, &first, "segment_recommendation", event.segment_recommendation);
    appendString(os, &first, "consensus_shadow_decision", event.consensus_shadow_decision);
    appendString(os, &first, "consensus_shadow_reason", event.consensus_shadow_reason);
    appendInt(os, &first, "consensus_valid_pair_count", event.consensus_valid_pair_count);
    appendInt(os, &first, "consensus_left_support_count", event.consensus_left_support_count);
    appendInt(os, &first, "consensus_right_support_count", event.consensus_right_support_count);
    appendInt(os, &first, "consensus_contradiction_count", event.consensus_contradiction_count);
    appendNumber(os, &first, "consensus_median_translation_delta",
                 event.consensus_median_translation_delta);
    appendNumber(os, &first, "consensus_mad_translation_delta",
                 event.consensus_mad_translation_delta);
    appendNumber(os, &first, "consensus_median_rotation_delta",
                 event.consensus_median_rotation_delta);
    appendNumber(os, &first, "consensus_mad_rotation_delta",
                 event.consensus_mad_rotation_delta);
    appendBool(os, &first, "consensus_estimator_valid", event.consensus_estimator_valid);
    appendInt(os, &first, "consensus_estimator_pair_count",
              event.consensus_estimator_pair_count);
    appendInt(os, &first, "consensus_estimator_inlier_count",
              event.consensus_estimator_inlier_count);
    appendNumber(os, &first, "consensus_estimator_inlier_ratio",
                 event.consensus_estimator_inlier_ratio);
    appendNumber(os, &first, "consensus_estimator_translation_median",
                 event.consensus_estimator_translation_median);
    appendNumber(os, &first, "consensus_estimator_z_median",
                 event.consensus_estimator_z_median);
    appendNumber(os, &first, "consensus_estimator_yaw_median",
                 event.consensus_estimator_yaw_median);
    appendNumber(os, &first, "consensus_estimator_translation_mad",
                 event.consensus_estimator_translation_mad);
    appendNumber(os, &first, "consensus_estimator_z_mad",
                 event.consensus_estimator_z_mad);
    appendNumber(os, &first, "consensus_estimator_yaw_mad",
                 event.consensus_estimator_yaw_mad);
    appendNumber(os, &first, "consensus_estimator_measurement_delta_translation",
                 event.consensus_estimator_measurement_delta_translation);
    appendNumber(os, &first, "consensus_estimator_measurement_delta_rotation",
                 event.consensus_estimator_measurement_delta_rotation);
    appendString(os, &first, "consensus_estimator_recommendation",
                 event.consensus_estimator_recommendation);
    appendString(os, &first, "loop_referee_recommendation", event.loop_referee_recommendation);
    appendString(os, &first, "loop_referee_reason", event.loop_referee_reason);
    appendString(os, &first, "loop_referee_risk_flags", event.loop_referee_risk_flags);
    appendString(os, &first, "gate_result", event.gate_result);
    appendString(os, &first, "reject_reason", event.reject_reason);
    appendInformationDiag(os, &first, event.loop_information);
    os << '}';
    return appendLine(path, os.str());
}

bool LoopDebugLogger::appendOptimizationSummary(const std::string& path, const LoopDebugOptimizationEvent& event)
{
    std::ostringstream os;
    bool first = true;
    os << '{';
    appendString(os, &first, "record_type", "optimization_summary");
    appendNumber(os, &first, "processing_time", event.processing_time);
    appendSize(os, &first, "accepted_edge_count", event.accepted_edge_count);
    appendAcceptedEdges(os, &first, event.accepted_edges);
    appendNumber(os, &first, "loop_residual_translation_before", event.loop_residual_translation_before);
    appendNumber(os, &first, "loop_residual_translation_after", event.loop_residual_translation_after);
    appendNumber(os, &first, "loop_residual_rotation_before", event.loop_residual_rotation_before);
    appendNumber(os, &first, "loop_residual_rotation_after", event.loop_residual_rotation_after);
    appendNumber(os, &first, "loop_residual_x_before", event.loop_residual_translation_axes_before.x());
    appendNumber(os, &first, "loop_residual_y_before", event.loop_residual_translation_axes_before.y());
    appendNumber(os, &first, "loop_residual_z_before", event.loop_residual_translation_axes_before.z());
    appendNumber(os, &first, "loop_residual_x_after", event.loop_residual_translation_axes_after.x());
    appendNumber(os, &first, "loop_residual_y_after", event.loop_residual_translation_axes_after.y());
    appendNumber(os, &first, "loop_residual_z_after", event.loop_residual_translation_axes_after.z());
    appendNumber(os, &first, "loop_residual_roll_before", event.loop_residual_rpy_axes_before.x());
    appendNumber(os, &first, "loop_residual_pitch_before", event.loop_residual_rpy_axes_before.y());
    appendNumber(os, &first, "loop_residual_yaw_before", event.loop_residual_rpy_axes_before.z());
    appendNumber(os, &first, "loop_residual_roll_after", event.loop_residual_rpy_axes_after.x());
    appendNumber(os, &first, "loop_residual_pitch_after", event.loop_residual_rpy_axes_after.y());
    appendNumber(os, &first, "loop_residual_yaw_after", event.loop_residual_rpy_axes_after.z());
    appendNumber(os, &first, "mean_pose_update_translation", event.mean_pose_update_translation);
    appendNumber(os, &first, "max_pose_update_translation", event.max_pose_update_translation);
    appendNumber(os, &first, "mean_pose_update_rotation", event.mean_pose_update_rotation);
    appendNumber(os, &first, "max_pose_update_rotation", event.max_pose_update_rotation);
    os << '}';
    return appendLine(path, os.str());
}

}  // namespace n3mapping
