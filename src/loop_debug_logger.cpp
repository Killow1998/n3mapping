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
    appendNumber(os, &first, "fitness_score", event.fitness_score);
    appendNumber(os, &first, "inlier_ratio", event.inlier_ratio);
    appendNumber(os, &first, "icp_translation_norm", event.icp_translation_norm);
    appendNumber(os, &first, "icp_rotation_norm", event.icp_rotation_norm);
    appendResidual(os, &first, event.residual);
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
