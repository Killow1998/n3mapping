#include "n3mapping/relocalization_debug_logger.h"

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

void appendNumberValue(std::ostream& os, double value)
{
    if (std::isfinite(value)) {
        os << std::setprecision(17) << value;
    } else {
        os << "null";
    }
}

void appendNumber(std::ostream& os, bool* first, const char* key, double value)
{
    appendComma(os, first);
    os << '"' << key << "\":";
    appendNumberValue(os, value);
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

void appendCandidateObject(std::ostream& os, const LoopCandidate& candidate)
{
    bool first = true;
    os << '{';
    appendInteger(os, &first, "match_id", candidate.match_id);
    appendString(os, &first, "candidate_source", candidateSourceName(candidate.candidate_source));
    appendNumber(os, &first, "rhpd_distance", candidate.rhpd_distance);
    appendNumber(os, &first, "sc_distance", candidate.sc_distance);
    appendNumber(os, &first, "fused_score", candidate.fused_score);
    appendNumber(os, &first, "yaw_diff_rad", static_cast<double>(candidate.yaw_diff_rad));
    os << '}';
}

void appendCandidates(std::ostream& os, bool* first, const char* key, const std::vector<LoopCandidate>& candidates)
{
    appendComma(os, first);
    os << '"' << key << "\":[";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        appendCandidateObject(os, candidates[i]);
    }
    os << ']';
}

void appendBasins(std::ostream& os, bool* first, const std::vector<RelocDebugBasinSummary>& basins)
{
    appendComma(os, first);
    os << "\"basins\":[";
    for (std::size_t i = 0; i < basins.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        bool basin_first = true;
        os << '{';
        appendInteger(os, &basin_first, "center_match_id", basins[i].center_match_id);
        appendSize(os, &basin_first, "member_count", basins[i].member_match_ids.size());
        appendComma(os, &basin_first);
        os << "\"member_match_ids\":[";
        for (std::size_t j = 0; j < basins[i].member_match_ids.size(); ++j) {
            if (j > 0) {
                os << ',';
            }
            os << basins[i].member_match_ids[j];
        }
        os << "]}";
    }
    os << ']';
}

void appendBasinBest(std::ostream& os, bool* first, const std::vector<RelocDebugBasinBestSummary>& results)
{
    appendComma(os, first);
    os << "\"per_basin_best\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        bool item_first = true;
        os << '{';
        appendInteger(os, &item_first, "basin_center_id", results[i].basin_center_id);
        appendInteger(os, &item_first, "matched_kf_id", results[i].matched_kf_id);
        appendComma(os, &item_first);
        os << "\"candidate\":";
        appendCandidateObject(os, results[i].candidate);
        appendNumber(os, &item_first, "fitness_score", results[i].fitness_score);
        appendNumber(os, &item_first, "inlier_ratio", results[i].inlier_ratio);
        appendNumber(os, &item_first, "selection_score", results[i].selection_score);
        appendNumber(os, &item_first, "log_likelihood", results[i].log_likelihood);
        os << '}';
    }
    os << ']';
}

void appendHypotheses(std::ostream& os, bool* first, const std::vector<RelocDebugHypothesisSummary>& hypotheses)
{
    appendComma(os, first);
    os << "\"hypotheses\":[";
    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        bool item_first = true;
        os << '{';
        appendInteger(os, &item_first, "seed_match_id", hypotheses[i].seed_match_id);
        appendInteger(os, &item_first, "last_match_id", hypotheses[i].last_match_id);
        appendNumber(os, &item_first, "cumulative_log_likelihood", hypotheses[i].cumulative_log_likelihood);
        appendInteger(os, &item_first, "num_updates", hypotheses[i].num_updates);
        appendInteger(os, &item_first, "converged_updates", hypotheses[i].converged_updates);
        appendBool(os, &item_first, "alive", hypotheses[i].alive);
        os << '}';
    }
    os << ']';
}

void appendQueryCloudSummary(std::ostream& os,
                             bool* first,
                             const char* prefix,
                             const RelocQueryCloudDebugSummary& summary)
{
    const std::string base(prefix);
    appendString(os, first, (base + "_mode").c_str(), summary.mode);
    appendInteger(os, first, (base + "_frame_count").c_str(), summary.frame_count);
    appendNumber(os, first, (base + "_motion_translation_m").c_str(), summary.motion_translation_m);
    appendNumber(os, first, (base + "_motion_rotation_rad").c_str(), summary.motion_rotation_rad);
    appendSize(os, first, (base + "_raw_points").c_str(), summary.raw_points);
    appendSize(os, first, (base + "_downsampled_points").c_str(), summary.downsampled_points);
    appendSize(os, first, (base + "_candidate_count").c_str(), summary.candidate_count);
    appendCandidates(os, first, (base + "_top_candidates").c_str(), summary.top_candidates);
}

void appendPose(std::ostream& os, bool* first, const char* key, const Eigen::Isometry3d& pose)
{
    appendComma(os, first);
    const Eigen::Quaterniond q(pose.rotation());
    os << '"' << key << "\":{";
    bool pose_first = true;
    appendNumber(os, &pose_first, "x", pose.translation().x());
    appendNumber(os, &pose_first, "y", pose.translation().y());
    appendNumber(os, &pose_first, "z", pose.translation().z());
    appendNumber(os, &pose_first, "qx", q.x());
    appendNumber(os, &pose_first, "qy", q.y());
    appendNumber(os, &pose_first, "qz", q.z());
    appendNumber(os, &pose_first, "qw", q.w());
    os << '}';
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

std::string RelocalizationDebugLogger::resolvePath(const Config& config)
{
    if (!config.reloc_debug_path.empty()) {
        return config.reloc_debug_path;
    }
    if (config.map_save_path.empty()) {
        return "relocalization_debug.jsonl";
    }
    return (std::filesystem::path(config.map_save_path) / "relocalization_debug.jsonl").string();
}

bool RelocalizationDebugLogger::appendRelocalization(const std::string& path,
                                                     const RelocalizationDebugEvent& event)
{
    std::ostringstream os;
    bool first = true;
    os << '{';
    appendString(os, &first, "record_type", "relocalize");
    appendNumber(os, &first, "processing_time", event.processing_time);
    appendSize(os, &first, "query_index", event.query_index);
    appendQueryCloudSummary(os, &first, "query", event.query_cloud);
    appendQueryCloudSummary(os, &first, "motion_query", event.motion_query_cloud);
    appendSize(os, &first, "candidate_count", event.candidate_count);
    appendCandidates(os, &first, "top_candidates", event.top_candidates);
    appendBasins(os, &first, event.basins);
    appendBasinBest(os, &first, event.basin_best_results);
    appendHypotheses(os, &first, event.hypotheses);
    appendNumber(os, &first, "temporal_hypothesis_score", event.temporal_hypothesis_score);
    appendNumber(os, &first, "log_likelihood", event.log_likelihood);
    appendInteger(os, &first, "winner_streak", event.winner_streak);
    appendNumber(os, &first, "margin", event.margin);
    appendNumber(os, &first, "ratio", event.ratio);
    appendNumber(os, &first, "basin_separation", event.basin_separation);
    appendBool(os, &first, "lock_accepted", event.lock_accepted);
    appendString(os, &first, "lock_result", event.lock_result);
    appendString(os, &first, "reject_reason", event.reject_reason);
    os << '}';
    return appendLine(path, os.str());
}

bool RelocalizationDebugLogger::appendTracking(const std::string& path,
                                               const RelocTrackingDebugEvent& event)
{
    std::ostringstream os;
    bool first = true;
    os << '{';
    appendString(os, &first, "record_type", "tracking");
    appendNumber(os, &first, "processing_time", event.processing_time);
    appendSize(os, &first, "query_index", event.query_index);
    appendPose(os, &first, "predicted_pose", event.predicted_pose);
    appendInteger(os, &first, "nearest_kf_id", event.nearest_kf_id);
    appendSize(os, &first, "submap_size", event.submap_size);
    appendBool(os, &first, "icp_converged", event.icp_converged);
    appendNumber(os, &first, "fitness_score", event.fitness_score);
    appendNumber(os, &first, "inlier_ratio", event.inlier_ratio);
    appendBool(os, &first, "retry_used", event.retry_used);
    appendInteger(os, &first, "consecutive_track_failures", event.consecutive_track_failures);
    appendBool(os, &first, "result_success", event.result_success);
    appendString(os, &first, "reject_reason", event.reject_reason);
    os << '}';
    return appendLine(path, os.str());
}

}  // namespace n3mapping
