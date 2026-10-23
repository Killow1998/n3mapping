// Spatial loop retrieval and its bounded boundary reserve.
#include "n3mapping/loop_detector.h"

#include <algorithm>
#include <cmath>
#include <glog/logging.h>

namespace n3mapping {

std::vector<LoopCandidate> LoopDetector::detectSpatialCandidates(
    int64_t query_id,
    const std::map<int64_t, Keyframe::Ptr>& keyframes,
    const MatchFilter& accept_match, bool include_boundary_ties) const {
    std::vector<LoopCandidate> candidates;
    if (!config_.loop_spatial_candidates_enable ||
        config_.loop_spatial_candidate_max_candidates <= 0) {
        return candidates;
    }

    auto query_it = keyframes.find(query_id);
    if (query_it == keyframes.end() || !query_it->second) {
        return candidates;
    }
    const auto& query_pose = query_it->second->pose_optimized;
    const double radius = std::max(1e-6, config_.loop_spatial_candidate_radius);
    const double min_path = std::max(0.0, config_.loop_min_path_length_m);

    // What makes a revisit worth registering is how far the platform walked
    // between the two visits, not how many keyframes it laid down. Asking for
    // fifty keyframes of index discarded the out-and-back at 95 to 104 m of
    // path on 0723 -- nine keyframes apart, nine metres of travel, and the
    // worst revisit disagreement in the map.
    std::map<int64_t, double> path_at;
    {
        double travelled = 0.0;
        const Eigen::Vector3d* previous = nullptr;
        Eigen::Vector3d previous_store;
        for (const auto& [id, keyframe] : keyframes) {
            if (keyframe) {
                const Eigen::Vector3d t = keyframe->pose_odom.translation();
                if (previous) {
                    travelled += (t - *previous).norm();
                }
                previous_store = t;
                previous = &previous_store;
            }
            path_at[id] = travelled;
        }
    }
    const auto query_path_it = path_at.find(query_id);
    const double query_path =
        query_path_it != path_at.end() ? query_path_it->second : 0.0;

    std::vector<std::pair<double, int64_t>> ranked;
    std::size_t spatial_considered = 0;
    std::size_t spatial_outside_radius = 0;
    for (const auto& [match_id, keyframe] : keyframes) {
        if (!keyframe || match_id >= query_id) {
            continue;
        }
        if (accept_match && !accept_match(match_id)) {
            continue;
        }
        const auto match_path_it = path_at.find(match_id);
        if (match_path_it == path_at.end() ||
            query_path - match_path_it->second < min_path) {
            continue;
        }
        const double squared_distance =
            (query_pose.translation() - keyframe->pose_optimized.translation()).squaredNorm();
        if (!std::isfinite(squared_distance)) {
            continue;
        }
        const double distance = std::sqrt(squared_distance);
        ++spatial_considered;
        // The radius is a hard bound, not a score normalizer: a candidate
        // farther than loop_spatial_candidate_radius must never enter the
        // returned set, however few candidates remain. Equal stays in.
        if (distance > radius) {
            ++spatial_outside_radius;
            continue;
        }
        ranked.emplace_back(distance, match_id);
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    const int primary_keep = std::min(config_.loop_spatial_candidate_max_candidates,
                                      static_cast<int>(ranked.size()));
    int keep = primary_keep;
    if (include_boundary_ties && primary_keep > 0) {
        // Graph updates can swap near-equal distances at the top-k boundary.
        // Keep the original nearest set and a bounded reserve within one
        // registration voxel. This is a retrieval heuristic, not an acceptance
        // tolerance or a calibrated pose uncertainty. The radius stays hard.
        const double boundary = ranked[primary_keep - 1].first +
            std::max(0.0, config_.gicp_downsampling_resolution);
        const std::size_t reserve_end = std::min(ranked.size(),
            2 * static_cast<std::size_t>(primary_keep));
        while (static_cast<std::size_t>(keep) < reserve_end &&
               ranked[keep].first <= boundary) ++keep;
    }
    if (config_.loop_debug_enable || VLOG_IS_ON(1)) {
        // Record the admission snapshot, including candidates outside top-k.
        // Final optimized poses cannot explain an earlier retrieval decision.
        // Ordinary verbose tracing needs only the boundary, and does not enable
        // the extra graph trials performed by the full debug stream.
        const auto trace_end = config_.loop_debug_enable ? ranked.size() :
            std::min(ranked.size(), static_cast<std::size_t>(keep) + 1);
        for (std::size_t i = 0; i < trace_end; ++i) {
            LOG(INFO) << "[Loop/SpatialRank] query=" << query_id
                      << " match=" << ranked[i].second << " rank=" << i + 1
                      << " distance=" << ranked[i].first
                      << " boundary_reserve=" << include_boundary_ties
                      << " selected=" << (i < static_cast<std::size_t>(keep));
        }
    }
    candidates.reserve(keep);
    for (int i = 0; i < keep; ++i) {
        LoopCandidate candidate;
        candidate.query_id = query_id;
        candidate.match_id = ranked[i].second;
        candidate.source_flags = LoopCandidate::SOURCE_SPATIAL;
        candidate.candidate_source = LoopCandidate::Source::SpatialRadius;
        candidate.fused_score = ranked[i].first / std::max(1e-6, radius);
        candidate.spatial_score = 1.0 / (1.0 + candidate.fused_score);
        candidate.fused_rank = i;
        candidates.push_back(candidate);
    }
    VLOG(1) << "[Loop/SpatialCandidates] considered=" << spatial_considered
            << " outside_radius=" << spatial_outside_radius
            << " kept=" << candidates.size() << " radius=" << radius;
    return candidates;
}

} // namespace n3mapping
