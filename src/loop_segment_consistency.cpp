#include "n3mapping/loop_segment_consistency.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace n3mapping {
namespace {

constexpr int kMinSegmentPairs = 2;

double normalizeAngle(double angle)
{
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d& transform)
{
    return transform.rotation().eulerAngles(0, 1, 2);
}

double median(std::vector<double> values)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if (values.size() % 2 == 1) {
        return *mid;
    }
    const double upper = *mid;
    std::nth_element(values.begin(), mid - 1, values.end());
    return 0.5 * (*(mid - 1) + upper);
}

double sampleStdDev(const std::vector<double>& values)
{
    if (values.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    double accum = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        accum += diff * diff;
    }
    return std::sqrt(accum / static_cast<double>(values.size() - 1));
}

struct DirectionStats {
    int pair_count = 0;
    int valid_pair_count = 0;
    int inlier_count = 0;
    std::vector<double> translation_errors;
    std::vector<double> yaw_errors;
    std::vector<double> z_errors;
    std::vector<double> roll_pitch_errors;
    std::string direction;
};

Keyframe::Ptr findLocalCorrespondence(const std::vector<Keyframe::Ptr>& keyframes,
                                      int64_t match_id,
                                      int query_offset,
                                      int local_id_radius,
                                      const Eigen::Isometry3d& expected_pose)
{
    Keyframe::Ptr best;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto& keyframe : keyframes) {
        if (!keyframe) {
            continue;
        }
        const int64_t match_delta = keyframe->id - match_id;
        if (match_delta == 0 || std::abs(match_delta) > local_id_radius) {
            continue;
        }
        // Keyframe ids are temporal order. Compare the same temporal side of
        // both segments; reverse traversal is represented by the inverse local
        // displacement above, not by walking match ids backward.
        if (query_offset * static_cast<int>(match_delta) <= 0) {
            continue;
        }
        const double distance =
            (keyframe->pose_optimized.translation() - expected_pose.translation()).norm();
        if (distance < best_distance) {
            best_distance = distance;
            best = keyframe;
        }
    }
    return best;
}

DirectionStats evaluateDirection(const Config& config,
                                 const KeyframeManager& keyframe_manager,
                                 const VerifiedLoop& loop,
                                 int half_window,
                                 int direction_sign)
{
    DirectionStats stats;
    stats.direction = direction_sign > 0 ? "same" : "reverse";
    const auto query = keyframe_manager.getKeyframe(loop.query_id);
    const auto match = keyframe_manager.getKeyframe(loop.match_id);
    if (!query || !match) {
        return stats;
    }

    const auto keyframes = keyframe_manager.getAllKeyframes();
    const int local_id_radius = std::max(half_window * 6, 12);
    const double translation_threshold = std::max(1.0, config.loop_max_icp_translation);
    const double yaw_threshold = std::max(0.5, config.loop_max_icp_rotation);

    for (int offset = -half_window; offset <= half_window; ++offset) {
        if (offset == 0) {
            continue;
        }
        ++stats.pair_count;
        const auto query_neighbor = keyframe_manager.getKeyframe(loop.query_id + offset);
        if (!query_neighbor) {
            continue;
        }
        const Eigen::Isometry3d T_query_query_neighbor =
            query->pose_optimized.inverse() * query_neighbor->pose_optimized;
        const Eigen::Isometry3d expected_match_neighbor =
            direction_sign > 0
                ? match->pose_optimized * T_query_query_neighbor
                : match->pose_optimized * T_query_query_neighbor.inverse();
        const auto match_neighbor = findLocalCorrespondence(
            keyframes, loop.match_id, offset,
            local_id_radius, expected_match_neighbor);
        if (!match_neighbor) {
            continue;
        }
        ++stats.valid_pair_count;

        const Eigen::Isometry3d residual =
            match_neighbor->pose_optimized.inverse() * expected_match_neighbor;
        const Eigen::Vector3d rpy = rollPitchYaw(residual);
        const double translation_error = residual.translation().norm();
        const double yaw_error = std::abs(normalizeAngle(rpy.z()));
        const double z_error = std::abs(residual.translation().z());
        const double roll_pitch_error = std::hypot(normalizeAngle(rpy.x()), normalizeAngle(rpy.y()));

        stats.translation_errors.push_back(translation_error);
        stats.yaw_errors.push_back(yaw_error);
        stats.z_errors.push_back(z_error);
        stats.roll_pitch_errors.push_back(roll_pitch_error);
        if (translation_error <= translation_threshold && yaw_error <= yaw_threshold) {
            ++stats.inlier_count;
        }
    }
    return stats;
}

DirectionStats evaluateCandidateDirection(const Config& config,
                                          const KeyframeManager& keyframe_manager,
                                          const LoopCandidate& candidate,
                                          int half_window,
                                          int direction_sign)
{
    DirectionStats stats;
    stats.direction = direction_sign > 0 ? "same" : "reverse";
    const auto query = keyframe_manager.getKeyframe(candidate.query_id);
    const auto match = keyframe_manager.getKeyframe(candidate.match_id);
    if (!query || !match) {
        return stats;
    }

    const double translation_threshold = std::max(1.0, config.loop_max_icp_translation);
    const double yaw_threshold = std::max(0.5, config.loop_max_icp_rotation);

    for (int offset = -half_window; offset <= half_window; ++offset) {
        if (offset == 0) {
            continue;
        }
        ++stats.pair_count;
        const auto query_neighbor = keyframe_manager.getKeyframe(candidate.query_id + offset);
        const auto match_neighbor = keyframe_manager.getKeyframe(
            candidate.match_id + direction_sign * offset);
        if (!query_neighbor || !match_neighbor) {
            continue;
        }
        ++stats.valid_pair_count;

        const Eigen::Isometry3d query_step =
            query->pose_optimized.inverse() * query_neighbor->pose_optimized;
        const Eigen::Isometry3d match_step =
            match->pose_optimized.inverse() * match_neighbor->pose_optimized;
        const Eigen::Isometry3d expected_match_step =
            direction_sign > 0 ? query_step : query_step.inverse();
        const Eigen::Isometry3d residual = match_step.inverse() * expected_match_step;
        const Eigen::Vector3d rpy = rollPitchYaw(residual);
        const double translation_error = residual.translation().norm();
        const double yaw_error = std::abs(normalizeAngle(rpy.z()));
        const double z_error = std::abs(residual.translation().z());
        const double roll_pitch_error = std::hypot(normalizeAngle(rpy.x()), normalizeAngle(rpy.y()));

        stats.translation_errors.push_back(translation_error);
        stats.yaw_errors.push_back(yaw_error);
        stats.z_errors.push_back(z_error);
        stats.roll_pitch_errors.push_back(roll_pitch_error);
        if (translation_error <= translation_threshold && yaw_error <= yaw_threshold) {
            ++stats.inlier_count;
        }
    }
    return stats;
}

double directionScore(const DirectionStats& stats)
{
    if (stats.valid_pair_count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    const double translation = median(stats.translation_errors);
    const double yaw = median(stats.yaw_errors);
    const double support_penalty = 1.0 / static_cast<double>(stats.valid_pair_count);
    return (std::isfinite(translation) ? translation : 1.0e6) +
           2.0 * (std::isfinite(yaw) ? yaw : 1.0e6) +
           support_penalty;
}

LoopSegmentConsistencyDiagnostics makeDiagnostics(const DirectionStats& stats)
{
    LoopSegmentConsistencyDiagnostics diagnostics;
    diagnostics.pair_count = stats.pair_count;
    diagnostics.valid_pair_count = stats.valid_pair_count;
    diagnostics.consensus_inlier_count = stats.inlier_count;
    diagnostics.consensus_ratio = stats.valid_pair_count > 0
        ? static_cast<double>(stats.inlier_count) / static_cast<double>(stats.valid_pair_count)
        : 0.0;
    diagnostics.translation_median = median(stats.translation_errors);
    diagnostics.translation_std = sampleStdDev(stats.translation_errors);
    diagnostics.yaw_median = median(stats.yaw_errors);
    diagnostics.yaw_std = sampleStdDev(stats.yaw_errors);
    diagnostics.z_std = sampleStdDev(stats.z_errors);
    diagnostics.roll_pitch_std = sampleStdDev(stats.roll_pitch_errors);
    diagnostics.direction = stats.direction;
    if (stats.valid_pair_count < kMinSegmentPairs) {
        diagnostics.recommendation = "insufficient_support";
    } else if (diagnostics.consensus_ratio >= 0.75) {
        diagnostics.recommendation = "consistent";
    } else {
        diagnostics.recommendation = "inconsistent";
    }
    return diagnostics;
}

}  // namespace

LoopSegmentConsistencyDiagnostics computeLoopSegmentConsistency(
    const Config& config,
    const KeyframeManager& keyframe_manager,
    const VerifiedLoop& loop,
    int half_window)
{
    if (half_window < 1 || loop.query_id < 0 || loop.match_id < 0) {
        return {};
    }
    const DirectionStats same = evaluateDirection(config, keyframe_manager, loop, half_window, 1);
    const DirectionStats reverse = evaluateDirection(config, keyframe_manager, loop, half_window, -1);
    return makeDiagnostics(directionScore(reverse) < directionScore(same) ? reverse : same);
}

LoopSegmentConsistencyDiagnostics computeLoopCandidateSegmentConsistency(
    const Config& config,
    const KeyframeManager& keyframe_manager,
    const LoopCandidate& candidate,
    int half_window)
{
    if (half_window < 1 || candidate.query_id < 0 || candidate.match_id < 0) {
        return {};
    }
    const DirectionStats same = evaluateCandidateDirection(config, keyframe_manager, candidate, half_window, 1);
    const DirectionStats reverse = evaluateCandidateDirection(config, keyframe_manager, candidate, half_window, -1);
    return makeDiagnostics(directionScore(reverse) < directionScore(same) ? reverse : same);
}

void assignLoopSegmentConsistency(
    VerifiedLoop* loop,
    const LoopSegmentConsistencyDiagnostics& diagnostics)
{
    if (!loop) {
        return;
    }
    loop->segment_pair_count = diagnostics.pair_count;
    loop->segment_valid_pair_count = diagnostics.valid_pair_count;
    loop->segment_consensus_inlier_count = diagnostics.consensus_inlier_count;
    loop->segment_consensus_ratio = diagnostics.consensus_ratio;
    loop->segment_translation_median = diagnostics.translation_median;
    loop->segment_translation_std = diagnostics.translation_std;
    loop->segment_yaw_median = diagnostics.yaw_median;
    loop->segment_yaw_std = diagnostics.yaw_std;
    loop->segment_z_std = diagnostics.z_std;
    loop->segment_roll_pitch_std = diagnostics.roll_pitch_std;
    loop->segment_direction = diagnostics.direction;
    loop->segment_recommendation = diagnostics.recommendation;
}

}  // namespace n3mapping
