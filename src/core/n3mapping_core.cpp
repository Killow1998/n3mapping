#include "n3mapping/core/n3mapping_core.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_heightmap_diagnostics.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/loop_referee.h"
#include "n3mapping/loop_segment_consistency.h"
#include "n3mapping/loop_verifier.h"
#include <pcl/common/transforms.h>
#include "n3mapping/pcl_compat.h"

namespace n3mapping {

namespace {

double rotationAngle(const Eigen::Isometry3d& transform)
{
    return std::abs(Eigen::AngleAxisd(transform.rotation()).angle());
}

bool graphTrialYawInconsistent(const LoopGraphTrialDiagnostics& diagnostics)
{
    constexpr double kNearHalfTurnYawRad = 2.8;
    return diagnostics.success &&
           std::isfinite(diagnostics.residual_yaw_after) &&
           std::abs(diagnostics.residual_yaw_after) >= kNearHalfTurnYawRad;
}

bool graphTrialTranslationInconsistent(const LoopGraphTrialDiagnostics& diagnostics)
{
    constexpr double kLargeGraphResidualTranslationM = 2.0;
    return diagnostics.success &&
           std::isfinite(diagnostics.residual_translation_norm_after) &&
           diagnostics.residual_translation_norm_after >= kLargeGraphResidualTranslationM;
}

std::string consensusRefereeRejectReason(const VerifiedLoop& loop)
{
    if (loop.consensus_estimator_recommendation == "insufficient_estimator_support" &&
        loop.vertical_hypothesis_edge_recommendation == "planar_xy_yaw") {
        return "consensus_insufficient_planar";
    }

    constexpr double kLargeConsensusMeasurementDeltaM = 5.0;
    if (loop.consensus_estimator_recommendation == "unstable_consensus_measurement" &&
        std::isfinite(loop.consensus_estimator_measurement_delta_translation) &&
        loop.consensus_estimator_measurement_delta_translation >= kLargeConsensusMeasurementDeltaM) {
        return "consensus_unstable_large_delta";
    }

    return {};
}

Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d& transform)
{
    return transform.rotation().eulerAngles(0, 1, 2);
}

struct LoopResidualAxisStats {
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
};

struct LoopRefereeDebugDecision {
    std::string recommendation = "not_available";
    std::string reason = "not_available";
    std::string risk_flags = "not_available";
};

double unitScoreBelow(double value, double limit)
{
    if (!std::isfinite(value) || limit <= 0.0) {
        return 0.0;
    }
    return std::clamp(1.0 - value / limit, 0.0, 1.0);
}

LoopFeatures makeSegmentAwareLoopFeatures(const Config& config,
                                          const LoopCandidate& candidate,
                                          const VerifiedLoop& loop,
                                          double icp_translation_norm)
{
    LoopFeatures features;
    features.descriptor_score = candidate.descriptor_score;
    features.spatial_score = candidate.spatial_score;
    features.geometric_overlap = loop.heightmap_vertical_consistency_score;
    features.temporal_gap = std::clamp(
        static_cast<double>(candidate.query_id - candidate.match_id) /
        std::max(1.0, static_cast<double>(config.sc_num_exclude_recent)), 0.0, 1.0);
    const double fitness_score = unitScoreBelow(loop.fitness_score, config.loop_fitness_threshold);
    const double inlier_score = config.loop_min_inlier_ratio > 0.0
        ? std::clamp(loop.inlier_ratio / config.loop_min_inlier_ratio, 0.0, 1.0)
        : std::clamp(loop.inlier_ratio, 0.0, 1.0);
    const double motion_score = unitScoreBelow(icp_translation_norm, config.loop_max_icp_translation);
    features.local_map_consistency = (fitness_score + inlier_score + motion_score) / 3.0;
    features.segment_consistency = loop.segment_consensus_ratio;
    features.segment_support = loop.segment_pair_count > 0
        ? static_cast<double>(loop.segment_valid_pair_count) / static_cast<double>(loop.segment_pair_count)
        : 0.0;
    const bool from_descriptor = candidate.fromRHPD() || candidate.fromSC();
    features.descriptor_supported = from_descriptor;
    features.spatial_only = candidate.fromSpatial() && !from_descriptor;
    features.predicted_translation_norm = loop.T_pred_match_query.translation().norm();
    features.icp_correction_yaw_abs = std::abs(rollPitchYaw(loop.T_icp_correction_match).z());
    features.segment_translation_median = loop.segment_translation_median;
    return features;
}

bool hasStrongLoopDescriptor(const Config& config, const LoopCandidate& candidate)
{
    return (candidate.fromSC() &&
            std::isfinite(candidate.sc_distance) &&
            candidate.sc_distance <= config.sc_dist_threshold) ||
           (candidate.fromRHPD() &&
            std::isfinite(candidate.rhpd_distance) &&
            candidate.rhpd_distance <= config.rhpd_dist_threshold);
}

bool hasPreIcpSegmentSupport(const LoopSegmentConsistencyDiagnostics& segment)
{
    return segment.valid_pair_count >= 2 && segment.consensus_ratio >= 0.75;
}

using CandidatePair = std::pair<int64_t, int64_t>;

struct PreIcpCandidate {
    LoopCandidate candidate;
    LoopSegmentConsistencyDiagnostics segment;
    double predicted_range_m = std::numeric_limits<double>::infinity();
    bool strong_descriptor = false;
};

struct SegmentClusterKey {
    int direction = 0;
    int64_t bucket = 0;

    bool operator<(const SegmentClusterKey& other) const
    {
        return std::tie(direction, bucket) < std::tie(other.direction, other.bucket);
    }
};

struct CandidateSelectionResult {
    std::vector<LoopCandidate> selected;
    std::map<CandidatePair, std::string> rejected;
};

CandidatePair candidatePair(const LoopCandidate& candidate)
{
    return {candidate.query_id, candidate.match_id};
}

int64_t floorDiv(int64_t value, int64_t divisor)
{
    if (divisor <= 1) {
        return value;
    }
    if (value >= 0) {
        return value / divisor;
    }
    return -(((-value) + divisor - 1) / divisor);
}

SegmentClusterKey segmentClusterKey(const Config& config,
                                    const LoopCandidate& candidate,
                                    const LoopSegmentConsistencyDiagnostics& segment)
{
    const bool reverse = segment.direction == "reverse";
    const int64_t axis = reverse
        ? candidate.query_id + candidate.match_id
        : candidate.query_id - candidate.match_id;
    const int64_t bucket_width = std::max<int64_t>(2, config.loop_kf_gap);
    return {reverse ? -1 : 1, floorDiv(axis, bucket_width)};
}

int sourceRank(const LoopCandidate& candidate)
{
    const bool descriptor = candidate.fromSC() || candidate.fromRHPD();
    const bool spatial = candidate.fromSpatial();
    if (descriptor && spatial) {
        return 3;
    }
    if (descriptor) {
        return 2;
    }
    if (spatial) {
        return 1;
    }
    return 0;
}

bool preferPreIcpRepresentative(const PreIcpCandidate& candidate,
                                const PreIcpCandidate& current)
{
    const int candidate_source = sourceRank(candidate.candidate);
    const int current_source = sourceRank(current.candidate);
    if (candidate_source != current_source) {
        return candidate_source > current_source;
    }
    if (candidate.strong_descriptor != current.strong_descriptor) {
        return candidate.strong_descriptor;
    }
    if (candidate.segment.consensus_ratio != current.segment.consensus_ratio) {
        return candidate.segment.consensus_ratio > current.segment.consensus_ratio;
    }
    if (std::abs(candidate.predicted_range_m - current.predicted_range_m) > 1.0e-6) {
        return candidate.predicted_range_m < current.predicted_range_m;
    }
    if (candidate.candidate.fused_score != current.candidate.fused_score) {
        return candidate.candidate.fused_score < current.candidate.fused_score;
    }
    return candidate.candidate.match_id < current.candidate.match_id;
}

CandidateSelectionResult selectPreIcpSegmentRepresentatives(
    const Config& config,
    const KeyframeManager& keyframe_manager,
    const std::vector<LoopCandidate>& raw_candidates)
{
    CandidateSelectionResult result;
    std::vector<PreIcpCandidate> eligible;
    eligible.reserve(raw_candidates.size());

    for (const auto& candidate : raw_candidates) {
        const CandidatePair key = candidatePair(candidate);
        const auto query = keyframe_manager.getKeyframe(candidate.query_id);
        const auto match = keyframe_manager.getKeyframe(candidate.match_id);
        if (!query || !match) {
            result.rejected[key] = "missing_keyframe_or_cloud";
            continue;
        }
        const double predicted_range =
            (match->pose_optimized.inverse() * query->pose_optimized).translation().norm();
        if (predicted_range > config.loop_max_range) {
            result.rejected[key] = "prediction_range_gate";
            continue;
        }

        PreIcpCandidate proposal;
        proposal.candidate = candidate;
        proposal.predicted_range_m = predicted_range;
        proposal.strong_descriptor = hasStrongLoopDescriptor(config, candidate);
        proposal.segment = computeLoopCandidateSegmentConsistency(
            config, keyframe_manager, candidate);
        if (!hasPreIcpSegmentSupport(proposal.segment) && !proposal.strong_descriptor) {
            result.rejected[key] = "pre_icp_segment_unconfirmed";
            continue;
        }
        eligible.push_back(std::move(proposal));
    }

    std::map<SegmentClusterKey, std::vector<std::size_t>> clusters;
    for (std::size_t i = 0; i < eligible.size(); ++i) {
        clusters[segmentClusterKey(config, eligible[i].candidate, eligible[i].segment)].push_back(i);
    }

    std::vector<bool> selected(eligible.size(), false);
    for (const auto& [cluster_key, indices] : clusters) {
        (void)cluster_key;
        if (indices.empty()) {
            continue;
        }
        std::set<int64_t> query_ids;
        for (const std::size_t index : indices) {
            query_ids.insert(eligible[index].candidate.query_id);
        }

        std::size_t best = indices.front();
        for (const std::size_t index : indices) {
            if (preferPreIcpRepresentative(eligible[index], eligible[best])) {
                best = index;
            }
        }

        const bool cluster_supported = query_ids.size() >= 2;
        if (!cluster_supported && !eligible[best].strong_descriptor) {
            for (const std::size_t index : indices) {
                result.rejected[candidatePair(eligible[index].candidate)] = "pre_icp_isolated_weak_cluster";
            }
            continue;
        }

        selected[best] = true;
        for (const std::size_t index : indices) {
            if (index != best) {
                result.rejected[candidatePair(eligible[index].candidate)] = "pre_icp_cluster_non_representative";
            }
        }
    }

    for (std::size_t i = 0; i < eligible.size(); ++i) {
        if (selected[i]) {
            result.selected.push_back(eligible[i].candidate);
        }
    }
    return result;
}

std::pair<double, double> meanLoopResidual(
    const std::vector<EdgeInfo>& edges,
    const std::map<int64_t, Eigen::Isometry3d>& poses)
{
    if (edges.empty()) {
        return {0.0, 0.0};
    }

    double translation_sum = 0.0;
    double rotation_sum = 0.0;
    std::size_t count = 0;
    for (const auto& edge : edges) {
        auto from_it = poses.find(edge.from_id);
        auto to_it = poses.find(edge.to_id);
        if (from_it == poses.end() || to_it == poses.end()) {
            continue;
        }
        const Eigen::Isometry3d predicted = from_it->second.inverse() * to_it->second;
        const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
        translation_sum += residual.translation().norm();
        rotation_sum += rotationAngle(residual);
        ++count;
    }

    if (count == 0) {
        return {0.0, 0.0};
    }
    return {translation_sum / static_cast<double>(count), rotation_sum / static_cast<double>(count)};
}

LoopResidualAxisStats meanLoopResidualAxes(
    const std::vector<EdgeInfo>& edges,
    const std::map<int64_t, Eigen::Isometry3d>& poses)
{
    LoopResidualAxisStats stats;
    std::size_t count = 0;
    for (const auto& edge : edges) {
        auto from_it = poses.find(edge.from_id);
        auto to_it = poses.find(edge.to_id);
        if (from_it == poses.end() || to_it == poses.end()) {
            continue;
        }
        const Eigen::Isometry3d predicted = from_it->second.inverse() * to_it->second;
        const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
        stats.translation += residual.translation().cwiseAbs();
        stats.rpy += rollPitchYaw(residual).cwiseAbs();
        ++count;
    }
    if (count == 0) {
        return stats;
    }
    const double inv_count = 1.0 / static_cast<double>(count);
    stats.translation *= inv_count;
    stats.rpy *= inv_count;
    return stats;
}

void accumulatePoseUpdateStats(const std::map<int64_t, Eigen::Isometry3d>& before,
                               const std::map<int64_t, Eigen::Isometry3d>& after,
                               CoreLoopClosureResult* result)
{
    double translation_sum = 0.0;
    double rotation_sum = 0.0;
    std::size_t count = 0;

    for (const auto& [id, before_pose] : before) {
        auto after_it = after.find(id);
        if (after_it == after.end()) {
            continue;
        }
        const Eigen::Isometry3d delta = before_pose.inverse() * after_it->second;
        const double translation = delta.translation().norm();
        const double rotation = rotationAngle(delta);
        translation_sum += translation;
        rotation_sum += rotation;
        result->max_pose_update_translation = std::max(result->max_pose_update_translation, translation);
        result->max_pose_update_rotation = std::max(result->max_pose_update_rotation, rotation);
        ++count;
    }

    if (count == 0) {
        return;
    }
    const std::size_t previous_count = result->pose_update_count;
    const std::size_t total_count = previous_count + count;
    result->mean_pose_update_translation =
        (result->mean_pose_update_translation * static_cast<double>(previous_count) + translation_sum) /
        static_cast<double>(total_count);
    result->mean_pose_update_rotation =
        (result->mean_pose_update_rotation * static_cast<double>(previous_count) + rotation_sum) /
        static_cast<double>(total_count);
    result->pose_update_count = total_count;
}

void assignGraphTrialDiagnostics(VerifiedLoop* loop, const LoopGraphTrialDiagnostics& diagnostics)
{
    if (!loop) {
        return;
    }
    loop->graph_trial_success = diagnostics.success;
    loop->graph_trial_residual_x_after = diagnostics.residual_x_after;
    loop->graph_trial_residual_y_after = diagnostics.residual_y_after;
    loop->graph_trial_residual_z_after = diagnostics.residual_z_after;
    loop->graph_trial_residual_roll_after = diagnostics.residual_roll_after;
    loop->graph_trial_residual_pitch_after = diagnostics.residual_pitch_after;
    loop->graph_trial_residual_yaw_after = diagnostics.residual_yaw_after;
    loop->graph_trial_residual_translation_norm_after = diagnostics.residual_translation_norm_after;
    loop->graph_trial_residual_rotation_norm_after = diagnostics.residual_rotation_norm_after;
    loop->graph_trial_mean_pose_update_translation = diagnostics.mean_pose_update_translation;
    loop->graph_trial_max_pose_update_translation = diagnostics.max_pose_update_translation;
    loop->graph_trial_mean_pose_update_rotation = diagnostics.mean_pose_update_rotation;
    loop->graph_trial_max_pose_update_rotation = diagnostics.max_pose_update_rotation;
    loop->graph_trial_existing_loop_residual_delta = diagnostics.existing_loop_residual_delta;
    loop->graph_trial_odom_residual_delta = diagnostics.odom_residual_delta;
    loop->graph_trial_consistency_score = diagnostics.consistency_score;
    loop->graph_trial_recommendation = diagnostics.recommendation;
}

void assignGraphTrialDiagnostics(LoopDebugCandidateEvent* event, const LoopGraphTrialDiagnostics& diagnostics)
{
    if (!event) {
        return;
    }
    event->graph_trial_success = diagnostics.success;
    event->graph_trial_residual_x_after = diagnostics.residual_x_after;
    event->graph_trial_residual_y_after = diagnostics.residual_y_after;
    event->graph_trial_residual_z_after = diagnostics.residual_z_after;
    event->graph_trial_residual_roll_after = diagnostics.residual_roll_after;
    event->graph_trial_residual_pitch_after = diagnostics.residual_pitch_after;
    event->graph_trial_residual_yaw_after = diagnostics.residual_yaw_after;
    event->graph_trial_residual_translation_norm_after = diagnostics.residual_translation_norm_after;
    event->graph_trial_residual_rotation_norm_after = diagnostics.residual_rotation_norm_after;
    event->graph_trial_mean_pose_update_translation = diagnostics.mean_pose_update_translation;
    event->graph_trial_max_pose_update_translation = diagnostics.max_pose_update_translation;
    event->graph_trial_mean_pose_update_rotation = diagnostics.mean_pose_update_rotation;
    event->graph_trial_max_pose_update_rotation = diagnostics.max_pose_update_rotation;
    event->graph_trial_existing_loop_residual_delta = diagnostics.existing_loop_residual_delta;
    event->graph_trial_odom_residual_delta = diagnostics.odom_residual_delta;
    event->graph_trial_consistency_score = diagnostics.consistency_score;
    event->graph_trial_recommendation = diagnostics.recommendation;
}

void assignConsensusEstimatorTrialDiagnostics(LoopDebugCandidateEvent* event,
                                              const LoopGraphTrialDiagnostics& diagnostics)
{
    if (!event) {
        return;
    }
    event->consensus_estimator_trial_success = diagnostics.success;
    event->consensus_estimator_trial_residual_x_after = diagnostics.residual_x_after;
    event->consensus_estimator_trial_residual_y_after = diagnostics.residual_y_after;
    event->consensus_estimator_trial_residual_z_after = diagnostics.residual_z_after;
    event->consensus_estimator_trial_residual_roll_after = diagnostics.residual_roll_after;
    event->consensus_estimator_trial_residual_pitch_after = diagnostics.residual_pitch_after;
    event->consensus_estimator_trial_residual_yaw_after = diagnostics.residual_yaw_after;
    event->consensus_estimator_trial_residual_translation_norm_after =
        diagnostics.residual_translation_norm_after;
    event->consensus_estimator_trial_residual_rotation_norm_after =
        diagnostics.residual_rotation_norm_after;
    event->consensus_estimator_trial_consistency_score = diagnostics.consistency_score;
    event->consensus_estimator_trial_recommendation = diagnostics.recommendation;
}

void assignSegmentDiagnostics(LoopDebugCandidateEvent* event, const VerifiedLoop& loop)
{
    if (!event) {
        return;
    }
    event->segment_pair_count = loop.segment_pair_count;
    event->segment_valid_pair_count = loop.segment_valid_pair_count;
    event->segment_consensus_inlier_count = loop.segment_consensus_inlier_count;
    event->segment_consensus_ratio = loop.segment_consensus_ratio;
    event->segment_translation_median = loop.segment_translation_median;
    event->segment_translation_std = loop.segment_translation_std;
    event->segment_yaw_median = loop.segment_yaw_median;
    event->segment_yaw_std = loop.segment_yaw_std;
    event->segment_z_std = loop.segment_z_std;
    event->segment_roll_pitch_std = loop.segment_roll_pitch_std;
    event->segment_direction = loop.segment_direction;
    event->segment_recommendation = loop.segment_recommendation;
}

void assignConsensusDiagnostics(LoopDebugCandidateEvent* event, const VerifiedLoop& loop)
{
    if (!event) {
        return;
    }
    event->consensus_shadow_decision = loop.consensus_shadow_decision;
    event->consensus_shadow_reason = loop.consensus_shadow_reason;
    event->consensus_valid_pair_count = loop.consensus_valid_pair_count;
    event->consensus_left_support_count = loop.consensus_left_support_count;
    event->consensus_right_support_count = loop.consensus_right_support_count;
    event->consensus_contradiction_count = loop.consensus_contradiction_count;
    event->consensus_median_translation_delta = loop.consensus_median_translation_delta;
    event->consensus_mad_translation_delta = loop.consensus_mad_translation_delta;
    event->consensus_median_rotation_delta = loop.consensus_median_rotation_delta;
    event->consensus_mad_rotation_delta = loop.consensus_mad_rotation_delta;
    event->consensus_estimator_valid = loop.consensus_estimator_valid;
    event->consensus_estimator_pair_count = loop.consensus_estimator_pair_count;
    event->consensus_estimator_inlier_count = loop.consensus_estimator_inlier_count;
    event->consensus_estimator_inlier_ratio = loop.consensus_estimator_inlier_ratio;
    event->consensus_estimator_translation_median = loop.consensus_estimator_translation_median;
    event->consensus_estimator_z_median = loop.consensus_estimator_z_median;
    event->consensus_estimator_yaw_median = loop.consensus_estimator_yaw_median;
    event->consensus_estimator_translation_mad = loop.consensus_estimator_translation_mad;
    event->consensus_estimator_z_mad = loop.consensus_estimator_z_mad;
    event->consensus_estimator_yaw_mad = loop.consensus_estimator_yaw_mad;
    event->consensus_estimator_measurement_delta_translation =
        loop.consensus_estimator_measurement_delta_translation;
    event->consensus_estimator_measurement_delta_rotation =
        loop.consensus_estimator_measurement_delta_rotation;
    event->consensus_estimator_recommendation = loop.consensus_estimator_recommendation;
}

void assignConsensusDiagnostics(LoopDebugCandidateEvent* event, const LoopConsensusResult& consensus)
{
    if (!event) {
        return;
    }
    event->consensus_shadow_decision = loopConsensusDecisionName(consensus.decision);
    event->consensus_shadow_reason = consensus.reason;
    event->consensus_valid_pair_count = consensus.valid_pair_count;
    event->consensus_left_support_count = consensus.left_support_count;
    event->consensus_right_support_count = consensus.right_support_count;
    event->consensus_contradiction_count = consensus.contradiction_count;
    event->consensus_median_translation_delta = consensus.median_translation_delta;
    event->consensus_mad_translation_delta = consensus.mad_translation_delta;
    event->consensus_median_rotation_delta = consensus.median_rotation_delta;
    event->consensus_mad_rotation_delta = consensus.mad_rotation_delta;
    event->consensus_estimator_valid = consensus.estimator_valid;
    event->consensus_estimator_pair_count = consensus.estimator_pair_count;
    event->consensus_estimator_inlier_count = consensus.estimator_inlier_count;
    event->consensus_estimator_inlier_ratio = consensus.estimator_inlier_ratio;
    event->consensus_estimator_translation_median = consensus.estimator_translation_median;
    event->consensus_estimator_z_median = consensus.estimator_z_median;
    event->consensus_estimator_yaw_median = consensus.estimator_yaw_median;
    event->consensus_estimator_translation_mad = consensus.estimator_translation_mad;
    event->consensus_estimator_z_mad = consensus.estimator_z_mad;
    event->consensus_estimator_yaw_mad = consensus.estimator_yaw_mad;
    event->consensus_estimator_measurement_delta_translation =
        consensus.estimator_measurement_delta_translation;
    event->consensus_estimator_measurement_delta_rotation =
        consensus.estimator_measurement_delta_rotation;
    event->consensus_estimator_recommendation = consensus.estimator_recommendation;
}

Config validateOrThrow(const Config& config)
{
    std::string error;
    if (!config.validate(&error)) {
        throw std::invalid_argument("Invalid N3MappingCore config: " + error);
    }
    return config;
}

double processingTimeSeconds()
{
    using Clock = std::chrono::system_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct ZDistributionStats {
    std::size_t count = 0;
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    double sum_z = 0.0;
    std::vector<double> samples;

    double span() const
    {
        return count > 0 ? max_z - min_z : std::numeric_limits<double>::quiet_NaN();
    }

    double mean() const
    {
        return count > 0 ? sum_z / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
    }

    double quantile(double q) const
    {
        if (samples.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::vector<double> values = samples;
        std::sort(values.begin(), values.end());
        const double index = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(index));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(index));
        if (lo == hi) {
            return values[lo];
        }
        const double t = index - static_cast<double>(lo);
        return values[lo] * (1.0 - t) + values[hi] * t;
    }

    double robustMin() const { return quantile(0.05); }
    double robustMax() const { return quantile(0.95); }

    double robustSpan() const
    {
        const double lo = robustMin();
        const double hi = robustMax();
        return std::isfinite(lo) && std::isfinite(hi) ? hi - lo : std::numeric_limits<double>::quiet_NaN();
    }
};

ZDistributionStats computeZStats(const core::LioFrame::PointCloud::Ptr& cloud)
{
    ZDistributionStats stats;
    if (!cloud) {
        return stats;
    }
    for (const auto& point : cloud->points) {
        if (!isFinitePoint(point)) {
            continue;
        }
        const double z = static_cast<double>(point.z);
        stats.min_z = std::min(stats.min_z, z);
        stats.max_z = std::max(stats.max_z, z);
        stats.sum_z += z;
        stats.samples.push_back(z);
        ++stats.count;
    }
    return stats;
}

ZDistributionStats computeTransformedZStats(const core::LioFrame::PointCloud::Ptr& cloud,
                                            const Eigen::Isometry3d& transform)
{
    ZDistributionStats stats;
    if (!cloud) {
        return stats;
    }
    for (const auto& point : cloud->points) {
        if (!isFinitePoint(point)) {
            continue;
        }
        const Eigen::Vector3d transformed =
            transform * Eigen::Vector3d(point.x, point.y, point.z);
        if (!std::isfinite(transformed.z())) {
            continue;
        }
        stats.min_z = std::min(stats.min_z, transformed.z());
        stats.max_z = std::max(stats.max_z, transformed.z());
        stats.sum_z += transformed.z();
        stats.samples.push_back(transformed.z());
        ++stats.count;
    }
    return stats;
}

double zOverlapRatio(const ZDistributionStats& a, const ZDistributionStats& b)
{
    if (a.count == 0 || b.count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double denominator = std::min(a.span(), b.span());
    if (!std::isfinite(denominator) || denominator <= 1e-9) {
        return 0.0;
    }
    const double overlap = std::min(a.max_z, b.max_z) - std::max(a.min_z, b.min_z);
    return std::max(0.0, std::min(1.0, overlap / denominator));
}

double robustZOverlapRatio(const ZDistributionStats& a, const ZDistributionStats& b)
{
    if (a.count == 0 || b.count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double a_min = a.robustMin();
    const double a_max = a.robustMax();
    const double b_min = b.robustMin();
    const double b_max = b.robustMax();
    if (!std::isfinite(a_min) || !std::isfinite(a_max) ||
        !std::isfinite(b_min) || !std::isfinite(b_max)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double denominator = std::min(a_max - a_min, b_max - b_min);
    if (denominator <= 1e-9) {
        return 0.0;
    }
    const double overlap = std::min(a_max, b_max) - std::max(a_min, b_min);
    return std::max(0.0, std::min(1.0, overlap / denominator));
}

double zCentroidDelta(const ZDistributionStats& source, const ZDistributionStats& target)
{
    const double source_mean = source.mean();
    const double target_mean = target.mean();
    if (!std::isfinite(source_mean) || !std::isfinite(target_mean)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return source_mean - target_mean;
}

struct VerticalHypothesisDiagnostics {
    int count = 0;
    double best_z_offset_m = std::numeric_limits<double>::quiet_NaN();
    double best_z_offset_fitness = std::numeric_limits<double>::quiet_NaN();
    double zero_z_fitness = std::numeric_limits<double>::quiet_NaN();
    double fitness_gap_zero_vs_best = std::numeric_limits<double>::quiet_NaN();
    double z_hypothesis_spread_m = std::numeric_limits<double>::quiet_NaN();
    double vertical_ambiguity_score = std::numeric_limits<double>::quiet_NaN();
    std::string edge_recommendation = "not_available";
};

bool hasUsableHypothesisFitness(const MatchResult& result)
{
    return result.converged && std::isfinite(result.fitness_score);
}

VerticalHypothesisDiagnostics computeVerticalHypothesisDiagnostics(
    PointCloudMatcher& matcher,
    const core::LioFrame::PointCloud::Ptr& target,
    const core::LioFrame::PointCloud::Ptr& source,
    const MatchResult& zero_result)
{
    VerticalHypothesisDiagnostics diagnostics;
    constexpr std::array<double, 7> kZOffsets = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
    struct Candidate {
        double z_offset = 0.0;
        double fitness = std::numeric_limits<double>::quiet_NaN();
    };
    std::vector<Candidate> usable;
    usable.reserve(kZOffsets.size());

    auto add_result = [&](double z_offset, const MatchResult& result) {
        if (!hasUsableHypothesisFitness(result)) {
            return;
        }
        ++diagnostics.count;
        usable.push_back({z_offset, result.fitness_score});
        if (!std::isfinite(diagnostics.best_z_offset_fitness) ||
            result.fitness_score < diagnostics.best_z_offset_fitness) {
            diagnostics.best_z_offset_m = z_offset;
            diagnostics.best_z_offset_fitness = result.fitness_score;
        }
    };

    diagnostics.zero_z_fitness =
        std::isfinite(zero_result.fitness_score) ? zero_result.fitness_score
                                                 : std::numeric_limits<double>::quiet_NaN();
    add_result(0.0, zero_result);
    for (double z_offset : kZOffsets) {
        if (z_offset == 0.0) {
            continue;
        }
        Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
        init.translation().z() = z_offset;
        try {
            add_result(z_offset, matcher.alignCloud(target, source, init));
        } catch (const std::exception&) {
            // Diagnostics must never affect loop-closure behavior.
        }
    }

    if (diagnostics.count == 0 || !std::isfinite(diagnostics.best_z_offset_fitness)) {
        diagnostics.edge_recommendation = "reject";
        return diagnostics;
    }
    if (std::isfinite(diagnostics.zero_z_fitness)) {
        diagnostics.fitness_gap_zero_vs_best =
            diagnostics.zero_z_fitness - diagnostics.best_z_offset_fitness;
    }

    const double near_best_band = std::max(0.02, 0.10 * std::max(1e-6, diagnostics.best_z_offset_fitness));
    double near_min = std::numeric_limits<double>::infinity();
    double near_max = -std::numeric_limits<double>::infinity();
    for (const auto& candidate : usable) {
        if (candidate.fitness <= diagnostics.best_z_offset_fitness + near_best_band) {
            near_min = std::min(near_min, candidate.z_offset);
            near_max = std::max(near_max, candidate.z_offset);
        }
    }
    if (std::isfinite(near_min) && std::isfinite(near_max)) {
        diagnostics.z_hypothesis_spread_m = near_max - near_min;
        diagnostics.vertical_ambiguity_score =
            std::clamp(diagnostics.z_hypothesis_spread_m / 4.0, 0.0, 1.0);
    } else {
        diagnostics.z_hypothesis_spread_m = 0.0;
        diagnostics.vertical_ambiguity_score = 0.0;
    }

    const double zero_best_gap =
        std::isfinite(diagnostics.fitness_gap_zero_vs_best) ? diagnostics.fitness_gap_zero_vs_best : 0.0;
    if (diagnostics.vertical_ambiguity_score >= 0.25 &&
        zero_best_gap <= std::max(0.02, 0.10 * std::max(1e-6, diagnostics.zero_z_fitness))) {
        diagnostics.edge_recommendation = "planar_xy_yaw";
    } else {
        diagnostics.edge_recommendation = "full6dof";
    }
    return diagnostics;
}

core::LioFrame::PointCloud::Ptr strideLimitCloud(const core::LioFrame::PointCloud::Ptr& cloud,
                                                 std::size_t max_points)
{
    if (!cloud || cloud->size() <= max_points || max_points == 0) {
        return cloud;
    }

    auto limited = pcl::make_shared<core::LioFrame::PointCloud>();
    limited->header = cloud->header;
    limited->reserve(max_points);
    const std::size_t stride =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(static_cast<double>(cloud->size()) /
                                                                    static_cast<double>(max_points))));
    for (std::size_t i = 0; i < cloud->size() && limited->size() < max_points; i += stride) {
        const auto& point = cloud->points[i];
        if (isFinitePoint(point)) {
            limited->push_back(point);
        }
    }
    limited->width = static_cast<std::uint32_t>(limited->size());
    limited->height = 1;
    limited->is_dense = true;
    return limited;
}

core::LioFrame::PointCloud::Ptr prepareLoopIcpCloud(const core::LioFrame::PointCloud::Ptr& cloud,
                                                    const Config& config)
{
    if (!cloud || cloud->empty() || config.loop_icp_max_points <= 0 ||
        cloud->size() <= static_cast<std::size_t>(config.loop_icp_max_points)) {
        return cloud;
    }

    core::LioFrame::PointCloud::Ptr prepared = cloud;
    const double voxel_size = std::max(config.loop_icp_prefilter_voxel_size, config.gicp_downsampling_resolution);
    if (voxel_size > 0.0) {
        core::LioFrame::PointCloud::Ptr filtered;
        if (safeVoxelGridFilter<pcl::PointXYZI>(cloud, voxel_size, &filtered) &&
            filtered && !filtered->empty()) {
            prepared = filtered;
        }
    }

    return strideLimitCloud(prepared, static_cast<std::size_t>(config.loop_icp_max_points));
}

}  // namespace

N3MappingCore::N3MappingCore(const Config& config)
  : config_(validateOrThrow(config))
  , session_(std::make_unique<core::N3MappingSession>(config_))
{
}

N3MappingCore::~N3MappingCore() = default;

void N3MappingCore::appendLoopDebugCandidate(const LoopDebugCandidateEvent& event) const
{
    if (!config_.loop_debug_enable) {
        return;
    }
    std::lock_guard<std::mutex> lock(loop_debug_mutex_);
    LoopDebugLogger::appendCandidate(LoopDebugLogger::resolvePath(config_), event);
}

void N3MappingCore::appendLoopDebugOptimization(const LoopDebugOptimizationEvent& event) const
{
    if (!config_.loop_debug_enable) {
        return;
    }
    std::lock_guard<std::mutex> lock(loop_debug_mutex_);
    LoopDebugLogger::appendOptimizationSummary(LoopDebugLogger::resolvePath(config_), event);
}

CoreRunMode parseCoreRunMode(const std::string& mode)
{
    if (mode == "mapping") {
        return CoreRunMode::MAPPING;
    }
    if (mode == "localization") {
        return CoreRunMode::LOCALIZATION;
    }
    if (mode == "map_extension") {
        return CoreRunMode::MAP_EXTENSION;
    }
    throw std::invalid_argument("Invalid n3mapping mode: " + mode);
}

const char* coreRunModeName(CoreRunMode mode)
{
    switch (mode) {
        case CoreRunMode::LOCALIZATION:
            return "localization";
        case CoreRunMode::MAP_EXTENSION:
            return "map_extension";
        case CoreRunMode::MAPPING:
        default:
            return "mapping";
    }
}

bool coreRunModeLoadsMap(CoreRunMode mode)
{
    return mode == CoreRunMode::LOCALIZATION || mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeSavesMap(CoreRunMode mode)
{
    return mode == CoreRunMode::MAPPING || mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeProcessesLoopClosures(CoreRunMode mode)
{
    return mode == CoreRunMode::MAPPING;
}

core::BackendOutput N3MappingCore::processFrame(CoreRunMode mode, const core::LioFrame& frame)
{
    switch (mode) {
        case CoreRunMode::LOCALIZATION:
            return processLocalizationFrame(frame);
        case CoreRunMode::MAP_EXTENSION:
            return processMapExtensionFrame(frame);
        case CoreRunMode::MAPPING:
        default:
            return processMappingFrame(frame);
    }
}

core::BackendOutput N3MappingCore::processMappingFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    auto output = makeOutput(true, frame.T_world_lidar, frame.undistorted_cloud);
    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    auto& keyframes = session_->keyframeManager();
    if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
        if (!external_dense_trajectory_recording_enabled_) {
            appendDenseTrajectorySampleWithLatestAnchor(timestamp, frame.T_world_lidar);
        }
        return output;
    }

    const int64_t keyframe_id = keyframes.addKeyframe(timestamp, frame.T_world_lidar, frame.undistorted_cloud);
    session_->loopDetector().addDescriptor(keyframe_id, frame.undistorted_cloud);
    addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

    if (keyframe_id == 0) {
        session_->graphOptimizer().addPriorFactor(keyframe_id, frame.T_world_lidar);
    } else {
        addOdometryConstraint(keyframe_id, frame.T_world_lidar);
    }

    session_->graphOptimizer().incrementalOptimize();
    refreshOptimizedPoses();

    Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
    if (session_->graphOptimizer().hasNode(keyframe_id)) {
        try {
            optimized_pose = session_->graphOptimizer().getOptimizedPose(keyframe_id);
        } catch (const std::exception&) {
            optimized_pose = frame.T_world_lidar;
        }
    }

    output.accepted_keyframe = true;
    output.keyframe_id = keyframe_id;
    output.T_world_lidar = optimized_pose;
    output.cloud_world = makeWorldCloud(frame.undistorted_cloud, optimized_pose);
    if (!external_dense_trajectory_recording_enabled_) {
        if (auto kf = keyframes.getKeyframe(keyframe_id)) {
            appendDenseTrajectorySample(timestamp, frame.T_world_lidar, keyframe_id, kf->pose_odom);
        }
    }

    {
        std::lock_guard<std::mutex> lock(loop_queue_mutex_);
        loop_detection_queue_.push_back(keyframe_id);
    }
    return output;
}

core::BackendOutput N3MappingCore::processLocalizationFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    if (!map_loaded_) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    Eigen::Isometry3d pose_map = frame.T_world_lidar;
    bool success = false;
    bool relocalization_locked = false;
    int64_t matched_keyframe_id = -1;
    auto& localizer = session_->worldLocalizing();

    if (localizer.isRelocalized()) {
        auto result = localizer.trackLocalization(frame.undistorted_cloud, frame.T_world_lidar);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
            matched_keyframe_id = result.matched_keyframe_id;
        }
    }

    if (!localizer.isRelocalized() || !success) {
        auto result = localizer.relocalize(frame.undistorted_cloud, frame.T_world_lidar);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
            relocalization_locked = true;
            matched_keyframe_id = result.matched_keyframe_id;
        }
    }

    if (!success) {
        pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
    }

    auto output = makeOutput(success, pose_map, frame.undistorted_cloud);
    output.relocalization_locked = relocalization_locked;
    output.matched_keyframe_id = matched_keyframe_id;
    return output;
}

core::BackendOutput N3MappingCore::processMapExtensionFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty() || !map_loaded_) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    auto& resuming = session_->mappingResuming();
    auto& localizer = session_->worldLocalizing();
    const auto state = resuming.getState();

    if (state == MappingResumingState::MAP_LOADED) {
        const bool locked = resuming.performInitialRelocalization(frame.undistorted_cloud, frame.T_world_lidar);
        auto output = makeOutput(locked,
                                 localizer.getMapToOdomTransform() * frame.T_world_lidar,
                                 frame.undistorted_cloud);
        output.relocalization_locked = locked;
        if (locked) {
            const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
            const int64_t matched_id = localizer.getLastMatchedKeyframeId();
            auto matched_kf = session_->keyframeManager().getKeyframe(matched_id);
            if (!external_dense_trajectory_recording_enabled_ && matched_kf) {
                appendDenseTrajectorySample(timestamp, output.T_world_lidar, matched_id, matched_kf->pose_optimized, false);
            }
        }
        return output;
    }

    if (state != MappingResumingState::RELOCALIZED && state != MappingResumingState::EXTENDING) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    Eigen::Isometry3d pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
    if (!session_->keyframeManager().shouldAddKeyframe(pose_map)) {
        if (!external_dense_trajectory_recording_enabled_) {
            const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
            appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_map, false);
        }
        return makeOutput(true, pose_map, frame.undistorted_cloud);
    }

    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    const int64_t keyframe_id = resuming.processNewKeyframe(timestamp, frame.T_world_lidar, frame.undistorted_cloud);
    if (keyframe_id >= 0) {
        resuming.detectCrossLoops(keyframe_id);
        session_->graphOptimizer().incrementalOptimize();
        refreshOptimizedPoses();
        if (session_->graphOptimizer().hasNode(keyframe_id)) {
            try {
                pose_map = session_->graphOptimizer().getOptimizedPose(keyframe_id);
            } catch (const std::exception&) {
                pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
            }
        }
    }

    auto output = makeOutput(keyframe_id >= 0, pose_map, frame.undistorted_cloud);
    output.accepted_keyframe = keyframe_id >= 0;
    output.keyframe_id = keyframe_id;
    if (!external_dense_trajectory_recording_enabled_ && keyframe_id >= 0) {
        if (auto kf = session_->keyframeManager().getKeyframe(keyframe_id)) {
            appendDenseTrajectorySample(timestamp, kf->pose_odom, keyframe_id, kf->pose_odom, false);
        }
    }
    return output;
}

CoreLoopClosureResult N3MappingCore::processPendingLoopClosures()
{
    CoreLoopClosureResult result;
    std::vector<int64_t> keyframes_to_check;
    {
        std::lock_guard<std::mutex> lock(loop_queue_mutex_);
        keyframes_to_check.swap(loop_detection_queue_);
    }

    std::map<int64_t, std::vector<LoopCandidate>> raw_candidates_by_query;
    std::map<int64_t, Keyframe::Ptr> query_keyframes;
    std::vector<LoopCandidate> raw_candidates;
    std::map<int64_t, Keyframe::Ptr> keyframe_map;
    if (config_.loop_spatial_candidates_enable) {
        for (const auto& keyframe : session_->keyframeManager().getAllKeyframes()) {
            if (keyframe) {
                keyframe_map[keyframe->id] = keyframe;
            }
        }
    }

    int64_t next_last_loop_check_id = last_loop_check_id_;
    for (int64_t query_id : keyframes_to_check) {
        if (query_id - next_last_loop_check_id < config_.loop_kf_gap) {
            continue;
        }

        auto query_kf = session_->keyframeManager().getKeyframe(query_id);
        if (!query_kf) {
            continue;
        }

        std::vector<LoopCandidate> candidates = session_->loopDetector().detectLoopCandidates(query_id);
        if (config_.loop_spatial_candidates_enable) {
            auto spatial_candidates =
                session_->loopDetector().detectSpatialCandidates(query_id, keyframe_map);
            for (const auto& spatial : spatial_candidates) {
                auto duplicate = std::find_if(candidates.begin(), candidates.end(),
                    [&](const LoopCandidate& existing) {
                        return existing.query_id == spatial.query_id &&
                               existing.match_id == spatial.match_id;
                    });
                if (duplicate == candidates.end()) {
                    candidates.push_back(spatial);
                } else {
                    duplicate->source_flags |= spatial.source_flags;
                    duplicate->spatial_score = std::max(duplicate->spatial_score, spatial.spatial_score);
                }
            }
        }
        if (candidates.empty()) {
            continue;
        }
        next_last_loop_check_id = query_id;
        query_keyframes[query_id] = query_kf;
        raw_candidates.insert(raw_candidates.end(), candidates.begin(), candidates.end());
        raw_candidates_by_query[query_id] = std::move(candidates);
    }
    last_loop_check_id_ = next_last_loop_check_id;

    if (raw_candidates.empty()) {
        return result;
    }

    const CandidateSelectionResult pre_icp_selection =
        selectPreIcpSegmentRepresentatives(config_, session_->keyframeManager(), raw_candidates);
    std::map<int64_t, std::vector<LoopCandidate>> candidates_by_query;
    for (const auto& candidate : pre_icp_selection.selected) {
        candidates_by_query[candidate.query_id].push_back(candidate);
    }

    for (const auto& [query_id, query_kf] : query_keyframes) {
        std::vector<LoopCandidate> candidates = candidates_by_query[query_id];

        std::vector<VerifiedLoop> verified_loops;
        LoopVerifier loop_verifier(config_);
        verified_loops.reserve(candidates.size());
        std::vector<LoopDebugCandidateEvent> debug_events;
        std::map<std::pair<int64_t, int64_t>, LoopGraphTrialDiagnostics> graph_trial_by_pair;
        std::map<std::pair<int64_t, int64_t>, LoopGraphTrialDiagnostics> consensus_estimator_trial_by_pair;
        std::map<std::pair<int64_t, int64_t>, LoopRefereeDebugDecision> referee_by_pair;
        std::map<std::pair<int64_t, int64_t>, LoopConsensusResult> consensus_by_pair;
        std::map<std::pair<int64_t, int64_t>, std::string> graph_reject_by_pair;
        const bool loop_debug_enabled = config_.loop_debug_enable;
        if (loop_debug_enabled) {
            debug_events.reserve(raw_candidates_by_query[query_id].size());
        }

        auto flush_debug_events = [&](const std::set<std::pair<int64_t, int64_t>>& accepted_pairs,
                                      const std::string& not_selected_reason) {
            if (!loop_debug_enabled) {
                return;
            }
            for (auto& event : debug_events) {
                const std::pair<int64_t, int64_t> key(event.candidate.query_id, event.candidate.match_id);
                auto trial_it = graph_trial_by_pair.find(key);
                if (trial_it != graph_trial_by_pair.end()) {
                    assignGraphTrialDiagnostics(&event, trial_it->second);
                }
                auto consensus_trial_it = consensus_estimator_trial_by_pair.find(key);
                if (consensus_trial_it != consensus_estimator_trial_by_pair.end()) {
                    assignConsensusEstimatorTrialDiagnostics(&event, consensus_trial_it->second);
                }
                auto referee_it = referee_by_pair.find(key);
                if (referee_it != referee_by_pair.end()) {
                    event.loop_referee_recommendation = referee_it->second.recommendation;
                    event.loop_referee_reason = referee_it->second.reason;
                    event.loop_referee_risk_flags = referee_it->second.risk_flags;
                }
                auto consensus_it = consensus_by_pair.find(key);
                if (consensus_it != consensus_by_pair.end()) {
                    assignConsensusDiagnostics(&event, consensus_it->second);
                }
                if (accepted_pairs.find(key) != accepted_pairs.end()) {
                    event.gate_result = "accepted";
                    event.reject_reason.clear();
                } else if (event.gate_result == "accepted") {
                    event.gate_result = "rejected";
                    auto graph_reject_it = graph_reject_by_pair.find(key);
                    if (graph_reject_it != graph_reject_by_pair.end()) {
                        event.reject_reason = graph_reject_it->second;
                    } else {
                        event.reject_reason = not_selected_reason.empty() ? "not_selected" : not_selected_reason;
                    }
                }
                appendLoopDebugCandidate(event);
            }
        };

        auto make_rejected_event = [&](const LoopCandidate& candidate,
                                       const std::string& reject_reason) {
            if (!loop_debug_enabled) {
                return;
            }
            LoopDebugCandidateEvent event;
            event.processing_time = processingTimeSeconds();
            event.query_timestamp = query_kf->timestamp;
            event.candidate = candidate;
            event.gate_result = "rejected";
            event.reject_reason = reject_reason;
            debug_events.push_back(event);
        };

        for (const auto& candidate : raw_candidates_by_query[query_id]) {
            auto rejected = pre_icp_selection.rejected.find(candidatePair(candidate));
            if (rejected != pre_icp_selection.rejected.end()) {
                make_rejected_event(candidate, rejected->second);
            }
        }

        if (candidates.empty()) {
            flush_debug_events({}, "pre_icp_cluster_empty");
            continue;
        }

        for (const auto& candidate : candidates) {
            auto match_kf = session_->keyframeManager().getKeyframe(candidate.match_id);
            if (!match_kf || !query_kf->cloud || !match_kf->cloud || query_kf->cloud->empty() || match_kf->cloud->empty()) {
                make_rejected_event(candidate, "missing_keyframe_or_cloud");
                continue;
            }

            auto source = session_->keyframeManager().buildSubmapInRootFrame(query_id, 0, candidate.match_id);
            auto target = session_->keyframeManager().buildSubmapInRootFrame(candidate.match_id, config_.gicp_submap_size, candidate.match_id);
            if (!source || source->empty() || !target || target->empty()) {
                make_rejected_event(candidate, "empty_submap");
                continue;
            }
            source = prepareLoopIcpCloud(source, config_);
            target = prepareLoopIcpCloud(target, config_);
            if (!source || source->size() < 10 || !target || target->size() < 10) {
                make_rejected_event(candidate, "empty_submap_after_prefilter");
                continue;
            }
            ZDistributionStats source_z_before;
            ZDistributionStats target_z;
            if (loop_debug_enabled) {
                source_z_before = computeZStats(source);
                target_z = computeZStats(target);
            }

            LoopVerification verification = loop_verifier.verifyPreparedSubmaps(
                candidate, query_kf, match_kf, source, target, session_->pointCloudMatcher());
            MatchResult match_result = verification.match_result;
            ZDistributionStats source_z_after;
            if (loop_debug_enabled) {
                source_z_after = computeTransformedZStats(source, verification.T_icp_correction_match);
            }

            VerifiedLoop loop = verification.loop;
            loop.source_z_span = source_z_before.span();
            loop.target_z_span = target_z.span();
            loop.z_overlap_ratio_before = zOverlapRatio(source_z_before, target_z);
            loop.z_overlap_ratio_after = zOverlapRatio(source_z_after, target_z);
            loop.source_z_robust_span = source_z_before.robustSpan();
            loop.target_z_robust_span = target_z.robustSpan();
            loop.z_robust_overlap_ratio_before = robustZOverlapRatio(source_z_before, target_z);
            loop.z_robust_overlap_ratio_after = robustZOverlapRatio(source_z_after, target_z);
            loop.source_target_z_centroid_delta_before = zCentroidDelta(source_z_before, target_z);
            loop.source_target_z_centroid_delta_after = zCentroidDelta(source_z_after, target_z);
            if (loop_debug_enabled && config_.loop_debug_vertical_hypotheses_enable &&
                match_result.converged && verification.fitness_ok && verification.inlier_ok && verification.geometry_ok) {
                const auto diagnostics = computeVerticalHypothesisDiagnostics(
                    session_->pointCloudMatcher(), target, source, match_result);
                loop.vertical_hypothesis_count = diagnostics.count;
                loop.best_z_offset_m = diagnostics.best_z_offset_m;
                loop.best_z_offset_fitness = diagnostics.best_z_offset_fitness;
                loop.zero_z_fitness = diagnostics.zero_z_fitness;
                loop.fitness_gap_zero_vs_best = diagnostics.fitness_gap_zero_vs_best;
                loop.z_hypothesis_spread_m = diagnostics.z_hypothesis_spread_m;
                loop.vertical_ambiguity_score = diagnostics.vertical_ambiguity_score;
                loop.vertical_hypothesis_edge_recommendation = diagnostics.edge_recommendation;
            }
            std::string reject_reason = verification.reject_reason;
            if (loop.verified) {
                loop = session_->loopClosureManager().applyEdgeModel(loop);
                if (!loop.verified) {
                    reject_reason = "edge_model";
                }
            }
            if (loop.verified) {
                const auto segment = computeLoopSegmentConsistency(
                    config_, session_->keyframeManager(), loop);
                assignLoopSegmentConsistency(&loop, segment);
                const LoopFeatures features = makeSegmentAwareLoopFeatures(
                    config_, candidate, loop, verification.icp_translation_norm);
                const LoopRefereeDecision referee = LoopReferee::evaluate(features);
                loop.loop_referee_energy = referee.energy;
                loop.loop_referee_recommendation =
                    referee.decision == LoopDecision::Accept ? "accept" : "reject";
                loop.loop_referee_reason = referee.reason;
                loop.loop_referee_risk_flags = referee.risk_flags;
                loop.verified = referee.decision == LoopDecision::Accept;
                if (!loop.verified) {
                    reject_reason = "loop_referee";
                }
            }
            if (loop_debug_enabled) {
                LoopDebugCandidateEvent event;
                event.processing_time = processingTimeSeconds();
                event.query_timestamp = query_kf->timestamp;
                event.candidate = candidate;
                event.icp_converged = match_result.converged;
                event.icp_iterations = match_result.iterations;
                event.icp_optimizer_error = match_result.optimizer_error;
                event.icp_termination = matchTerminationName(match_result.termination);
                event.fitness_score = match_result.fitness_score;
                event.inlier_ratio = match_result.inlier_ratio;
                event.icp_translation_norm = verification.icp_translation_norm;
                event.icp_rotation_norm = verification.icp_rotation_norm;
                event.residual = verification.T_measurement_residual;
                event.T_pred_match_query = verification.T_pred_match_query;
                event.T_icp_correction_match = verification.T_icp_correction_match;
                event.T_measured_match_query = verification.T_measured_match_query;
                event.has_loop_measurement = true;
                event.loop_measurement_match_query = verification.T_measured_match_query;
                event.loop_information = loop.information;
                event.edge_mode = loop.verified ? loopEdgeModeName(loop.edge_mode) : "not_applicable";
                event.vertical_observability_score =
                    loop.verified ? loop.vertical_observability_score : std::numeric_limits<double>::quiet_NaN();
                event.vertical_downweighted = loop.vertical_downweighted;
                event.source_z_span = loop.source_z_span;
                event.target_z_span = loop.target_z_span;
                event.z_overlap_ratio_before = loop.z_overlap_ratio_before;
                event.z_overlap_ratio_after = loop.z_overlap_ratio_after;
                event.source_z_robust_span = loop.source_z_robust_span;
                event.target_z_robust_span = loop.target_z_robust_span;
                event.z_robust_overlap_ratio_before = loop.z_robust_overlap_ratio_before;
                event.z_robust_overlap_ratio_after = loop.z_robust_overlap_ratio_after;
                event.source_target_z_centroid_delta_before = loop.source_target_z_centroid_delta_before;
                event.source_target_z_centroid_delta_after = loop.source_target_z_centroid_delta_after;
                event.vertical_information_ratio = loop.vertical_information_ratio;
                event.vertical_hypothesis_count = loop.vertical_hypothesis_count;
                event.best_z_offset_m = loop.best_z_offset_m;
                event.best_z_offset_fitness = loop.best_z_offset_fitness;
                event.zero_z_fitness = loop.zero_z_fitness;
                event.fitness_gap_zero_vs_best = loop.fitness_gap_zero_vs_best;
                event.z_hypothesis_spread_m = loop.z_hypothesis_spread_m;
                event.vertical_ambiguity_score = loop.vertical_ambiguity_score;
                event.vertical_hypothesis_edge_recommendation = loop.vertical_hypothesis_edge_recommendation;
                event.heightmap_overlap_cell_count = loop.heightmap_overlap_cell_count;
                event.heightmap_overlap_ratio = loop.heightmap_overlap_ratio;
                event.heightmap_ground_dz_median = loop.heightmap_ground_dz_median;
                event.heightmap_ground_dz_p90 = loop.heightmap_ground_dz_p90;
                event.heightmap_ground_dz_max = loop.heightmap_ground_dz_max;
                event.heightmap_ground_support_ratio = loop.heightmap_ground_support_ratio;
                event.heightmap_vertical_consistency_score = loop.heightmap_vertical_consistency_score;
                assignSegmentDiagnostics(&event, loop);
                assignConsensusDiagnostics(&event, loop);
                event.loop_referee_recommendation = loop.loop_referee_recommendation;
                event.loop_referee_reason = loop.loop_referee_reason;
                event.loop_referee_risk_flags = loop.loop_referee_risk_flags;
                event.gate_result = loop.verified ? "accepted" : "rejected";
                event.reject_reason = reject_reason;
                debug_events.push_back(event);
            }
            verified_loops.push_back(loop);
        }

        if (verified_loops.empty()) {
            flush_debug_events({}, "not_selected");
            continue;
        }

        auto valid_loops = session_->loopClosureManager().filterValidLoops(verified_loops);
        auto best_loops = session_->loopClosureManager().selectBestPerQuery(valid_loops);
        if (best_loops.empty()) {
            flush_debug_events({}, "not_selected");
            continue;
        }

        LoopConsensusVerifier consensus_verifier(config_);
        std::vector<VerifiedLoop> consensus_gated_loops;
        consensus_gated_loops.reserve(best_loops.size());
        for (auto& loop : best_loops) {
            const auto consensus = consensus_verifier.evaluate(
                session_->keyframeManager(), session_->pointCloudMatcher(), loop,
                std::max(2, config_.gicp_submap_size));
            assignLoopConsensus(&loop, consensus);
            const auto key = std::make_pair(loop.query_id, loop.match_id);
            consensus_by_pair[key] = consensus;
            if (const std::string reason = consensusRefereeRejectReason(loop); !reason.empty()) {
                graph_reject_by_pair[key] = reason;
                continue;
            }
            consensus_gated_loops.push_back(loop);
        }
        best_loops.swap(consensus_gated_loops);
        if (best_loops.empty()) {
            flush_debug_events({}, "consensus_referee_rejected");
            continue;
        }

        auto edges = session_->loopClosureManager().buildLoopEdges(best_loops, LoopEdgeDirection::MatchToQuery);
        if (edges.empty()) {
            flush_debug_events({}, "edge_build_empty");
            continue;
        }
        const auto poses_before = session_->graphOptimizer().getOptimizedPoses();
        const auto committed_edges = session_->graphOptimizer().getEdges();
        std::vector<EdgeInfo> gated_edges;
        std::vector<VerifiedLoop> gated_best_loops;
        gated_edges.reserve(edges.size());
        gated_best_loops.reserve(best_loops.size());

        for (const auto& edge : edges) {
            auto loop_it = std::find_if(
                best_loops.begin(), best_loops.end(),
                [&](const VerifiedLoop& loop) {
                    return loop.match_id == edge.from_id && loop.query_id == edge.to_id;
                });
            if (loop_it == best_loops.end()) {
                continue;
            }

            VerifiedLoop loop = *loop_it;
            const std::vector<EdgeInfo> candidate_edges{edge};
            const auto diagnostics = computeLoopGraphTrialDiagnostics(
                config_, poses_before, committed_edges, candidate_edges);
            const auto key = std::make_pair(loop.query_id, loop.match_id);
            auto consensus_it = consensus_by_pair.find(key);
            if (consensus_it != consensus_by_pair.end() &&
                consensus_it->second.estimator_pair_count >= 3) {
                EdgeInfo estimator_edge = edge;
                estimator_edge.measurement =
                    consensus_it->second.estimator_measurement_match_query;
                const std::vector<EdgeInfo> estimator_edges{estimator_edge};
                const auto estimator_diagnostics = computeLoopGraphTrialDiagnostics(
                    config_, poses_before, committed_edges, estimator_edges);
                consensus_estimator_trial_by_pair[key] = estimator_diagnostics;
            }
            assignGraphTrialDiagnostics(&loop, diagnostics);
            graph_trial_by_pair[key] = diagnostics;
            referee_by_pair[key] = {
                loop.loop_referee_recommendation,
                loop.loop_referee_reason,
                loop.loop_referee_risk_flags
            };
            if (graphTrialYawInconsistent(diagnostics)) {
                graph_reject_by_pair[key] = "graph_inconsistent_yaw";
                continue;
            }
            if (graphTrialTranslationInconsistent(diagnostics)) {
                graph_reject_by_pair[key] = "graph_inconsistent_translation";
                continue;
            }
            gated_edges.push_back(edge);
            gated_best_loops.push_back(loop);
        }

        edges.swap(gated_edges);
        best_loops.swap(gated_best_loops);
        if (edges.empty()) {
            flush_debug_events({}, "graph_trial_rejected");
            continue;
        }
        result.place_candidate_count += best_loops.size();
        const auto residual_before = meanLoopResidual(edges, poses_before);
        const auto residual_axes_before = meanLoopResidualAxes(edges, poses_before);
        const bool optimization_committed =
            session_->loopClosureManager().applyEdges(edges, session_->graphOptimizer());
        if (!optimization_committed) {
            flush_debug_events({}, "optimization_failed");
            continue;
        }

        std::set<std::pair<int64_t, int64_t>> accepted_debug_pairs;
        for (const auto& loop : best_loops) {
            accepted_debug_pairs.insert({loop.query_id, loop.match_id});
        }
        flush_debug_events(accepted_debug_pairs, "not_selected");

        loop_count_ += edges.size();
        refreshOptimizedPoses();
        const auto poses_after = session_->graphOptimizer().getOptimizedPoses();
        const auto residual_after = meanLoopResidual(edges, poses_after);
        const auto residual_axes_after = meanLoopResidualAxes(edges, poses_after);

        result.optimized = true;
        result.edge_count += edges.size();
        result.graph_edge_count += edges.size();
        result.loop_residual_translation_before = residual_before.first;
        result.loop_residual_rotation_before = residual_before.second;
        result.loop_residual_translation_after = residual_after.first;
        result.loop_residual_rotation_after = residual_after.second;
        accumulatePoseUpdateStats(poses_before, poses_after, &result);
        result.accepted_loops.insert(result.accepted_loops.end(), best_loops.begin(), best_loops.end());

        if (loop_debug_enabled) {
            LoopDebugOptimizationEvent event;
            event.processing_time = processingTimeSeconds();
            event.accepted_edge_count = edges.size();
            event.accepted_edges.reserve(edges.size());
            for (const auto& edge : edges) {
                event.accepted_edges.emplace_back(edge.from_id, edge.to_id);
            }
            event.loop_residual_translation_before = residual_before.first;
            event.loop_residual_rotation_before = residual_before.second;
            event.loop_residual_translation_after = residual_after.first;
            event.loop_residual_rotation_after = residual_after.second;
            event.loop_residual_translation_axes_before = residual_axes_before.translation;
            event.loop_residual_translation_axes_after = residual_axes_after.translation;
            event.loop_residual_rpy_axes_before = residual_axes_before.rpy;
            event.loop_residual_rpy_axes_after = residual_axes_after.rpy;
            event.mean_pose_update_translation = result.mean_pose_update_translation;
            event.max_pose_update_translation = result.max_pose_update_translation;
            event.mean_pose_update_rotation = result.mean_pose_update_rotation;
            event.max_pose_update_rotation = result.max_pose_update_rotation;
            appendLoopDebugOptimization(event);
        }
    }

    return result;
}

bool N3MappingCore::loadMap(const std::string& map_path)
{
    std::vector<core::DenseTrajectoryPose> loaded_dense_optimized;
    core::DenseTrajectoryMetadata loaded_dense_metadata;
    const bool loaded_for_dense = session_->mapSerializer().loadMap(
        map_path,
        session_->keyframeManager(),
        session_->loopDetector(),
        session_->graphOptimizer(),
        &loaded_dense_optimized,
        &loaded_dense_metadata);
    if (!loaded_for_dense) {
        return false;
    }
    if (!session_->mappingResuming().initializeFromLoadedMap()) {
        return false;
    }
    session_->worldLocalizing().reset();
    dense_trajectory_metadata_ = loaded_dense_metadata;
    initializeDenseSamplesFromOptimized(loaded_dense_optimized);
    map_loaded_ = true;
    return true;
}

bool N3MappingCore::saveMap(const std::string& map_path)
{
    const auto dense_optimized_trajectory = buildDenseOptimizedTrajectory();
    core::DenseTrajectoryMetadata metadata = dense_trajectory_metadata_;
    if (!dense_optimized_trajectory.empty() && (metadata.source.empty() || metadata.source == "none")) {
        metadata.source = "native";
        metadata.degraded = false;
    }
    return session_->mapSerializer().saveMap(
        map_path,
        session_->keyframeManager(),
        session_->loopDetector(),
        session_->graphOptimizer(),
        dense_optimized_trajectory,
        metadata);
}

bool N3MappingCore::saveGlobalMap(const std::string& pcd_path)
{
    return session_->mapSerializer().saveGlobalMap(
        pcd_path, session_->keyframeManager(), config_.save_global_map_voxel_size);
}

bool N3MappingCore::saveMapSnapshot(std::string* error)
{
    if (session_->keyframeManager().size() < 1) {
        if (error) *error = "no_keyframes";
        return false;
    }

    const std::string map_file = config_.map_save_path + "/n3map.pbstream";
    if (!saveMap(map_file)) {
        if (error) *error = "save_pbstream_failed";
        return false;
    }

    if (config_.save_global_map_on_shutdown) {
        const std::string global_map_file = config_.map_save_path + "/global_map.pcd";
        if (!saveGlobalMap(global_map_file)) {
            if (error) *error = "save_global_map_failed";
            return false;
        }
    }

    return true;
}

core::LioFrame::PointCloud::Ptr N3MappingCore::buildGlobalMap() const
{
    return session_->mapSerializer().buildGlobalMap(
        session_->keyframeManager(), config_.global_map_voxel_size);
}

bool N3MappingCore::mapLoaded() const
{
    return map_loaded_;
}

Keyframe::Ptr N3MappingCore::getKeyframe(int64_t id) const
{
    return session_->keyframeManager().getKeyframe(id);
}

std::vector<Keyframe::Ptr> N3MappingCore::getAllKeyframes() const
{
    return session_->keyframeManager().getAllKeyframes();
}

std::map<int64_t, Eigen::Isometry3d> N3MappingCore::getOptimizedPoses() const
{
    return session_->graphOptimizer().getOptimizedPoses();
}

std::vector<core::DenseTrajectoryPose> N3MappingCore::getDenseOptimizedTrajectory() const
{
    return buildDenseOptimizedTrajectory();
}

void N3MappingCore::setExternalDenseTrajectoryRecordingEnabled(bool enabled)
{
    external_dense_trajectory_recording_enabled_ = enabled;
}

void N3MappingCore::recordDenseTrajectoryPose(CoreRunMode mode,
                                              double timestamp,
                                              const Eigen::Isometry3d& pose_world_lidar)
{
    if (!coreRunModeSavesMap(mode) || !std::isfinite(timestamp) || !isFinitePose(pose_world_lidar)) {
        return;
    }

    if (mode == CoreRunMode::MAPPING) {
        appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_world_lidar);
        return;
    }

    if (mode != CoreRunMode::MAP_EXTENSION || !map_loaded_) {
        return;
    }

    const auto state = session_->mappingResuming().getState();
    if (state != MappingResumingState::RELOCALIZED && state != MappingResumingState::EXTENDING) {
        return;
    }
    if (!session_->worldLocalizing().isRelocalized()) {
        return;
    }

    const Eigen::Isometry3d pose_map = session_->worldLocalizing().getMapToOdomTransform() * pose_world_lidar;
    if (!isFinitePose(pose_map)) {
        return;
    }
    appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_map, false);
}

core::BackendOutput N3MappingCore::makeOutput(bool success,
                                              const Eigen::Isometry3d& pose,
                                              const PointCloud::Ptr& cloud) const
{
    core::BackendOutput output;
    output.success = success;
    output.T_world_lidar = pose;
    output.cloud_body = cloud;
    output.cloud_world = makeWorldCloud(cloud, pose);
    return output;
}

N3MappingCore::PointCloud::Ptr N3MappingCore::makeWorldCloud(const PointCloud::Ptr& cloud,
                                                             const Eigen::Isometry3d& pose) const
{
    if (!cloud || cloud->empty()) {
        return pcl::make_shared<PointCloud>();
    }
    auto transformed = pcl::make_shared<PointCloud>();
    pcl::transformPointCloud(*cloud, *transformed, pose.matrix().cast<float>());
    return transformed;
}

void N3MappingCore::appendDenseTrajectorySample(double timestamp,
                                                const Eigen::Isometry3d& raw_pose,
                                                int64_t anchor_keyframe_id,
                                                const Eigen::Isometry3d& anchor_raw_pose,
                                                bool use_bracketing_correction)
{
    core::AnchoredDenseTrajectorySample sample;
    sample.seq = static_cast<uint64_t>(dense_trajectory_samples_.size());
    sample.timestamp = timestamp;
    sample.pose_world_lidar_raw = raw_pose;
    sample.anchor_keyframe_id = anchor_keyframe_id;
    sample.anchor_pose_world_lidar_raw = anchor_raw_pose;
    sample.has_anchor = anchor_keyframe_id >= 0;
    sample.use_bracketing_correction = use_bracketing_correction;
    if (dense_trajectory_metadata_.source == "keyframe_fallback") {
        dense_trajectory_metadata_.source = "mixed_keyframe_fallback_and_high_rate";
        dense_trajectory_metadata_.degraded = true;
    } else if (dense_trajectory_metadata_.source.empty() || dense_trajectory_metadata_.source == "none") {
        dense_trajectory_metadata_.source = "native";
        dense_trajectory_metadata_.degraded = false;
    }
    dense_trajectory_samples_.push_back(sample);
}

void N3MappingCore::appendDenseTrajectorySampleWithLatestAnchor(double timestamp,
                                                                const Eigen::Isometry3d& raw_pose,
                                                                bool use_bracketing_correction)
{
    auto latest = session_->keyframeManager().getLatestKeyframe();
    if (!latest) {
        appendDenseTrajectorySample(timestamp, raw_pose, -1, Eigen::Isometry3d::Identity(), use_bracketing_correction);
        return;
    }

    const Eigen::Isometry3d anchor_raw_pose =
        latest->is_from_loaded_map ? latest->pose_optimized : latest->pose_odom;
    appendDenseTrajectorySample(timestamp, raw_pose, latest->id, anchor_raw_pose, use_bracketing_correction);
}

void N3MappingCore::initializeDenseSamplesFromOptimized(
    const std::vector<core::DenseTrajectoryPose>& dense_optimized)
{
    dense_trajectory_samples_.clear();
    dense_trajectory_samples_.reserve(dense_optimized.size());

    const auto keyframes = session_->keyframeManager().getAllKeyframes();
    for (const auto& dense_pose : dense_optimized) {
        Keyframe::Ptr anchor;
        double best_time = -std::numeric_limits<double>::infinity();
        for (const auto& kf : keyframes) {
            if (!kf) {
                continue;
            }
            if (kf->timestamp <= dense_pose.timestamp && kf->timestamp >= best_time) {
                anchor = kf;
                best_time = kf->timestamp;
            }
        }
        if (!anchor && !keyframes.empty()) {
            anchor = keyframes.front();
        }

        core::AnchoredDenseTrajectorySample sample;
        sample.seq = dense_pose.seq;
        sample.timestamp = dense_pose.timestamp;
        sample.pose_world_lidar_raw = dense_pose.pose_world_lidar;
        if (anchor) {
            sample.anchor_keyframe_id = anchor->id;
            sample.anchor_pose_world_lidar_raw = anchor->pose_optimized;
            sample.has_anchor = true;
            sample.use_bracketing_correction = false;
        }
        dense_trajectory_samples_.push_back(sample);
    }
}

std::vector<core::DenseTrajectoryPose> N3MappingCore::buildDenseOptimizedTrajectory() const
{
    std::vector<core::DenseTrajectoryPose> dense_optimized;
    dense_optimized.reserve(dense_trajectory_samples_.size());
    for (const auto& sample : dense_trajectory_samples_) {
        core::DenseTrajectoryPose pose;
        pose.seq = sample.seq;
        pose.timestamp = sample.timestamp;
        pose.pose_world_lidar = sample.pose_world_lidar_raw;
        if (sample.use_bracketing_correction) {
            pose.pose_world_lidar = interpolateDenseCorrection(sample.timestamp) * sample.pose_world_lidar_raw;
        } else if (sample.has_anchor) {
            auto anchor = session_->keyframeManager().getKeyframe(sample.anchor_keyframe_id);
            if (anchor) {
                pose.pose_world_lidar =
                    anchor->pose_optimized * sample.anchor_pose_world_lidar_raw.inverse() * sample.pose_world_lidar_raw;
            }
        }
        dense_optimized.push_back(pose);
    }
    return dense_optimized;
}

Eigen::Isometry3d N3MappingCore::interpolateDenseCorrection(double timestamp) const
{
    const auto keyframes = session_->keyframeManager().getAllKeyframes();
    Keyframe::Ptr before;
    Keyframe::Ptr after;

    for (const auto& kf : keyframes) {
        if (!kf) {
            continue;
        }
        if (kf->timestamp <= timestamp && (!before || kf->timestamp > before->timestamp)) {
            before = kf;
        }
        if (kf->timestamp >= timestamp && (!after || kf->timestamp < after->timestamp)) {
            after = kf;
        }
    }

    if (!before && !after) {
        return Eigen::Isometry3d::Identity();
    }
    if (!before) {
        before = after;
    }
    if (!after) {
        after = before;
    }

    const Eigen::Isometry3d correction_before = before->pose_optimized * before->pose_odom.inverse();
    const Eigen::Isometry3d correction_after = after->pose_optimized * after->pose_odom.inverse();
    double alpha = 0.0;
    const double dt = after->timestamp - before->timestamp;
    if (std::isfinite(dt) && dt > 1e-9) {
        alpha = std::clamp((timestamp - before->timestamp) / dt, 0.0, 1.0);
    }

    Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
    correction.translation() =
        (1.0 - alpha) * correction_before.translation() + alpha * correction_after.translation();
    Eigen::Quaterniond qb(correction_before.rotation());
    Eigen::Quaterniond qa(correction_after.rotation());
    qb.normalize();
    qa.normalize();
    correction.linear() = qb.slerp(alpha, qa).toRotationMatrix();
    return correction;
}

void N3MappingCore::addRhpdDescriptorForKeyframe(int64_t keyframe_id, const PointCloud::Ptr& fallback_cloud)
{
    auto kf = session_->keyframeManager().getKeyframe(keyframe_id);
    if (!kf) {
        return;
    }

    const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
    PointCloud::Ptr rhpd_cloud = fallback_cloud;
    if (submap_radius > 0) {
        rhpd_cloud = session_->keyframeManager().buildCausalSubmapInRootFrame(keyframe_id, submap_radius, keyframe_id);
    }

    if (rhpd_cloud && !rhpd_cloud->empty() && config_.rhpd_submap_voxel_size > 1e-4) {
        PointCloud::Ptr filtered;
        if (safeVoxelGridFilter<pcl::PointXYZI>(rhpd_cloud, config_.rhpd_submap_voxel_size, &filtered) &&
            filtered && !filtered->empty()) {
            rhpd_cloud = filtered;
        }
    }

    kf->rhpd_descriptor = session_->loopDetector().addRHPD(keyframe_id, rhpd_cloud);
}

bool N3MappingCore::addOdometryConstraint(int64_t keyframe_id, const Eigen::Isometry3d& pose)
{
    auto prev_kf = session_->keyframeManager().getKeyframe(keyframe_id - 1);
    if (!prev_kf) {
        return false;
    }

    EdgeInfo edge;
    edge.from_id = keyframe_id - 1;
    edge.to_id = keyframe_id;
    edge.measurement = prev_kf->pose_odom.inverse() * pose;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
    edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
    edge.type = EdgeType::ODOMETRY;
    session_->graphOptimizer().addOdometryEdge(edge);
    return true;
}

void N3MappingCore::refreshOptimizedPoses()
{
    session_->keyframeManager().updateOptimizedPoses(session_->graphOptimizer().getOptimizedPoses());
}

}  // namespace n3mapping
