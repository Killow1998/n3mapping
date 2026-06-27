#include "n3mapping/loop_consensus_verifier.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace n3mapping {
namespace {

double rotationAngle(const Eigen::Isometry3d& transform)
{
    return Eigen::AngleAxisd(transform.rotation()).angle();
}

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
    auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if (values.size() % 2 == 1) {
        return *mid;
    }
    const double upper = *mid;
    std::nth_element(values.begin(), mid - 1, values.end());
    return 0.5 * (*(mid - 1) + upper);
}

double mad(const std::vector<double>& values, double center)
{
    if (values.empty() || !std::isfinite(center)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) {
        deviations.push_back(std::abs(value - center));
    }
    return median(std::move(deviations));
}

Eigen::Isometry3d composeFromTranslationRpy(const Eigen::Vector3d& translation,
                                            const Eigen::Vector3d& rpy)
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = translation;
    transform.linear() =
        (Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ())).toRotationMatrix();
    return transform;
}

void estimateConsensusMeasurement(const Config& config,
                                  const std::vector<LoopConsensusPairEvidence>& pairs,
                                  const Eigen::Isometry3d* central_measurement,
                                  LoopConsensusResult* result)
{
    if (!result) {
        return;
    }

    std::vector<Eigen::Vector3d> translations;
    std::vector<Eigen::Vector3d> rpys;
    translations.reserve(pairs.size());
    rpys.reserve(pairs.size());
    for (const auto& pair : pairs) {
        if (!pair.valid || !pair.converged || !pair.has_estimated_measurement) {
            continue;
        }
        translations.push_back(pair.estimated_match_query.translation());
        rpys.push_back(rollPitchYaw(pair.estimated_match_query));
    }

    result->estimator_pair_count = static_cast<int>(translations.size());
    if (translations.size() < 3) {
        result->estimator_recommendation = "insufficient_estimator_support";
        return;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<double> rolls;
    std::vector<double> pitches;
    std::vector<double> yaws;
    xs.reserve(translations.size());
    ys.reserve(translations.size());
    zs.reserve(translations.size());
    rolls.reserve(rpys.size());
    pitches.reserve(rpys.size());
    yaws.reserve(rpys.size());
    for (std::size_t i = 0; i < translations.size(); ++i) {
        xs.push_back(translations[i].x());
        ys.push_back(translations[i].y());
        zs.push_back(translations[i].z());
        rolls.push_back(normalizeAngle(rpys[i].x()));
        pitches.push_back(normalizeAngle(rpys[i].y()));
        yaws.push_back(normalizeAngle(rpys[i].z()));
    }

    const Eigen::Vector3d median_translation(
        median(xs), median(ys), median(zs));
    const Eigen::Vector3d median_rpy(
        median(rolls), median(pitches), median(yaws));
    result->estimator_measurement_match_query =
        composeFromTranslationRpy(median_translation, median_rpy);
    result->estimator_translation_median = median_translation.norm();
    result->estimator_z_median = median_translation.z();
    result->estimator_yaw_median = median_rpy.z();

    std::vector<double> translation_errors;
    std::vector<double> z_errors;
    std::vector<double> yaw_errors;
    translation_errors.reserve(translations.size());
    z_errors.reserve(translations.size());
    yaw_errors.reserve(translations.size());
    for (std::size_t i = 0; i < translations.size(); ++i) {
        const double translation_error = (translations[i] - median_translation).norm();
        const double z_error = std::abs(translations[i].z() - median_translation.z());
        const double yaw_error = std::abs(normalizeAngle(rpys[i].z() - median_rpy.z()));
        translation_errors.push_back(translation_error);
        z_errors.push_back(z_error);
        yaw_errors.push_back(yaw_error);
        if (translation_error <= config.keyframe_distance_threshold &&
            yaw_error <= config.keyframe_angle_threshold) {
            ++result->estimator_inlier_count;
        }
    }
    result->estimator_inlier_ratio =
        translations.empty() ? 0.0
                             : static_cast<double>(result->estimator_inlier_count) /
                                   static_cast<double>(translations.size());
    result->estimator_translation_mad = median(translation_errors);
    result->estimator_z_mad = median(z_errors);
    result->estimator_yaw_mad = median(yaw_errors);
    result->estimator_valid = result->estimator_inlier_count >= 3 &&
                              result->estimator_inlier_ratio >= 0.75 &&
                              result->estimator_translation_mad <= config.keyframe_distance_threshold &&
                              result->estimator_yaw_mad <= config.keyframe_angle_threshold;
    result->estimator_recommendation =
        result->estimator_valid ? "stable_consensus_measurement"
                                : "unstable_consensus_measurement";

    if (central_measurement) {
        const Eigen::Isometry3d delta =
            central_measurement->inverse() * result->estimator_measurement_match_query;
        result->estimator_measurement_delta_translation = delta.translation().norm();
        result->estimator_measurement_delta_rotation = rotationAngle(delta);
    }
}

std::string consensusReason(LoopConsensusDecision decision,
                            int valid_pair_count,
                            int contradiction_count)
{
    if (decision == LoopConsensusDecision::Commit) {
        return "neighborhood_consensus";
    }
    if (decision == LoopConsensusDecision::Reject) {
        return "neighborhood_contradiction";
    }
    if (valid_pair_count < 4) {
        return "insufficient_support";
    }
    if (contradiction_count > 0) {
        return "unstable_consensus";
    }
    return "weak_consensus";
}

}  // namespace

const char* loopConsensusDecisionName(LoopConsensusDecision decision)
{
    switch (decision) {
        case LoopConsensusDecision::Commit:
            return "commit";
        case LoopConsensusDecision::Defer:
            return "defer";
        case LoopConsensusDecision::Reject:
            return "reject";
        default:
            return "unknown";
    }
}

LoopConsensusVerifier::LoopConsensusVerifier(const Config& config) : config_(config)
{
    std::string error;
    if (!config_.validate(&error)) {
        throw std::invalid_argument("Invalid N3Mapping loop consensus config: " + error);
    }
}

Eigen::Isometry3d LoopConsensusVerifier::predictNeighborTransform(
    const Eigen::Isometry3d& T_world_query,
    const Eigen::Isometry3d& T_world_match,
    const Eigen::Isometry3d& T_world_query_neighbor,
    const Eigen::Isometry3d& T_world_match_neighbor,
    const Eigen::Isometry3d& T_match_query)
{
    const Eigen::Isometry3d T_query_query_neighbor =
        T_world_query.inverse() * T_world_query_neighbor;
    const Eigen::Isometry3d T_match_neighbor_match =
        T_world_match_neighbor.inverse() * T_world_match;
    return T_match_neighbor_match * T_match_query * T_query_query_neighbor;
}

LoopConsensusResult LoopConsensusVerifier::summarizePairs(
    const Config& config,
    const std::vector<LoopConsensusPairEvidence>& pairs)
{
    return summarizePairs(config, pairs, Eigen::Isometry3d::Identity());
}

LoopConsensusResult LoopConsensusVerifier::summarizePairs(
    const Config& config,
    const std::vector<LoopConsensusPairEvidence>& pairs,
    const Eigen::Isometry3d& central_measurement)
{
    LoopConsensusResult result;
    result.pairs = pairs;

    const double translation_threshold = 2.0 * config.keyframe_distance_threshold;
    const double rotation_threshold = 2.0 * config.keyframe_angle_threshold;
    const double contradiction_translation = 4.0 * config.keyframe_distance_threshold;
    const double contradiction_rotation = 4.0 * config.keyframe_angle_threshold;

    std::vector<double> translation_deltas;
    std::vector<double> rotation_deltas;
    int support_count = 0;
    for (const auto& pair : pairs) {
        if (!pair.valid) {
            continue;
        }
        ++result.valid_pair_count;
        if (pair.offset < 0) {
            ++result.left_support_count;
        } else if (pair.offset > 0) {
            ++result.right_support_count;
        }
        if (!pair.converged ||
            !std::isfinite(pair.delta_translation_norm) ||
            !std::isfinite(pair.delta_rotation_norm)) {
            ++result.contradiction_count;
            continue;
        }
        translation_deltas.push_back(pair.delta_translation_norm);
        rotation_deltas.push_back(pair.delta_rotation_norm);
        if (pair.delta_translation_norm <= translation_threshold &&
            pair.delta_rotation_norm <= rotation_threshold) {
            ++support_count;
        }
        if (pair.delta_translation_norm > contradiction_translation ||
            pair.delta_rotation_norm > contradiction_rotation) {
            ++result.contradiction_count;
        }
    }

    result.median_translation_delta = median(translation_deltas);
    result.median_rotation_delta = median(rotation_deltas);
    result.mad_translation_delta = mad(translation_deltas, result.median_translation_delta);
    result.mad_rotation_delta = mad(rotation_deltas, result.median_rotation_delta);

    const bool enough_support = result.valid_pair_count >= 4;
    const bool has_side_support = result.left_support_count > 0 || result.right_support_count > 0;
    const int required_support = static_cast<int>(std::ceil(0.75 * result.valid_pair_count));
    const bool tight_center =
        std::isfinite(result.median_translation_delta) &&
        std::isfinite(result.median_rotation_delta) &&
        result.median_translation_delta <= translation_threshold &&
        result.median_rotation_delta <= rotation_threshold;
    const bool stable_spread =
        (!std::isfinite(result.mad_translation_delta) ||
         result.mad_translation_delta <= config.keyframe_distance_threshold) &&
        (!std::isfinite(result.mad_rotation_delta) ||
         result.mad_rotation_delta <= config.keyframe_angle_threshold);

    if (enough_support && result.contradiction_count > result.valid_pair_count / 2) {
        result.decision = LoopConsensusDecision::Reject;
    } else if (enough_support && has_side_support && support_count >= required_support &&
               tight_center && stable_spread && result.contradiction_count == 0) {
        result.decision = LoopConsensusDecision::Commit;
    } else {
        result.decision = LoopConsensusDecision::Defer;
    }
    result.reason = consensusReason(result.decision,
                                    result.valid_pair_count,
                                    result.contradiction_count);
    estimateConsensusMeasurement(config, pairs, &central_measurement, &result);
    return result;
}

LoopConsensusResult LoopConsensusVerifier::evaluate(const KeyframeManager& keyframes,
                                                    PointCloudMatcher& matcher,
                                                    const VerifiedLoop& central_loop,
                                                    int half_window) const
{
    std::vector<LoopConsensusPairEvidence> pairs;
    if (half_window < 1 || central_loop.query_id < 0 || central_loop.match_id < 0) {
        return summarizePairs(config_, pairs);
    }

    const auto query = keyframes.getKeyframe(central_loop.query_id);
    const auto match = keyframes.getKeyframe(central_loop.match_id);
    if (!query || !match) {
        return summarizePairs(config_, pairs);
    }

    for (int offset = -half_window; offset <= half_window; ++offset) {
        if (offset == 0) {
            continue;
        }
        LoopConsensusPairEvidence evidence;
        evidence.offset = offset;
        evidence.query_neighbor_id = central_loop.query_id + offset;
        evidence.match_neighbor_id = central_loop.match_id + offset;

        const auto query_neighbor = keyframes.getKeyframe(evidence.query_neighbor_id);
        const auto match_neighbor = keyframes.getKeyframe(evidence.match_neighbor_id);
        if (!query_neighbor || !match_neighbor ||
            std::abs(evidence.query_neighbor_id - evidence.match_neighbor_id) <
                config_.sc_num_exclude_recent) {
            evidence.reject_reason = "missing_or_recent_neighbor";
            pairs.push_back(evidence);
            continue;
        }

        auto source = keyframes.buildSubmapInRootFrame(
            evidence.query_neighbor_id, 0, evidence.query_neighbor_id);
        auto target = keyframes.buildSubmapInRootFrame(
            evidence.match_neighbor_id, config_.gicp_submap_size, evidence.match_neighbor_id);
        if (!source || source->size() < 10 || !target || target->size() < 10) {
            evidence.reject_reason = "empty_neighbor_submap";
            pairs.push_back(evidence);
            continue;
        }

        const Eigen::Isometry3d predicted = predictNeighborTransform(
            query->pose_optimized,
            match->pose_optimized,
            query_neighbor->pose_optimized,
            match_neighbor->pose_optimized,
            central_loop.T_measured_match_query);
        const MatchResult match_result = matcher.alignCloud(target, source, predicted);
        evidence.valid = true;
        evidence.converged = match_result.converged;
        evidence.fitness_score = match_result.fitness_score;
        evidence.inlier_ratio = match_result.inlier_ratio;
        if (match_result.converged) {
            const Eigen::Isometry3d delta = predicted.inverse() * match_result.T_target_source;
            evidence.delta_translation_norm = delta.translation().norm();
            evidence.delta_rotation_norm = rotationAngle(delta);
            const Eigen::Isometry3d T_match_neighbor_match =
                match_neighbor->pose_optimized.inverse() * match->pose_optimized;
            const Eigen::Isometry3d T_query_query_neighbor =
                query->pose_optimized.inverse() * query_neighbor->pose_optimized;
            evidence.estimated_match_query =
                T_match_neighbor_match.inverse() *
                match_result.T_target_source *
                T_query_query_neighbor.inverse();
            evidence.has_estimated_measurement = true;
        } else {
            evidence.reject_reason = "icp_not_converged";
        }
        pairs.push_back(evidence);
    }

    return summarizePairs(config_, pairs, central_loop.T_measured_match_query);
}

void assignLoopConsensus(VerifiedLoop* loop, const LoopConsensusResult& result)
{
    if (!loop) {
        return;
    }
    loop->consensus_shadow_decision = loopConsensusDecisionName(result.decision);
    loop->consensus_shadow_reason = result.reason;
    loop->consensus_valid_pair_count = result.valid_pair_count;
    loop->consensus_left_support_count = result.left_support_count;
    loop->consensus_right_support_count = result.right_support_count;
    loop->consensus_contradiction_count = result.contradiction_count;
    loop->consensus_median_translation_delta = result.median_translation_delta;
    loop->consensus_mad_translation_delta = result.mad_translation_delta;
    loop->consensus_median_rotation_delta = result.median_rotation_delta;
    loop->consensus_mad_rotation_delta = result.mad_rotation_delta;
    loop->consensus_estimator_valid = result.estimator_valid;
    loop->consensus_estimator_pair_count = result.estimator_pair_count;
    loop->consensus_estimator_inlier_count = result.estimator_inlier_count;
    loop->consensus_estimator_inlier_ratio = result.estimator_inlier_ratio;
    loop->consensus_estimator_translation_median = result.estimator_translation_median;
    loop->consensus_estimator_z_median = result.estimator_z_median;
    loop->consensus_estimator_yaw_median = result.estimator_yaw_median;
    loop->consensus_estimator_translation_mad = result.estimator_translation_mad;
    loop->consensus_estimator_z_mad = result.estimator_z_mad;
    loop->consensus_estimator_yaw_mad = result.estimator_yaw_mad;
    loop->consensus_estimator_measurement_delta_translation =
        result.estimator_measurement_delta_translation;
    loop->consensus_estimator_measurement_delta_rotation =
        result.estimator_measurement_delta_rotation;
    loop->consensus_estimator_recommendation = result.estimator_recommendation;
}

}  // namespace n3mapping
