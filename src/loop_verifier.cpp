#include "n3mapping/loop_verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "n3mapping/loop_heightmap_diagnostics.h"

namespace n3mapping {
namespace {

std::string loopRejectReason(bool icp_converged,
                             bool fitness_ok,
                             bool inlier_ok,
                             bool geom_ok)
{
    if (!icp_converged) {
        return "icp_not_converged";
    }
    if (!fitness_ok) {
        return "fitness_threshold";
    }
    if (!inlier_ok) {
        return "inlier_threshold";
    }
    if (!geom_ok) {
        return "geometry_gate";
    }
    return "";
}

double verticalInformationRatio(const Eigen::Matrix<double, 6, 6>& information)
{
    const double x = information(0, 0);
    const double y = information(1, 1);
    const double z = information(2, 2);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        x <= 0.0 || y <= 0.0 || z <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double xy_reference = std::sqrt(x * y);
    return xy_reference > 0.0 ? z / xy_reference : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

LoopVerifier::LoopVerifier(const Config& config) : config_(config)
{
    std::string config_error;
    if (!config_.validate(&config_error)) {
        throw std::invalid_argument("Invalid N3Mapping loop verifier config: " + config_error);
    }
}

Eigen::Isometry3d LoopVerifier::measurementResidual(const Eigen::Isometry3d& predicted_match_query,
                                                    const Eigen::Isometry3d& measured_match_query)
{
    return predicted_match_query.inverse() * measured_match_query;
}

LoopVerification LoopVerifier::verifyPreparedSubmaps(
    const LoopCandidate& candidate,
    const Keyframe::Ptr& query_keyframe,
    const Keyframe::Ptr& match_keyframe,
    const core::LioFrame::PointCloud::Ptr& source_in_match_frame,
    const core::LioFrame::PointCloud::Ptr& target_in_match_frame,
    PointCloudMatcher& matcher) const
{
    LoopVerification verification;
    verification.loop.query_id = candidate.query_id;
    verification.loop.match_id = candidate.match_id;
    verification.loop.candidate_yaw_diff_rad = static_cast<double>(candidate.yaw_diff_rad);

    if (!query_keyframe || !match_keyframe || !source_in_match_frame || source_in_match_frame->empty() ||
        !target_in_match_frame || target_in_match_frame->empty()) {
        verification.reject_reason = "missing_keyframe_or_cloud";
        return verification;
    }

    verification.T_pred_match_query = match_keyframe->pose_optimized.inverse() * query_keyframe->pose_optimized;
    verification.match_result = matcher.alignCloud(
        target_in_match_frame, source_in_match_frame, Eigen::Isometry3d::Identity());
    verification.T_icp_correction_match = verification.match_result.T_target_source;
    verification.T_measured_match_query =
        verification.T_icp_correction_match * verification.T_pred_match_query;
    verification.T_measurement_residual =
        measurementResidual(verification.T_pred_match_query, verification.T_measured_match_query);

    auto& loop = verification.loop;
    loop.T_pred_match_query = verification.T_pred_match_query;
    loop.T_icp_correction_match = verification.T_icp_correction_match;
    loop.T_measured_match_query = verification.T_measured_match_query;
    loop.T_measurement_residual = verification.T_measurement_residual;
    loop.fitness_score = verification.match_result.fitness_score;
    loop.inlier_ratio = verification.match_result.inlier_ratio;
    loop.information = config_.loop_use_icp_information
        ? verification.match_result.information
        : Eigen::Matrix<double, 6, 6>::Identity();
    loop.vertical_information_ratio = verticalInformationRatio(verification.match_result.information);

    const auto predicted_overlap = computeSubmapOverlapConsistency(
        target_in_match_frame, source_in_match_frame, Eigen::Isometry3d::Identity());
    loop.submap_pred_overlap_cell_count = predicted_overlap.overlap_cell_count;
    loop.submap_pred_overlap_ratio = predicted_overlap.overlap_ratio;
    loop.submap_pred_support_ratio = predicted_overlap.support_ratio;
    loop.submap_pred_consistency_score = predicted_overlap.consistency_score;

    verification.fitness_ok = verification.match_result.fitness_score < config_.loop_fitness_threshold;
    verification.inlier_ok = verification.match_result.inlier_ratio >= config_.loop_min_inlier_ratio;
    verification.icp_translation_norm = verification.T_icp_correction_match.translation().norm();
    verification.icp_rotation_norm = Eigen::AngleAxisd(verification.T_icp_correction_match.rotation()).angle();
    verification.geometry_ok =
        verification.icp_translation_norm <= config_.loop_max_icp_translation &&
        verification.icp_rotation_norm <= config_.loop_max_icp_rotation;

    if (verification.match_result.converged) {
        const auto heightmap = computeHeightmapConsistency(
            target_in_match_frame, source_in_match_frame, verification.T_icp_correction_match);
        loop.heightmap_overlap_cell_count = heightmap.overlap_cell_count;
        loop.heightmap_overlap_ratio = heightmap.overlap_ratio;
        loop.heightmap_ground_dz_median = heightmap.ground_dz_median;
        loop.heightmap_ground_dz_p90 = heightmap.ground_dz_p90;
        loop.heightmap_ground_dz_max = heightmap.ground_dz_max;
        loop.heightmap_ground_support_ratio = heightmap.ground_support_ratio;
        loop.heightmap_vertical_consistency_score = heightmap.vertical_consistency_score;

        const auto measured_overlap = computeSubmapOverlapConsistency(
            target_in_match_frame, source_in_match_frame, verification.T_icp_correction_match);
        loop.submap_measured_overlap_cell_count = measured_overlap.overlap_cell_count;
        loop.submap_measured_overlap_ratio = measured_overlap.overlap_ratio;
        loop.submap_measured_support_ratio = measured_overlap.support_ratio;
        loop.submap_measured_consistency_score = measured_overlap.consistency_score;
        loop.submap_overlap_gain =
            loop.submap_measured_consistency_score - loop.submap_pred_consistency_score;
    }

    verification.reject_reason = loopRejectReason(
        verification.match_result.converged, verification.fitness_ok,
        verification.inlier_ok, verification.geometry_ok);

    loop.verified = verification.match_result.converged &&
                    verification.fitness_ok &&
                    verification.inlier_ok &&
                    verification.geometry_ok;
    if (loop.verified) {
        loop.T_match_query = verification.T_measured_match_query;
    }
    return verification;
}

LoopVerification LoopVerifier::verifyKeyframesLegacy(const LoopCandidate& candidate,
                                                     const Keyframe::Ptr& query_keyframe,
                                                     const Keyframe::Ptr& match_keyframe,
                                                     PointCloudMatcher& matcher) const
{
    LoopVerification verification;
    verification.loop.query_id = candidate.query_id;
    verification.loop.match_id = candidate.match_id;
    verification.loop.candidate_yaw_diff_rad = static_cast<double>(candidate.yaw_diff_rad);
    if (!query_keyframe || !match_keyframe) {
        verification.reject_reason = "missing_keyframe_or_cloud";
        return verification;
    }

    verification.T_pred_match_query = match_keyframe->pose_optimized.inverse() * query_keyframe->pose_optimized;
    Eigen::Isometry3d init_guess = verification.T_pred_match_query;
    Eigen::AngleAxisd yaw_correction(candidate.yaw_diff_rad, Eigen::Vector3d::UnitZ());
    init_guess.linear() = init_guess.linear() * yaw_correction.toRotationMatrix();
    verification.match_result = matcher.align(match_keyframe, query_keyframe, init_guess);
    verification.T_measured_match_query = verification.match_result.T_target_source;
    verification.T_icp_correction_match =
        verification.T_measured_match_query * verification.T_pred_match_query.inverse();
    verification.T_measurement_residual =
        measurementResidual(verification.T_pred_match_query, verification.T_measured_match_query);

    auto& loop = verification.loop;
    loop.T_match_query = verification.T_measured_match_query;
    loop.T_pred_match_query = verification.T_pred_match_query;
    loop.T_icp_correction_match = verification.T_icp_correction_match;
    loop.T_measured_match_query = verification.T_measured_match_query;
    loop.T_measurement_residual = verification.T_measurement_residual;
    loop.fitness_score = verification.match_result.fitness_score;
    loop.inlier_ratio = verification.match_result.inlier_ratio;
    loop.information = verification.match_result.information;
    loop.verified = verification.match_result.success;
    verification.reject_reason = loop.verified ? "" : "registration_invalid";
    return verification;
}

}  // namespace n3mapping
