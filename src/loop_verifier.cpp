#include "n3mapping/loop_verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "n3mapping/loop_heightmap_diagnostics.h"

namespace n3mapping {
namespace {

// Only the registration quality decides. `converged` describes the path the
// solver took, not the answer it reached: a run that spends its whole
// iteration budget and lands on an excellent alignment reports converged
// false. The geometry check compares the correction against the prior, which
// is the quantity the loop exists to fix.
std::string loopRejectReason(bool fitness_ok, bool inlier_ok)
{
    if (!fitness_ok) {
        return "fitness_threshold";
    }
    if (!inlier_ok) {
        return "inlier_threshold";
    }
    return "";
}

// How each axis of one registration stood relative to the others, normalised so
// the three factors multiply to one. Returns all-ones when the Hessian has
// nothing usable to say, which leaves the configured noise untouched.
Eigen::Vector3d axisWeights(const Eigen::Matrix<double, 6, 6>& information,
                            int offset, double max_ratio)
{
    Eigen::Vector3d ones = Eigen::Vector3d::Ones();
    Eigen::Vector3d diag;
    for (int i = 0; i < 3; ++i) {
        diag(i) = information(offset + i, offset + i);
        if (!std::isfinite(diag(i)) || diag(i) <= 0.0) {
            return ones;
        }
    }
    const double geo = std::cbrt(diag(0) * diag(1) * diag(2));
    if (!std::isfinite(geo) || geo <= 0.0) {
        return ones;
    }
    const double lo = 1.0 / std::max(1.0, max_ratio);
    const double hi = std::max(1.0, max_ratio);
    Eigen::Vector3d weights;
    for (int i = 0; i < 3; ++i) {
        weights(i) = std::clamp(diag(i) / geo, lo, hi);
    }
    return weights;
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
    // Identity is not "no information" -- it is a metre and a radian of sigma,
    // and it silently displaces the configured fallback because every consumer
    // downstream tests the matrix for zero, not for meaning. With ICP's own
    // information switched off, the configured loop noise is what belongs here.
    // Block order is (translation, rotation), matching addOdometryConstraint;
    // createRobustNoiseModel swaps them for GTSAM.
    if (config_.loop_use_icp_information) {
        loop.information = verification.match_result.information;
    } else {
        const double sigma_xy = config_.loop_noise_position;
        const double sigma_z = config_.loop_noise_position_z > 0.0
                                   ? config_.loop_noise_position_z
                                   : config_.loop_noise_position;
        const double rot_info =
            1.0 / (config_.loop_noise_rotation * config_.loop_noise_rotation);
        // The configured sigmas set how much a loop is trusted overall; the
        // registration's own Hessian says which of its axes deserve more of
        // that trust and which less. The weights multiply to one, so this moves
        // stiffness between axes without adding or removing any.
        Eigen::Vector3d w_pos = Eigen::Vector3d::Ones();
        Eigen::Vector3d w_rot = Eigen::Vector3d::Ones();
        if (config_.loop_axis_weighting_enable) {
            w_pos = axisWeights(verification.match_result.information, 0,
                                config_.loop_axis_weighting_max);
            w_rot = axisWeights(verification.match_result.information, 3,
                                config_.loop_axis_weighting_max);
        }
        loop.information = Eigen::Matrix<double, 6, 6>::Identity();
        loop.information(0, 0) = w_pos(0) / (sigma_xy * sigma_xy);
        loop.information(1, 1) = w_pos(1) / (sigma_xy * sigma_xy);
        loop.information(2, 2) = w_pos(2) / (sigma_z * sigma_z);
        loop.information(3, 3) = w_rot(0) * rot_info;
        loop.information(4, 4) = w_rot(1) * rot_info;
        loop.information(5, 5) = w_rot(2) * rot_info;
        // Reported as an observability, so capped at one: an axis constrained
        // better than its neighbours is not more than fully observable.
        loop.vertical_observability_score = std::min(1.0, w_pos(2));
    }
    loop.vertical_information_ratio = verticalInformationRatio(verification.match_result.information);

    verification.fitness_ok = verification.match_result.fitness_score < config_.loop_fitness_threshold;
    verification.inlier_ok = verification.match_result.inlier_ratio >= config_.loop_min_inlier_ratio;
    verification.icp_translation_norm = verification.T_icp_correction_match.translation().norm();
    verification.icp_rotation_norm = Eigen::AngleAxisd(verification.T_icp_correction_match.rotation()).angle();
    verification.geometry_ok =
        verification.icp_translation_norm <= config_.loop_max_icp_translation &&
        verification.icp_rotation_norm <= config_.loop_max_icp_rotation;

    // Computed for every candidate the registration quality admits, so the
    // referee downstream sees them. Previously this was gated on `converged`,
    // which is why 541 of 588 candidates reached the referee with no evidence
    // attached at all.
    if (verification.fitness_ok && verification.inlier_ok) {
        const auto heightmap = computeHeightmapConsistency(
            target_in_match_frame, source_in_match_frame, verification.T_icp_correction_match);
        loop.heightmap_overlap_cell_count = heightmap.overlap_cell_count;
        loop.heightmap_overlap_ratio = heightmap.overlap_ratio;
        loop.heightmap_ground_dz_median = heightmap.ground_dz_median;
        loop.heightmap_ground_dz_p90 = heightmap.ground_dz_p90;
        loop.heightmap_ground_dz_max = heightmap.ground_dz_max;
        loop.heightmap_ground_support_ratio = heightmap.ground_support_ratio;
        loop.heightmap_vertical_consistency_score = heightmap.vertical_consistency_score;
    }

    verification.reject_reason =
        loopRejectReason(verification.fitness_ok, verification.inlier_ok);

    // icp_translation_norm, icp_rotation_norm and match_result.converged stay
    // populated above and still reach the debug stream; they are evidence
    // about the loop, not grounds to refuse it. A loop that moves the prior by
    // five metres is not suspect -- on a map that has drifted by two and a
    // half, it is the correction.
    loop.verified = verification.fitness_ok && verification.inlier_ok;
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
