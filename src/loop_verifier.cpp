#include "n3mapping/loop_verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "n3mapping/loop_heightmap_diagnostics.h"
#include "n3mapping/n3map_proto_utils.h"

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

void LoopVerifier::finalizeRegistrationEvidence(
    LoopVerification* verification) const
{
    if (!verification) {
        return;
    }

    auto& loop = verification->loop;
    const auto& match = verification->match_result;
    loop.fitness_score = match.fitness_score;
    loop.inlier_ratio = match.inlier_ratio;

    // An ICP Hessian is theoretically symmetric, but the numerical result is
    // not guaranteed to be bit-for-bit symmetric. Canonicalize it before it can
    // enter GTSAM or the serialized map. The default product path deliberately
    // uses configured loop noise instead of this Hessian.
    const Eigen::Matrix<double, 6, 6> symmetric_icp_information =
        0.5 * (match.information + match.information.transpose());
    std::string icp_information_error;
    const bool icp_information_valid =
        isValidInformationMatrix(symmetric_icp_information,
                                 &icp_information_error);
    const bool use_icp_information =
        config_.loop_use_icp_information && icp_information_valid;
    if (use_icp_information) {
        loop.information = symmetric_icp_information;
    } else {
        const double sigma_xy = config_.loop_noise_position;
        const double sigma_z = config_.loop_noise_position_z > 0.0
                                   ? config_.loop_noise_position_z
                                   : config_.loop_noise_position;
        const double rot_info =
            1.0 / (config_.loop_noise_rotation * config_.loop_noise_rotation);

        Eigen::Vector3d w_pos = Eigen::Vector3d::Ones();
        Eigen::Vector3d w_rot = Eigen::Vector3d::Ones();
        if (config_.loop_axis_weighting_enable && icp_information_valid) {
            w_pos = axisWeights(symmetric_icp_information, 0,
                                config_.loop_axis_weighting_max);
            w_rot = axisWeights(symmetric_icp_information, 3,
                                config_.loop_axis_weighting_max);
        }
        loop.information = Eigen::Matrix<double, 6, 6>::Identity();
        loop.information(0, 0) = w_pos(0) / (sigma_xy * sigma_xy);
        loop.information(1, 1) = w_pos(1) / (sigma_xy * sigma_xy);
        loop.information(2, 2) = w_pos(2) / (sigma_z * sigma_z);
        loop.information(3, 3) = w_rot(0) * rot_info;
        loop.information(4, 4) = w_rot(1) * rot_info;
        loop.information(5, 5) = w_rot(2) * rot_info;
        loop.vertical_observability_score = std::min(1.0, w_pos(2));
    }
    loop.vertical_information_ratio =
        verticalInformationRatio(symmetric_icp_information);

    verification->fitness_ok =
        match.fitness_score < config_.loop_fitness_threshold;
    verification->inlier_ok =
        match.inlier_ratio >= config_.loop_min_inlier_ratio;
    verification->icp_translation_norm =
        verification->T_icp_correction_match.translation().norm();
    verification->icp_rotation_norm = Eigen::AngleAxisd(
        verification->T_icp_correction_match.rotation()).angle();
    verification->geometry_ok =
        verification->icp_translation_norm <= config_.loop_max_icp_translation &&
        verification->icp_rotation_norm <= config_.loop_max_icp_rotation;
    verification->reject_reason =
        loopRejectReason(verification->fitness_ok, verification->inlier_ok);
    loop.verified = verification->fitness_ok && verification->inlier_ok;
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
    if (!query_keyframe || !match_keyframe || !source_in_match_frame || source_in_match_frame->empty() ||
        !target_in_match_frame || target_in_match_frame->empty()) {
        LoopVerification verification;
        verification.loop.query_id = candidate.query_id;
        verification.loop.match_id = candidate.match_id;
        verification.reject_reason = "missing_keyframe_or_cloud";
        return verification;
    }

    const Eigen::Isometry3d predicted_match_query =
        match_keyframe->pose_optimized.inverse() * query_keyframe->pose_optimized;
    const MatchResult match_result = matcher.alignCloud(
        target_in_match_frame, source_in_match_frame, Eigen::Isometry3d::Identity());
    const Eigen::Isometry3d measured_match_query =
        match_result.T_target_source * predicted_match_query;
    return finalizePreparedRegistration(
        candidate, query_keyframe, match_keyframe, source_in_match_frame,
        target_in_match_frame, match_result, measured_match_query,
        match_result.T_target_source);
}

LoopVerification LoopVerifier::verifyPreparedQueryToMatch(
    const LoopCandidate& candidate,
    const Keyframe::Ptr& query_keyframe,
    const Keyframe::Ptr& match_keyframe,
    const core::LioFrame::PointCloud::Ptr& source_in_query_frame,
    const core::LioFrame::PointCloud::Ptr& target_in_match_frame,
    const Eigen::Isometry3d& initial_match_query,
    PointCloudMatcher& matcher) const
{
    if (!query_keyframe || !match_keyframe || !source_in_query_frame ||
        source_in_query_frame->empty() || !target_in_match_frame ||
        target_in_match_frame->empty()) {
        LoopVerification verification;
        verification.loop.query_id = candidate.query_id;
        verification.loop.match_id = candidate.match_id;
        verification.reject_reason = "missing_keyframe_or_cloud";
        return verification;
    }

    const MatchResult match_result = matcher.alignCloud(
        target_in_match_frame, source_in_query_frame, initial_match_query);
    return finalizePreparedRegistration(
        candidate, query_keyframe, match_keyframe, source_in_query_frame,
        target_in_match_frame, match_result, match_result.T_target_source,
        match_result.T_target_source);
}

LoopVerification LoopVerifier::finalizePreparedRegistration(
    const LoopCandidate& candidate,
    const Keyframe::Ptr& query_keyframe,
    const Keyframe::Ptr& match_keyframe,
    const core::LioFrame::PointCloud::Ptr& source_registration_cloud,
    const core::LioFrame::PointCloud::Ptr& target_registration_cloud,
    const MatchResult& match_result,
    const Eigen::Isometry3d& measured_match_query,
    const Eigen::Isometry3d& cloud_alignment) const
{
    LoopVerification verification;
    verification.loop.query_id = candidate.query_id;
    verification.loop.match_id = candidate.match_id;
    verification.loop.candidate_yaw_diff_rad =
        static_cast<double>(candidate.yaw_diff_rad);
    verification.T_pred_match_query =
        match_keyframe->pose_optimized.inverse() * query_keyframe->pose_optimized;
    verification.match_result = match_result;
    verification.T_measured_match_query = measured_match_query;
    verification.T_icp_correction_match =
        measured_match_query * verification.T_pred_match_query.inverse();
    verification.T_measurement_residual = measurementResidual(
        verification.T_pred_match_query, verification.T_measured_match_query);

    auto& loop = verification.loop;
    loop.T_pred_match_query = verification.T_pred_match_query;
    loop.T_icp_correction_match = verification.T_icp_correction_match;
    loop.T_measured_match_query = verification.T_measured_match_query;
    loop.T_measurement_residual = verification.T_measurement_residual;
    finalizeRegistrationEvidence(&verification);

    // Computed for every candidate the registration quality admits, so the
    // referee downstream sees them. Previously this was gated on `converged`,
    // which is why 541 of 588 candidates reached the referee with no evidence
    // attached at all.
    if (verification.fitness_ok && verification.inlier_ok) {
        const auto heightmap = computeHeightmapConsistency(
            target_registration_cloud, source_registration_cloud,
            cloud_alignment);
        loop.heightmap_overlap_cell_count = heightmap.overlap_cell_count;
        loop.heightmap_overlap_ratio = heightmap.overlap_ratio;
        loop.heightmap_ground_dz_median = heightmap.ground_dz_median;
        loop.heightmap_ground_dz_p90 = heightmap.ground_dz_p90;
        loop.heightmap_ground_dz_max = heightmap.ground_dz_max;
        loop.heightmap_ground_support_ratio = heightmap.ground_support_ratio;
        loop.heightmap_vertical_consistency_score = heightmap.vertical_consistency_score;
    }

    // icp_translation_norm, icp_rotation_norm and match_result.converged stay
    // populated above and still reach the debug stream; they are evidence
    // about the loop, not grounds to refuse it. A loop that moves the prior by
    // five metres is not suspect -- on a map that has drifted by two and a
    // half, it is the correction.
    if (loop.verified) {
        loop.T_match_query = verification.T_measured_match_query;
    }
    return verification;
}

}  // namespace n3mapping
