#pragma once

#include <string>
#include <limits>

#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/core/types.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

struct LoopVerification {
    VerifiedLoop loop;
    MatchResult match_result;
    std::string reject_reason;
    bool fitness_ok = false;
    bool inlier_ok = false;
    bool geometry_ok = false;
    double icp_translation_norm = std::numeric_limits<double>::quiet_NaN();
    double icp_rotation_norm = std::numeric_limits<double>::quiet_NaN();
    Eigen::Isometry3d T_pred_match_query = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_icp_correction_match = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_measured_match_query = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_measurement_residual = Eigen::Isometry3d::Identity();
};

class LoopVerifier {
public:
    explicit LoopVerifier(const Config& config);

    LoopVerification verifyPreparedSubmaps(const LoopCandidate& candidate,
                                           const Keyframe::Ptr& query_keyframe,
                                           const Keyframe::Ptr& match_keyframe,
                                           const core::LioFrame::PointCloud::Ptr& source_in_match_frame,
                                           const core::LioFrame::PointCloud::Ptr& target_in_match_frame,
                                           PointCloudMatcher& matcher) const;

    LoopVerification verifyPreparedQueryToMatch(
        const LoopCandidate& candidate,
        const Keyframe::Ptr& query_keyframe,
        const Keyframe::Ptr& match_keyframe,
        const core::LioFrame::PointCloud::Ptr& source_in_query_frame,
        const core::LioFrame::PointCloud::Ptr& target_in_match_frame,
        const Eigen::Isometry3d& initial_match_query,
        PointCloudMatcher& matcher) const;

    // Compatibility test helper. Production loop paths use prepared submaps
    // through LoopVerificationPipeline so Mapping and Map Extension share the
    // same evidence and constraint gates.
    LoopVerification verifyKeyframesLegacy(const LoopCandidate& candidate,
                                           const Keyframe::Ptr& query_keyframe,
                                           const Keyframe::Ptr& match_keyframe,
                                           PointCloudMatcher& matcher) const;

    static Eigen::Isometry3d measurementResidual(const Eigen::Isometry3d& predicted_match_query,
                                                 const Eigen::Isometry3d& measured_match_query);

private:
    // Both the prepared-submap path and the legacy keyframe path must attach
    // identical quality evidence and information semantics to a registration.
    // Keeping this in one place prevents Map Extension from silently bypassing
    // loop_use_icp_information and the product loop thresholds.
    void finalizeRegistrationEvidence(LoopVerification* verification) const;
    LoopVerification finalizePreparedRegistration(
        const LoopCandidate& candidate,
        const Keyframe::Ptr& query_keyframe,
        const Keyframe::Ptr& match_keyframe,
        const core::LioFrame::PointCloud::Ptr& source_registration_cloud,
        const core::LioFrame::PointCloud::Ptr& target_registration_cloud,
        const MatchResult& match_result,
        const Eigen::Isometry3d& measured_match_query,
        const Eigen::Isometry3d& cloud_alignment) const;

    Config config_;
};

}  // namespace n3mapping
