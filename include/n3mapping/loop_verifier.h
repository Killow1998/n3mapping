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

    LoopVerification verifyKeyframesLegacy(const LoopCandidate& candidate,
                                           const Keyframe::Ptr& query_keyframe,
                                           const Keyframe::Ptr& match_keyframe,
                                           PointCloudMatcher& matcher) const;

    static Eigen::Isometry3d measurementResidual(const Eigen::Isometry3d& predicted_match_query,
                                                 const Eigen::Isometry3d& measured_match_query);

private:
    Config config_;
};

}  // namespace n3mapping
