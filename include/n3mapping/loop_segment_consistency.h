#pragma once

#include <limits>
#include <string>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {

struct LoopSegmentConsistencyDiagnostics {
    int pair_count = 0;
    int valid_pair_count = 0;
    int consensus_inlier_count = 0;
    double consensus_ratio = 0.0;
    double translation_median = std::numeric_limits<double>::quiet_NaN();
    double translation_std = std::numeric_limits<double>::quiet_NaN();
    double yaw_median = std::numeric_limits<double>::quiet_NaN();
    double yaw_std = std::numeric_limits<double>::quiet_NaN();
    double z_std = std::numeric_limits<double>::quiet_NaN();
    double roll_pitch_std = std::numeric_limits<double>::quiet_NaN();
    std::string direction = "not_available";
    std::string recommendation = "insufficient_support";
};

LoopSegmentConsistencyDiagnostics computeLoopSegmentConsistency(
    const Config& config,
    const KeyframeManager& keyframe_manager,
    const VerifiedLoop& loop,
    int half_window = 2);

LoopSegmentConsistencyDiagnostics computeLoopCandidateSegmentConsistency(
    const Config& config,
    const KeyframeManager& keyframe_manager,
    const LoopCandidate& candidate,
    int half_window = 2);

void assignLoopSegmentConsistency(
    VerifiedLoop* loop,
    const LoopSegmentConsistencyDiagnostics& diagnostics);

}  // namespace n3mapping
