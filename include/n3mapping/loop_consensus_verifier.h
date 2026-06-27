#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

enum class LoopConsensusDecision {
    Commit,
    Defer,
    Reject
};

const char* loopConsensusDecisionName(LoopConsensusDecision decision);

struct LoopConsensusPairEvidence {
    int64_t query_neighbor_id = -1;
    int64_t match_neighbor_id = -1;
    int offset = 0;
    bool valid = false;
    bool converged = false;
    double fitness_score = std::numeric_limits<double>::quiet_NaN();
    double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
    double delta_translation_norm = std::numeric_limits<double>::quiet_NaN();
    double delta_rotation_norm = std::numeric_limits<double>::quiet_NaN();
    bool has_estimated_measurement = false;
    Eigen::Isometry3d estimated_match_query = Eigen::Isometry3d::Identity();
    std::string reject_reason;
};

struct LoopConsensusResult {
    LoopConsensusDecision decision = LoopConsensusDecision::Defer;
    std::string reason = "not_available";

    int valid_pair_count = 0;
    int left_support_count = 0;
    int right_support_count = 0;
    int contradiction_count = 0;

    double median_translation_delta = std::numeric_limits<double>::quiet_NaN();
    double mad_translation_delta = std::numeric_limits<double>::quiet_NaN();
    double median_rotation_delta = std::numeric_limits<double>::quiet_NaN();
    double mad_rotation_delta = std::numeric_limits<double>::quiet_NaN();

    bool estimator_valid = false;
    int estimator_pair_count = 0;
    int estimator_inlier_count = 0;
    double estimator_inlier_ratio = 0.0;
    Eigen::Isometry3d estimator_measurement_match_query = Eigen::Isometry3d::Identity();
    double estimator_translation_median = std::numeric_limits<double>::quiet_NaN();
    double estimator_z_median = std::numeric_limits<double>::quiet_NaN();
    double estimator_yaw_median = std::numeric_limits<double>::quiet_NaN();
    double estimator_translation_mad = std::numeric_limits<double>::quiet_NaN();
    double estimator_z_mad = std::numeric_limits<double>::quiet_NaN();
    double estimator_yaw_mad = std::numeric_limits<double>::quiet_NaN();
    double estimator_measurement_delta_translation =
        std::numeric_limits<double>::quiet_NaN();
    double estimator_measurement_delta_rotation =
        std::numeric_limits<double>::quiet_NaN();
    std::string estimator_recommendation = "not_available";

    std::vector<LoopConsensusPairEvidence> pairs;
};

class LoopConsensusVerifier {
public:
    explicit LoopConsensusVerifier(const Config& config);

    LoopConsensusResult evaluate(const KeyframeManager& keyframes,
                                 PointCloudMatcher& matcher,
                                 const VerifiedLoop& central_loop,
                                 int half_window = 2) const;

    static Eigen::Isometry3d predictNeighborTransform(const Eigen::Isometry3d& T_world_query,
                                                      const Eigen::Isometry3d& T_world_match,
                                                      const Eigen::Isometry3d& T_world_query_neighbor,
                                                      const Eigen::Isometry3d& T_world_match_neighbor,
                                                      const Eigen::Isometry3d& T_match_query);

    static LoopConsensusResult summarizePairs(const Config& config,
                                              const std::vector<LoopConsensusPairEvidence>& pairs);
    static LoopConsensusResult summarizePairs(const Config& config,
                                              const std::vector<LoopConsensusPairEvidence>& pairs,
                                              const Eigen::Isometry3d& central_measurement);

private:
    Config config_;
};

void assignLoopConsensus(VerifiedLoop* loop, const LoopConsensusResult& result);

}  // namespace n3mapping
