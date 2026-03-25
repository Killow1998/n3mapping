// WorldLocalizing: global relocalization via ScanContext + ICP, and tracking localization with T_map_odom.
#pragma once

#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

struct RelocResult {
    bool success = false;
    int64_t matched_keyframe_id = -1;
    Eigen::Isometry3d pose_in_map = Eigen::Isometry3d::Identity();
    double confidence = 0.0;
    double fitness_score = 0.0;
};

class WorldLocalizing {
public:
    using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

    WorldLocalizing(const Config& config, KeyframeManager& keyframe_manager,
                    LoopDetector& loop_detector, PointCloudMatcher& matcher);

    RelocResult relocalize(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);
    RelocResult trackLocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);
    bool isRelocalized() const;
    Eigen::Isometry3d getMapToOdomTransform() const;
    void reset();
    void setMapToOdomTransform(const Eigen::Isometry3d& T_map_odom);
    int64_t getLastMatchedKeyframeId() const;

private:
    struct RelocHypothesis {
        int64_t seed_match_id = -1;
        int64_t last_match_id = -1;
        Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
        double cumulative_log_likelihood = 0.0;
        int num_updates = 0;
        bool alive = true;
    };

    std::vector<LoopCandidate> searchCandidates(const PointCloudT::Ptr& cloud);
    RelocResult verifyCandidates(const PointCloudT::Ptr& cloud, const std::vector<LoopCandidate>& candidates);
    bool evaluateSingleCandidate(const PointCloudT::Ptr& cloud,
                                 const LoopCandidate& candidate,
                                 MatchResult& best_match,
                                 int64_t& matched_kf_id);
    double computeRelocLogLikelihood(const LoopCandidate& candidate, const MatchResult& match_result) const;
    double computeTrackLogLikelihood(const MatchResult& match_result,
                                     const Eigen::Isometry3d& predicted_pose) const;
    void clearRelocHypotheses();
    int64_t findNearestKeyframe(const Eigen::Isometry3d& pose) const;

    Config config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    PointCloudMatcher& matcher_;

    bool is_relocalized_;
    Eigen::Isometry3d T_map_odom_;
    int64_t last_matched_id_;
    Eigen::Isometry3d last_odom_pose_;
    int consecutive_track_failures_;
    std::vector<RelocHypothesis> pending_hypotheses_;
    int hypothesis_window_count_;
    mutable std::mutex mutex_;
};

} // namespace n3mapping
