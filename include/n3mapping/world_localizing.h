// WorldLocalizing: global relocalization via RHPD + ICP, and tracking
// localization with T_map_odom.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_debug_logger.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

struct RelocResult {
  bool success = false;
  int64_t seed_keyframe_id = -1;
  int64_t support_keyframe_id = -1;
  // Legacy alias for support_keyframe_id.
  int64_t matched_keyframe_id = -1;
  Eigen::Isometry3d pose_in_map = Eigen::Isometry3d::Identity();
  double confidence = 0.0;
  double fitness_score = 0.0;
};

class WorldLocalizing {
public:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  WorldLocalizing(const Config &config, KeyframeManager &keyframe_manager,
                  LoopDetector &loop_detector, PointCloudMatcher &matcher);

  RelocResult relocalize(const PointCloudT::Ptr &cloud,
                         const Eigen::Isometry3d &odom_pose);
  RelocResult trackLocalization(const PointCloudT::Ptr &cloud,
                                const Eigen::Isometry3d &odom_pose);
  bool isRelocalized() const;
  Eigen::Isometry3d getMapToOdomTransform() const;
  void reset();
  void setMapToOdomTransform(const Eigen::Isometry3d &T_map_odom);
  int64_t getLastMatchedKeyframeId() const;
  bool loadLocalizationAtlas(const std::string &map_path,
                             std::string *error = nullptr);
  bool localizationAtlasLoaded() const;

private:
  struct RelocHypothesis {
    int64_t seed_match_id = -1;
    int64_t last_match_id = -1;
    Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
    double cumulative_log_likelihood = 0.0;
    int num_updates = 0;
    int converged_updates = 0;
    double visibility_consistency_sum = 0.0;
    double visibility_evidence_sum = 0.0;
    int visibility_updates = 0;
    bool alive = true;
  };

  struct QueryFrame {
    PointCloudT::Ptr cloud;
    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
  };

  struct CandidatePoseEvaluation {
    MatchResult match;
    int64_t matched_kf_id = -1;
    VisibilityConsistencyResult visibility;
  };

  std::vector<LoopCandidate> searchCandidates(const PointCloudT::Ptr &cloud);
  std::vector<CandidatePoseEvaluation> evaluateCandidatePoses(
      const PointCloudT::Ptr &cloud,
      const PointCloudMatcher::PreparedSource &prepared_cloud,
      const LoopCandidate &candidate);
  void rebuildRelocMapCacheIfNeeded();
  PointCloudT::Ptr buildRelocTargetCloud(int64_t center_id);
  VisibilityConsistencyResult
  evaluatePoseVisibility(const PointCloudT::Ptr &target_cloud,
                         const PointCloudT::Ptr &query_cloud,
                         const Eigen::Isometry3d &T_map_lidar) const;
  void rebuildFrameRHPDIndexIfNeeded();
  void appendFrameRHPDCandidates(const Eigen::VectorXd &query_rhpd,
                                 const Eigen::MatrixXd &query_sc,
                                 std::vector<LoopCandidate> &candidates);
  double computeRelocLogLikelihood(const LoopCandidate &candidate,
                                   const MatchResult &match_result) const;
  double
  computeTrackLogLikelihood(const MatchResult &match_result,
                            const Eigen::Isometry3d &predicted_pose) const;
  PointCloudT::Ptr
  buildRelocQueryCloud(const PointCloudT::Ptr &cloud,
                       const Eigen::Isometry3d &odom_pose,
                       RelocQueryCloudDebugSummary *debug_summary = nullptr);
  PointCloudT::Ptr buildRelocMotionQueryCloudForDebug(
      const Eigen::Isometry3d &odom_pose,
      RelocQueryCloudDebugSummary *debug_summary) const;
  void appendRelocalizationDebug(const RelocalizationDebugEvent &event) const;
  void appendTrackingDebug(const RelocTrackingDebugEvent &event) const;
  void clearRelocHypotheses();
  int64_t findNearestKeyframe(const Eigen::Isometry3d &pose) const;

  Config config_;
  KeyframeManager &keyframe_manager_;
  LoopDetector &loop_detector_;
  PointCloudMatcher &matcher_;
  std::unique_ptr<LocalizationAtlas> localization_atlas_;
  RHPDManager frame_rhpd_manager_;
  size_t frame_rhpd_indexed_keyframes_;
  PointCloudT::Ptr reloc_map_cache_;
  size_t reloc_map_cached_keyframes_;

  bool is_relocalized_;
  Eigen::Isometry3d T_map_odom_;
  int64_t last_matched_id_;
  int64_t relocalization_seed_id_;
  Eigen::Isometry3d last_odom_pose_;
  int consecutive_track_failures_;
  std::vector<RelocHypothesis> pending_hypotheses_;
  std::deque<QueryFrame> query_frame_buffer_;
  int hypothesis_window_count_;
  int64_t last_window_winner_match_id_;
  int winner_streak_;
  uint64_t relocalize_debug_query_index_;
  uint64_t track_debug_query_index_;
  mutable std::mutex debug_mutex_;
  mutable std::mutex mutex_;
};

} // namespace n3mapping
