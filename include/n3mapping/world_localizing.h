// WorldLocalizing: global relocalization via RHPD + ICP, and tracking
// localization with T_map_odom.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/free_space_grid.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_candidate_evaluator.h"
#include "n3mapping/relocalization_debug_logger.h"
#include "n3mapping/relocalization_place_index.h"
#include "n3mapping/relocalization_query_builder.h"
#include "n3mapping/relocalization_state.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

struct RelocResult {
  bool success = false;
  RelocalizationState state = RelocalizationState::SEARCHING;
  PoseSource pose_source = PoseSource::NONE;
  // Machine-readable terminal or pending decision for this relocalize() call.
  // Examples: no_candidates, temporal_window_pending, log_likelihood,
  // accepted. This is evidence output; it does not participate in decisions.
  std::string decision = "not_attempted";
  int64_t seed_keyframe_id = -1;
  int64_t support_keyframe_id = -1;
  // Legacy alias for support_keyframe_id.
  int64_t matched_keyframe_id = -1;
  Eigen::Isometry3d pose_in_map = Eigen::Isometry3d::Identity();
  double confidence = 0.0;
  double fitness_score = 0.0;
};

// Offline-only registration evidence. The oracle pose is never used by the
// runtime relocalization path; this result exists to locate which stage fails.
struct RegistrationSeedProbeAttempt {
  std::string seed_kind;
  int64_t seed_keyframe_id = -1;
  double yaw_offset_rad = 0.0;
  Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
  MatchResult match;
  bool fitness_pass = false;
  bool inlier_pass = false;
  double derived_confidence = 0.0;
  bool confidence_pass = false;
  bool production_quality_pass = false;
  VisibilityConsistencyResult initial_visibility;
  VisibilityConsistencyResult refined_visibility;
  bool production_kept_initial_pose = false;
  Eigen::Isometry3d production_pose = Eigen::Isometry3d::Identity();
};

struct RegistrationSeedProbeResult {
  bool valid = false;
  std::string error;
  int64_t oracle_nearest_keyframe_id = -1;
  int64_t descriptor_candidate_keyframe_id = -1;
  std::vector<RegistrationSeedProbeAttempt> attempts;
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
  // Strict local tracking used while extending a loaded map. Registration
  // targets and visibility evidence come only from immutable loaded-map
  // keyframes; corrections are bounded and applied without lag. A frame with
  // no geometric evidence fails closed instead of silently accepting odometry.
  RelocResult trackLoadedMap(const PointCloudT::Ptr &cloud,
                             const Eigen::Isometry3d &odom_pose);
  bool isRelocalized() const;
  Eigen::Isometry3d getMapToOdomTransform() const;
  void reset();
  // WP-02: 地图替换时旧地图派生的缓存不得泄漏到新地图。
  // resetLocalizationState() 只清运行状态（假设、锁定、窗口、buffer、
  // streak/persistence），保留地图派生缓存 —— 单地图内重置后可复用缓存。
  // notifyMapReplaced() = resetLocalizationState() + 清全部地图派生缓存
  // （frame RHPD 索引、reloc map cache、free-space grid 及失败锁存、atlas），
  // 用于"加载成功一张新地图"之后。reset() 保持为兼容入口（= state reset）。
  void resetLocalizationState();
  void notifyMapReplaced();
  // 只读诊断：外部测试可观察缓存状态，但不允许修改内部状态。
  struct WorldLocalizingCacheDiagnostics {
    bool atlas_loaded = false;
    std::size_t frame_rhpd_indexed_keyframes = 0;
    std::size_t reloc_map_cached_keyframes = 0;
    std::size_t free_space_grid_keyframes = 0;
    bool free_space_grid_valid = false;
    bool free_space_grid_failed = false;
  };
  WorldLocalizingCacheDiagnostics cacheDiagnostics() const;
  void setMapToOdomTransform(const Eigen::Isometry3d &T_map_odom);
  int64_t getLastMatchedKeyframeId() const;
  bool loadLocalizationAtlas(const std::string &map_path,
                             std::string *error = nullptr);
  bool localizationAtlasLoaded() const;
  RegistrationSeedProbeResult
  probeRegistrationSeeds(const PointCloudT::Ptr &cloud,
                         const Eigen::Isometry3d &odom_pose,
                         const Eigen::Isometry3d &oracle_pose);
  // Cross-session constraints must explain the current scan in the immutable
  // loaded map, not in keyframes added by the extension they are judging.
  VisibilityConsistencyResult evaluateLoadedMapPoseVisibility(
      const PointCloudT::Ptr &query_cloud,
      const Eigen::Isometry3d &T_map_lidar);

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
    double last_rot_info_min = 0.0;
    double last_trans_info_min = 0.0;
    int last_iterations = 0;
    // Diagnostics only: which pose the recorded registration numbers
    // describe, and whether that registration would have passed the gate
    // used when a hypothesis is first created.
    bool last_pose_is_refined = false;
    bool last_production_quality = false;
    double last_rot_info_marginal_min = 0.0;
    int last_termination = 0;
    double last_inlier_ratio = 0.0;
    double last_fitness = 0.0;
  };

  struct RelocMatchQuality {
    bool fitness_pass = false;
    bool inlier_pass = false;
    double confidence = 0.0;
    bool confidence_pass = false;
    bool accepted = false;
  };

  RelocMatchQuality evaluateRelocMatchQuality(const MatchResult &match) const;

  void rebuildRelocMapCacheIfNeeded();
  void rebuildLoadedMapVisibilityCacheIfNeeded();
  void rebuildFreeSpaceGridIfNeeded();
  void killFreeSpaceDominatedHypotheses(const PointCloudT::Ptr &query_cloud,
                                        const Eigen::Isometry3d &odom_pose);
  const RelocHypothesis *freeSpaceBestHypothesis(
      const PointCloudT::Ptr &query_cloud, const Eigen::Isometry3d &odom_pose);
  PointCloudT::Ptr buildRelocTargetCloud(int64_t center_id);
  VisibilityConsistencyResult
  evaluatePoseVisibility(const PointCloudT::Ptr &target_cloud,
                         const PointCloudT::Ptr &query_cloud,
                         const Eigen::Isometry3d &T_map_lidar) const;
  double computeRelocLogLikelihood(const LoopCandidate &candidate,
                                   const MatchResult &match_result) const;
  double
  computeTrackLogLikelihood(const MatchResult &match_result,
                            const Eigen::Isometry3d &predicted_pose) const;
  void appendRelocalizationDebug(const RelocalizationDebugEvent &event) const;
  void appendTrackingDebug(const RelocTrackingDebugEvent &event) const;
  void clearRelocHypotheses();
  RelocResult trackLocalizationImpl(const PointCloudT::Ptr &cloud,
                                    const Eigen::Isometry3d &odom_pose,
                                    bool strict_loaded_map);
  int64_t findNearestKeyframe(const Eigen::Isometry3d &pose) const;
  int64_t findNearestLoadedKeyframe(const Eigen::Isometry3d &pose) const;

  Config config_;
  KeyframeManager &keyframe_manager_;
  LoopDetector &loop_detector_;
  PointCloudMatcher &matcher_;
  RelocalizationQueryBuilder query_builder_;
  RelocalizationPlaceIndex place_index_;
  std::unique_ptr<LocalizationAtlas> localization_atlas_;
  RelocalizationCandidateEvaluator candidate_evaluator_;
  PointCloudT::Ptr reloc_map_cache_;
  size_t reloc_map_cached_keyframes_;
  KeyframeMapRevision reloc_map_revision_;
  PointCloudT::Ptr loaded_map_visibility_cache_;
  size_t loaded_map_visibility_cached_keyframes_ = 0;
  KeyframeMapRevision loaded_map_visibility_revision_;
  // Frames the current hypothesis set has survived across rejected
  // windows. Only a valve: a set that never resolves must not wedge the
  // episode forever.
  int hypothesis_persist_frames_ = 0;
  FreeSpaceGrid free_space_grid_;
  size_t free_space_grid_keyframes_ = 0;
  KeyframeMapRevision free_space_grid_revision_;
  bool free_space_grid_failed_ = false;

  bool is_relocalized_;
  Eigen::Isometry3d T_map_odom_;
  int64_t last_matched_id_;
  int64_t relocalization_seed_id_;
  Eigen::Isometry3d last_odom_pose_;
  int consecutive_track_failures_;
  std::vector<RelocHypothesis> pending_hypotheses_;
  int hypothesis_window_count_;
  Eigen::Isometry3d hypothesis_window_start_odom_pose_;
  bool has_last_window_winner_transform_;
  Eigen::Isometry3d last_window_winner_map_odom_;
  int winner_streak_;
  uint64_t relocalize_debug_query_index_;
  uint64_t track_debug_query_index_;
  mutable std::mutex debug_mutex_;
  mutable std::mutex mutex_;
};

} // namespace n3mapping
