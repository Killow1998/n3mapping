#pragma once
#include "n3mapping/relocalization_performance.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/registration_probe.h"
#include "n3mapping/free_space_grid.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_candidate_evaluator.h"
#include "n3mapping/relocalization_debug_emitter.h"
#include "n3mapping/relocalization_decision_policy.h"
#include "n3mapping/relocalization_hypothesis_manager.h"
#include "n3mapping/relocalization_quality.h"
#include "n3mapping/relocalization_place_index.h"
#include "n3mapping/relocalization_query_builder.h"
#include "n3mapping/relocalization_state.h"
#include "n3mapping/relocalization_target_provider.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

class TrackingTargetCache;
struct RelocalizationSeedPlan;
struct RelocalizationObservation;
namespace test { class FreeSpaceVetoTest; }

struct RelocResult {
  RelocalizationPerformance performance;
  TrackingPerformance tracking_performance;
  bool success = false;
  RelocalizationState state = RelocalizationState::SEARCHING;
  PoseSource pose_source = PoseSource::NONE;
  WeakTranslationAxes weak_translation_axes;

  std::string decision = "not_attempted";
  int64_t seed_keyframe_id = -1;
  int64_t support_keyframe_id = -1;

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
  ~WorldLocalizing();

  RelocResult
  relocalize(const PointCloudT::Ptr &cloud, const Eigen::Isometry3d &odom_pose,
             double query_timestamp = std::numeric_limits<double>::quiet_NaN());
  RelocResult trackLocalization(const PointCloudT::Ptr &cloud,
                                const Eigen::Isometry3d &odom_pose);

  RelocResult trackLoadedMap(const PointCloudT::Ptr &cloud,
                             const Eigen::Isometry3d &odom_pose);
  bool isRelocalized() const;
  Eigen::Isometry3d getMapToOdomTransform() const;
  void reset();

  void resetLocalizationState();
  void notifyMapReplaced();

  struct WorldLocalizingCacheDiagnostics {
    bool atlas_loaded = false;
    std::size_t frame_rhpd_indexed_keyframes = 0;
    std::size_t reloc_map_cached_keyframes = 0;
    std::size_t free_space_grid_keyframes = 0;
    bool free_space_grid_valid = false;
    bool free_space_grid_failed = false;
    std::size_t reloc_target_cache_hits = 0;
    std::size_t reloc_target_cache_misses = 0;
    std::size_t reloc_target_cache_bytes = 0;
    std::size_t reloc_target_cache_entries = 0;
    std::size_t localization_target_cache_hits = 0;
    std::size_t localization_target_cache_misses = 0;
    std::size_t localization_target_cache_bytes = 0;
    std::size_t localization_target_cache_entries = 0;
    std::size_t loaded_map_target_cache_hits = 0;
    std::size_t loaded_map_target_cache_misses = 0;
    std::size_t loaded_map_target_cache_entries = 0;
  };
  WorldLocalizingCacheDiagnostics cacheDiagnostics() const;
  void setMapToOdomTransform(const Eigen::Isometry3d &T_map_odom);
  void installOptimizedReference(LocalizationAtlas& atlas,
                                 const Eigen::Isometry3d& pose_correction);
  int64_t getLastMatchedKeyframeId() const;
  bool loadLocalizationAtlas(const std::string &map_path,
                             std::string *error = nullptr,
                             LoopDetector* restore_descriptors = nullptr);
  bool localizationAtlasLoaded() const;
  bool compactLoadedKeyframeClouds();
  PointCloudT::ConstPtr atlasGlobalMap() const;
  std::string atlasSourceMapSha256() const;
  RegistrationSeedProbeResult
  probeRegistrationSeeds(const PointCloudT::Ptr &cloud,
                         const Eigen::Isometry3d &odom_pose,
                         const Eigen::Isometry3d &oracle_pose);

  LoadedMapVisibilityEvaluator snapshotLoadedMapVisibility() const;

  void warmLoadedMapTrackingTarget(int64_t anchor_id);

private:
  friend class test::FreeSpaceVetoTest;

  std::map<int64_t, std::vector<std::uint32_t>> reference_atlas_indices_;
  using RelocHypothesis = RelocalizationHypothesis;

  void appendTrackingRecovery(
      const Eigen::Isometry3d& odom_pose, RelocalizationObservation& observation,
      RelocalizationPerformance& cost, std::vector<RelocHypothesis>& hypotheses);

  void updateRelocalizationCandidates(
      const PointCloudT::Ptr& cloud, const std::vector<LoopCandidate>& candidates,
      const Eigen::Isometry3d& odom_pose, RelocalizationObservation& observation,
      RelocalizationPerformance& cost, RelocalizationDebugEvent* event);

  std::vector<const RelocHypothesis*> confirmRelocalization(
      const PointCloudT::Ptr& cloud,
      const Eigen::Isometry3d& odom_pose, RelocalizationObservation& observation,
      RelocalizationPerformance& cost, bool debug_enabled);

  std::vector<RelocHypothesis> seedRelocalization(
      RelocalizationSeedPlan plan,
      const PointCloudMatcher::PreparedSource& prepared_query,
      const PreparedVisibilityObservation& visibility_observation,
      const Eigen::Isometry3d& odom_pose, RelocalizationPerformance& cost,
      RelocalizationDebugEvent* event);

  void rebuildRelocMapCacheIfNeeded();
  void rebuildFreeSpaceGridIfNeeded();
  void clearFreeSpaceReference();
  void killFreeSpaceDominatedHypotheses(const PointCloudT::Ptr &query_cloud,
                                        const Eigen::Isometry3d &odom_pose);
  const RelocHypothesis *
  freeSpaceBestHypothesis(const PointCloudT::Ptr &query_cloud,
                          const Eigen::Isometry3d &odom_pose,
                          const RelocHypothesis &ranked_winner,
                          FreeSpaceGrid::Score& ranked_score);
  PointCloudT::Ptr buildRelocTargetCloud(int64_t center_id);
  RelocTargetRequest makeRelocTargetRequest(int64_t center_id);
  void releaseSearchPreparation();
  double computeRelocLogLikelihood(const LoopCandidate &candidate,
                                   const MatchResult &match_result) const;
  double
  computeTrackLogLikelihood(const MatchResult &match_result,
                            const Eigen::Isometry3d &predicted_pose) const;
  RelocResult trackLocalizationImpl(const PointCloudT::Ptr &cloud,
                                    const Eigen::Isometry3d &odom_pose,
                                    bool strict_loaded_map);
  void resetTrackingStateLocked();
  void recoverWeakTranslation(
      const PointCloudMatcher::PreparedTarget& target, const PointCloudMatcher::PreparedSource& prepared_source,
      const Eigen::Isometry3d& predicted_pose, const std::optional<Eigen::Vector3d>& previous_proposal,
      MatchResult& match_result, WeakTranslationAxes& weak_axes, Eigen::Vector3d& confirmed_shift);
  MatchResult registerTracking(
      const PointCloudMatcher::PreparedTarget* target,
      const PointCloudMatcher::PreparedSource& source,
      const PointCloudT& query_cloud,
      const Eigen::Isometry3d& prediction, const Eigen::Isometry3d& odom_pose,
      bool strict_loaded_map,
      RelocTrackingDebugEvent& diagnostics, WeakTranslationAxes& weak_axes,
      const std::optional<Eigen::Vector3d>& previous_proposal, Eigen::Vector3d& confirmed_shift);
  int64_t findNearestKeyframe(const Eigen::Isometry3d &pose,
                              bool loaded_only = false) const;

  Config config_;
  KeyframeManager &keyframe_manager_;
  PointCloudMatcher &matcher_;
  RelocalizationQueryBuilder query_builder_;
  RelocalizationPlaceIndex place_index_;
  std::size_t search_offset_ = 0;
  int relocalization_queries_ = 0;
  Eigen::Isometry3d search_anchor_ = Eigen::Isometry3d::Identity();
  std::unique_ptr<LocalizationAtlas> localization_atlas_;
  KeyframeMapRevision loaded_atlas_revision_;
  std::unique_ptr<RelocTargetProvider> target_provider_;
  RelocalizationCandidateEvaluator candidate_evaluator_;
  RelocalizationHypothesisManager hypothesis_manager_;
  RelocalizationDecisionPolicy decision_policy_;
  RelocalizationDebugEmitter debug_emitter_;
  PointCloudT::Ptr reloc_map_cache_;
  size_t reloc_map_cached_keyframes_;
  KeyframeMapRevision reloc_map_revision_;
  std::unique_ptr<TrackingTargetCache> tracking_targets_;
  FreeSpaceGrid free_space_grid_;
  size_t free_space_grid_keyframes_ = 0;
  KeyframeMapRevision free_space_grid_revision_;
  bool free_space_grid_failed_ = false;

  bool is_relocalized_;

  bool has_ever_relocalized_;
  Eigen::Isometry3d T_map_odom_;
  Eigen::Vector3d remaining_map_shift_ = Eigen::Vector3d::Zero();
  std::optional<Eigen::Vector3d> directional_proposal_;
  int64_t last_matched_id_;
  int64_t relocalization_seed_id_;
  Eigen::Isometry3d last_odom_pose_;
  int consecutive_track_failures_;
  mutable std::mutex mutex_;
};

}
