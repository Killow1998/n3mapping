// Localization lifecycle and public entry points. Search, tracking, map
// evidence and offline probes are implemented in separate translation units.
#include "n3mapping/world_localizing.h"
#include "n3mapping/tracking_target_cache.h"

#include <cmath>
#include <glog/logging.h>

namespace n3mapping {
WorldLocalizing::WorldLocalizing(const Config &config,
                                 KeyframeManager &keyframe_manager,
                                 LoopDetector &loop_detector,
                                 PointCloudMatcher &matcher)
    : config_(config), keyframe_manager_(keyframe_manager),
      matcher_(matcher),
      local_map_selector_(keyframe_manager_), query_builder_(config_),
      place_index_(config_, keyframe_manager_, loop_detector),
      localization_atlas_(
          std::make_unique<LocalizationAtlas>(config_, matcher_)),
      target_provider_(
          makeRelocTargetProvider(config_, matcher_, *localization_atlas_)),
      candidate_evaluator_(config_, keyframe_manager_, matcher_,
                           *target_provider_),
      decision_policy_(config_), debug_emitter_(config_),
      reloc_map_cache_(pcl::make_shared<PointCloudT>()),
      reloc_map_cached_keyframes_(0),
      tracking_targets_(std::make_unique<TrackingTargetCache>(
          config_, keyframe_manager_, matcher_)),
      is_relocalized_(false), has_ever_relocalized_(false),
      T_map_odom_(Eigen::Isometry3d::Identity()), last_matched_id_(-1),
      relocalization_seed_id_(-1),
      last_odom_pose_(Eigen::Isometry3d::Identity()),
      consecutive_track_failures_(0) {
}

WorldLocalizing::~WorldLocalizing() = default;

bool WorldLocalizing::isRelocalized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_relocalized_;
}

Eigen::Isometry3d WorldLocalizing::getMapToOdomTransform() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return T_map_odom_;
}

void WorldLocalizing::resetLocalizationState() {
  std::lock_guard<std::mutex> lock(mutex_);
  is_relocalized_ = false;
  has_ever_relocalized_ = false;
  T_map_odom_ = Eigen::Isometry3d::Identity();
  last_matched_id_ = -1;
  relocalization_seed_id_ = -1;
  last_odom_pose_ = Eigen::Isometry3d::Identity();
  consecutive_track_failures_ = 0;
  query_builder_.reset();
  hypothesis_manager_.reset();
  releaseSearchPreparation();
}

void WorldLocalizing::releaseSearchPreparation() {
  candidate_evaluator_.clear();
  target_provider_->clear();
  reloc_map_cache_ = pcl::make_shared<PointCloudT>();
  reloc_map_cached_keyframes_ = 0;
  reloc_map_revision_ = {};
  free_space_grid_ = FreeSpaceGrid();
  free_space_grid_keyframes_ = 0;
  free_space_grid_revision_ = {};
  free_space_grid_failed_ = false;
}

void WorldLocalizing::notifyMapReplaced() {
  std::lock_guard<std::mutex> lock(mutex_);
  is_relocalized_ = false;
  has_ever_relocalized_ = false;
  T_map_odom_ = Eigen::Isometry3d::Identity();
  last_matched_id_ = -1;
  relocalization_seed_id_ = -1;
  last_odom_pose_ = Eigen::Isometry3d::Identity();
  consecutive_track_failures_ = 0;
  query_builder_.reset();
  hypothesis_manager_.reset();
  // Map-derived state: none of it may survive into the next map. A map that
  // happens to have the same keyframe count as the previous one must still
  // rebuild its frame-RHPD index, reloc map cache, free-space grid and atlas.
  place_index_.resetMapDerivedState();
  releaseSearchPreparation();
  tracking_targets_->clearMap();
  if (localization_atlas_) {
    localization_atlas_->clear();
  }
}

void WorldLocalizing::reset() { resetLocalizationState(); }

WorldLocalizing::WorldLocalizingCacheDiagnostics
WorldLocalizing::cacheDiagnostics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  WorldLocalizingCacheDiagnostics d;
  d.atlas_loaded = localization_atlas_ && localization_atlas_->loaded();
  d.frame_rhpd_indexed_keyframes = place_index_.indexedKeyframes();
  d.reloc_map_cached_keyframes = reloc_map_cached_keyframes_;
  d.free_space_grid_keyframes = free_space_grid_keyframes_;
  d.free_space_grid_valid = free_space_grid_.valid();
  d.free_space_grid_failed = free_space_grid_failed_;
  if (target_provider_) {
    const auto target_diag = target_provider_->diagnostics();
    d.reloc_target_cache_hits = target_diag.cache_hits;
    d.reloc_target_cache_misses = target_diag.cache_misses;
    d.reloc_target_cache_bytes = target_diag.cache_total_bytes;
    d.reloc_target_cache_entries = target_diag.cache_entries;
  }
  const auto tracking = tracking_targets_->diagnostics();
  d.localization_target_cache_hits = tracking.local_hits;
  d.localization_target_cache_misses = tracking.local_misses;
  d.localization_target_cache_bytes = tracking.local_bytes;
  d.localization_target_cache_entries = tracking.local_entries;
  d.loaded_map_target_cache_hits = tracking.loaded_hits;
  d.loaded_map_target_cache_misses = tracking.loaded_misses;
  d.loaded_map_target_cache_entries = tracking.loaded_entries;
  return d;
}

void WorldLocalizing::setMapToOdomTransform(
    const Eigen::Isometry3d &T_map_odom) {
  std::lock_guard<std::mutex> lock(mutex_);
  T_map_odom_ = T_map_odom;
  is_relocalized_ = true;
  has_ever_relocalized_ = true;
  relocalization_seed_id_ = -1;
  query_builder_.reset();
  hypothesis_manager_.reset();
  releaseSearchPreparation();
}

int64_t WorldLocalizing::getLastMatchedKeyframeId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_matched_id_;
}

void WorldLocalizing::installOptimizedReference(
    LocalizationAtlas& atlas, const Eigen::Isometry3d& correction) {
  std::lock_guard<std::mutex> lock(mutex_);
  localization_atlas_->swapData(atlas);
  tracking_targets_->installLoadedReference(
      localization_atlas_->globalMap(), localization_atlas_->preparedTarget());
  T_map_odom_ = correction * T_map_odom_;
  query_builder_.reset();
  hypothesis_manager_.reset();
  releaseSearchPreparation();
  // Keep lock state, failure counters and the last raw odometry sample.
}

bool WorldLocalizing::loadLocalizationAtlas(const std::string &map_path,
                                            std::string *error) {
  if (!localization_atlas_) {
    if (error)
      *error = "localization atlas object is unavailable";
    return false;
  }
  localization_atlas_->clear();
  if (target_provider_) {
    target_provider_->clear();
  }
  if (!config_.reloc_atlas_enable) {
    place_index_.prepare();
    return true;
  }
  const std::string atlas_path =
      config_.reloc_atlas_path.empty()
          ? LocalizationAtlas::defaultAtlasPath(map_path)
          : config_.reloc_atlas_path;
  LocalizationAtlasStats stats;
  if (!localization_atlas_->load(map_path, atlas_path, &stats, error))
    return false;
  DescriptorSearchPerformance index_cost;
  const auto *catalog = localization_atlas_->placeCatalog();
  if (catalog) {
    for (const auto &members : *catalog) {
      for (const auto id : members) {
        if (!keyframe_manager_.getKeyframe(id)) {
          if (error) *error = "atlas place catalog refers to an unknown keyframe";
          return false;
        }
      }
    }
  }
  place_index_.prepare(localization_atlas_->frameDescriptors(), &index_cost, catalog);
  LOG(INFO) << "[Reloc/PlaceIndex] ready keyframes=" << place_index_.indexedKeyframes()
            << " retrieval_views=" << place_index_.indexedViews()
            << " compiled_places=" << (catalog ? catalog->size() : 0)
            << " cached_frames=" << index_cost.cached_frames
            << " cache_decode_ms=" << stats.descriptor_load_ms
            << " prepare_ms=" << index_cost.index_rebuild.wall_ms;
  tracking_targets_->installLoadedReference(
      localization_atlas_->globalMap(), localization_atlas_->preparedTarget());
  LOG(INFO) << "[LocalizationAtlas] Loaded " << atlas_path
            << " global_points=" << stats.global_point_count
            << " prepared_points=" << stats.prepared_point_count
            << " bytes=" << stats.sidecar_bytes << " load_ms=" << stats.load_ms
            << " kdtree_ms=" << stats.kdtree_ms
            << " free_space_load_ms=" << stats.free_space_load_ms;
  loaded_atlas_revision_ = keyframe_manager_.revision();
  free_space_grid_ = FreeSpaceGrid{};
  free_space_grid_revision_ = loaded_atlas_revision_;
  free_space_grid_keyframes_ = 0;
  free_space_grid_failed_ = false;
  return true;
}

bool WorldLocalizing::localizationAtlasLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return localization_atlas_ && localization_atlas_->loaded();
}

void WorldLocalizing::warmLoadedMapTrackingTarget(int64_t anchor_id) {
  if (anchor_id < 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  tracking_targets_->acquireLoaded(anchor_id);
}

VisibilityConsistencyResult WorldLocalizing::evaluateLoadedMapPoseVisibility(
    const PointCloudT::Ptr &query_cloud, const Eigen::Isometry3d &T_map_lidar) {
  if (!query_cloud || query_cloud->empty() ||
      !T_map_lidar.matrix().allFinite()) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto target = tracking_targets_->loadedVisibilityTarget(T_map_lidar.translation());
  if (!target) return {};
  return evaluateVisibilityConsistency(*target, *query_cloud, T_map_lidar,
                                       visibilityOptionsFromConfig(config_));
}

} // namespace n3mapping
