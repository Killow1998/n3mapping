// Tracking target geometry and bounded preparation caches. Calls are serialized
// by WorldLocalizing; only requested targets are prepared.
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

class TrackingTargetCache {
public:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  struct TargetResult {
    std::shared_ptr<const PointCloudMatcher::PreparedTarget> target;
    // Exact raw local visibility crop from the same frozen map as the shared
    // registration target; never a downsampled registration representation.
    PointCloudT::Ptr visibility_cloud;
    bool enabled = false;
    bool hit = false;
    bool miss = false;
    std::size_t entry_bytes = 0;
    std::size_t total_bytes = 0;
    std::size_t entries = 0;
    std::size_t submap_points = 0;
    double loaded_map_cache_ms = 0.0;
    double submap_build_ms = 0.0;
    double target_prepare_ms = 0.0;
  };

  struct Diagnostics {
    std::size_t local_hits = 0;
    std::size_t local_misses = 0;
    std::size_t local_bytes = 0;
    std::size_t local_entries = 0;
    std::size_t loaded_hits = 0;
    std::size_t loaded_misses = 0;
    std::size_t loaded_entries = 0;
  };

  TrackingTargetCache(const Config &config, KeyframeManager &keyframe_manager,
                      PointCloudMatcher &matcher);

  TargetResult acquireLocal(int64_t anchor_id, int submap_range);
  TargetResult acquireLoaded(int64_t anchor_id);
  PointCloudT::Ptr loadedVisibilityTarget(const Eigen::Vector3d &center);
  // A map replacement invalidates all targets. Session reset deliberately
  // leaves these map-derived targets available for reuse.
  void clearMap();
  void installLoadedReference(const PointCloudT::Ptr& cloud,
                             const PointCloudMatcher::PreparedTarget& target);
  Diagnostics diagnostics() const;

private:
  struct LocalEntry {
    int64_t anchor_id = -1;
    int submap_range = 0;
    KeyframeMapRevision revision;
    std::shared_ptr<const PointCloudMatcher::PreparedTarget> target;
    std::size_t bytes = 0;
    std::size_t submap_points = 0;
  };

  struct LoadedEntry {
    int64_t anchor_id = -1;
    PointCloudT::Ptr visibility_cloud;
  };

  static bool usable(const PointCloudMatcher::PreparedTarget &target);
  void clearLocal(bool reset_statistics);
  void rebuildLoadedMapIfNeeded();
  TargetResult acquireLoadedFromSnapshot(int64_t anchor_id);

  Config config_;
  KeyframeManager &keyframe_manager_;
  PointCloudMatcher &matcher_;
  std::deque<LocalEntry> local_entries_;
  KeyframeMapRevision local_revision_;
  bool have_local_revision_ = false;
  std::size_t local_bytes_ = 0;
  std::size_t local_hits_ = 0;
  std::size_t local_misses_ = 0;

  PointCloudT::Ptr loaded_map_;
  std::shared_ptr<const PointCloudMatcher::PreparedTarget> loaded_prepared_;
  // Crop identity belongs to the frozen loaded geometry, not live graph poses.
  std::map<int64_t, Eigen::Vector3d> loaded_anchor_positions_;
  std::size_t loaded_keyframe_count_ = 0;
  KeyframeMapRevision loaded_revision_;
  std::deque<LoadedEntry> loaded_entries_;
  std::size_t loaded_hits_ = 0;
  std::size_t loaded_misses_ = 0;

  static constexpr std::size_t kLoadedEntries = 4;
};

} // namespace n3mapping
