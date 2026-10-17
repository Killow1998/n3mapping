// Tracking target geometry and bounded preparation caches. Foreground calls are
// serialized by WorldLocalizing; workers own their matcher and immutable inputs.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
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
    // The exact original crop, never a registration/downsampled representation.
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
    std::size_t prefetch_requests = 0;
    std::size_t prefetch_pending = 0;
    std::size_t prefetch_queued = 0;
    std::size_t prefetch_workers = 0;
    std::size_t prefetch_threads_per_worker = 0;
  };

  TrackingTargetCache(const Config &config, KeyframeManager &keyframe_manager,
                      PointCloudMatcher &matcher);
  ~TrackingTargetCache();

  TargetResult acquireLocal(int64_t anchor_id, int submap_range);
  TargetResult acquireLoaded(int64_t anchor_id);
  void prefetchLoaded(const Eigen::Vector3d &query_position,
                      int64_t current_anchor_id);
  void warmLoaded(int64_t anchor_id, const Eigen::Vector3d &query_position,
                  bool wait_for_prefetch);
  PointCloudT::Ptr loadedVisibilityTarget(const Eigen::Vector3d &center);
  // A map replacement invalidates queued and in-flight work. Session reset
  // deliberately leaves these map-derived targets available for reuse.
  void clearMap();
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
    std::size_t loaded_keyframe_count = 0;
    KeyframeMapRevision revision;
    std::shared_ptr<const PointCloudMatcher::PreparedTarget> target;
    PointCloudT::Ptr visibility_cloud;
  };

  // Epoch is part of request identity: a stale worker cannot erase a newly
  // queued request with identical map revision and anchor after invalidation.
  using RequestKey = std::tuple<std::uint64_t, int64_t, std::uint64_t,
                                std::uint64_t, std::size_t>;
  struct PrefetchRequest {
    int64_t anchor_id = -1;
    Eigen::Vector3d crop_center = Eigen::Vector3d::Zero();
    PointCloudT::Ptr loaded_map;
    std::size_t loaded_keyframe_count = 0;
    KeyframeMapRevision revision;
    std::uint64_t epoch = 0;
  };

  static bool usable(const PointCloudMatcher::PreparedTarget &target);
  static bool matches(const LoadedEntry &entry, int64_t anchor_id,
                      const KeyframeMapRevision &revision, std::size_t count);
  static RequestKey requestKey(const PrefetchRequest &request);
  bool hasLoadedLocked(int64_t anchor_id, const KeyframeMapRevision &revision,
                       std::size_t count) const;
  void clearLocal(bool reset_statistics);
  void invalidateLoaded();
  void rebuildLoadedMapIfNeeded();
  TargetResult acquireLoadedFromSnapshot(int64_t anchor_id);
  void startPrefetchWorkers();
  void prefetchLoop(std::size_t worker_index);

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
  std::size_t loaded_keyframe_count_ = 0;
  KeyframeMapRevision loaded_revision_;
  std::deque<LoadedEntry> loaded_entries_;
  std::deque<PrefetchRequest> prefetch_queue_;
  std::set<RequestKey> prefetch_pending_;
  mutable std::mutex loaded_mutex_;
  std::condition_variable prefetch_cv_;
  std::vector<std::unique_ptr<PointCloudMatcher>> prefetch_matchers_;
  std::vector<std::thread> prefetch_threads_;
  std::size_t worker_count_ = 0;
  int threads_per_worker_ = 1;
  bool stop_ = false;
  std::uint64_t epoch_ = 0;
  std::size_t loaded_hits_ = 0;
  std::size_t loaded_misses_ = 0;
  std::size_t prefetch_requests_ = 0;

  static constexpr std::size_t kLoadedEntries = 4;
  static constexpr std::size_t kMaxWorkers = 2;
  static constexpr std::size_t kMaxQueued = kLoadedEntries * 2;
};

} // namespace n3mapping
