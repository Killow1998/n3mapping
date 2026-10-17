#include "n3mapping/tracking_target_cache.h"

#include <algorithm>
#include <utility>

#include "n3mapping/localization_atlas.h"

namespace n3mapping {

void TrackingTargetCache::startPrefetchWorkers() {
  if (!prefetch_threads_.empty()) return;
  prefetch_matchers_.reserve(worker_count_);
  prefetch_threads_.reserve(worker_count_);
  // Finalize the matcher vector before any worker reads it.
  for (std::size_t index = 0; index < worker_count_; ++index) {
    Config worker_config = config_;
    worker_config.num_threads = threads_per_worker_;
    prefetch_matchers_.push_back(std::make_unique<PointCloudMatcher>(worker_config));
  }
  for (std::size_t index = 0; index < worker_count_; ++index) {
    prefetch_threads_.emplace_back(&TrackingTargetCache::prefetchLoop, this, index);
  }
}

TrackingTargetCache::RequestKey
TrackingTargetCache::requestKey(const PrefetchRequest &request) {
  return {request.epoch, request.anchor_id, request.revision.generation,
          request.revision.pose_revision, request.loaded_keyframe_count};
}

bool TrackingTargetCache::hasLoadedLocked(
    int64_t anchor_id, const KeyframeMapRevision &revision, std::size_t count) const {
  return std::any_of(loaded_entries_.begin(), loaded_entries_.end(),
                     [&](const LoadedEntry &entry) {
    return matches(entry, anchor_id, revision, count);
  });
}

void TrackingTargetCache::prefetchLoaded(const Eigen::Vector3d &query_position,
                                        int64_t current_anchor_id) {
  if (!query_position.allFinite() || !loaded_map_ || loaded_map_->empty()) return;
  startPrefetchWorkers();
  std::vector<Keyframe::Ptr> nearest;
  for (const auto &keyframe : keyframe_manager_.getAllKeyframes()) {
    if (keyframe && keyframe->is_from_loaded_map &&
        keyframe->pose_optimized.translation().allFinite()) {
      nearest.push_back(keyframe);
    }
  }
  std::sort(nearest.begin(), nearest.end(),
            [&](const Keyframe::Ptr &lhs, const Keyframe::Ptr &rhs) {
    return (lhs->pose_optimized.translation() - query_position).squaredNorm() <
           (rhs->pose_optimized.translation() - query_position).squaredNorm();
  });
  std::size_t alternatives_considered = 0;
  for (const auto &keyframe : nearest) {
    if (keyframe->id == current_anchor_id) continue;
    // Cached/pending neighbors consume a working-set slot too, preventing a
    // stationary tracker from continually preparing ever more distant anchors.
    if (alternatives_considered >= kLoadedEntries - 1u) break;
    ++alternatives_considered;
    PrefetchRequest request;
    request.anchor_id = keyframe->id;
    request.crop_center = keyframe->pose_optimized.translation();
    request.loaded_map = loaded_map_;
    request.loaded_keyframe_count = loaded_keyframe_count_;
    request.revision = loaded_revision_;
    {
      std::lock_guard<std::mutex> lock(loaded_mutex_);
      request.epoch = epoch_;
      const auto key = requestKey(request);
      if (hasLoadedLocked(request.anchor_id, request.revision,
                          request.loaded_keyframe_count) ||
          prefetch_pending_.count(key) != 0u) continue;
      while (prefetch_queue_.size() >= kMaxQueued) {
        prefetch_pending_.erase(requestKey(prefetch_queue_.back()));
        prefetch_queue_.pop_back();
      }
      prefetch_queue_.push_back(std::move(request));
      prefetch_pending_.insert(key);
      ++prefetch_requests_;
    }
    prefetch_cv_.notify_one();
  }
}

void TrackingTargetCache::prefetchLoop(std::size_t worker_index) {
  if (worker_index >= prefetch_matchers_.size() || !prefetch_matchers_[worker_index]) return;
  auto &matcher = *prefetch_matchers_[worker_index];
  while (true) {
    PrefetchRequest request;
    {
      std::unique_lock<std::mutex> lock(loaded_mutex_);
      prefetch_cv_.wait(lock, [this] { return stop_ || !prefetch_queue_.empty(); });
      if (stop_) return;
      request = std::move(prefetch_queue_.front());
      prefetch_queue_.pop_front();
    }
    const auto submap = LocalizationAtlas::cropGlobalMap(
        config_, request.loaded_map, request.crop_center);
    std::shared_ptr<const PointCloudMatcher::PreparedTarget> target;
    if (submap && !submap->empty()) {
      target = std::make_shared<const PointCloudMatcher::PreparedTarget>(
          matcher.prepareTargetCloud(submap));
    }
    std::lock_guard<std::mutex> lock(loaded_mutex_);
    prefetch_pending_.erase(requestKey(request));
    prefetch_cv_.notify_all();
    if (request.epoch != epoch_ || !target || !usable(*target) ||
        hasLoadedLocked(request.anchor_id, request.revision,
                        request.loaded_keyframe_count)) continue;
    while (loaded_entries_.size() >= kLoadedEntries) loaded_entries_.pop_back();
    loaded_entries_.push_back({request.anchor_id, request.loaded_keyframe_count,
                              request.revision, std::move(target), submap});
  }
}

void TrackingTargetCache::warmLoaded(int64_t anchor_id,
                                    const Eigen::Vector3d &query_position,
                                    bool wait_for_prefetch) {
  if (anchor_id < 0 || !query_position.allFinite()) return;
  rebuildLoadedMapIfNeeded();
  const auto anchor = keyframe_manager_.getKeyframe(anchor_id);
  if (!anchor || !anchor->is_from_loaded_map ||
      !anchor->pose_optimized.translation().allFinite() || !loaded_map_ ||
      loaded_map_->empty()) return;
  // Let alternatives build concurrently with the authoritative foreground
  // target. Only initial lock waits; steady-state graph updates remain bounded.
  prefetchLoaded(query_position, anchor_id);
  const auto target = acquireLoadedFromSnapshot(anchor_id);
  if (target.submap_points == 0) return;
  if (wait_for_prefetch) {
    std::unique_lock<std::mutex> lock(loaded_mutex_);
    prefetch_cv_.wait(lock, [this] { return stop_ || prefetch_pending_.empty(); });
  }
}

} // namespace n3mapping
