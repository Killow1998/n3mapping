#include "n3mapping/tracking_target_cache.h"

#include <algorithm>
#include <utility>
#include <chrono>
#include <ctime>
#include <glog/logging.h>

#include "n3mapping/localization_atlas.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {
using Clock = std::chrono::steady_clock;
double elapsedMs(const Clock::time_point &started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}
} // namespace

TrackingTargetCache::TrackingTargetCache(const Config &config,
                                        KeyframeManager &keyframe_manager,
                                        PointCloudMatcher &matcher)
    : config_(config), keyframe_manager_(keyframe_manager), matcher_(matcher),
      loaded_map_(pcl::make_shared<PointCloudT>()) {}

bool TrackingTargetCache::usable(const PointCloudMatcher::PreparedTarget &target) {
  return target.plane_levels.size() >= 2 &&
         std::all_of(target.plane_levels.begin(), target.plane_levels.end(),
                     [](const PointCloudMatcher::PreparedTargetLevel &level) {
                       return level.cloud && level.kdtree && level.cloud->size() >= 10;
                     });
}

TrackingTargetCache::TargetResult
TrackingTargetCache::acquireLocal(int64_t anchor_id, int submap_range) {
  TargetResult result;
  const std::size_t max_bytes = static_cast<std::size_t>(
      std::max(0, config_.localization_tracking_target_cache_max_bytes));
  const std::size_t max_entries = static_cast<std::size_t>(
      std::max(0, config_.localization_tracking_target_cache_max_entries));
  result.enabled = max_bytes > 0 && max_entries > 0;
  const KeyframeMapRevision revision = keyframe_manager_.revision();
  if (result.enabled) {
    if (!have_local_revision_ || local_revision_ != revision) {
      clearLocal(false);
      local_revision_ = revision;
      have_local_revision_ = true;
    }
    const auto found = std::find_if(local_entries_.begin(), local_entries_.end(),
                                    [&](const LocalEntry &entry) {
      return entry.anchor_id == anchor_id && entry.submap_range == submap_range &&
             entry.revision == revision && entry.target && usable(*entry.target);
    });
    if (found != local_entries_.end()) {
      result.hit = true;
      ++local_hits_;
      if (found != local_entries_.begin()) {
        auto entry = std::move(*found);
        local_entries_.erase(found);
        local_entries_.push_front(std::move(entry));
      }
      const auto &entry = local_entries_.front();
      result.target = entry.target;
      result.submap_points = entry.submap_points;
      result.entry_bytes = entry.bytes;
      result.total_bytes = local_bytes_;
      result.entries = local_entries_.size();
      return result;
    }
    result.miss = true;
    ++local_misses_;
  }

  const auto build_started = Clock::now();
  const auto submap = keyframe_manager_.buildLocalSubmap(anchor_id, submap_range);
  result.submap_points = submap ? submap->size() : 0;
  result.submap_build_ms = elapsedMs(build_started);
  const auto prepare_started = Clock::now();
  auto prepared = std::make_shared<const PointCloudMatcher::PreparedTarget>(
      matcher_.prepareTargetCloud(submap));
  result.target_prepare_ms = elapsedMs(prepare_started);
  result.target = prepared;
  result.entry_bytes = estimatePreparedTargetMemoryBytes(*prepared);
  if (!result.enabled || !usable(*prepared) || result.entry_bytes > max_bytes) {
    result.total_bytes = local_bytes_;
    result.entries = local_entries_.size();
    return result;
  }
  while (!local_entries_.empty() &&
         (local_entries_.size() >= max_entries ||
          local_bytes_ > max_bytes - result.entry_bytes)) {
    local_bytes_ -= local_entries_.back().bytes;
    local_entries_.pop_back();
  }
  local_entries_.push_front({anchor_id, submap_range, revision, prepared,
                            result.entry_bytes, result.submap_points});
  local_bytes_ += result.entry_bytes;
  result.total_bytes = local_bytes_;
  result.entries = local_entries_.size();
  return result;
}

void TrackingTargetCache::clearLocal(bool reset_statistics) {
  local_entries_.clear();
  local_bytes_ = 0;
  local_revision_ = {};
  have_local_revision_ = false;
  if (reset_statistics) {
    local_hits_ = 0;
    local_misses_ = 0;
  }
}

void TrackingTargetCache::rebuildLoadedMapIfNeeded() {
  const KeyframeMapRevision revision = keyframe_manager_.revision();
  std::vector<Keyframe::Ptr> loaded_keyframes;
  for (const auto &keyframe : keyframe_manager_.getAllKeyframes()) {
    if (keyframe && keyframe->is_from_loaded_map) loaded_keyframes.push_back(keyframe);
  }
  if (loaded_map_ && !loaded_map_->empty() &&
      loaded_revision_.generation == revision.generation &&
      loaded_keyframe_count_ == loaded_keyframes.size()) {
    // Keep the loaded reference immutable across extension graph corrections.
    return;
  }
  loaded_entries_.clear();
  loaded_prepared_.reset();
  loaded_map_ = LocalizationAtlas::buildGlobalMap(config_, loaded_keyframes);
  loaded_anchor_positions_.clear();
  for (const auto &keyframe : loaded_keyframes)
    loaded_anchor_positions_.emplace(keyframe->id,
                                     keyframe->pose_optimized.translation());
  loaded_keyframe_count_ = loaded_keyframes.size();
  loaded_revision_ = revision;
}

TrackingTargetCache::TargetResult
TrackingTargetCache::acquireLoaded(int64_t anchor_id) {
  TargetResult result;
  const auto cache_started = Clock::now();
  rebuildLoadedMapIfNeeded();
  const double cache_ms = elapsedMs(cache_started);
  result = acquireLoadedFromSnapshot(anchor_id);
  result.loaded_map_cache_ms = cache_ms;
  return result;
}

TrackingTargetCache::TargetResult
TrackingTargetCache::acquireLoadedFromSnapshot(int64_t anchor_id) {
  TargetResult result;
  const auto lookup_started = Clock::now();
  const auto found = std::find_if(loaded_entries_.begin(), loaded_entries_.end(),
                                [&](const LoadedEntry &entry) {
    return entry.anchor_id == anchor_id;
  });
  if (found != loaded_entries_.end()) {
    ++loaded_hits_;
    result.hit = true;
    if (found != loaded_entries_.begin()) {
      auto entry = std::move(*found);
      loaded_entries_.erase(found);
      loaded_entries_.push_front(std::move(entry));
    }
    result.target = loaded_prepared_;
    result.visibility_cloud = loaded_entries_.front().visibility_cloud;
    result.submap_points = result.visibility_cloud ? result.visibility_cloud->size() : 0;
    result.target_prepare_ms = elapsedMs(lookup_started);
    return result;
  }

  const auto anchor = loaded_anchor_positions_.find(anchor_id);
  const auto build_started = Clock::now();
  if (anchor != loaded_anchor_positions_.end()) {
    result.visibility_cloud = LocalizationAtlas::cropGlobalMap(
        config_, loaded_map_, anchor->second);
  }
  result.submap_build_ms = elapsedMs(build_started);
  result.submap_points = result.visibility_cloud ? result.visibility_cloud->size() : 0;
  if (result.submap_points == 0) return result;
  result.miss = true;
  ++loaded_misses_;
  // The map does not change when a different anchor becomes nearest. Build
  // its exact-search trees and normal/covariance levels once per generation.
  // Local raw crops remain available for the independent visibility gate.
  if (!loaded_prepared_) {
    const auto prepare_started = Clock::now();
    const auto cpu_started = std::clock();
    loaded_prepared_ = std::make_shared<const PointCloudMatcher::PreparedTarget>(
        matcher_.prepareTargetCloud(loaded_map_));
    result.target_prepare_ms = elapsedMs(prepare_started);
    LOG(INFO) << "[LoadedMapTarget] generation=" << loaded_revision_.generation
              << " points=" << loaded_map_->size()
              << " prepare_ms=" << result.target_prepare_ms
              << " process_cpu_ms_during_prepare="
              << 1000.0 * (std::clock() - cpu_started) / CLOCKS_PER_SEC
              << " estimated_bytes="
              << estimatePreparedTargetMemoryBytes(*loaded_prepared_);
  }
  result.target = loaded_prepared_;
  if (!usable(*result.target)) return result;

  loaded_entries_.push_front({anchor_id, result.visibility_cloud});
  while (loaded_entries_.size() > kLoadedEntries) loaded_entries_.pop_back();
  return result;
}

TrackingTargetCache::PointCloudT::Ptr
TrackingTargetCache::loadedVisibilityTarget(const Eigen::Vector3d &center) {
  rebuildLoadedMapIfNeeded();
  return LocalizationAtlas::cropGlobalMap(config_, loaded_map_, center);
}

void TrackingTargetCache::clearMap() {
  clearLocal(true);
  loaded_map_ = pcl::make_shared<PointCloudT>();
  loaded_prepared_.reset();
  loaded_anchor_positions_.clear();
  loaded_keyframe_count_ = 0;
  loaded_revision_ = {};
  loaded_entries_.clear();
  loaded_hits_ = 0;
  loaded_misses_ = 0;
}

void TrackingTargetCache::installLoadedReference(
    const PointCloudT::Ptr& cloud,
    const PointCloudMatcher::PreparedTarget& target) {
  clearMap();
  loaded_map_ = cloud;
  loaded_prepared_ = std::make_shared<const PointCloudMatcher::PreparedTarget>(target);
  for (const auto& frame : keyframe_manager_.getAllKeyframes())
    if (frame && frame->is_from_loaded_map) {
      loaded_anchor_positions_.emplace(frame->id, frame->pose_optimized.translation());
      ++loaded_keyframe_count_;
    }
  loaded_revision_ = keyframe_manager_.revision();
}

TrackingTargetCache::Diagnostics TrackingTargetCache::diagnostics() const {
  Diagnostics result;
  result.local_hits = local_hits_;
  result.local_misses = local_misses_;
  result.local_bytes = local_bytes_;
  result.local_entries = local_entries_.size();
  result.loaded_hits = loaded_hits_;
  result.loaded_misses = loaded_misses_;
  result.loaded_entries = loaded_entries_.size();
  return result;
}

} // namespace n3mapping
