#include "n3mapping/relocalization_target_provider.h"

#include <algorithm>
#include <chrono>
#include <list>
#include <limits>
#include <mutex>

#include <Eigen/Geometry>
#include <glog/logging.h>

namespace n3mapping {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kSearchCacheBytes = 128 * 1024 * 1024;

double elapsedMs(const Clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

struct CacheKey {
  int64_t anchor_id = -1;
  KeyframeMapRevision map_revision;
  Eigen::Vector3d crop_center = Eigen::Vector3d::Zero();
  bool visibility_from_atlas = false;

  bool operator==(const CacheKey &other) const {
    return anchor_id == other.anchor_id && map_revision == other.map_revision &&
           crop_center == other.crop_center && visibility_from_atlas == other.visibility_from_atlas;
  }
};

struct CacheEntry {
  CacheKey key;
  std::shared_ptr<const PointCloudMatcher::PreparedTarget> target;
  pcl::PointCloud<pcl::PointXYZI>::Ptr local_target;
  std::shared_ptr<const std::vector<std::uint32_t>> indices;
  std::size_t bytes = 0;
};

class ConfiguredRelocTargetProvider final : public RelocTargetProvider {
public:
  ConfiguredRelocTargetProvider(const Config &config,
                                PointCloudMatcher &matcher,
                                LocalizationAtlas &localization_atlas)
      : config_(config), matcher_(matcher), localization_atlas_(localization_atlas) {}

  PreparedRelocTarget getTarget(const RelocTargetRequest &input) override {
    const auto &request = input;
    PreparedRelocTarget result;
    result.visibility_target = request.local_target;
    result.metrics.anchor_id = request.anchor_id;
    result.metrics.target_build_ms = request.target_build_ms;
    result.metrics.crop_center = request.crop_center;
    result.metrics.map_revision = request.map_revision;

    getCachedTarget(request, &result);
    logMetrics(result.metrics);
    return result;
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    clearEntriesLocked();
    ++epoch_;
    cache_hits_ = 0;
    cache_misses_ = 0;
    have_cache_revision_ = false;
  }

  RelocTargetProviderDiagnostics diagnostics() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return {cache_hits_, cache_misses_, cache_total_bytes_, entries_.size()};
  }

private:
  bool buildLocalTarget(RelocTargetRequest *request, RelocTargetMetrics *metrics) {
    if (!request->local_target && request->build_local_target) {
      const auto start = Clock::now();
      request->local_target = request->build_local_target();
      metrics->target_build_ms += elapsedMs(start);
      ++metrics->target_builds;
    }
    if ((!request->local_target || request->local_target->empty()) &&
        request->build_fallback_target) {
      const auto start = Clock::now();
      request->local_target = request->build_fallback_target();
      metrics->target_build_ms += elapsedMs(start);
      ++metrics->target_builds;
      return false;
    }
    return true;
  }

  void getCachedTarget(RelocTargetRequest request,
                       PreparedRelocTarget *result) {
    auto *metrics = &result->metrics;
    const bool use_atlas = localization_atlas_.loaded();
    metrics->registration_source = use_atlas ? "global_atlas" : "local_lru";
    const std::size_t max_bytes = use_atlas ? kSearchCacheBytes :
        static_cast<std::size_t>(config_.reloc_target_cache_max_bytes);
    const std::size_t max_entries = use_atlas ? std::numeric_limits<std::size_t>::max() :
        static_cast<std::size_t>(config_.reloc_target_cache_max_entries);
    metrics->effective_max_bytes = max_bytes;
    metrics->effective_max_entries = max_entries;

    const CacheKey key{request.anchor_id, request.map_revision, request.crop_center,
                       request.visibility_from_atlas};
    std::size_t build_epoch = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      synchronizeRevisionLocked(request.map_revision);
      const auto found = findLocked(key);
      if (found != entries_.end()) {
        entries_.splice(entries_.begin(), entries_, found);
        ++cache_hits_;
        metrics->cache_hit = true;
        metrics->cache_entry_bytes = entries_.front().bytes;
        fillCacheMetricsLocked(metrics);
        metrics->target_points = use_atlas ? localization_atlas_.globalMap()->size() :
                                           entries_.front().local_target->size();
        result->visibility_target = entries_.front().local_target;
        result->visibility_indices = entries_.front().indices;
        result->registration_target = entries_.front().target;
        return;
      }
      ++cache_misses_;
      metrics->cache_miss = true;
      build_epoch = epoch_;
    }

    bool retainable = true;
    if (use_atlas && request.visibility_from_atlas) {
      const auto started = Clock::now();
      result->visibility_indices = localization_atlas_.cropVisibilityIndices(request.crop_center);
      metrics->target_build_ms += elapsedMs(started);
      ++metrics->target_builds;
      if (result->visibility_indices && !result->visibility_indices->empty()) {
        request.local_target = localization_atlas_.globalMap();
      } else if (result->visibility_indices) {

        request.local_target = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        request.build_local_target = {};
        result->visibility_indices.reset();
      }
    }
    if (!result->visibility_indices) retainable = buildLocalTarget(&request, metrics);
    result->visibility_target = request.local_target;
    if (!request.local_target || request.local_target->empty()) return;
    if (use_atlas) {
      result->registration_target =
          std::make_shared<const PointCloudMatcher::PreparedTarget>(
              localization_atlas_.preparedTarget());
      metrics->target_points = localization_atlas_.globalMap()->size();
    } else {
      const auto started = Clock::now();
      result->registration_target =
          std::make_shared<const PointCloudMatcher::PreparedTarget>(
              matcher_.prepareTargetCloud(request.local_target));
      ++metrics->target_preparations;
      metrics->target_prepare_ms = elapsedMs(started);
      metrics->target_points = request.local_target->size();
    }
    const auto &prepared = result->registration_target;
    const auto &indices = result->visibility_indices;
    const std::size_t crop_bytes = indices
        ? sizeof(*indices) + indices->capacity() * sizeof(std::uint32_t)
        : sizeof(*request.local_target) + request.local_target->points.capacity() * sizeof(pcl::PointXYZI);
    const std::size_t prepared_bytes = use_atlas
        ? sizeof(*prepared) + prepared->plane_levels.capacity() *
              sizeof(PointCloudMatcher::PreparedTargetLevel)
        : estimatePreparedTargetMemoryBytes(*prepared);
    const std::size_t payload_bytes = prepared_bytes >
        std::numeric_limits<std::size_t>::max() - crop_bytes
        ? std::numeric_limits<std::size_t>::max() : prepared_bytes + crop_bytes;
    constexpr std::size_t metadata_bytes = sizeof(CacheEntry) + 2 * sizeof(void*);
    const std::size_t entry_bytes = payload_bytes >
        std::numeric_limits<std::size_t>::max() - metadata_bytes
        ? std::numeric_limits<std::size_t>::max() : payload_bytes + metadata_bytes;
    metrics->cache_entry_bytes = entry_bytes;

    std::lock_guard<std::mutex> lock(mutex_);

    if (build_epoch != epoch_) return;
    const auto raced = findLocked(key);
    if (raced != entries_.end()) {
      entries_.splice(entries_.begin(), entries_, raced);
      metrics->cache_entry_bytes = entries_.front().bytes;
      fillCacheMetricsLocked(metrics);
      result->visibility_target = entries_.front().local_target;
      result->visibility_indices = entries_.front().indices;
      result->registration_target = entries_.front().target;
      return;
    }

    if (!retainable || max_bytes == 0 || max_entries == 0 || entry_bytes > max_bytes) {
      fillCacheMetricsLocked(metrics);
      return;
    }

    while (!entries_.empty() &&
           (entries_.size() >= max_entries ||
            cache_total_bytes_ + entry_bytes > max_bytes)) {
      cache_total_bytes_ -= entries_.back().bytes;
      entries_.pop_back();
    }
    entries_.push_front({key, prepared, request.local_target, indices, entry_bytes});
    cache_total_bytes_ += entry_bytes;
    fillCacheMetricsLocked(metrics);
  }

  using EntryIterator = std::list<CacheEntry>::iterator;

  EntryIterator findLocked(const CacheKey &key) {
    return std::find_if(
        entries_.begin(), entries_.end(),
        [&](const CacheEntry &entry) { return entry.key == key; });
  }

  void synchronizeRevisionLocked(const KeyframeMapRevision &revision) {
    if (!have_cache_revision_ || cache_revision_ != revision) {
      clearEntriesLocked();
      ++epoch_;
      cache_revision_ = revision;
      have_cache_revision_ = true;
    }
  }

  void clearEntriesLocked() {
    entries_.clear();
    cache_total_bytes_ = 0;
  }

  void fillCacheMetricsLocked(RelocTargetMetrics *metrics) const {
    metrics->cache_total_bytes = cache_total_bytes_;
    metrics->cache_entries = entries_.size();
  }

  void logMetrics(const RelocTargetMetrics &metrics) const {
    VLOG(1) << "[RelocTarget] registration_source=" << metrics.registration_source
            << " anchor_id=" << metrics.anchor_id
            << " target_build_ms=" << metrics.target_build_ms
            << " target_prepare_ms=" << metrics.target_prepare_ms
            << " target_points=" << metrics.target_points
            << " cache_hit=" << metrics.cache_hit
            << " cache_miss=" << metrics.cache_miss
            << " cache_entry_bytes=" << metrics.cache_entry_bytes
            << " cache_total_bytes=" << metrics.cache_total_bytes
            << " cache_entries=" << metrics.cache_entries
            << " target_builds=" << metrics.target_builds
            << " target_preparations=" << metrics.target_preparations
            << " effective_max_bytes=" << metrics.effective_max_bytes
            << " effective_max_entries=" << metrics.effective_max_entries
            << " crop_center=" << metrics.crop_center.transpose()
            << " map_revision=" << metrics.map_revision.generation << ":"
            << metrics.map_revision.structure_revision << ":"
            << metrics.map_revision.pose_revision;
  }

  const Config config_;
  PointCloudMatcher &matcher_;
  LocalizationAtlas &localization_atlas_;
  mutable std::mutex mutex_;
  std::list<CacheEntry> entries_;
  std::size_t cache_total_bytes_ = 0;
  std::size_t cache_hits_ = 0;
  std::size_t cache_misses_ = 0;
  std::size_t epoch_ = 0;
  bool have_cache_revision_ = false;
  KeyframeMapRevision cache_revision_;
};

}

std::unique_ptr<RelocTargetProvider>
makeRelocTargetProvider(const Config &config, PointCloudMatcher &matcher,
                        LocalizationAtlas &localization_atlas) {
  return std::make_unique<ConfiguredRelocTargetProvider>(config, matcher,
                                                         localization_atlas);
}

}
