#include "n3mapping/relocalization_target_provider.h"

#include <algorithm>
#include <chrono>
#include <list>
#include <limits>
#include <mutex>
#include <stdexcept>

#include <Eigen/Geometry>
#include <glog/logging.h>

namespace n3mapping {
namespace {

using Clock = std::chrono::steady_clock;

// Search-only working set, independent of the tracking cache. Oversized
// targets are evaluated normally without retention.
constexpr std::size_t kSearchLocalCacheBytes = 128 * 1024 * 1024;
constexpr std::size_t kSearchLocalCacheEntries = 8;

double elapsedMs(const Clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

RelocTargetMode parseMode(const std::string &mode) {
  if (mode == "legacy_global_atlas") {
    return RelocTargetMode::LEGACY_GLOBAL_ATLAS;
  }
  if (mode == "local_no_cache") {
    return RelocTargetMode::LOCAL_NO_CACHE;
  }
  if (mode == "local_lru") {
    return RelocTargetMode::LOCAL_LRU;
  }
  if (mode == "shadow_local_lru") {
    return RelocTargetMode::SHADOW_LOCAL_LRU;
  }
  throw std::invalid_argument("Unknown relocalization target mode: " + mode);
}

struct CacheKey {
  int64_t anchor_id = -1;
  KeyframeMapRevision map_revision;
  Eigen::Vector3d crop_center = Eigen::Vector3d::Zero();

  bool operator==(const CacheKey &other) const {
    return anchor_id == other.anchor_id && map_revision == other.map_revision &&
           crop_center == other.crop_center;
  }
};

struct CacheEntry {
  CacheKey key;
  std::shared_ptr<const PointCloudMatcher::PreparedTarget> target;
  pcl::PointCloud<pcl::PointXYZI>::Ptr local_target;
  std::size_t bytes = 0;
};

class ConfiguredRelocTargetProvider final : public RelocTargetProvider {
public:
  ConfiguredRelocTargetProvider(const Config &config,
                                PointCloudMatcher &matcher,
                                LocalizationAtlas &localization_atlas)
      : config_(config), mode_(parseMode(config.reloc_target_mode)),
        matcher_(matcher), localization_atlas_(localization_atlas) {}

  PreparedRelocTarget getTarget(const RelocTargetRequest &input) override {
    RelocTargetRequest request = input;
    PreparedRelocTarget result;
    result.visibility_target = request.local_target;
    result.metrics.mode = relocTargetModeName(mode_);
    result.metrics.anchor_id = request.anchor_id;
    result.metrics.target_build_ms = request.target_build_ms;
    result.metrics.crop_center = request.crop_center;
    result.metrics.map_revision = request.map_revision;

    const bool automatic_local =
        mode_ == RelocTargetMode::LEGACY_GLOBAL_ATLAS &&
        !localization_atlas_.loaded();
    if (automatic_local || mode_ == RelocTargetMode::LOCAL_LRU) {
      getCachedLocalTarget(request, &result);
      result.metrics.registration_source = automatic_local
          ? "legacy_local_lru" : "local_lru";
      logMetrics(result.metrics);
      return result;
    }
    const bool primary_target = buildLocalTarget(&request, &result.metrics);
    result.visibility_target = request.local_target;
    if (!request.local_target || request.local_target->empty()) {
      logMetrics(result.metrics);
      return result;
    }

    switch (mode_) {
    case RelocTargetMode::LEGACY_GLOBAL_ATLAS:
      setLegacyTarget(request, &result);
      break;
    case RelocTargetMode::LOCAL_NO_CACHE:
      setUncachedLocalTarget(request, &result);
      break;
    case RelocTargetMode::LOCAL_LRU:
      break;
    case RelocTargetMode::SHADOW_LOCAL_LRU: {
      PreparedRelocTarget shadow;
      shadow.metrics = result.metrics;
      if (primary_target) {
        getCachedLocalTarget(request, &shadow);
      } else {
        setUncachedLocalTarget(request, &shadow);
      }
      const auto &shadow_metrics = shadow.metrics;
      setLegacyTarget(request, &result);
      result.metrics.target_prepare_ms = shadow_metrics.target_prepare_ms;
      result.metrics.target_points = shadow_metrics.target_points;
      result.metrics.cache_hit = shadow_metrics.cache_hit;
      result.metrics.cache_miss = shadow_metrics.cache_miss;
      result.metrics.cache_entry_bytes = shadow_metrics.cache_entry_bytes;
      result.metrics.cache_total_bytes = shadow_metrics.cache_total_bytes;
      result.metrics.cache_entries = shadow_metrics.cache_entries;
      result.metrics.registration_source = localization_atlas_.loaded()
                                               ? "legacy_global_atlas_shadow"
                                               : "legacy_local_no_cache_shadow";
      break;
    }
    }

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
  void setLegacyTarget(const RelocTargetRequest &request,
                       PreparedRelocTarget *result) {
    if (localization_atlas_.loaded()) {
      result->registration_target =
          std::make_shared<const PointCloudMatcher::PreparedTarget>(
              localization_atlas_.preparedTarget());
      result->metrics.registration_source = "legacy_global_atlas";
      const auto &global_map = localization_atlas_.globalMap();
      result->metrics.target_points = global_map ? global_map->size() : 0;
      return;
    }
    setUncachedLocalTarget(request, result);
    result->metrics.registration_source = "legacy_local_no_cache";
  }

  void setUncachedLocalTarget(const RelocTargetRequest &request,
                              PreparedRelocTarget *result) {
    const auto start = Clock::now();
    auto prepared = matcher_.prepareTargetCloud(request.local_target);
    ++result->metrics.target_preparations;
    result->metrics.target_prepare_ms = elapsedMs(start);
    result->metrics.target_points = request.local_target->size();
    result->registration_target =
        std::make_shared<const PointCloudMatcher::PreparedTarget>(
            std::move(prepared));
    result->metrics.registration_source = "local_no_cache";
  }

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

  void getCachedLocalTarget(RelocTargetRequest request,
                            PreparedRelocTarget *result) {
    auto *metrics = &result->metrics;
    const bool automatic_local = mode_ == RelocTargetMode::LEGACY_GLOBAL_ATLAS;
    const std::size_t max_bytes = automatic_local ? kSearchLocalCacheBytes :
        static_cast<std::size_t>(std::max(0, config_.reloc_target_cache_max_bytes));
    const std::size_t max_entries = automatic_local ? kSearchLocalCacheEntries :
        static_cast<std::size_t>(std::max(0, config_.reloc_target_cache_max_entries));
    metrics->effective_max_bytes = max_bytes;
    metrics->effective_max_entries = max_entries;
    // Config identity is the provider itself: its config is immutable and no
    // entries are shared between providers.
    const CacheKey key{request.anchor_id, request.map_revision, request.crop_center};
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
        metrics->target_points = entries_.front().local_target->size();
        result->visibility_target = entries_.front().local_target;
        result->registration_target = entries_.front().target;
        return;
      }
      ++cache_misses_;
      metrics->cache_miss = true;
      build_epoch = epoch_;
    }

    const bool retainable = buildLocalTarget(&request, metrics);
    result->visibility_target = request.local_target;
    if (!request.local_target || request.local_target->empty()) return;
    const auto start = Clock::now();
    auto prepared = std::make_shared<const PointCloudMatcher::PreparedTarget>(
        matcher_.prepareTargetCloud(request.local_target));
    metrics->target_prepare_ms = elapsedMs(start);
    ++metrics->target_preparations;
    result->registration_target = prepared;
    metrics->target_points = request.local_target->size();
    const std::size_t crop_bytes = sizeof(*request.local_target) +
        request.local_target->points.capacity() * sizeof(pcl::PointXYZI);
    const std::size_t prepared_bytes = estimatePreparedTargetMemoryBytes(*prepared);
    const std::size_t entry_bytes = prepared_bytes >
        std::numeric_limits<std::size_t>::max() - crop_bytes
        ? std::numeric_limits<std::size_t>::max() : prepared_bytes + crop_bytes;
    metrics->cache_entry_bytes = entry_bytes;

    std::lock_guard<std::mutex> lock(mutex_);
    // A reset or a newer map wins over work that started before it.
    if (build_epoch != epoch_) return;
    const auto raced = findLocked(key);
    if (raced != entries_.end()) {
      entries_.splice(entries_.begin(), entries_, raced);
      metrics->cache_entry_bytes = entries_.front().bytes;
      fillCacheMetricsLocked(metrics);
      result->visibility_target = entries_.front().local_target;
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
    entries_.push_front({key, prepared, request.local_target, entry_bytes});
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
    VLOG(1) << "[RelocTarget] mode=" << metrics.mode
            << " registration_source=" << metrics.registration_source
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
  RelocTargetMode mode_;
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

} // namespace

const char *relocTargetModeName(RelocTargetMode mode) {
  switch (mode) {
  case RelocTargetMode::LEGACY_GLOBAL_ATLAS:
    return "legacy_global_atlas";
  case RelocTargetMode::LOCAL_NO_CACHE:
    return "local_no_cache";
  case RelocTargetMode::LOCAL_LRU:
    return "local_lru";
  case RelocTargetMode::SHADOW_LOCAL_LRU:
    return "shadow_local_lru";
  }
  return "unknown";
}

std::unique_ptr<RelocTargetProvider>
makeRelocTargetProvider(const Config &config, PointCloudMatcher &matcher,
                        LocalizationAtlas &localization_atlas) {
  return std::make_unique<ConfiguredRelocTargetProvider>(config, matcher,
                                                         localization_atlas);
}

} // namespace n3mapping
