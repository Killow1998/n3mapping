#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

enum class RelocTargetMode {
  LEGACY_GLOBAL_ATLAS,
  LOCAL_NO_CACHE,
  LOCAL_LRU,
  SHADOW_LOCAL_LRU,
};

const char *relocTargetModeName(RelocTargetMode mode);

struct RelocTargetRequest {
  int64_t anchor_id = -1;
  KeyframeMapRevision map_revision;
  Eigen::Vector3d crop_center = Eigen::Vector3d::Zero();
  pcl::PointCloud<pcl::PointXYZI>::Ptr local_target;
  double target_build_ms = 0.0;
};

struct RelocTargetMetrics {
  std::string mode;
  std::string registration_source;
  double target_build_ms = 0.0;
  double target_prepare_ms = 0.0;
  std::size_t target_points = 0;
  bool cache_hit = false;
  bool cache_miss = false;
  std::size_t cache_entry_bytes = 0;
  std::size_t cache_total_bytes = 0;
  std::size_t cache_entries = 0;
  Eigen::Vector3d crop_center = Eigen::Vector3d::Zero();
  KeyframeMapRevision map_revision;
};

struct PreparedRelocTarget {
  pcl::PointCloud<pcl::PointXYZI>::Ptr visibility_target;
  std::shared_ptr<const PointCloudMatcher::PreparedTarget> registration_target;
  RelocTargetMetrics metrics;

  bool valid() const {
    return visibility_target && !visibility_target->empty() &&
           registration_target && !registration_target->plane_levels.empty();
  }
};

struct RelocTargetProviderDiagnostics {
  std::size_t cache_hits = 0;
  std::size_t cache_misses = 0;
  std::size_t cache_total_bytes = 0;
  std::size_t cache_entries = 0;
};

class RelocTargetProvider {
public:
  virtual ~RelocTargetProvider() = default;

  virtual PreparedRelocTarget getTarget(const RelocTargetRequest &request) = 0;
  virtual void clear() = 0;
  virtual RelocTargetProviderDiagnostics diagnostics() const = 0;
};

std::unique_ptr<RelocTargetProvider>
makeRelocTargetProvider(const Config &config, PointCloudMatcher &matcher,
                        LocalizationAtlas &localization_atlas);

} // namespace n3mapping
