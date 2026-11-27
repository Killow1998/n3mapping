#include "continuation_cleaning.h"
#include "visibility_angular_bins.h"
#include "n3mapping/config.h"
#include "n3mapping/visibility_consistency.h"

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <optional>
#include <unordered_map>

namespace n3mapping {
namespace {

class SphericalDepthBins {
public:
  explicit SphericalDepthBins(double resolution)
      : azimuth_count_(static_cast<int>(std::ceil(360.0 / resolution))),
        elevation_count_(static_cast<int>(std::ceil(180.0 / resolution))),
        azimuth_(180.0, resolution, azimuth_count_, true),
        elevation_(90.0, resolution, elevation_count_, false) {}

  using Depths = std::unordered_map<uint64_t, double>;

  Depths project(const Keyframe::PointCloudT& cloud, double min_range, double max_range) const {
    Depths depths;
    depths.reserve(cloud.size());
    for (const auto& point : cloud) {
      const auto position = point.getVector3fMap().cast<double>();
      const double range = position.norm();
      if (!(range >= min_range && range <= max_range)) continue;
      const auto [azimuth, elevation] = bins(position);
      const auto [it, inserted] = depths.try_emplace(index(azimuth, elevation), range);
      if (!inserted) it->second = std::min(it->second, range);
    }
    return depths;
  }

  bool seesBehind(const Depths& depths, const Eigen::Vector3d& point, double limit) const {
    const auto [azimuth, elevation] = bins(point);
    const auto center = depths.find(index(azimuth, elevation));
    return center != depths.end() && center->second > limit &&
           !hasNearbyReturn(depths, azimuth, elevation, 0.0, limit);
  }

  bool seesSurface(const Depths& depths, const Eigen::Vector3d& point,
                   double range, double margin) const {
    const auto [azimuth, elevation] = bins(point);
    return hasNearbyReturn(depths, azimuth, elevation, range - margin, range + margin);
  }

private:
  bool hasNearbyReturn(const Depths& depths, int azimuth, int elevation,
                       double min_depth, double max_depth) const {
    const int previous = azimuth == 0 ? azimuth_count_ - 1 : azimuth - 1;
    const int next = azimuth == azimuth_count_ - 1 ? 0 : azimuth + 1;
    for (int row = std::max(0, elevation - 1);
         row <= std::min(elevation_count_ - 1, elevation + 1); ++row) {
      for (const int column : {previous, azimuth, next}) {
        const auto neighbor = depths.find(index(column, row));
        if (neighbor != depths.end() && neighbor->second >= min_depth &&
            neighbor->second <= max_depth) return true;
      }
    }
    return false;
  }

  std::pair<int, int> bins(const Eigen::Vector3d& point) const {
    return {azimuth_(point.y(), point.x()), elevation_.elevation(point.z(), point.x(), point.y())};
  }

  uint64_t index(int azimuth, int elevation) const {
    return static_cast<uint64_t>(elevation) * azimuth_count_ + azimuth;
  }

  int azimuth_count_;
  int elevation_count_;
  visibility_detail::AngularBinLookup azimuth_;
  visibility_detail::AngularBinLookup elevation_;
};

}

ContinuationCleaning computeContinuationCleaning(
    const Config& config, const std::vector<Keyframe::Ptr>& keyframes) {
  const auto min_views = static_cast<std::size_t>(config.continuation_penetration_min_views);
  const double min_range = visibilityOptionsFromConfig(config).range_min_m;
  const double max_range = config.continuation_penetration_max_range_m;
  const double max_range_squared = max_range * max_range;
  const double margin = config.continuation_penetration_margin_m;

  struct View {
    Keyframe::Ptr frame;
    Eigen::Isometry3d sensor_from_map;
    std::optional<SphericalDepthBins::Depths> depths;
  };
  std::vector<View> views;
  for (const auto& frame : keyframes) {
    if (!frame->is_from_loaded_map && frame->isValid())
      views.push_back({frame, frame->pose_optimized.inverse(), std::nullopt});
  }
  ContinuationCleaning result;
  if (views.size() < min_views) return result;

  const SphericalDepthBins depth_bins(config.continuation_penetration_resolution_deg);
  for (const auto& frame : keyframes) {
    if (!frame->is_from_loaded_map || !frame->isValid()) continue;
    Eigen::AlignedBox3d bounds;
    for (const auto& point : *frame->cloud)
      bounds.extend(frame->pose_optimized * point.getVector3fMap().cast<double>());
    std::vector<std::pair<View*, Eigen::Isometry3d>> overlapping;
    for (auto& view : views) {
      if (bounds.squaredExteriorDistance(view.frame->pose_optimized.translation()) <= max_range_squared)
        overlapping.emplace_back(&view, view.sensor_from_map * frame->pose_optimized);
    }
    if (overlapping.size() < min_views) continue;
    ++result.inspected_keyframes;
    for (const auto& [view, transform] : overlapping) {
      if (!view->depths) view->depths = depth_bins.project(*view->frame->cloud, min_range, max_range);
    }

    Keyframe::PointCloudT::Ptr kept;
    for (auto point = frame->cloud->begin(); point != frame->cloud->end(); ++point) {
      std::size_t penetrated_views = 0;
      for (const auto& [view, transform] : overlapping) {
        const Eigen::Vector3d position = transform * point->getVector3fMap().cast<double>();
        const double range = position.norm();
        if (range >= min_range && range <= max_range &&
            depth_bins.seesBehind(*view->depths, position, range + margin) &&
            ++penetrated_views == min_views) break;
      }
      const bool observed_surface = penetrated_views == min_views &&
          std::any_of(overlapping.begin(), overlapping.end(), [&](const auto& candidate) {
            const auto& [view, transform] = candidate;
            const Eigen::Vector3d position = transform * point->getVector3fMap().cast<double>();
            const double range = position.norm();
            return range >= min_range && range <= max_range &&
                depth_bins.seesSurface(*view->depths, position, range, margin);
          });
      if (penetrated_views < min_views || observed_surface) {
        if (kept) kept->push_back(*point);
      } else if (!kept) {
        kept = pcl::make_shared<Keyframe::PointCloudT>();
        kept->reserve(frame->cloud->size());
        kept->insert(kept->end(), frame->cloud->begin(), point);
        kept->header = frame->cloud->header;
        kept->sensor_origin_ = frame->cloud->sensor_origin_;
        kept->sensor_orientation_ = frame->cloud->sensor_orientation_;
        kept->is_dense = frame->cloud->is_dense;
      }
    }
    if (!kept || kept->empty()) {
      result.kept_points += frame->cloud->size();
      if (kept) {
        ++result.skipped_empty_keyframes;
        LOG(WARNING) << "[ContinuationCleaning] keeping fully penetrated keyframe=" << frame->id;
      }
      continue;
    }
    result.removed_points += frame->cloud->size() - kept->size();
    result.kept_points += kept->size();
    result.clouds.emplace(frame->id, std::move(kept));
  }
  return result;
}

}
