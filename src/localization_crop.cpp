#include "n3mapping/localization_atlas.h"
#include "mapped_target.h"
#include "spatial_point_blocks.h"
#include <algorithm>
#include <limits>
#include <pcl/common/point_tests.h>

namespace n3mapping {
namespace {
template <typename Append>
void cropPoints(const Config &config, const LocalizationAtlas::PointCloudT::Ptr &map,
                const Eigen::Vector3d &center, const SpatialPointBlocks* blocks, Append append) {
  if (!map || map->empty() || !center.allFinite()) return;
  const double radius = std::max(1.0, config.rhpd_max_range) +
                        std::max(0.0, config.gicp_max_correspondence_distance);
  const double squared_radius = radius * radius;
  const auto test = [&](std::size_t i) {
    const auto &point = (*map)[i];
    const Eigen::Vector3d position(point.x, point.y, point.z);
    if ((position - center).squaredNorm() <= squared_radius) append(i);
  };
  if (blocks) blocks->forEachCandidate(center, squared_radius, test);
  else for (std::size_t i = 0; i < map->size(); ++i) test(i);
}
} // namespace

void LocalizationAtlas::prefetchTarget(
    const PointCloudMatcher::PreparedSource& source,
    const std::vector<Eigen::Isometry3d>& poses) {
  if (!mapping_ || source.plane_levels.empty() || source.plane_levels.front().cloud->empty())
    return;
  Eigen::AlignedBox3d observed;
  for (const auto& point : source.plane_levels.front().cloud->points)
    observed.extend(point.head<3>());
  std::vector<Eigen::AlignedBox3d> regions;
  const double margin = config_.gicp_max_correspondence_distance *
                        std::max(1.0, config_.reloc_track_retry_corr_scale);
  for (const auto& pose : poses) {
    Eigen::AlignedBox3d region;
    for (int corner = 0; corner < 8; ++corner) {
      Eigen::Vector3d point;
      for (int axis = 0; axis < 3; ++axis)
        point[axis] = (corner & (1 << axis)) ? observed.max()[axis] : observed.min()[axis];
      region.extend(pose * point);
    }
    region.min().array() -= margin;
    region.max().array() += margin;
    regions.push_back(region);
  }
  mapping_->prefetch(regions);
}

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::cropGlobalMap(
    const Config &config, const PointCloudT::Ptr &map, const Eigen::Vector3d &center,
    const SpatialPointBlocks* blocks) {
  auto target = pcl::make_shared<PointCloudT>();
  if (!map || map->empty() || !center.allFinite()) return target;
  cropPoints(config, map, center, blocks, [&](std::size_t i) { target->push_back((*map)[i]); });
  target->width = static_cast<std::uint32_t>(target->size());
  target->height = 1;
  target->is_dense = true;
  return target;
}

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::cropVisibilityTarget(
    const Eigen::Vector3d &center) const {
  return cropGlobalMap(config_, global_map_, center, blocks_.get());
}

std::shared_ptr<const std::vector<std::uint32_t>>
LocalizationAtlas::cropVisibilityIndices(const Eigen::Vector3d &center) const {
  if (!global_map_ || global_map_->size() > std::numeric_limits<std::uint32_t>::max())
    return {}; // Caller uses the full-width copied crop for oversized maps.
  auto indices = std::make_shared<std::vector<std::uint32_t>>();
  cropPoints(config_, global_map_, center, blocks_.get(),
             [&](std::size_t i) { indices->push_back(static_cast<std::uint32_t>(i)); });
  return indices;
}
} // namespace n3mapping
