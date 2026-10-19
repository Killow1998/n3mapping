#include "n3mapping/localization_atlas.h"
#include <algorithm>
#include <limits>
#include <pcl/common/point_tests.h>

namespace n3mapping {
namespace {
template <typename Append>
void cropPoints(const Config &config, const LocalizationAtlas::PointCloudT::Ptr &map,
                const Eigen::Vector3d &center, Append append) {
  if (!map || map->empty() || !center.allFinite()) return;
  const double radius = std::max(1.0, config.rhpd_max_range) +
                        std::max(0.0, config.gicp_max_correspondence_distance);
  const double squared_radius = radius * radius;
  for (std::size_t i = 0; i < map->size(); ++i) {
    const auto &point = (*map)[i];
    if (!pcl::isFinite(point)) continue;
    const Eigen::Vector3d position(point.x, point.y, point.z);
    if ((position - center).squaredNorm() <= squared_radius) append(i);
  }
}
} // namespace

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::cropGlobalMap(
    const Config &config, const PointCloudT::Ptr &map, const Eigen::Vector3d &center) {
  auto target = pcl::make_shared<PointCloudT>();
  if (!map || map->empty() || !center.allFinite()) return target;
  cropPoints(config, map, center, [&](std::size_t i) { target->push_back((*map)[i]); });
  target->width = static_cast<std::uint32_t>(target->size());
  target->height = 1;
  target->is_dense = true;
  return target;
}

LocalizationAtlas::PointCloudT::Ptr LocalizationAtlas::cropVisibilityTarget(
    const Eigen::Vector3d &center) const {
  return cropGlobalMap(config_, global_map_, center);
}

std::shared_ptr<const std::vector<std::uint32_t>>
LocalizationAtlas::cropVisibilityIndices(const Eigen::Vector3d &center) const {
  if (!global_map_ || global_map_->size() > std::numeric_limits<std::uint32_t>::max())
    return {}; // Caller uses the full-width copied crop for oversized maps.
  auto indices = std::make_shared<std::vector<std::uint32_t>>();
  cropPoints(config_, global_map_, center,
             [&](std::size_t i) { indices->push_back(static_cast<std::uint32_t>(i)); });
  return indices;
}
} // namespace n3mapping
