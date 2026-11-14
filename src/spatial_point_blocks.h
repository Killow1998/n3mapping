#pragma once
#include <algorithm>
#include <limits>
#include <vector>
#include <Eigen/Geometry>
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
// Spatial rejection over immutable, consecutive raw points preserves crop order.
class SpatialPointBlocks {
public:
  explicit SpatialPointBlocks(const PointCloudMatcher::PointCloudT& cloud)
      : points_(cloud.size()), boxes_((points_ + kPoints - 1) / kPoints) {
    for (std::size_t block = 0; block < boxes_.size(); ++block) {
      auto& box = boxes_[block];
      box.setEmpty();
      for (auto i = block * kPoints; i < std::min(points_, (block + 1) * kPoints); ++i)
        box.extend(cloud[i].getVector3fMap());
    }
  }
  template <class Visitor>
  void forEachCandidate(const Eigen::Vector3d& center, double radius_squared,
                        Visitor visit) const {
    const double guard = 16 * std::numeric_limits<double>::epsilon() * (1 + radius_squared);
    for (std::size_t block = 0; block < boxes_.size(); ++block) {
      const auto& box = boxes_[block];
      double lower = 0;
      for (int k = 0; k < 3; ++k) {
        const auto d = std::max({0.0, double(box.min()[k]) - center[k],
                                 center[k] - double(box.max()[k])});
        lower += d * d;
      }
      if (lower > radius_squared + guard) continue;
      for (auto i = block * kPoints; i < std::min(points_, (block + 1) * kPoints); ++i)
        visit(i);
    }
  }
private:
  static constexpr std::size_t kPoints = 512;
  std::size_t points_;
  std::vector<Eigen::AlignedBox3f> boxes_;
};
} // namespace n3mapping
