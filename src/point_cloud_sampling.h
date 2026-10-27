#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

namespace n3mapping::registration_detail {
// Sampling writes coordinates only. Allocate normal/covariance arrays later,
// at the retained size, when feature estimation actually needs them.
struct SamplingCloud : small_gicp::PointCloud {};
} // namespace n3mapping::registration_detail

namespace small_gicp::traits {
template <>
struct Traits<n3mapping::registration_detail::SamplingCloud> : Traits<PointCloud> {
  static void resize(n3mapping::registration_detail::SamplingCloud& cloud, size_t n) {
    cloud.points.resize(n);
  }
};
} // namespace small_gicp::traits

namespace n3mapping::registration_detail {
inline small_gicp::PointCloud::Ptr sampleCoordinates(
    const small_gicp::PointCloud& cloud, double resolution, int threads) {
  auto result = small_gicp::voxelgrid_sampling_omp<
      small_gicp::PointCloud, SamplingCloud>(cloud, resolution, threads);
  result->points.shrink_to_fit();
  return result;
}
// Prepared finite coordinates only. Keep an existing coarse point per cell,
// so this sampling step cannot add a centroid between surfaces. Coordinate ties
// are deterministic even when parallel preparation changes the input order.
inline small_gicp::PointCloud::Ptr sampleRepresentativeCoordinates(
    const small_gicp::PointCloud& cloud, double resolution) {
  using Cell = std::array<int, 3>;
  std::vector<std::pair<Cell, std::size_t>> cells;
  cells.reserve(cloud.size());
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const auto& point = cloud.point(i);
    cells.push_back({{static_cast<int>(std::floor(point.x() / resolution)),
                      static_cast<int>(std::floor(point.y() / resolution)),
                      static_cast<int>(std::floor(point.z() / resolution))}, i});
  }
  std::sort(cells.begin(), cells.end());
  auto points = std::make_shared<small_gicp::PointCloud>();
  points->points.reserve(cells.size());
  for (std::size_t first = 0; first < cells.size();) {
    const auto& cell = cells[first].first;
    const Eigen::Vector4d center((cell[0] + 0.5) * resolution,
        (cell[1] + 0.5) * resolution, (cell[2] + 0.5) * resolution, 1.0);
    auto best = cloud.point(cells[first].second);
    double distance = (best - center).squaredNorm();
    auto end = first + 1;
    for (; end < cells.size() && cells[end].first == cell; ++end) {
      const auto& point = cloud.point(cells[end].second);
      const double next_distance = (point - center).squaredNorm();
      if (next_distance < distance || (next_distance == distance &&
          std::array<double, 3>{point.x(), point.y(), point.z()} <
              std::array<double, 3>{best.x(), best.y(), best.z()})) {
        best = point;
        distance = next_distance;
      }
    }
    points->points.push_back(best);
    first = end;
  }
  points->points.shrink_to_fit();
  return points;
}
} // namespace n3mapping::registration_detail
