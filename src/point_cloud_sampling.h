#pragma once
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
} // namespace n3mapping::registration_detail
