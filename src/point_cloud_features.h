// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
// Adapted from small_gicp::estimate_local_features: retain its search,
// accumulation and eigen decomposition, but allocate scratch once per worker.
#pragma once
#include <small_gicp/util/normal_estimation.hpp>
#include <type_traits>

namespace n3mapping::registration_detail {
template <typename Setter, typename Tree>
void estimateFeatures(small_gicp::PointCloud& cloud, Tree& tree,
                      int neighbors, int threads) {
  if constexpr (!std::is_same_v<Setter, small_gicp::CovarianceSetter<small_gicp::PointCloud>>)
    cloud.normals.resize(cloud.size());
  if constexpr (!std::is_same_v<Setter, small_gicp::NormalSetter<small_gicp::PointCloud>>)
    cloud.covs.resize(cloud.size());
#pragma omp parallel num_threads(threads)
  {
    std::vector<size_t> indices(neighbors);
    std::vector<double> distances(neighbors);
#pragma omp for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(cloud.size()); ++i) {
      const auto n = tree.knn_search(cloud.point(i), neighbors,
                                    indices.data(), distances.data());
      if (n < 5) { Setter::set_invalid(cloud, i); continue; }
      Eigen::Vector4d sum = Eigen::Vector4d::Zero();
      Eigen::Matrix4d square = Eigen::Matrix4d::Zero();
      for (size_t k = 0; k < n; ++k) {
        const auto& point = cloud.point(indices[k]);
        sum += point;
        square += point * point.transpose();
      }
      const Eigen::Vector4d mean = sum / n;
      const Eigen::Matrix4d covariance = (square - mean * sum.transpose()) / n;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen;
      eigen.computeDirect(covariance.block<3, 3>(0, 0));
      Setter::set(cloud, i, eigen.eigenvectors());
    }
  }
}
} // namespace n3mapping::registration_detail
